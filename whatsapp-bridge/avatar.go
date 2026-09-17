package main

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/types"
)

// avatarFreshness is how long a cached avatar (including a cached "no
// picture"/"unauthorized" fact) is trusted with zero contact to WhatsApp.
// Profile pictures change far less often than messages, so this can be
// generous: at most 4 revalidation round-trips per contact per day even
// under continuous traffic, while still picking up a changed photo the same
// day it changes. Past this window we still don't do a full re-download —
// see refreshAvatar, which revalidates via ExistingID and only pulls image
// bytes again if WhatsApp reports the photo actually changed.
const avatarFreshness = 6 * time.Hour

// avatarBrowserMaxAge is what /api/contacts/{jid}/avatar tells browsers via
// Cache-Control. Deliberately shorter than avatarFreshness: once a browser's
// own cache entry expires, its next request carries If-None-Match, which
// lands well inside our server-side freshness window and gets answered with
// a 304 straight from our SQLite cache — the common case never reaches
// WhatsApp at all. `private` because this is a single-consumer image, not
// something an intermediate shared cache should store.
const avatarBrowserMaxAge = 1 * time.Hour

// avatarFetchTimeout bounds both the whatsmeow round-trip and the CDN image
// download. Some JIDs are observed to hang on the full profile-picture
// fetch; without a bound that stalls the HTTP handler (and whoever is
// waiting on it) indefinitely. On timeout, a stale cache entry (if any) is
// served instead of failing the request outright — see makeGetAvatarHandler.
const avatarFetchTimeout = 10 * time.Second

// avatarDB is a dedicated SQLite store for cached profile-picture bytes,
// separate from messages.db/whatsapp.db. It holds derived, disposable data
// (nothing here can't be re-fetched from WhatsApp), so it gets its own file
// rather than a new table wedged into an existing schema.
var avatarDB *sql.DB

// avatarRecord is one cached avatar (or cached absence of one) for a JID.
type avatarRecord struct {
	JID           string
	PictureID     string // whatsmeow's stable picture id; "" when HasPicture is false
	ContentType   string
	Data          []byte
	HasPicture    bool
	FetchedAt     time.Time // when the stored bytes (or "no picture" fact) were last actually established
	RevalidatedAt time.Time // when we last confirmed the above is still current, even if nothing changed
}

func initAvatarDB() error {
	if err := os.MkdirAll("store", 0755); err != nil {
		return fmt.Errorf("failed to create store directory: %v", err)
	}

	db, err := sql.Open("sqlite3", "file:store/avatars.db?_journal_mode=WAL&_busy_timeout=5000")
	if err != nil {
		return fmt.Errorf("failed to open avatar database: %v", err)
	}
	// Single-connection discipline mirrors store.go/query.go. Not load-bearing
	// here the way it is for those (no ATTACH state to lose), but this store
	// is low-traffic and it costs nothing to keep one writer instead of
	// letting database/sql open a pool.
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)
	db.SetConnMaxLifetime(0)

	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS avatars (
			jid            TEXT PRIMARY KEY,
			picture_id     TEXT NOT NULL DEFAULT '',
			content_type   TEXT NOT NULL DEFAULT '',
			data           BLOB,
			has_picture    BOOLEAN NOT NULL DEFAULT 0,
			fetched_at     TIMESTAMP NOT NULL,
			revalidated_at TIMESTAMP NOT NULL
		);
	`)
	if err != nil {
		db.Close()
		return fmt.Errorf("failed to create avatars table: %v", err)
	}

	avatarDB = db
	return nil
}

func getAvatarRecord(jid string) (*avatarRecord, error) {
	row := avatarDB.QueryRow(`
		SELECT jid, picture_id, content_type, data, has_picture, fetched_at, revalidated_at
		FROM avatars WHERE jid = ?
	`, jid)
	var rec avatarRecord
	// mattn/go-sqlite3 converts a BOOLEAN-declared column straight to Go
	// bool (based on the declared column type, not the stored affinity), so
	// this must scan into a bool, not an int — an earlier version of this
	// scanned into int and failed on every row with "converting driver.Value
	// type bool (\"true\") to a int: invalid syntax".
	if err := row.Scan(&rec.JID, &rec.PictureID, &rec.ContentType, &rec.Data, &rec.HasPicture, &rec.FetchedAt, &rec.RevalidatedAt); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}
	return &rec, nil
}

func upsertAvatarRecord(rec *avatarRecord) error {
	_, err := avatarDB.Exec(`
		INSERT INTO avatars (jid, picture_id, content_type, data, has_picture, fetched_at, revalidated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(jid) DO UPDATE SET
			picture_id     = excluded.picture_id,
			content_type   = excluded.content_type,
			data           = excluded.data,
			has_picture    = excluded.has_picture,
			fetched_at     = excluded.fetched_at,
			revalidated_at = excluded.revalidated_at
	`, rec.JID, rec.PictureID, rec.ContentType, rec.Data, rec.HasPicture, rec.FetchedAt, rec.RevalidatedAt)
	return err
}

// touchAvatarRevalidated bumps revalidated_at only, for the cheap "WhatsApp
// confirmed nothing changed" path — the stored bytes/content-type/id are
// left untouched.
func touchAvatarRevalidated(jid string, at time.Time) error {
	_, err := avatarDB.Exec(`UPDATE avatars SET revalidated_at = ? WHERE jid = ?`, at, jid)
	return err
}

// refreshAvatar revalidates (or, if rec is nil, fully fetches) the avatar
// for jid. When rec already has a picture, this asks whatsmeow to confirm it
// via ExistingID — a cheap call that returns nil with no image data when the
// photo hasn't changed. Image bytes are only downloaded from the CDN when
// whatsmeow reports the photo is new/changed, or when there's no cache yet.
// A "this contact has no picture" (or "hidden from us") result is persisted
// too, so repeated requests don't re-hit WhatsApp every time.
func refreshAvatar(ctx context.Context, client *whatsmeow.Client, jidStr string, jid types.JID, rec *avatarRecord, now time.Time) (*avatarRecord, error) {
	ctx, cancel := context.WithTimeout(ctx, avatarFetchTimeout)
	defer cancel()

	params := &whatsmeow.GetProfilePictureParams{}
	if rec != nil && rec.HasPicture && rec.PictureID != "" {
		params.ExistingID = rec.PictureID
	}

	info, err := client.GetProfilePictureInfo(ctx, jid, params)
	if err != nil {
		if errors.Is(err, whatsmeow.ErrProfilePictureNotSet) || errors.Is(err, whatsmeow.ErrProfilePictureUnauthorized) {
			// No picture (or the contact hid it from us) — cache that fact
			// too so we don't ask WhatsApp again every request.
			newRec := &avatarRecord{JID: jidStr, HasPicture: false, FetchedAt: now, RevalidatedAt: now}
			if uerr := upsertAvatarRecord(newRec); uerr != nil {
				logger.Warnf("avatar: failed to persist no-picture cache for %s: %v", jidStr, uerr)
			}
			return newRec, nil
		}
		return rec, err
	}

	if info == nil {
		// Unchanged: WhatsApp confirmed ExistingID still matches. rec is
		// necessarily non-nil here (ExistingID was only set from rec).
		if terr := touchAvatarRevalidated(jidStr, now); terr != nil {
			logger.Warnf("avatar: failed to touch revalidation for %s: %v", jidStr, terr)
		}
		rec.RevalidatedAt = now
		return rec, nil
	}

	// New or changed photo — download the actual bytes.
	data, contentType, derr := fetchAvatarBytes(ctx, info.URL)
	if derr != nil {
		return rec, fmt.Errorf("fetched changed picture info but failed to download image: %w", derr)
	}
	newRec := &avatarRecord{
		JID:           jidStr,
		PictureID:     info.ID,
		ContentType:   contentType,
		Data:          data,
		HasPicture:    true,
		FetchedAt:     now,
		RevalidatedAt: now,
	}
	if uerr := upsertAvatarRecord(newRec); uerr != nil {
		return rec, fmt.Errorf("failed to persist avatar cache: %w", uerr)
	}
	return newRec, nil
}

// fetchAvatarBytes downloads the actual image from WhatsApp's CDN. The URL
// in ProfilePictureInfo is a plain, short-lived signed link that (per
// whatsmeow's own doc comment) "can be downloaded with a simple HTTP
// request" — no auth headers or whatsmeow-side decryption needed, unlike
// message media.
func fetchAvatarBytes(ctx context.Context, url string) ([]byte, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, "", err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("unexpected status %d fetching avatar image", resp.StatusCode)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", err
	}
	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "image/jpeg" // WhatsApp profile pictures are JPEGs in practice
	}
	return data, contentType, nil
}
