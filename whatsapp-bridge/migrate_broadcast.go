package main

import (
	"database/sql"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/blevesearch/bleve/v2"
)

// RunMigrateBroadcastChats moves messages that were stored under a broadcast
// list's JID (before directChatJID existed) into the sender's DM chat, removes
// the list chats and their search docs, and renames chats whose name is just
// their own number when a contact name is now resolvable. It is idempotent: a
// second run finds nothing to move and leaves everything as it is.
//
// Run it with the service stopped. bleve holds an exclusive lock on the index.
func RunMigrateBroadcastChats(store *MessageStore) error {
	rows, err := store.db.Query(`SELECT jid FROM chats WHERE jid LIKE '%@broadcast' AND jid <> 'status@broadcast'`)
	if err != nil {
		return fmt.Errorf("list broadcast chats: %w", err)
	}
	var lists []string
	for rows.Next() {
		var jid string
		if err := rows.Scan(&jid); err == nil {
			lists = append(lists, jid)
		}
	}
	rows.Close()

	var moved, deduped, skipped, emptied int
	var targets []string
	for _, list := range lists {
		res, err := migrateBroadcastChat(store, list)
		if err != nil {
			logger.Warnf("migrate %s: %v", list, err)
			skipped++
			continue
		}
		switch {
		case res.skipReason != "":
			logger.Warnf("migrate %s: skipped, %s", list, res.skipReason)
			skipped++
		case res.target == "":
			logger.Infof("migrate %s: nothing left to move (%d already in DM chat), chat deleted", list, res.deduped)
			deduped += res.deduped
			emptied++
		default:
			logger.Infof("migrate %s → %s (%q): %d moved, %d already in DM chat", list, res.target, res.name, res.moved, res.deduped)
			moved += res.moved
			deduped += res.deduped
			targets = append(targets, res.target)
		}
	}

	// Search docs. The delete matches on the index itself rather than on the
	// chats just migrated, so a rerun after a crash between the SQL commit and
	// this step still removes what was left behind. status@broadcast doesn't
	// match the pattern.
	listDocs := bleve.NewRegexpQuery(`[0-9]+@broadcast`)
	listDocs.SetField("chat_jid")
	docsDeleted, err := deleteDocsMatching(store.index, listDocs)
	if err != nil {
		return fmt.Errorf("delete broadcast-list search docs: %w", err)
	}
	for _, target := range targets {
		if err := reIndexAllMessages(store, 0, target); err != nil {
			logger.Warnf("reindex %s failed: %v (rerun with --reindex %s)", target, err, target)
		}
	}

	renamed, err := renameRawNumberChats(store.db)
	if err != nil {
		return fmt.Errorf("rename chats: %w", err)
	}

	logger.Infof("Broadcast migration done: %d messages moved, %d duplicates dropped, %d chats deleted with nothing to move, %d chats skipped, %d search docs deleted, %d chats renamed, targets reindexed: %v",
		moved, deduped, emptied, skipped, docsDeleted, renamed, targets)
	return nil
}

type broadcastMigration struct {
	target, name   string
	moved, deduped int
	skipReason     string
}

func migrateBroadcastChat(store *MessageStore, list string) (broadcastMigration, error) {
	var res broadcastMigration

	// All queries below go through tx: the store holds a single connection
	// (it's what keeps wdb attached), so a store.db call here would block.
	tx, err := store.db.Begin()
	if err != nil {
		return res, err
	}
	defer tx.Rollback()

	var senders []string
	var fromMe bool
	rows, err := tx.Query(`SELECT DISTINCT sender, is_from_me FROM messages WHERE chat_jid = ?`, list)
	if err != nil {
		return res, err
	}
	for rows.Next() {
		var s string
		var me bool
		if err := rows.Scan(&s, &me); err != nil {
			rows.Close()
			return res, err
		}
		senders = append(senders, s)
		fromMe = fromMe || me
	}
	rows.Close()

	switch {
	case len(senders) == 0:
		if _, err := tx.Exec(`DELETE FROM chats WHERE jid = ?`, list); err != nil {
			return res, err
		}
		return res, tx.Commit()
	case fromMe:
		// One of our own lists: its messages really do belong to the list chat.
		res.skipReason = "contains our own messages"
		return res, nil
	case len(senders) > 1:
		res.skipReason = fmt.Sprintf("more than one sender (%v)", senders)
		return res, nil
	}
	sender := senders[0]

	// The target is the sender's DM chat, in the same form live DMs use: LID
	// when the sender is a LID, otherwise the phone-number JID. twin is the same
	// person's other-form chat, which can already hold a copy of a message.
	var target, twin string
	var pn string
	switch err := tx.QueryRow(`SELECT pn FROM wdb.whatsmeow_lid_map WHERE lid = ?`, sender).Scan(&pn); {
	case err == nil:
		target, twin = sender+"@lid", pn+"@s.whatsapp.net"
	case errors.Is(err, sql.ErrNoRows):
		var n int
		if err := tx.QueryRow(`SELECT COUNT(*) FROM wdb.whatsmeow_contacts WHERE their_jid = ?`, sender+"@s.whatsapp.net").Scan(&n); err != nil {
			return res, err
		}
		if n == 0 {
			res.skipReason = fmt.Sprintf("sender %s is neither a known LID nor a known contact", sender)
			return res, nil
		}
		target = sender + "@s.whatsapp.net"
		var lid string
		if err := tx.QueryRow(`SELECT lid FROM wdb.whatsmeow_lid_map WHERE pn = ?`, sender).Scan(&lid); err == nil {
			twin = lid + "@lid"
		}
	default:
		return res, err
	}

	r, err := tx.Exec(`DELETE FROM messages WHERE chat_jid = ? AND id IN (SELECT id FROM messages WHERE chat_jid IN (?, ?))`, list, target, twin)
	if err != nil {
		return res, err
	}
	n, _ := r.RowsAffected()
	res.deduped = int(n)

	var left int
	if err := tx.QueryRow(`SELECT COUNT(*) FROM messages WHERE chat_jid = ?`, list).Scan(&left); err != nil {
		return res, err
	}
	if left == 0 {
		// Every message was already in the DM chat: nothing to move, and no
		// reason to create an empty target chat.
		if _, err := tx.Exec(`DELETE FROM chats WHERE jid = ?`, list); err != nil {
			return res, err
		}
		return res, tx.Commit()
	}

	// messages.chat_jid references chats(jid), so the target row has to exist
	// before the move. Its name and time are set properly below.
	if _, err := tx.Exec(`INSERT OR IGNORE INTO chats (jid, name) VALUES (?, ?)`, target, sender); err != nil {
		return res, err
	}
	r, err = tx.Exec(`UPDATE messages SET chat_jid = ? WHERE chat_jid = ?`, target, list)
	if err != nil {
		return res, err
	}
	n, _ = r.RowsAffected()
	res.moved = int(n)

	name, err := storedContactName(tx, target)
	if err != nil {
		return res, err
	}
	if name == "" {
		// Best sender name recorded on the messages themselves.
		_ = tx.QueryRow(`SELECT full_name FROM messages WHERE chat_jid = ? AND full_name != '' AND full_name != sender
			ORDER BY timestamp DESC LIMIT 1`, target).Scan(&name)
	}
	if name == "" {
		name = sender
	}
	if _, err := tx.Exec(`
		INSERT INTO chats (jid, name, last_message_time)
		VALUES (?, ?, (SELECT MAX(timestamp) FROM messages WHERE chat_jid = ?))
		ON CONFLICT(jid) DO UPDATE SET name = excluded.name, last_message_time = excluded.last_message_time`,
		target, name, target); err != nil {
		return res, err
	}
	if _, err := tx.Exec(`DELETE FROM chats WHERE jid = ?`, list); err != nil {
		return res, err
	}
	if err := tx.Commit(); err != nil {
		return res, err
	}
	res.target, res.name = target, name

	if err := moveMediaDir(list, target); err != nil {
		logger.Warnf("move media %s → %s: %v", list, target, err)
	}
	return res, nil
}

// storedContactName is contactDisplayName over the attached whatsmeow tables:
// address-book name first, a @lid resolved to its phone-number contact, then
// push names. Returns "" when nothing is stored.
func storedContactName(q interface {
	QueryRow(string, ...any) *sql.Row
}, jid string) (string, error) {
	user, server, _ := strings.Cut(jid, "@")
	if server != "lid" && server != "s.whatsapp.net" {
		return "", nil
	}
	pnJID := ""
	if server == "lid" {
		var pn string
		if err := q.QueryRow(`SELECT pn FROM wdb.whatsmeow_lid_map WHERE lid = ?`, user).Scan(&pn); err == nil {
			pnJID = pn + "@s.whatsapp.net"
		}
	}
	var name string
	err := q.QueryRow(`
		SELECT COALESCE(
			(SELECT NULLIF(full_name, '') FROM wdb.whatsmeow_contacts WHERE their_jid = ?1),
			(SELECT NULLIF(full_name, '') FROM wdb.whatsmeow_contacts WHERE their_jid = ?2),
			(SELECT NULLIF(push_name, '') FROM wdb.whatsmeow_contacts WHERE their_jid = ?1),
			(SELECT NULLIF(push_name, '') FROM wdb.whatsmeow_contacts WHERE their_jid = ?2),
			'')`,
		jid, pnJID).Scan(&name)
	return name, err
}

// renameRawNumberChats gives a contact name to DM chats still named after
// their own number, wherever one is now resolvable.
func renameRawNumberChats(db *sql.DB) (int, error) {
	rows, err := db.Query(`SELECT jid, name FROM chats
		WHERE (jid LIKE '%@lid' OR jid LIKE '%@s.whatsapp.net')
		  AND name = substr(jid, 1, instr(jid, '@') - 1)`)
	if err != nil {
		return 0, err
	}
	type chat struct{ jid, name string }
	var chats []chat
	for rows.Next() {
		var c chat
		if err := rows.Scan(&c.jid, &c.name); err == nil {
			chats = append(chats, c)
		}
	}
	rows.Close()

	renamed := 0
	for _, c := range chats {
		name, err := storedContactName(db, c.jid)
		if err != nil {
			return renamed, err
		}
		if name == "" || name == c.name {
			continue
		}
		if _, err := db.Exec(`UPDATE chats SET name = ? WHERE jid = ?`, name, c.jid); err != nil {
			return renamed, err
		}
		logger.Infof("renamed %s: %q → %q", c.jid, c.name, name)
		renamed++
	}
	return renamed, nil
}

// moveMediaDir moves downloaded media from one chat's directory to another's,
// never overwriting a file already there.
func moveMediaDir(fromJID, toJID string) error {
	src := filepath.Join("store", strings.ReplaceAll(fromJID, ":", "_"))
	dst := filepath.Join("store", strings.ReplaceAll(toJID, ":", "_"))
	entries, err := os.ReadDir(src)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dst, 0755); err != nil {
		return err
	}
	for _, e := range entries {
		to := filepath.Join(dst, e.Name())
		if _, err := os.Stat(to); err == nil {
			logger.Warnf("media %s already exists, leaving %s in place", to, filepath.Join(src, e.Name()))
			continue
		}
		if err := os.Rename(filepath.Join(src, e.Name()), to); err != nil {
			return err
		}
	}
	_ = os.Remove(src) // stays when a file was left behind above
	return nil
}
