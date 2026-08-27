package main

import (
	"database/sql"
	"sync"
	"time"
)

// BroadcastMessage is the payload sent to WebSocket subscribers for each
// incoming WhatsApp message.
type BroadcastMessage struct {
	ChatJID  string        `json:"chat_jid"`
	ChatName string        `json:"chat_name"`
	Message  MessageWithID `json:"message"`
}

// subscriber holds a WebSocket client's channel and its JID filter.
// An empty jids slice means no filtering — all messages are delivered.
type subscriber struct {
	ch   chan BroadcastMessage
	jids []string
}

// MessageBroadcaster fan-outs incoming messages to all connected WebSocket
// clients. It is safe for concurrent use.
type MessageBroadcaster struct {
	clients map[*subscriber]struct{}
	mu      sync.RWMutex
}

// NewMessageBroadcaster creates an empty broadcaster.
func NewMessageBroadcaster() *MessageBroadcaster {
	return &MessageBroadcaster{
		clients: make(map[*subscriber]struct{}),
	}
}

// Subscribe registers a new subscriber and returns its receive channel.
// If jids is non-empty, only messages matching those JIDs are delivered.
// The caller must call Unsubscribe when done to avoid a goroutine/channel leak.
func (b *MessageBroadcaster) Subscribe(jids []string) chan BroadcastMessage {
	ch := make(chan BroadcastMessage, 64)
	b.mu.Lock()
	b.clients[&subscriber{ch: ch, jids: jids}] = struct{}{}
	b.mu.Unlock()
	return ch
}

// Unsubscribe removes the channel from the subscriber map and closes it.
// The delete happens before the close so that a concurrent Broadcast call
// (holding only a read-lock over the same map snapshot) will never see the
// already-closed channel.
func (b *MessageBroadcaster) Unsubscribe(ch chan BroadcastMessage) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for sub := range b.clients {
		if sub.ch == ch {
			delete(b.clients, sub)
			break
		}
	}
	close(ch)
}

// jidMatches checks if msg.ChatJID matches any of the subscriber's JID filters.
// If the subscriber has no filters (empty slice), all messages match.
func (b *MessageBroadcaster) jidMatches(sub *subscriber, msg BroadcastMessage) bool {
	if len(sub.jids) == 0 {
		return true
	}
	for _, jid := range sub.jids {
		if jid == msg.ChatJID {
			return true
		}
	}
	return false
}

// Broadcast delivers msg to all current subscribers. Sends are non-blocking:
// if a subscriber's buffer is full the message is dropped for that client so
// that a slow consumer never stalls the WhatsApp event handler goroutine.
func (b *MessageBroadcaster) Broadcast(msg BroadcastMessage) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	subscriberCount := len(b.clients)
	logger.Infof("Broadcast: chat=%q jid=%s subscribers=%d", msg.ChatName, msg.ChatJID, subscriberCount)

	delivered := 0
	dropped := 0
	filtered := 0
	for sub := range b.clients {
		if !b.jidMatches(sub, msg) {
			filtered++
			continue
		}
		select {
		case sub.ch <- msg:
			delivered++
		default:
			dropped++
			logger.Warnf("Broadcast dropped: chat=%q subscriber buffer full", msg.ChatName)
		}
	}
	logger.Infof("Broadcast done: chat=%q delivered=%d dropped=%d filtered=%d", msg.ChatName, delivered, dropped, filtered)
}

// ClientRegistry persists the last message timestamp delivered to each named
// WebSocket client, tracked per chat JID. It survives server restarts via the
// shared SQLite database.
//
// Unfiltered clients (no jids query param) use the empty string "" as their
// JID bucket: a single global cursor is correct for them because they want
// every message, not a per-chat replay window. Filtered clients get one
// cursor per subscribed JID, so that catch-up for one chat is never skipped
// because a different chat advanced the client's marker first.
type ClientRegistry struct {
	db    *sql.DB
	mu    sync.Mutex
	cache map[string]map[string]time.Time // client_name -> chat_jid -> last_seen
}

// NewClientRegistry migrates the client_last_seen table to its per-JID
// schema if needed, creates it if missing, loads all existing rows into the
// in-memory cache, and returns a ready registry.
func NewClientRegistry(db *sql.DB) (*ClientRegistry, error) {
	if err := migrateClientLastSeenTable(db); err != nil {
		return nil, err
	}

	_, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS client_last_seen (
			client_name TEXT NOT NULL,
			chat_jid    TEXT NOT NULL DEFAULT '',
			last_seen   TIMESTAMP NOT NULL,
			PRIMARY KEY (client_name, chat_jid)
		)`)
	if err != nil {
		return nil, err
	}

	rows, err := db.Query("SELECT client_name, chat_jid, last_seen FROM client_last_seen")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	cache := make(map[string]map[string]time.Time)
	for rows.Next() {
		var name, jid string
		var t time.Time
		if err := rows.Scan(&name, &jid, &t); err != nil {
			return nil, err
		}
		if cache[name] == nil {
			cache[name] = make(map[string]time.Time)
		}
		cache[name][jid] = t
	}

	return &ClientRegistry{db: db, cache: cache}, nil
}

// migrateClientLastSeenTable upgrades a pre-existing client_last_seen table
// (PRIMARY KEY client_name, one global cursor per client) to the per-JID
// schema, preserving old rows as each client's "" (unfiltered) cursor.
// No-op if the table doesn't exist yet or already has the new schema.
func migrateClientLastSeenTable(db *sql.DB) error {
	rows, err := db.Query(`PRAGMA table_info(client_last_seen)`)
	if err != nil {
		return err
	}
	exists := false
	hasChatJID := false
	for rows.Next() {
		exists = true
		var cid int
		var name, ctype string
		var notnull, pk int
		var dflt sql.NullString
		if err := rows.Scan(&cid, &name, &ctype, &notnull, &dflt, &pk); err != nil {
			rows.Close()
			return err
		}
		if name == "chat_jid" {
			hasChatJID = true
		}
	}
	rows.Close()
	if !exists || hasChatJID {
		return nil
	}

	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.Exec(`ALTER TABLE client_last_seen RENAME TO client_last_seen_old`); err != nil {
		return err
	}
	if _, err := tx.Exec(`
		CREATE TABLE client_last_seen (
			client_name TEXT NOT NULL,
			chat_jid    TEXT NOT NULL DEFAULT '',
			last_seen   TIMESTAMP NOT NULL,
			PRIMARY KEY (client_name, chat_jid)
		)`); err != nil {
		return err
	}
	if _, err := tx.Exec(`
		INSERT INTO client_last_seen (client_name, chat_jid, last_seen)
		SELECT client_name, '', last_seen FROM client_last_seen_old`); err != nil {
		return err
	}
	if _, err := tx.Exec(`DROP TABLE client_last_seen_old`); err != nil {
		return err
	}
	return tx.Commit()
}

// GetLastSeen returns the last timestamp recorded for the given client name
// and JID bucket ("" for an unfiltered client's global cursor). Returns
// false if that (name, jid) pair has never been recorded.
func (r *ClientRegistry) GetLastSeen(name, jid string) (time.Time, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	t, ok := r.cache[name][jid]
	return t, ok
}

// UpdateLastSeen records t as the new last-seen timestamp for (name, jid),
// but only if t is strictly after the current value. Writes through to SQLite.
func (r *ClientRegistry) UpdateLastSeen(name, jid string, t time.Time) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if existing, ok := r.cache[name][jid]; ok && !t.After(existing) {
		return nil
	}
	if r.cache[name] == nil {
		r.cache[name] = make(map[string]time.Time)
	}
	r.cache[name][jid] = t
	_, err := r.db.Exec(`
		INSERT INTO client_last_seen (client_name, chat_jid, last_seen) VALUES (?, ?, ?)
		ON CONFLICT(client_name, chat_jid) DO UPDATE SET last_seen = excluded.last_seen
		WHERE excluded.last_seen > client_last_seen.last_seen`,
		name, jid, t)
	return err
}
