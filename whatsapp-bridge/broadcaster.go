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
// WebSocket client. It survives server restarts via the shared SQLite database.
type ClientRegistry struct {
	db    *sql.DB
	mu    sync.Mutex
	cache map[string]time.Time // write-through cache
}

// NewClientRegistry creates the client_last_seen table if needed, loads all
// existing rows into the in-memory cache, and returns a ready registry.
func NewClientRegistry(db *sql.DB) (*ClientRegistry, error) {
	_, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS client_last_seen (
			client_name TEXT PRIMARY KEY,
			last_seen   TIMESTAMP NOT NULL
		)`)
	if err != nil {
		return nil, err
	}

	rows, err := db.Query("SELECT client_name, last_seen FROM client_last_seen")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	cache := make(map[string]time.Time)
	for rows.Next() {
		var name string
		var t time.Time
		if err := rows.Scan(&name, &t); err != nil {
			return nil, err
		}
		cache[name] = t
	}

	return &ClientRegistry{db: db, cache: cache}, nil
}

// GetLastSeen returns the last timestamp recorded for the given client name.
// Returns false if the client has never connected before.
func (r *ClientRegistry) GetLastSeen(name string) (time.Time, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	t, ok := r.cache[name]
	return t, ok
}

// UpdateLastSeen records t as the new last-seen timestamp for name, but only
// if t is strictly after the current value. Writes through to SQLite.
func (r *ClientRegistry) UpdateLastSeen(name string, t time.Time) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if existing, ok := r.cache[name]; ok && !t.After(existing) {
		return nil
	}
	r.cache[name] = t
	_, err := r.db.Exec(`
		INSERT INTO client_last_seen (client_name, last_seen) VALUES (?, ?)
		ON CONFLICT(client_name) DO UPDATE SET last_seen = excluded.last_seen
		WHERE excluded.last_seen > client_last_seen.last_seen`,
		name, t)
	return err
}
