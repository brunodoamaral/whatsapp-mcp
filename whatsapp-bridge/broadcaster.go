package main

import (
	"database/sql"
	"sync"
	"time"

	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
)

// BroadcastMessage is the payload sent to WebSocket subscribers for each
// incoming WhatsApp message.
type BroadcastMessage struct {
	ChatJID  string        `json:"chat_jid"`
	ChatName string        `json:"chat_name"`
	Message  MessageWithID `json:"message"`
}

// TypingMessage is the payload sent to WebSocket subscribers who opted in
// (typing=true) for a chat-presence (typing/paused) update. JID identifies
// whoever's presence changed: the other party in the chat, or — when
// IsFromMe is true — one of the account's own other linked devices
// composing/pausing in that chat.
type TypingMessage struct {
	ChatJID  string `json:"chat_jid"`
	JID      string `json:"jid"`
	IsFromMe bool   `json:"is_from_me"`
	State    string `json:"state"` // "composing" or "paused"
}

// GroupInfoMessage is the payload sent to WebSocket subscribers who opted in
// (groupinfo=true) for a group metadata change: rename, topic, membership,
// or settings change. Only the field(s) that actually changed in a given
// event are non-nil/non-empty; everything else is omitted from the JSON.
type GroupInfoMessage struct {
	ChatJID   string    `json:"chat_jid"`
	Sender    string    `json:"sender,omitempty"`
	Timestamp time.Time `json:"timestamp"`

	Name                       *GroupNameChange      `json:"name,omitempty"`
	Topic                      *GroupTopicChange     `json:"topic,omitempty"`
	Locked                     *bool                 `json:"locked,omitempty"`
	Announce                   *bool                 `json:"announce,omitempty"`
	Ephemeral                  *GroupEphemeralChange `json:"ephemeral,omitempty"`
	MembershipApprovalRequired *bool                 `json:"membership_approval_required,omitempty"`
	Deleted                    *GroupDeleteChange    `json:"deleted,omitempty"`

	NewInviteLink *string  `json:"new_invite_link,omitempty"`
	Join          []string `json:"join,omitempty"`
	Leave         []string `json:"leave,omitempty"`
	Promote       []string `json:"promote,omitempty"`
	Demote        []string `json:"demote,omitempty"`
	Suspended     bool     `json:"suspended,omitempty"`
	Unsuspended   bool     `json:"unsuspended,omitempty"`
}

type GroupNameChange struct {
	Name  string `json:"name"`
	SetBy string `json:"set_by,omitempty"`
}

type GroupTopicChange struct {
	Topic   string `json:"topic"`
	Deleted bool   `json:"deleted,omitempty"`
	SetBy   string `json:"set_by,omitempty"`
}

type GroupEphemeralChange struct {
	Enabled                  bool   `json:"enabled"`
	DisappearingTimerSeconds uint32 `json:"disappearing_timer_seconds,omitempty"`
}

type GroupDeleteChange struct {
	Deleted bool   `json:"deleted"`
	Reason  string `json:"reason,omitempty"`
}

// buildGroupInfoMessage flattens a whatsmeow *events.GroupInfo into the wire
// format above, converting JIDs to strings (types.JID has no MarshalJSON).
// Link/Unlink/ParticipantVersionID/UnknownChanges are intentionally omitted —
// community group-linking plumbing and internal versioning, not meaningful
// to a downstream consumer.
func buildGroupInfoMessage(v *events.GroupInfo) GroupInfoMessage {
	msg := GroupInfoMessage{
		ChatJID:     v.JID.String(),
		Timestamp:   v.Timestamp,
		Join:        jidsToStrings(v.Join),
		Leave:       jidsToStrings(v.Leave),
		Promote:     jidsToStrings(v.Promote),
		Demote:      jidsToStrings(v.Demote),
		Suspended:   v.Suspended,
		Unsuspended: v.Unsuspended,
	}
	if v.Sender != nil {
		msg.Sender = v.Sender.String()
	}
	if v.Name != nil {
		msg.Name = &GroupNameChange{Name: v.Name.Name, SetBy: v.Name.NameSetBy.String()}
	}
	if v.Topic != nil {
		msg.Topic = &GroupTopicChange{Topic: v.Topic.Topic, Deleted: v.Topic.TopicDeleted, SetBy: v.Topic.TopicSetBy.String()}
	}
	if v.Locked != nil {
		msg.Locked = &v.Locked.IsLocked
	}
	if v.Announce != nil {
		msg.Announce = &v.Announce.IsAnnounce
	}
	if v.Ephemeral != nil {
		msg.Ephemeral = &GroupEphemeralChange{Enabled: v.Ephemeral.IsEphemeral, DisappearingTimerSeconds: v.Ephemeral.DisappearingTimer}
	}
	if v.MembershipApprovalMode != nil {
		msg.MembershipApprovalRequired = &v.MembershipApprovalMode.IsJoinApprovalRequired
	}
	if v.Delete != nil {
		msg.Deleted = &GroupDeleteChange{Deleted: v.Delete.Deleted, Reason: v.Delete.DeleteReason}
	}
	msg.NewInviteLink = v.NewInviteLink
	return msg
}

func jidsToStrings(jids []types.JID) []string {
	if len(jids) == 0 {
		return nil
	}
	out := make([]string, len(jids))
	for i, j := range jids {
		out[i] = j.String()
	}
	return out
}

// PushNameMessage is the payload sent to WebSocket subscribers who opted in
// (pushname=true) for a contact's WhatsApp display-name change.
type PushNameMessage struct {
	JID         string `json:"jid"`
	OldPushName string `json:"old_push_name"`
	NewPushName string `json:"new_push_name"`
}

// SubscribeOptions selects which optional live-event streams a WebSocket
// subscriber wants, alongside the always-on message stream.
type SubscribeOptions struct {
	WantTyping    bool
	WantGroupInfo bool
	WantPushName  bool
}

// subscriber holds a WebSocket client's channels and its JID filter.
// An empty jids slice means no filtering — all messages are delivered.
// typingCh/groupInfoCh/pushNameCh are nil unless the client opted into that
// event stream, and a nil channel is never selected on, so no extra
// guarding is needed elsewhere.
type subscriber struct {
	ch          chan BroadcastMessage
	typingCh    chan TypingMessage
	groupInfoCh chan GroupInfoMessage
	pushNameCh  chan PushNameMessage
	jids        []string
}

// MessageBroadcaster fan-outs incoming messages to all connected WebSocket
// clients. It is safe for concurrent use.
type MessageBroadcaster struct {
	clients map[*subscriber]struct{}
	mu      sync.RWMutex

	// Voice notes awaiting transcription (see holdback.go).
	holdMu   sync.Mutex
	held     map[string]*heldBroadcast // holdKey -> held message
	released []releasedBroadcast       // recent releases, oldest first
	// refresh reloads a held message's content and transcript status from the
	// database just before it is released.
	refresh func(*BroadcastMessage)
}

// NewMessageBroadcaster creates an empty broadcaster. refresh may be nil.
func NewMessageBroadcaster(refresh func(*BroadcastMessage)) *MessageBroadcaster {
	return &MessageBroadcaster{
		clients: make(map[*subscriber]struct{}),
		held:    make(map[string]*heldBroadcast),
		refresh: refresh,
	}
}

// Subscribe registers a new subscriber and returns its receive channels.
// If jids is non-empty, only messages matching those JIDs are delivered.
// Each optional channel is nil unless the corresponding opts field is true.
// The caller must call Unsubscribe when done to avoid a goroutine/channel leak.
func (b *MessageBroadcaster) Subscribe(jids []string, opts SubscribeOptions) (
	ch chan BroadcastMessage, typingCh chan TypingMessage,
	groupInfoCh chan GroupInfoMessage, pushNameCh chan PushNameMessage,
) {
	ch = make(chan BroadcastMessage, 64)
	if opts.WantTyping {
		typingCh = make(chan TypingMessage, 64)
	}
	if opts.WantGroupInfo {
		groupInfoCh = make(chan GroupInfoMessage, 64)
	}
	if opts.WantPushName {
		pushNameCh = make(chan PushNameMessage, 64)
	}
	b.mu.Lock()
	b.clients[&subscriber{ch: ch, typingCh: typingCh, groupInfoCh: groupInfoCh, pushNameCh: pushNameCh, jids: jids}] = struct{}{}
	b.mu.Unlock()
	return ch, typingCh, groupInfoCh, pushNameCh
}

// Unsubscribe removes the channel from the subscriber map and closes it and
// its optional channels, if any. The delete happens before the close so that
// a concurrent Broadcast call (holding only a read-lock over the same map
// snapshot) will never see the already-closed channel.
func (b *MessageBroadcaster) Unsubscribe(ch chan BroadcastMessage) {
	b.mu.Lock()
	defer b.mu.Unlock()
	var typingCh chan TypingMessage
	var groupInfoCh chan GroupInfoMessage
	var pushNameCh chan PushNameMessage
	for sub := range b.clients {
		if sub.ch == ch {
			typingCh = sub.typingCh
			groupInfoCh = sub.groupInfoCh
			pushNameCh = sub.pushNameCh
			delete(b.clients, sub)
			break
		}
	}
	close(ch)
	if typingCh != nil {
		close(typingCh)
	}
	if groupInfoCh != nil {
		close(groupInfoCh)
	}
	if pushNameCh != nil {
		close(pushNameCh)
	}
}

// jidMatches checks if chatJID matches any of the subscriber's JID filters.
// If the subscriber has no filters (empty slice), all messages match.
func (b *MessageBroadcaster) jidMatches(sub *subscriber, chatJID string) bool {
	if len(sub.jids) == 0 {
		return true
	}
	for _, jid := range sub.jids {
		if jid == chatJID {
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
	logger.Infof("Broadcast: chat=%q jid=%s id=%s subscribers=%d", msg.ChatName, msg.ChatJID, msg.Message.ID, subscriberCount)

	delivered := 0
	dropped := 0
	filtered := 0
	for sub := range b.clients {
		if !b.jidMatches(sub, msg.ChatJID) {
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

// BroadcastTyping delivers a typing/paused chat-presence update to
// subscribers who opted into typing events and whose JID filter (if any)
// matches the chat. Sends are non-blocking, same as Broadcast: a slow
// consumer never stalls the WhatsApp event handler goroutine.
func (b *MessageBroadcaster) BroadcastTyping(msg TypingMessage) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	for sub := range b.clients {
		if sub.typingCh == nil || !b.jidMatches(sub, msg.ChatJID) {
			continue
		}
		select {
		case sub.typingCh <- msg:
		default:
			logger.Warnf("BroadcastTyping dropped: chat=%q subscriber buffer full", msg.ChatJID)
		}
	}
}

// BroadcastGroupInfo delivers a group metadata change to subscribers who
// opted into group-info events and whose JID filter (if any) matches the
// group. Sends are non-blocking, same as Broadcast.
func (b *MessageBroadcaster) BroadcastGroupInfo(msg GroupInfoMessage) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	for sub := range b.clients {
		if sub.groupInfoCh == nil || !b.jidMatches(sub, msg.ChatJID) {
			continue
		}
		select {
		case sub.groupInfoCh <- msg:
		default:
			logger.Warnf("BroadcastGroupInfo dropped: chat=%q subscriber buffer full", msg.ChatJID)
		}
	}
}

// BroadcastPushName delivers a contact display-name change to every
// subscriber who opted into push-name events. Unlike Broadcast/BroadcastTyping
// there is no chat JID to filter on — a contact's name isn't scoped to one
// chat — so every push-name subscriber receives it. Sends are non-blocking,
// same as Broadcast.
func (b *MessageBroadcaster) BroadcastPushName(msg PushNameMessage) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	for sub := range b.clients {
		if sub.pushNameCh == nil {
			continue
		}
		select {
		case sub.pushNameCh <- msg:
		default:
			logger.Warnf("BroadcastPushName dropped: jid=%q subscriber buffer full", msg.JID)
		}
	}
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
	// disconnectedAt is in-memory only: it lets a reconnecting client be sent
	// held voice notes that were released while it was away (see
	// MessageBroadcaster.catchUpView). Lost on restart, as are the holds.
	disconnectedAt map[string]time.Time
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

	return &ClientRegistry{db: db, cache: cache, disconnectedAt: make(map[string]time.Time)}, nil
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

// MarkDisconnected records that clientName's connection just ended.
func (r *ClientRegistry) MarkDisconnected(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.disconnectedAt[name] = time.Now()
}

// LastDisconnected returns when clientName last disconnected during this
// process's lifetime, or the zero time if it hasn't.
func (r *ClientRegistry) LastDisconnected(name string) time.Time {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.disconnectedAt[name]
}
