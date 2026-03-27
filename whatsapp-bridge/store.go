package main

import (
	"database/sql"
	"fmt"
	"os"
	"time"

	"github.com/blevesearch/bleve/v2"
	_ "github.com/mattn/go-sqlite3"
)

// Message represents a chat message for our client
type Message struct {
	Time      time.Time
	Sender    string
	FullName  string
	Content   string
	IsFromMe  bool
	MediaType string
	Filename  string
}

// MessageStore handles message persistence (SQLite) and search (bleve + embeddings).
type MessageStore struct {
	db       *sql.DB
	index    bleve.Index
	embedder *Embedder
}

// NewMessageStore initialises the SQLite database, the bleve index (with vector
// support when available), and the ONNX embedding model.
func NewMessageStore() (*MessageStore, error) {
	if err := os.MkdirAll("store", 0755); err != nil {
		return nil, fmt.Errorf("failed to create store directory: %v", err)
	}

	db, err := sql.Open("sqlite3", "file:store/messages.db?_foreign_keys=on&_journal_mode=WAL&_busy_timeout=5000")
	if err != nil {
		return nil, fmt.Errorf("failed to open message database: %v", err)
	}
	db.SetMaxOpenConns(1)

	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS chats (
			jid TEXT PRIMARY KEY,
			name TEXT,
			last_message_time TIMESTAMP,
			muted BOOLEAN DEFAULT FALSE
		);

		CREATE TABLE IF NOT EXISTS messages (
			id TEXT,
			chat_jid TEXT,
			sender TEXT,
			full_name TEXT,
			content TEXT,
			timestamp TIMESTAMP,
			is_from_me BOOLEAN,
			media_type TEXT,
			filename TEXT,
			url TEXT,
			media_key BLOB,
			file_sha256 BLOB,
			file_enc_sha256 BLOB,
			file_length INTEGER,
			PRIMARY KEY (id, chat_jid),
			FOREIGN KEY (chat_jid) REFERENCES chats(jid)
		);

		CREATE INDEX IF NOT EXISTS idx_messages_chat_timestamp ON messages(chat_jid, timestamp);
	`)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create tables: %v", err)
	}

	// Initialise embedder (best-effort — search still works text-only without it).
	logger.Infof("Initialising embedding model...")
	embedder, err := NewEmbedder(defaultEmbeddingModelID, defaultEmbeddingOnnxFile)
	if err != nil {
		logger.Warnf("Embedding model unavailable, falling back to text-only search: %v", err)
		embedder = nil
	} else {
		searchEnabled = true
		logger.Infof("Embedding model ready (dim=%d)", embedder.EmbDim())
	}

	// If we have an embedder, we need a vector-capable index. If the existing
	// index was created without vector fields, delete and recreate it.
	embDim := 0
	if embedder != nil {
		embDim = embedder.EmbDim()
	}

	index, err := openOrCreateIndex(embDim)
	if err != nil {
		if embedder != nil {
			embedder.Close()
		}
		db.Close()
		return nil, err
	}

	return &MessageStore{db: db, index: index, embedder: embedder}, nil
}

// Close releases all resources.
func (store *MessageStore) Close() error {
	if store.embedder != nil {
		store.embedder.Close()
	}
	if store.index != nil {
		store.index.Close()
	}
	return store.db.Close()
}

// Store a chat in the database
func (store *MessageStore) StoreChat(jid, name string, lastMessageTime time.Time) error {
	_, err := store.db.Exec(
		"INSERT OR REPLACE INTO chats (jid, name, last_message_time) VALUES (?, ?, ?)",
		jid, name, lastMessageTime,
	)
	return err
}

// Helper to dereference string pointers for database storage
func stringPtrValue(s *string) interface{} {
	if s == nil {
		return nil
	}
	return *s
}

// Store a message in the database
func (store *MessageStore) StoreMessage(id, chatJID, sender, fullName string, content string, timestamp time.Time, isFromMe bool,
	mediaType, filename, url string, mediaKey, fileSHA256, fileEncSHA256 []byte, fileLength uint64) error {
	// Only store if there's actual content or media
	if content == "" && mediaType == "" {
		return nil
	}

	_, err := store.db.Exec(
		`INSERT OR REPLACE INTO messages
		(id, chat_jid, sender, full_name, content, timestamp, is_from_me, media_type, filename, url, media_key, file_sha256, file_enc_sha256, file_length)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		id, chatJID, sender, fullName, content, timestamp, isFromMe, mediaType, filename, url, mediaKey, fileSHA256, fileEncSHA256, fileLength,
	)
	if err != nil {
		return err
	}

	// Index message in bleve (with embedding if available).
	indexMessage(store.index, store.embedder, store.db, id, chatJID, sender, fullName, content, timestamp, isFromMe, mediaType, filename)

	return nil
}

// Get a single message by ID
func (store *MessageStore) GetMessage(id string, chatJID string) (*Message, error) {
	var msg Message
	var timestamp time.Time
	err := store.db.QueryRow(
		"SELECT sender, full_name, content, timestamp, is_from_me, media_type, filename FROM messages WHERE id = ? AND chat_jid = ?",
		id, chatJID,
	).Scan(&msg.Sender, &msg.FullName, &msg.Content, &timestamp, &msg.IsFromMe, &msg.MediaType, &msg.Filename)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("message not found")
		}
		return nil, err
	}

	msg.Time = timestamp
	return &msg, nil
}

// Get messages from a chat
func (store *MessageStore) GetMessages(chatJID string, limit int) ([]Message, error) {
	rows, err := store.db.Query(
		"SELECT sender, full_name, content, timestamp, is_from_me, media_type, filename FROM messages WHERE chat_jid = ? ORDER BY timestamp DESC LIMIT ?",
		chatJID, limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var messages []Message
	for rows.Next() {
		var msg Message
		var timestamp time.Time
		err := rows.Scan(&msg.Sender, &msg.FullName, &msg.Content, &timestamp, &msg.IsFromMe, &msg.MediaType, &msg.Filename)
		if err != nil {
			return nil, err
		}
		msg.Time = timestamp
		messages = append(messages, msg)
	}

	return messages, nil
}

// Get all chats
func (store *MessageStore) GetChats() (map[string]time.Time, error) {
	rows, err := store.db.Query("SELECT jid, last_message_time FROM chats ORDER BY last_message_time DESC")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	chats := make(map[string]time.Time)
	for rows.Next() {
		var jid string
		var lastMessageTime time.Time
		err := rows.Scan(&jid, &lastMessageTime)
		if err != nil {
			return nil, err
		}
		chats[jid] = lastMessageTime
	}

	return chats, nil
}

// ReIndexAllMessages delegates to the search module.
func (store *MessageStore) ReIndexAllMessages(maxRows int) error {
	return reIndexAllMessages(store, maxRows)
}

// GetChatNameByJID returns the stored name for a chat, or "" if not found.
func (store *MessageStore) GetChatNameByJID(chatJID string) string {
	var name sql.NullString
	err := store.db.QueryRow("SELECT name FROM chats WHERE jid = ?", chatJID).Scan(&name)
	if err != nil || !name.Valid {
		return ""
	}
	return name.String
}

// Get mute status for a chat
func (store *MessageStore) IsChatMuted(chatJID string) (bool, error) {
	var muted bool
	err := store.db.QueryRow("SELECT muted FROM chats WHERE jid = ?", chatJID).Scan(&muted)
	if err == sql.ErrNoRows {
		return false, nil // Default to not muted
	}
	return muted, err
}

// Set mute status for a chat
func (store *MessageStore) SetChatMuted(chatJID string, muted bool) error {
	_, err := store.db.Exec("UPDATE chats SET muted = ? WHERE jid = ?", muted, chatJID)
	return err
}

// calculateUserMessageRatio calculates the ratio of user's messages in a chat.
func (store *MessageStore) calculateUserMessageRatio(chatJID string) (float64, error) {
	var totalMessages, userMessages int
	err := store.db.QueryRow(`
		SELECT COUNT(*) as total,
		       SUM(CASE WHEN is_from_me = 1 THEN 1 ELSE 0 END) as user
		FROM messages
		WHERE chat_jid = ?
	`, chatJID).Scan(&totalMessages, &userMessages)
	if err != nil {
		return 0, err
	}
	if totalMessages == 0 {
		return 0, nil
	}
	return float64(userMessages) / float64(totalMessages), nil
}

// SearchMessages delegates to the search module.
func (store *MessageStore) SearchMessages(queryStr string, chatJID string, limit int, offset int) ([]SearchResult, error) {
	return searchMessages(store, queryStr, chatJID, limit, offset)
}

// Store additional media info in the database
func (store *MessageStore) StoreMediaInfo(id, chatJID, url string, mediaKey, fileSHA256, fileEncSHA256 []byte, fileLength uint64) error {
	_, err := store.db.Exec(
		"UPDATE messages SET url = ?, media_key = ?, file_sha256 = ?, file_enc_sha256 = ?, file_length = ? WHERE id = ? AND chat_jid = ?",
		url, mediaKey, fileSHA256, fileEncSHA256, fileLength, id, chatJID,
	)
	return err
}

// Get media info from the database
func (store *MessageStore) GetMediaInfo(id, chatJID string) (string, string, string, []byte, []byte, []byte, uint64, error) {
	var mediaType, filename, url string
	var mediaKey, fileSHA256, fileEncSHA256 []byte
	var fileLength uint64

	err := store.db.QueryRow(
		"SELECT media_type, filename, url, media_key, file_sha256, file_enc_sha256, file_length FROM messages WHERE id = ? AND chat_jid = ?",
		id, chatJID,
	).Scan(&mediaType, &filename, &url, &mediaKey, &fileSHA256, &fileEncSHA256, &fileLength)

	return mediaType, filename, url, mediaKey, fileSHA256, fileEncSHA256, fileLength, err
}
