package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/coder/websocket"
	"github.com/go-chi/chi/v5"
	"go.mau.fi/whatsmeow"
)

// MessageWithID is a Message enriched with its database primary key, used for
// HTTP API responses and WebSocket broadcasts.
type MessageWithID struct {
	ID        string    `json:"id"`
	Time      time.Time `json:"time"`
	Sender    string    `json:"sender"`
	FullName  string    `json:"full_name"`
	Content   string    `json:"content"`
	IsFromMe  bool      `json:"is_from_me"`
	MediaType string    `json:"media_type,omitempty"`
	Filename  string    `json:"filename,omitempty"`
	ReplyToID string    `json:"reply_to_id,omitempty"`
}

// splitAndTrim splits a string by a delimiter and trims whitespace from each element.
func splitAndTrim(s, sep string) []string {
	parts := strings.Split(s, sep)
	for i, part := range parts {
		parts[i] = strings.TrimSpace(part)
	}
	return parts
}

// startRESTServer initialises the chi router with all API routes and starts the server.
func startRESTServer(client *whatsmeow.Client, messageStore *MessageStore, broadcaster *MessageBroadcaster, port int) {
	r := chi.NewRouter()

	r.Post("/api/send", makeSendHandler(client))
	r.Post("/api/download", makeDownloadHandler(client, messageStore))
	r.Get("/api/search", makeSearchHandler(messageStore))
	r.Post("/api/chats/{jid}/mute", makeMuteHandler(messageStore))
	r.Get("/api/chats/{jid}/messages", makeGetMessagesHandler(messageStore))
	r.Get("/ws/messages", makeWSHandler(broadcaster))

	serverAddr := fmt.Sprintf(":%d", port)
	logger.Infof("Starting REST API server on %s...", serverAddr)

	go func() {
		if err := http.ListenAndServe(serverAddr, r); err != nil {
			logger.Errorf("REST API server error: %v", err)
		}
	}()
}

func makeSendHandler(client *whatsmeow.Client) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req SendMessageRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request format", http.StatusBadRequest)
			return
		}

		if req.Recipient == "" {
			http.Error(w, "Recipient is required", http.StatusBadRequest)
			return
		}

		if req.Message == "" && req.MediaPath == "" {
			http.Error(w, "Message or media path is required", http.StatusBadRequest)
			return
		}

		logger.Debugf("Received request to send message: %s %s", req.Message, req.MediaPath)

		success, message := sendWhatsAppMessage(client, req.Recipient, req.Message, req.MediaPath)
		logger.Debugf("Message sent: success=%v %s", success, message)

		w.Header().Set("Content-Type", "application/json")
		if !success {
			w.WriteHeader(http.StatusInternalServerError)
		}

		json.NewEncoder(w).Encode(SendMessageResponse{
			Success: success,
			Message: message,
		})
	}
}

func makeDownloadHandler(client *whatsmeow.Client, messageStore *MessageStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req DownloadMediaRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request format", http.StatusBadRequest)
			return
		}

		if req.MessageID == "" || req.ChatJID == "" {
			http.Error(w, "Message ID and Chat JID are required", http.StatusBadRequest)
			return
		}

		success, mediaType, filename, path, err := downloadMedia(client, messageStore, req.MessageID, req.ChatJID)

		w.Header().Set("Content-Type", "application/json")

		if !success || err != nil {
			errMsg := "Unknown error"
			if err != nil {
				errMsg = err.Error()
			}
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(DownloadMediaResponse{
				Success: false,
				Message: fmt.Sprintf("Failed to download media: %s", errMsg),
			})
			return
		}

		json.NewEncoder(w).Encode(DownloadMediaResponse{
			Success:  true,
			Message:  fmt.Sprintf("Successfully downloaded %s media", mediaType),
			Filename: filename,
			Path:     path,
		})
	}
}

func makeSearchHandler(messageStore *MessageStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		query := r.URL.Query().Get("q")
		if query == "" {
			http.Error(w, "Query parameter 'q' is required", http.StatusBadRequest)
			return
		}

		chatJIDsStr := r.URL.Query().Get("chat_jid")
		limitStr := r.URL.Query().Get("limit")
		semanticWeightStr := r.URL.Query().Get("semantic_weight")
		daysSinceStr := r.URL.Query().Get("days_since")

		daysSince := 0 // default to no time filter
		if daysSinceStr != "" {
			if d, err := strconv.Atoi(daysSinceStr); err == nil && d >= 0 {
				daysSince = d
			}
		}

		// Split chatJIDs by comma if provided
		var chatJIDs []string
		if chatJIDsStr != "" {
			chatJIDs = splitAndTrim(chatJIDsStr, ",")
		}

		limit := 10 // default
		if limitStr != "" {
			if l, err := strconv.Atoi(limitStr); err == nil && l > 0 && l <= 100 {
				limit = l
			}
		}

		semanticWeight := 0.5 // default
		if semanticWeightStr != "" {
			if w, err := strconv.ParseFloat(semanticWeightStr, 64); err == nil && w >= 0 && w <= 1 {
				semanticWeight = w
			}
		}

		// Limit semanticWeight
		if semanticWeight > 1.0 {
			semanticWeight = 1.0
		} else if semanticWeight < 0.0 {
			semanticWeight = 0.0
		}

		results, err := messageStore.SearchMessages(query, chatJIDs, limit, semanticWeight, daysSince)
		if err != nil {
			http.Error(w, fmt.Sprintf("Search failed: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"query":   query,
			"results": results,
			"total":   len(results),
		})
	}
}

func makeMuteHandler(messageStore *MessageStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		chatJID := chi.URLParam(r, "jid")
		if chatJID == "" {
			http.Error(w, "Chat JID required", http.StatusBadRequest)
			return
		}

		var req struct {
			Muted bool `json:"muted"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request format", http.StatusBadRequest)
			return
		}

		err := messageStore.SetChatMuted(chatJID, req.Muted)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to update mute status: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"message": fmt.Sprintf("Chat %s muted status updated", chatJID),
		})
	}
}

func makeGetMessagesHandler(messageStore *MessageStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		chatJID := chi.URLParam(r, "jid")
		if chatJID == "" {
			http.Error(w, "Chat JID required", http.StatusBadRequest)
			return
		}

		q := r.URL.Query()

		limit := 50
		if s := q.Get("limit"); s != "" {
			if v, err := strconv.Atoi(s); err != nil || v < 0 {
				http.Error(w, "Invalid limit", http.StatusBadRequest)
				return
			} else if v > 200 {
				limit = 200
			} else {
				limit = v
			}
		}

		offset := 0
		if s := q.Get("offset"); s != "" {
			if v, err := strconv.Atoi(s); err != nil || v < 0 {
				http.Error(w, "Invalid offset", http.StatusBadRequest)
				return
			} else {
				offset = v
			}
		}

		filter := MessageFilter{Limit: limit, Offset: offset}

		if s := q.Get("start"); s != "" {
			t, err := time.Parse(time.RFC3339, s)
			if err != nil {
				http.Error(w, "Invalid start timestamp (use RFC3339)", http.StatusBadRequest)
				return
			}
			filter.Start = &t
		}

		if s := q.Get("end"); s != "" {
			t, err := time.Parse(time.RFC3339, s)
			if err != nil {
				http.Error(w, "Invalid end timestamp (use RFC3339)", http.StatusBadRequest)
				return
			}
			filter.End = &t
		}

		if filter.Start != nil && filter.End != nil && filter.Start.After(*filter.End) {
			http.Error(w, "start must be before end", http.StatusBadRequest)
			return
		}

		messages, err := messageStore.GetMessagesFiltered(chatJID, filter)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to retrieve messages: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"chat_jid": chatJID,
			"messages": messages,
			"count":    len(messages),
		})
	}
}

func makeWSHandler(broadcaster *MessageBroadcaster) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{
			OriginPatterns: []string{"*"},
		})
		if err != nil {
			return
		}
		defer conn.CloseNow()

		ch := broadcaster.Subscribe()
		defer broadcaster.Unsubscribe(ch)

		ctx := r.Context()
		for {
			select {
			case <-ctx.Done():
				conn.Close(websocket.StatusNormalClosure, "client disconnected")
				return
			case msg, ok := <-ch:
				if !ok {
					conn.Close(websocket.StatusNormalClosure, "broadcaster closed")
					return
				}
				data, err := json.Marshal(msg)
				if err != nil {
					continue
				}
				writeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
				err = conn.Write(writeCtx, websocket.MessageText, data)
				cancel()
				if err != nil {
					return
				}
			}
		}
	}
}
