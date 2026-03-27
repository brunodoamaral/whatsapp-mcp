package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"

	"github.com/go-chi/chi/v5"
	"go.mau.fi/whatsmeow"
)

// startRESTServer initialises the chi router with all API routes and starts the server.
func startRESTServer(client *whatsmeow.Client, messageStore *MessageStore, port int) {
	r := chi.NewRouter()

	r.Post("/api/send", makeSendHandler(client))
	r.Post("/api/download", makeDownloadHandler(client, messageStore))
	r.Get("/api/search", makeSearchHandler(messageStore))
	r.Post("/api/chats/{jid}/mute", makeMuteHandler(messageStore))

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

		chatJID := r.URL.Query().Get("chat_jid")
		limitStr := r.URL.Query().Get("limit")
		offsetStr := r.URL.Query().Get("offset")

		limit := 20 // default
		if limitStr != "" {
			if l, err := strconv.Atoi(limitStr); err == nil && l > 0 && l <= 100 {
				limit = l
			}
		}

		offset := 0 // default
		if offsetStr != "" {
			if o, err := strconv.Atoi(offsetStr); err == nil && o >= 0 {
				offset = o
			}
		}

		results, err := messageStore.SearchMessages(query, chatJID, limit, offset)
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
