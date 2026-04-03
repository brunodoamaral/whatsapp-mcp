package main

import (
	"context"
	"fmt"
	"reflect"
	"runtime/debug"
	"time"

	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
	waLog "go.mau.fi/whatsmeow/util/log"
)

// Handle regular incoming messages with media support
func handleMessage(client *whatsmeow.Client, messageStore *MessageStore, msg *events.Message, logger waLog.Logger) *BroadcastMessage {
	// Save message to database
	chatJID := msg.Info.Chat.String()
	sender := msg.Info.Sender.User
	var senderName string

	// Just use contact info (full name) - try this first
	contact, err := client.Store.Contacts.GetContact(context.Background(), msg.Info.Sender)
	if err == nil && contact.FullName != "" {
		senderName = contact.FullName
	} else if contact.PushName != "" {
		senderName = contact.PushName
	} else if sender != "" {
		// Fallback to sender
		senderName = sender
	} else {
		// Last fallback to JID
		senderName = msg.Info.Sender.User
	}

	// Check if this is a forwarded message and adjust chat JID if needed
	if msg.Message != nil && msg.Message.MessageContextInfo != nil {
		// This is a forwarded message - the chat JID should be the current chat, not the original
		// msg.Info.Chat should already contain the correct current chat JID
		tracef("Detected forwarded message from %s to chat %s", sender, chatJID)
	}

	// Get appropriate chat name (pass nil for conversation since we don't have one for regular messages)
	name := GetChatName(client, messageStore, msg.Info.Chat, chatJID, nil, sender, logger)

	// Update chat in database with the message timestamp (keeps last message time updated)
	errChat := messageStore.StoreChat(chatJID, name, msg.Info.Timestamp)
	if errChat != nil {
		logger.Warnf("Failed to store chat: %v", errChat)
	}

	// Extract text content
	content := extractTextContent(msg.Message)

	// Extract media info
	mediaType, filename, url, mediaKey, fileSHA256, fileEncSHA256, fileLength := extractMediaInfo(msg.Message)

	// Skip if there's no content and no media
	if content == "" && mediaType == "" {
		return nil
	}

	// Extract reply-to message ID if this is a reply
	replyToID := extractReplyToID(msg.Message)

	// Store message in database with contact's full name for indexing
	err = messageStore.StoreMessage(
		msg.Info.ID,
		chatJID,
		sender,
		senderName,
		content,
		msg.Info.Timestamp,
		msg.Info.IsFromMe,
		mediaType,
		filename,
		url,
		mediaKey,
		fileSHA256,
		fileEncSHA256,
		fileLength,
		replyToID,
	)

	if err != nil {
		logger.Warnf("Failed to store message: %v", err)
		return nil
	}

	// Log message reception
	timestamp := msg.Info.Timestamp.Format("2006-01-02 15:04:05")
	direction := "←"
	if msg.Info.IsFromMe {
		direction = "→"
	}

	// Log based on message type
	if mediaType != "" {
		logger.Debugf("[%s] %s %s: [%s: %s] %s", timestamp, direction, sender, mediaType, filename, content)
	} else if content != "" {
		logger.Debugf("[%s] %s %s: %s", timestamp, direction, sender, content)
	}

	return &BroadcastMessage{
		ChatJID:  chatJID,
		ChatName: name,
		Message: MessageWithID{
			ID:        msg.Info.ID,
			Time:      msg.Info.Timestamp,
			Sender:    sender,
			FullName:  senderName,
			Content:   content,
			IsFromMe:  msg.Info.IsFromMe,
			MediaType: mediaType,
			Filename:  filename,
			ReplyToID: replyToID,
		},
	}
}

// GetChatName determines the appropriate name for a chat based on JID and other info
func GetChatName(client *whatsmeow.Client, messageStore *MessageStore, jid types.JID, chatJID string, conversation interface{}, sender string, logger waLog.Logger) string {
	// Always try to get the current name first, then check database as fallback
	var name string

	if jid.Server == "g.us" {
		// This is a group chat
		tracef("Getting name for group: %s", chatJID)

		// Use conversation data if provided (from history sync)
		if conversation != nil {
			// Extract name from conversation if available
			var displayName, convName *string
			v := reflect.ValueOf(conversation)
			if v.Kind() == reflect.Ptr && !v.IsNil() {
				v = v.Elem()

				// Try to find DisplayName field
				if displayNameField := v.FieldByName("DisplayName"); displayNameField.IsValid() && displayNameField.Kind() == reflect.Ptr && !displayNameField.IsNil() {
					dn := displayNameField.Elem().String()
					displayName = &dn
				}

				// Try to find Name field
				if nameField := v.FieldByName("Name"); nameField.IsValid() && nameField.Kind() == reflect.Ptr && !nameField.IsNil() {
					n := nameField.Elem().String()
					convName = &n
				}
			}

			// Use the name we found
			if displayName != nil && *displayName != "" {
				name = *displayName
			} else if convName != nil && *convName != "" {
				name = *convName
			}
		}

		// If we didn't get a name, try group info
		if name == "" {
			groupInfo, err := client.GetGroupInfo(context.Background(), jid)
			if err == nil && groupInfo.Name != "" {
				name = groupInfo.Name
			} else {
				// Fallback name for groups
				name = fmt.Sprintf("Group %s", jid.User)
			}
		}

		tracef("Using group name: %s", name)
	} else {
		// This is an individual contact
		tracef("Getting name for contact: %s", chatJID)

		// Just use contact info (full name) - try this first
		contact, err := client.Store.Contacts.GetContact(context.Background(), jid)
		if err == nil && contact.FullName != "" {
			name = contact.FullName
			tracef("Found contact name: %s", name)
		} else if contact.PushName != "" {
			name = contact.PushName
			tracef("Using push name: %s", name)
		} else if sender != "" {
			// Fallback to sender
			name = sender
			logger.Warnf("Using sender as fallback: %s", name)
		} else {
			// Last fallback to JID
			name = jid.User
			logger.Warnf("Using JID as last fallback: %s", name)
		}
	}

	// Update the database with the current name
	if name != "" {
		_, err := messageStore.db.Exec("INSERT OR REPLACE INTO chats (jid, name, last_message_time) VALUES (?, ?, ?)",
			chatJID, name, time.Now())
		if err != nil {
			logger.Warnf("Failed to update chat name in database: %v\n%s", err, debug.Stack())
		}
	}

	return name
}

// Handle history sync events
func handleHistorySync(client *whatsmeow.Client, messageStore *MessageStore, historySync *events.HistorySync, logger waLog.Logger) {
	logger.Infof("Received history sync event with %d conversations", len(historySync.Data.Conversations))

	syncedCount := 0
	for _, conversation := range historySync.Data.Conversations {
		// Parse JID from the conversation
		if conversation.ID == nil {
			continue
		}

		chatJID := *conversation.ID

		// Try to parse the JID
		jid, err := types.ParseJID(chatJID)
		if err != nil {
			logger.Warnf("Failed to parse JID %s: %v", chatJID, err)
			continue
		}

		// Get appropriate chat name by passing the history sync conversation directly
		name := GetChatName(client, messageStore, jid, chatJID, conversation, "", logger)

		// Process messages
		messages := conversation.Messages
		if len(messages) > 0 {
			// Update chat with latest message timestamp
			latestMsg := messages[0]
			if latestMsg == nil || latestMsg.Message == nil {
				continue
			}

			// Get timestamp from message info
			timestamp := time.Time{}
			if ts := latestMsg.Message.GetMessageTimestamp(); ts != 0 {
				timestamp = time.Unix(int64(ts), 0)
			} else {
				continue
			}

			messageStore.StoreChat(chatJID, name, timestamp)

			// Store messages
			for _, msg := range messages {
				if msg == nil || msg.Message == nil {
					continue
				}

				// Extract text content
				var content string
				if msg.Message.Message != nil {
					if conv := msg.Message.Message.GetConversation(); conv != "" {
						content = conv
					} else if ext := msg.Message.Message.GetExtendedTextMessage(); ext != nil {
						content = ext.GetText()
					}
				}

				// Extract media info
				var mediaType, filename, url string
				var mediaKey, fileSHA256, fileEncSHA256 []byte
				var fileLength uint64

				if msg.Message.Message != nil {
					mediaType, filename, url, mediaKey, fileSHA256, fileEncSHA256, fileLength = extractMediaInfo(msg.Message.Message)
				}

				// Log the message content for debugging
				tracef("Message content: %v, Media Type: %v", content, mediaType)

				// Skip messages with no content and no media
				if content == "" && mediaType == "" {
					continue
				}

				// Determine sender
				var sender string
				isFromMe := false
				if msg.Message.Key != nil {
					if msg.Message.Key.FromMe != nil {
						isFromMe = *msg.Message.Key.FromMe
					}
					if !isFromMe && msg.Message.Key.Participant != nil && *msg.Message.Key.Participant != "" {
						sender = *msg.Message.Key.Participant
					} else if isFromMe {
						sender = client.Store.ID.User
					} else {
						sender = jid.User
					}
				} else {
					sender = jid.User
				}

				// Store message
				msgID := ""
				if msg.Message.Key != nil && msg.Message.Key.ID != nil {
					msgID = *msg.Message.Key.ID
				}

				// Get message timestamp
				timestamp := time.Time{}
				if ts := msg.Message.GetMessageTimestamp(); ts != 0 {
					timestamp = time.Unix(int64(ts), 0)
				} else {
					continue
				}

				// Determine sender's full name for indexing
				var senderFullName string
				if msg.Message.Key != nil && msg.Message.Key.Participant != nil && *msg.Message.Key.Participant != "" {
					// Try to get contact info for this participant
					jid, err := types.ParseJID(*msg.Message.Key.Participant)
					if err != nil {
						logger.Warnf("Failed to parse JID: %v", err)
						continue
					}
					contact, err := client.Store.Contacts.GetContact(context.Background(), jid)
					if err == nil && contact.FullName != "" {
						senderFullName = contact.FullName
					} else if contact.PushName != "" {
						senderFullName = contact.PushName
					} else if sender != "" {
						// Fallback to sender
						senderFullName = sender
					} else {
						// Last fallback to JID
						senderFullName = *msg.Message.Key.Participant
					}
				} else if isFromMe {
					senderFullName = "You"
				} else {
					senderFullName = sender
				}

				histReplyToID := ""
				if msg.Message.Message != nil {
					histReplyToID = extractReplyToID(msg.Message.Message)
				}

				err = messageStore.StoreMessage(
					msgID,
					chatJID,
					sender,
					senderFullName,
					content,
					timestamp,
					isFromMe,
					mediaType,
					filename,
					url,
					mediaKey,
					fileSHA256,
					fileEncSHA256,
					fileLength,
					histReplyToID,
				)
				if err != nil {
					logger.Warnf("Failed to store history message: %v", err)
				} else {
					syncedCount++
					// Log successful message storage
					if mediaType != "" {
						tracef("Stored message: [%s] %s -> %s: [%s: %s] %s",
							timestamp.Format("2006-01-02 15:04:05"), sender, chatJID, mediaType, filename, content)
					} else {
						tracef("Stored message: [%s] %s -> %s: %s",
							timestamp.Format("2006-01-02 15:04:05"), sender, chatJID, content)
					}
				}
			}
		}
	}

	logger.Infof("History sync complete. Stored %d messages.", syncedCount)
}

// Request history sync from the server
func requestHistorySync(client *whatsmeow.Client) {
	if client == nil {
		logger.Errorf("Client is not initialized. Cannot request history sync.")
		return
	}

	if !client.IsConnected() {
		logger.Errorf("Client is not connected. Please ensure you are connected to WhatsApp first.")
		return
	}

	if client.Store.ID == nil {
		logger.Errorf("Client is not logged in. Please scan the QR code first.")
		return
	}

	// Build and send a history sync request
	historyMsg := client.BuildHistorySyncRequest(nil, 100)
	if historyMsg == nil {
		logger.Errorf("Failed to build history sync request.")
		return
	}

	_, err := client.SendMessage(context.Background(), types.JID{
		Server: "s.whatsapp.net",
		User:   "status",
	}, historyMsg)

	if err != nil {
		logger.Errorf("Failed to request history sync: %v", err)
	} else {
		logger.Infof("History sync requested. Waiting for server response...")
	}
}

