package main

import "sync"

// BroadcastMessage is the payload sent to WebSocket subscribers for each
// incoming WhatsApp message.
type BroadcastMessage struct {
	ChatJID  string        `json:"chat_jid"`
	ChatName string        `json:"chat_name"`
	Message  MessageWithID `json:"message"`
}

// MessageBroadcaster fan-outs incoming messages to all connected WebSocket
// clients. It is safe for concurrent use.
type MessageBroadcaster struct {
	clients map[chan BroadcastMessage]struct{}
	mu      sync.RWMutex
}

// NewMessageBroadcaster creates an empty broadcaster.
func NewMessageBroadcaster() *MessageBroadcaster {
	return &MessageBroadcaster{
		clients: make(map[chan BroadcastMessage]struct{}),
	}
}

// Subscribe registers a new subscriber and returns its receive channel.
// The caller must call Unsubscribe when done to avoid a goroutine/channel leak.
func (b *MessageBroadcaster) Subscribe() chan BroadcastMessage {
	ch := make(chan BroadcastMessage, 64)
	b.mu.Lock()
	b.clients[ch] = struct{}{}
	b.mu.Unlock()
	return ch
}

// Unsubscribe removes the channel from the subscriber map and closes it.
// The delete happens before the close so that a concurrent Broadcast call
// (holding only a read-lock over the same map snapshot) will never see the
// already-closed channel.
func (b *MessageBroadcaster) Unsubscribe(ch chan BroadcastMessage) {
	b.mu.Lock()
	delete(b.clients, ch)
	b.mu.Unlock()
	close(ch)
}

// Broadcast delivers msg to all current subscribers. Sends are non-blocking:
// if a subscriber's buffer is full the message is dropped for that client so
// that a slow consumer never stalls the WhatsApp event handler goroutine.
func (b *MessageBroadcaster) Broadcast(msg BroadcastMessage) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	for ch := range b.clients {
		select {
		case ch <- msg:
		default:
		}
	}
}
