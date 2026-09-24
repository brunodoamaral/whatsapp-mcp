package main

import (
	"time"
)

// Voice notes are held back from the WebSocket stream until their
// transcription settles, then broadcast once with the transcript already in
// content. Broadcasting them on arrival (empty content) and filling content in
// later without any event meant a consumer that reacts to each message
// (WhatsDoing) analysed an empty audio and never saw the text — see CLAUDE.md.
const (
	// broadcastHoldCap bounds how long a voice note's broadcast may wait for
	// its transcript. An average note transcribes in ~30s on this Pi, a long
	// one in a few minutes; past this the message goes out with whatever
	// status it has rather than sitting behind whisper's 20-minute timeout.
	broadcastHoldCap = 4 * time.Minute
	// releasedReplayWindow is how long a released hold is remembered so that a
	// client that was disconnected at release time still gets it on reconnect,
	// even if its catch-up cursor has already moved past the message's
	// timestamp (later messages were delivered while this one was held).
	releasedReplayWindow = 30 * time.Minute
)

type heldBroadcast struct {
	msg    BroadcastMessage
	heldAt time.Time
	timer  *time.Timer
}

type releasedBroadcast struct {
	msg BroadcastMessage
	at  time.Time
}

func holdKey(chatJID, id string) string { return chatJID + "\x00" + id }

// awaitsTranscript reports whether a message should be held rather than
// broadcast now: a voice note with no text yet whose transcription is still
// in flight, with the pipeline actually running.
func awaitsTranscript(p *AudioPipeline, msg MessageWithID) bool {
	if p == nil || !p.cfg.enabled || msg.MediaType != "audio" || msg.Content != "" {
		return false
	}
	switch msg.TranscriptStatus {
	case tsPending, tsDownloaded, tsRetrying:
		return true
	}
	return false
}

// Hold parks msg until Release is called for it or deadline passes, whichever
// comes first. Holding a message that is already held is a no-op, so a
// redelivered event can never produce a second broadcast.
func (b *MessageBroadcaster) Hold(msg BroadcastMessage, deadline time.Time) {
	key := holdKey(msg.ChatJID, msg.Message.ID)
	b.holdMu.Lock()
	if _, ok := b.held[key]; ok {
		b.holdMu.Unlock()
		logger.Infof("Broadcast hold skipped: chat=%q id=%s already held", msg.ChatName, msg.Message.ID)
		return
	}
	h := &heldBroadcast{msg: msg, heldAt: time.Now()}
	b.held[key] = h
	h.timer = time.AfterFunc(time.Until(deadline), func() {
		b.Release(msg.ChatJID, msg.Message.ID, "timeout")
	})
	b.holdMu.Unlock()
	logger.Infof("Broadcast held: chat=%q jid=%s id=%s status=%s reason=awaiting transcript release_by=%s",
		msg.ChatName, msg.ChatJID, msg.Message.ID, msg.Message.TranscriptStatus, deadline.Format("15:04:05"))
}

// Release broadcasts a held message, refreshed from the database so it
// carries the transcript and final status. reason is the terminal
// transcript status that triggered it, or "timeout". Releasing a message
// that isn't held (never held, or already released) does nothing — this is
// what keeps a transcription that finishes after the cap, or a sweep over
// old rows, from ever broadcasting.
func (b *MessageBroadcaster) Release(chatJID, id, reason string) {
	key := holdKey(chatJID, id)
	b.holdMu.Lock()
	h, ok := b.held[key]
	if !ok {
		b.holdMu.Unlock()
		return
	}
	h.timer.Stop()
	msg := h.msg
	if b.refresh != nil {
		b.refresh(&msg)
	}
	delete(b.held, key)
	// Recorded before the broadcast, under the same lock as the delete, so a
	// concurrent catch-up sees the message as either held or released —
	// never neither.
	now := time.Now()
	b.released = append(b.released, releasedBroadcast{msg: msg, at: now})
	b.pruneReleasedLocked(now)
	b.holdMu.Unlock()

	logger.Infof("Broadcast released: chat=%q jid=%s id=%s reason=%s status=%s held=%.1fs chars=%d",
		msg.ChatName, msg.ChatJID, id, reason, msg.Message.TranscriptStatus,
		now.Sub(h.heldAt).Seconds(), len(msg.Message.Content))
	b.Broadcast(msg)
}

func (b *MessageBroadcaster) pruneReleasedLocked(now time.Time) {
	cut := 0
	for cut < len(b.released) && now.Sub(b.released[cut].at) > releasedReplayWindow {
		cut++
	}
	b.released = b.released[cut:]
}

// catchUpView returns, in one consistent snapshot, the set of currently held
// message keys (to be withheld from a catch-up replay — they'll arrive live)
// and the messages released after `since` (to be added to it).
func (b *MessageBroadcaster) catchUpView(since time.Time) (held map[string]bool, released []BroadcastMessage) {
	b.holdMu.Lock()
	defer b.holdMu.Unlock()
	held = make(map[string]bool, len(b.held))
	for k := range b.held {
		held[k] = true
	}
	if !since.IsZero() {
		for _, r := range b.released {
			if r.at.After(since) {
				released = append(released, r.msg)
			}
		}
	}
	return held, released
}

// refreshFromStore is the broadcaster's refresh hook: it reloads the fields
// transcription changes, so a released message carries the transcript.
func refreshFromStore(store *MessageStore) func(*BroadcastMessage) {
	return func(bm *BroadcastMessage) {
		var content, status string
		if err := store.db.QueryRow(
			`SELECT COALESCE(content, ''), COALESCE(transcript_status, '') FROM messages WHERE id = ? AND chat_jid = ?`,
			bm.Message.ID, bm.ChatJID,
		).Scan(&content, &status); err != nil {
			logger.Warnf("Broadcast refresh failed for %s: %v", bm.Message.ID, err)
			return
		}
		bm.Message.Content = content
		bm.Message.TranscriptStatus = status
	}
}

// rehold re-parks voice notes that were held when the bridge last stopped.
// Only rows younger than broadcastHoldCap qualify, each with just the time it
// has left under the cap — anything older either already went out before the
// restart or is left to WS catch-up, so resuming the sweep over a backlog
// never turns into a flood of live broadcasts.
func rehold(b *MessageBroadcaster, store *MessageStore) {
	rows, err := store.db.Query(`
		SELECT m.id, m.chat_jid, COALESCE(c.name, m.chat_jid),
			m.sender, COALESCE(m.full_name, ''), COALESCE(m.content, ''), m.timestamp,
			m.is_from_me, COALESCE(m.filename, ''), COALESCE(m.reply_to_id, ''), m.transcript_status
		FROM messages m
		LEFT JOIN chats c ON c.jid = m.chat_jid
		WHERE m.media_type = 'audio' AND m.transcript_status IN (?, ?, ?)
		  AND m.timestamp > ?`,
		tsPending, tsDownloaded, tsRetrying, time.Now().Add(-broadcastHoldCap))
	if err != nil {
		logger.Warnf("Broadcast re-hold query failed: %v", err)
		return
	}
	var msgs []BroadcastMessage
	for rows.Next() {
		var bm BroadcastMessage
		m := &bm.Message
		if err := rows.Scan(&m.ID, &bm.ChatJID, &bm.ChatName, &m.Sender, &m.FullName, &m.Content, &m.Time,
			&m.IsFromMe, &m.Filename, &m.ReplyToID, &m.TranscriptStatus); err != nil {
			logger.Warnf("Broadcast re-hold scan failed: %v", err)
			continue
		}
		m.MediaType = "audio"
		msgs = append(msgs, bm)
	}
	rows.Close()
	for _, bm := range msgs {
		b.Hold(bm, bm.Message.Time.Add(broadcastHoldCap))
	}
	if len(msgs) > 0 {
		logger.Infof("Broadcast re-hold: %d voice note(s) from before restart", len(msgs))
	}
}

// dispatchMessage sends a freshly handled message out: straight to the
// broadcaster, or — for a voice note still being transcribed — into a hold.
// The hold is registered before the note is queued for download, so the
// pipeline can never settle it before there is a hold to release.
func dispatchMessage(b *MessageBroadcaster, bm BroadcastMessage) {
	m := bm.Message
	if awaitsTranscript(audioPipeline, m) {
		b.Hold(bm, time.Now().Add(broadcastHoldCap))
	} else {
		b.Broadcast(bm)
	}
	// Voice notes go straight into the download queue: WhatsApp media URLs
	// expire, so the audio has to reach local disk well before transcription
	// gets around to it.
	if m.MediaType == "audio" && m.Content == "" {
		audioPipeline.Enqueue(m.ID, bm.ChatJID, m.Time)
	}
}
