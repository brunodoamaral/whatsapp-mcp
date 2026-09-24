package main

import (
	"testing"
	"time"

	waLog "go.mau.fi/whatsmeow/util/log"
)

func voiceNote(id string) BroadcastMessage {
	return BroadcastMessage{
		ChatJID:  "1@lid",
		ChatName: "c",
		Message:  MessageWithID{ID: id, Time: time.Now(), MediaType: "audio", TranscriptStatus: tsPending},
	}
}

func recv(ch chan BroadcastMessage, wait time.Duration) (BroadcastMessage, bool) {
	select {
	case m := <-ch:
		return m, true
	case <-time.After(wait):
		return BroadcastMessage{}, false
	}
}

func TestHoldReleaseOnce(t *testing.T) {
	logger = waLog.Noop
	b := NewMessageBroadcaster(func(bm *BroadcastMessage) {
		bm.Message.Content = "transcript"
		bm.Message.TranscriptStatus = tsDone
	})
	ch, _, _, _ := b.Subscribe(nil, SubscribeOptions{})

	b.Hold(voiceNote("A"), time.Now().Add(time.Hour))
	b.Hold(voiceNote("A"), time.Now().Add(time.Hour)) // redelivery: no second hold
	if _, ok := recv(ch, 50*time.Millisecond); ok {
		t.Fatal("held message was broadcast before release")
	}

	b.Release("1@lid", "A", tsDone)
	m, ok := recv(ch, time.Second)
	if !ok || m.Message.Content != "transcript" || m.Message.TranscriptStatus != tsDone {
		t.Fatalf("release: got %+v ok=%v", m, ok)
	}

	b.Release("1@lid", "A", tsDone) // late settle after release
	if _, ok := recv(ch, 50*time.Millisecond); ok {
		t.Fatal("second release produced a second broadcast")
	}
}

func TestHoldTimeoutThenLateSettle(t *testing.T) {
	logger = waLog.Noop
	b := NewMessageBroadcaster(nil)
	ch, _, _, _ := b.Subscribe(nil, SubscribeOptions{})

	b.Hold(voiceNote("B"), time.Now().Add(50*time.Millisecond))
	m, ok := recv(ch, time.Second)
	if !ok || m.Message.ID != "B" || m.Message.TranscriptStatus != tsPending {
		t.Fatalf("timeout release: got %+v ok=%v", m, ok)
	}
	b.Release("1@lid", "B", tsDone)
	if _, ok := recv(ch, 50*time.Millisecond); ok {
		t.Fatal("transcription finishing after the cap produced a second broadcast")
	}
}

func TestCatchUpWithholdsHeldAndAddsReleased(t *testing.T) {
	logger = waLog.Noop
	b := NewMessageBroadcaster(nil)
	disconnected := time.Now()

	b.Hold(voiceNote("H"), time.Now().Add(time.Hour))
	b.Hold(voiceNote("R"), time.Now().Add(time.Hour))
	b.Release("1@lid", "R", tsDone)

	held, released := b.catchUpView(disconnected)
	missed := []BroadcastMessage{voiceNote("H"), {ChatJID: "1@lid", Message: MessageWithID{ID: "T", Time: time.Now()}}}
	out := mergeCatchUp(missed, held, released, nil)

	ids := map[string]bool{}
	for _, m := range out {
		ids[m.Message.ID] = true
	}
	if ids["H"] || !ids["R"] || !ids["T"] || len(out) != 2 {
		t.Fatalf("catch-up ids = %v", ids)
	}

	// A client filtered to another chat doesn't get the released note.
	if out := mergeCatchUp(nil, held, released, []string{"2@lid"}); len(out) != 0 {
		t.Fatalf("filtered catch-up leaked %d messages", len(out))
	}
	// A client that never disconnected during this process gets nothing extra.
	if _, rel := b.catchUpView(time.Time{}); len(rel) != 0 {
		t.Fatal("released replay without a disconnect time")
	}
}

func TestAwaitsTranscript(t *testing.T) {
	p := &AudioPipeline{cfg: transcribeConfig{enabled: true}}
	m := voiceNote("X").Message
	if !awaitsTranscript(p, m) {
		t.Fatal("pending voice note should be held")
	}
	for _, s := range []string{tsDone, tsEmpty, tsNoMedia, tsFailed} {
		m.TranscriptStatus = s
		if awaitsTranscript(p, m) {
			t.Fatalf("status %s should not be held", s)
		}
	}
	m.TranscriptStatus = tsPending
	if awaitsTranscript(&AudioPipeline{}, m) || awaitsTranscript(nil, m) {
		t.Fatal("disabled pipeline should not hold")
	}
	m.Content = "caption"
	if awaitsTranscript(p, m) {
		t.Fatal("voice note with text should not be held")
	}
}
