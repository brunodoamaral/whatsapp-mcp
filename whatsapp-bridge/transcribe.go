package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"go.mau.fi/whatsmeow"
)

// Voice-note transcription pipeline.
//
// Two stages, deliberately separated:
//
//  1. Download (network-bound, 2 workers). WhatsApp media URLs expire after a
//     couple of weeks, so audio has to hit local disk as soon as it arrives —
//     it must not sit behind a slow CPU queue. Roughly 97% of the existing
//     voice notes in this database are already unrecoverable for this reason.
//  2. Transcribe (CPU-bound, 1 worker). ~33s of CPU per average voice note on
//     this Pi, which is far below the arrival rate but must not contend with
//     the event loop, so it runs niced and single-file.
//
// State lives in messages.transcript_status, which is both the audit trail and
// the work queue — a restart resumes from it via sweep().
const (
	tsPending    = "pending"    // audio row, media not yet on disk
	tsDownloaded = "downloaded" // file on disk, awaiting transcription
	tsDone       = "done"       // transcript written to content
	tsEmpty      = "empty"      // ran, produced nothing usable (silence/noise/hallucination)
	tsNoMedia    = "no_media"   // media gone from WhatsApp servers, unrecoverable
	tsFailed     = "failed"     // ffmpeg/whisper error, retried on next sweep
)

const (
	// repeatRunLimit truncates a transcript at the point where whisper falls
	// into a repetition loop (observed with tiny on a 48s clip: "é, é, é, …",
	// which also made it slower than base on that sample).
	repeatRunLimit = 6
	// minTranscriptWords drops output too short to carry meaning.
	minTranscriptWords = 2
	// whisperTimeout bounds a single transcription. The longest voice note in
	// this database is ~4min, which small-q5_1 handles in ~3.5min.
	whisperTimeout = 20 * time.Minute
	// sweepInterval re-scans for stuck or newly-eligible rows.
	sweepInterval = 10 * time.Minute
	// sweepBatch bounds how much a single sweep enqueues.
	sweepBatch = 500
)

// bracketTag matches whisper's non-speech annotations, e.g. "[MÚSICA DE FUNDO]"
// or "[Music]". These are hallucinations on short/noisy clips and must not be
// indexed as if they were speech.
var bracketTag = regexp.MustCompile(`\[[^\]]*\]`)

type transcribeConfig struct {
	enabled bool
	cli     string
	model   string
	threads int
	lang    string
	nice    string // path to nice(1), empty if unavailable
}

func loadTranscribeConfig() transcribeConfig {
	cfg := transcribeConfig{
		enabled: os.Getenv("TRANSCRIBE_DISABLED") == "",
		cli:     envOr("WHISPER_CLI", "/home/bruno/git/whisper.cpp/build/bin/whisper-cli"),
		model:   envOr("WHISPER_MODEL", "/home/bruno/git/whisper.cpp/models/ggml-small-q5_1.bin"),
		lang:    envOr("WHISPER_LANG", "pt"),
		threads: 4,
	}
	if s := os.Getenv("WHISPER_THREADS"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			cfg.threads = n
		}
	}
	if p, err := exec.LookPath("nice"); err == nil {
		cfg.nice = p
	}
	return cfg
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// audioJob identifies one voice note to process.
type audioJob struct {
	id      string
	chatJID string
	ts      time.Time
}

// AudioPipeline downloads and transcribes voice notes in the background.
type AudioPipeline struct {
	client *whatsmeow.Client
	store  *MessageStore
	cfg    transcribeConfig

	downloads   chan audioJob
	transcripts chan audioJob
	wg          sync.WaitGroup
	stop        chan struct{}
	stopOnce    sync.Once
}

func NewAudioPipeline(client *whatsmeow.Client, store *MessageStore) *AudioPipeline {
	return &AudioPipeline{
		client:      client,
		store:       store,
		cfg:         loadTranscribeConfig(),
		downloads:   make(chan audioJob, 256),
		transcripts: make(chan audioJob, 256),
		stop:        make(chan struct{}),
	}
}

// Start launches the workers and the periodic sweeper. Safe to call when
// transcription is disabled or misconfigured — it logs and does nothing.
func (p *AudioPipeline) Start() {
	if !p.cfg.enabled {
		logger.Infof("Voice-note transcription disabled (TRANSCRIBE_DISABLED set)")
		return
	}
	if _, err := os.Stat(p.cfg.cli); err != nil {
		logger.Warnf("Voice-note transcription disabled: whisper CLI not found at %s", p.cfg.cli)
		p.cfg.enabled = false
		return
	}
	if _, err := os.Stat(p.cfg.model); err != nil {
		logger.Warnf("Voice-note transcription disabled: model not found at %s", p.cfg.model)
		p.cfg.enabled = false
		return
	}
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		logger.Warnf("Voice-note transcription disabled: ffmpeg not found in PATH")
		p.cfg.enabled = false
		return
	}

	logger.Infof("Voice-note transcription enabled (model=%s lang=%s threads=%d)",
		filepath.Base(p.cfg.model), p.cfg.lang, p.cfg.threads)

	for i := 0; i < 2; i++ {
		p.wg.Add(1)
		go p.downloadWorker()
	}
	p.wg.Add(1)
	go p.transcribeWorker()
	p.wg.Add(1)
	go p.sweeper()
}

// Stop signals the workers and waits for the in-flight job to finish.
func (p *AudioPipeline) Stop() {
	p.stopOnce.Do(func() { close(p.stop) })
	p.wg.Wait()
}

// Enqueue schedules a newly received voice note. The row is already marked
// pending by StoreMessage, so a dropped enqueue (full queue, crash) is picked
// up by the next sweep rather than lost.
func (p *AudioPipeline) Enqueue(id, chatJID string, ts time.Time) {
	if p == nil || !p.cfg.enabled {
		return
	}
	select {
	case p.downloads <- audioJob{id: id, chatJID: chatJID, ts: ts}:
	default:
		logger.Warnf("Audio download queue full, deferring %s to next sweep", id)
	}
}

func (p *AudioPipeline) setStatus(id, chatJID, status string) {
	if _, err := p.store.db.Exec(
		`UPDATE messages SET transcript_status = ? WHERE id = ? AND chat_jid = ?`,
		status, id, chatJID,
	); err != nil {
		logger.Warnf("Failed to set transcript_status=%s for %s: %v", status, id, err)
	}
}

func (p *AudioPipeline) downloadWorker() {
	defer p.wg.Done()
	for {
		select {
		case <-p.stop:
			return
		case job := <-p.downloads:
			p.download(job)
		}
	}
}

func (p *AudioPipeline) download(job audioJob) {
	ok, _, _, path, err := downloadMedia(p.client, p.store, job.id, job.chatJID)
	if !ok || err != nil {
		// Expired URL or missing media keys. Not retried aggressively: once
		// WhatsApp drops the blob it is gone for good.
		logger.Warnf("Audio download failed for %s: %v", job.id, err)
		p.setStatus(job.id, job.chatJID, tsNoMedia)
		return
	}
	tracef("Audio downloaded: %s -> %s", job.id, path)
	p.setStatus(job.id, job.chatJID, tsDownloaded)

	select {
	case p.transcripts <- job:
	default:
		logger.Warnf("Transcription queue full, deferring %s to next sweep", job.id)
	}
}

func (p *AudioPipeline) transcribeWorker() {
	defer p.wg.Done()
	for {
		select {
		case <-p.stop:
			return
		case job := <-p.transcripts:
			p.transcribe(job)
		}
	}
}

// transcribe converts one voice note to text and folds it into the index.
func (p *AudioPipeline) transcribe(job audioJob) {
	path, err := p.localMediaPath(job.id, job.chatJID)
	if err != nil {
		logger.Warnf("Transcription skipped for %s: %v", job.id, err)
		p.setStatus(job.id, job.chatJID, tsPending) // re-download on next sweep
		return
	}

	start := time.Now()
	raw, err := p.runWhisper(path)
	if err != nil {
		logger.Warnf("Transcription failed for %s: %v", job.id, err)
		p.setStatus(job.id, job.chatJID, tsFailed)
		return
	}

	text := cleanTranscript(raw)
	if text == "" {
		tracef("Transcription empty for %s (%.1fs)", job.id, time.Since(start).Seconds())
		p.setStatus(job.id, job.chatJID, tsEmpty)
		return
	}

	// The transcript goes into content unprefixed: every read path already does
	// COALESCE(content, ''), so it flows into the context window, the embedding
	// and the API for free. Provenance lives in transcript_status, not in a
	// marker string that would be embedded into the vector 16 times per group.
	// The content guard means a message that acquired real text between enqueue
	// and now (a caption, an edit) is never overwritten by machine output.
	if _, err := p.store.db.Exec(
		`UPDATE messages SET content = ?, transcript_status = ?
		 WHERE id = ? AND chat_jid = ? AND (content IS NULL OR content = '')`,
		text, tsDone, job.id, job.chatJID,
	); err != nil {
		logger.Warnf("Failed to store transcript for %s: %v", job.id, err)
		return
	}

	logger.Infof("Transcribed %s in %.1fs (%d chars)", job.id, time.Since(start).Seconds(), len(text))

	// Opened without an index (backfill --skip-index): the transcript is in
	// SQLite and a following full reindex will pick it up.
	if p.store.index == nil {
		return
	}

	// Fold the new text into the context group that contains this message.
	group, err := groupIndexForMessage(p.store.db, job.chatJID, job.id)
	if err != nil {
		logger.Warnf("Could not locate group for %s: %v", job.id, err)
		return
	}
	if err := rebuildGroupDoc(p.store, job.chatJID, group); err != nil {
		logger.Warnf("Failed to re-index %s group %d: %v", job.chatJID, group, err)
	}
}

// localMediaPath returns the on-disk path of a message's media, mirroring the
// layout used by downloadMedia.
func (p *AudioPipeline) localMediaPath(id, chatJID string) (string, error) {
	var filename string
	if err := p.store.db.QueryRow(
		`SELECT COALESCE(filename, '') FROM messages WHERE id = ? AND chat_jid = ?`,
		id, chatJID,
	).Scan(&filename); err != nil {
		return "", fmt.Errorf("lookup filename: %w", err)
	}
	if filename == "" {
		return "", fmt.Errorf("no filename recorded")
	}
	path := filepath.Join("store", strings.ReplaceAll(chatJID, ":", "_"), filename)
	if _, err := os.Stat(path); err != nil {
		return "", fmt.Errorf("media not on disk: %w", err)
	}
	return path, nil
}

// runWhisper decodes the audio to 16kHz mono PCM and runs whisper.cpp over it.
func (p *AudioPipeline) runWhisper(audioPath string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), whisperTimeout)
	defer cancel()

	wav, err := os.CreateTemp("", "wa-transcribe-*.wav")
	if err != nil {
		return "", fmt.Errorf("create temp wav: %w", err)
	}
	wavPath := wav.Name()
	wav.Close()
	defer os.Remove(wavPath)

	ff := exec.CommandContext(ctx, "ffmpeg", "-v", "error", "-y",
		"-i", audioPath, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wavPath)
	if out, err := ff.CombinedOutput(); err != nil {
		return "", fmt.Errorf("ffmpeg: %w: %s", err, strings.TrimSpace(string(out)))
	}

	// -nt drops timestamps, -np drops progress chatter, leaving bare text on stdout.
	args := []string{"-m", p.cfg.model, "-f", wavPath,
		"-l", p.cfg.lang, "-t", strconv.Itoa(p.cfg.threads), "-nt", "-np"}
	name := p.cfg.cli
	if p.cfg.nice != "" {
		// Keep whisper off the event loop's back on a 4-core box.
		args = append([]string{"-n", "10", p.cfg.cli}, args...)
		name = p.cfg.nice
	}

	cmd := exec.CommandContext(ctx, name, args...)
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("whisper: %w", err)
	}
	return string(out), nil
}

// cleanTranscript strips whisper artefacts that would otherwise be indexed as
// speech. Both failure modes here were observed while benchmarking this corpus:
// bracketed non-speech tags on short noisy clips, and repetition loops.
func cleanTranscript(s string) string {
	s = bracketTag.ReplaceAllString(s, " ")
	words := strings.Fields(s)

	run := 1
	for i := 1; i < len(words); i++ {
		if strings.EqualFold(words[i], words[i-1]) {
			run++
			if run >= repeatRunLimit {
				words = words[:i-run+1]
				break
			}
		} else {
			run = 1
		}
	}

	if len(words) < minTranscriptWords {
		return ""
	}
	return strings.Join(words, " ")
}

// RunTranscribeBackfill transcribes every voice note whose media is already on
// disk, sequentially, then returns. Used by --transcribe-backfill to validate
// the pipeline end to end against existing data before it matters in the live
// path. No WhatsApp client is needed: expired media is simply skipped.
// maxRows caps how many voice notes are processed (0 = all), so the full run can
// be taken in chunks. sinceDays limits the window to recent messages (0 = all).
// When store was opened without an index, each transcript is written to SQLite
// only and a subsequent full --reindex is required to make it searchable.
func RunTranscribeBackfill(store *MessageStore, maxRows, sinceDays int) error {
	p := NewAudioPipeline(nil, store)
	if !p.cfg.enabled {
		return fmt.Errorf("transcription disabled")
	}
	if _, err := os.Stat(p.cfg.cli); err != nil {
		return fmt.Errorf("whisper CLI not found at %s", p.cfg.cli)
	}
	if _, err := os.Stat(p.cfg.model); err != nil {
		return fmt.Errorf("model not found at %s", p.cfg.model)
	}

	window := ""
	args := []interface{}{tsPending, tsDownloaded, tsFailed}
	if sinceDays > 0 {
		window = " AND timestamp >= datetime('now', ?)"
		args = append(args, fmt.Sprintf("-%d days", sinceDays))
	}

	rows, err := store.db.Query(
		`SELECT id, chat_jid, timestamp FROM messages
		 WHERE media_type = 'audio' AND (content IS NULL OR content = '')
		   AND (transcript_status IS NULL OR transcript_status IN (?, ?, ?))`+window+`
		 ORDER BY timestamp DESC`, args...)
	if err != nil {
		return fmt.Errorf("query audio messages: %w", err)
	}
	var jobs []audioJob
	for rows.Next() {
		var j audioJob
		if err := rows.Scan(&j.id, &j.chatJID, &j.ts); err == nil {
			jobs = append(jobs, j)
		}
	}
	rows.Close()

	// Only rows whose media survived are actionable.
	var local []audioJob
	for _, j := range jobs {
		if _, err := p.localMediaPath(j.id, j.chatJID); err == nil {
			local = append(local, j)
			if maxRows > 0 && len(local) >= maxRows {
				break
			}
		}
	}
	if maxRows > 0 {
		logger.Infof("Backfill: %d audio messages, processing %d (capped by --max-rows)", len(jobs), len(local))
	} else {
		logger.Infof("Backfill: %d audio messages, %d with media on disk (%d expired)",
			len(jobs), len(local), len(jobs)-len(local))
	}

	start := time.Now()
	for i, j := range local {
		p.transcribe(j)
		if (i+1)%10 == 0 || i+1 == len(local) {
			elapsed := time.Since(start)
			perItem := elapsed / time.Duration(i+1)
			logger.Infof("Backfill: %d/%d done, %.1f min elapsed, ~%.1f min remaining",
				i+1, len(local), elapsed.Minutes(),
				(perItem * time.Duration(len(local)-i-1)).Minutes())
		}
	}
	logger.Infof("Backfill complete in %.1f min", time.Since(start).Minutes())
	return nil
}

// sweeper re-enqueues rows the in-memory queues never got to: dropped enqueues,
// work interrupted by a restart, and transient failures.
func (p *AudioPipeline) sweeper() {
	defer p.wg.Done()
	p.sweep()
	ticker := time.NewTicker(sweepInterval)
	defer ticker.Stop()
	for {
		select {
		case <-p.stop:
			return
		case <-ticker.C:
			p.sweep()
		}
	}
}

func (p *AudioPipeline) sweep() {
	// tsFailed is retried; tsNoMedia, tsDone and tsEmpty are terminal.
	queued := p.enqueueByStatus([]string{tsPending, tsFailed}, p.downloads)
	ready := p.enqueueByStatus([]string{tsDownloaded}, p.transcripts)
	if queued+ready > 0 {
		logger.Infof("Transcription sweep: %d to download, %d to transcribe", queued, ready)
	}
}

func (p *AudioPipeline) enqueueByStatus(statuses []string, dest chan audioJob) int {
	placeholders := strings.TrimSuffix(strings.Repeat("?,", len(statuses)), ",")
	args := make([]interface{}, 0, len(statuses)+1)
	for _, s := range statuses {
		args = append(args, s)
	}
	args = append(args, sweepBatch)

	rows, err := p.store.db.Query(
		`SELECT id, chat_jid, timestamp FROM messages
		 WHERE transcript_status IN (`+placeholders+`)
		 ORDER BY timestamp DESC LIMIT ?`, args...)
	if err != nil {
		logger.Warnf("Transcription sweep query failed: %v", err)
		return 0
	}

	var jobs []audioJob
	for rows.Next() {
		var j audioJob
		if err := rows.Scan(&j.id, &j.chatJID, &j.ts); err == nil {
			jobs = append(jobs, j)
		}
	}
	rows.Close()

	n := 0
	for _, j := range jobs {
		select {
		case dest <- j:
			n++
		default:
			return n // queue full; the next sweep continues where this stopped
		}
	}
	return n
}
