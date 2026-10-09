package main

import (
	"encoding/json"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"

	waLog "go.mau.fi/whatsmeow/util/log"
)

// TestEmbedderMemory embeds one batch per sequence-length bucket, then many
// mixed batches, and logs process anonymous memory (RssAnon+swap) after each step. It guards against
// the 2026-10 OOM: one ORT session (= one full model copy) per bucket grew the
// bridge to ~2.5 GiB of C heap. Memory must step once at load, then stay flat.
//
// Opt-in because it loads the real model:
//
//	EMB_MEM_TEST=1 go test -tags "vectors full" -run TestEmbedderMemory -v
//
// EMB_VEC_OUT=<file> writes the vectors; EMB_VEC_REF=<file> compares against
// an earlier run's file (cosine >= 0.9999) to check a refactor didn't change
// the embeddings.
func TestEmbedderMemory(t *testing.T) {
	if os.Getenv("EMB_MEM_TEST") != "1" {
		t.Skip("set EMB_MEM_TEST=1 to run")
	}
	logger = waLog.Stdout("Test", "INFO", false)
	before := procMemKB()
	e, err := NewEmbedder("", "")
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	defer e.Close()
	t.Logf("after load:      rss+swap=%6d MB (+%d)", procMemKB()/1024, (procMemKB()-before)/1024)

	// One text per bucket, sized to land in exactly that bucket.
	var texts []string
	words := strings.Fields("hoje vamos comprar o bolo de aniversario para a festa da escola amanha cedo")
	for _, target := range seqLenBuckets {
		var sb strings.Builder
		for i := 0; len(e.tok.Encode(sb.String())) < target-2; i++ {
			sb.WriteString(words[i%len(words)])
			sb.WriteByte(' ')
		}
		texts = append(texts, sb.String())
	}

	base := procMemKB()
	var vecs [][]float32
	for i, text := range texts {
		v, err := e.Embed(text)
		if err != nil {
			t.Fatalf("Embed bucket %d: %v", seqLenBuckets[i], err)
		}
		vecs = append(vecs, v)
		t.Logf("bucket %3d:      rss+swap=%6d MB (+%d since first embed)", seqLenBuckets[i], procMemKB()/1024, (procMemKB()-base)/1024)
	}

	// Mixed batches of varying size and length, as live indexing produces.
	rounds := 100
	if v, err := strconv.Atoi(os.Getenv("EMB_MEM_ROUNDS")); err == nil && v > 0 {
		rounds = v
	}
	for round := 0; round < rounds; round++ {
		if round > 0 && round%50 == 0 {
			t.Logf("mixed round %4d: rss+swap=%6d MB", round, procMemKB()/1024)
		}
		n := 1 + round%embBatch
		batch := make([]string, n)
		for j := range batch {
			batch[j] = texts[(round*3+j)%len(texts)]
		}
		if _, err := e.EmbedBatch(batch); err != nil {
			t.Fatalf("EmbedBatch round %d: %v", round, err)
		}
	}
	t.Logf("after mixed:     rss+swap=%6d MB (+%d since first embed)", procMemKB()/1024, (procMemKB()-base)/1024)

	// A text padded to a longer bucket by its batch-mates embeds almost, not
	// exactly, the same: the qint8 model quantizes activations per tensor, so
	// padding shifts the scale slightly (~0.997 cosine, same before the fix).
	batched, err := e.EmbedBatch([]string{texts[0], texts[len(texts)-1]})
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	if c := cosine(batched[0], vecs[0]); c < 0.99 {
		t.Errorf("batched vs single embedding differ: cosine=%.6f", c)
	}

	if out := os.Getenv("EMB_VEC_OUT"); out != "" {
		b, _ := json.Marshal(vecs)
		if err := os.WriteFile(out, b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if ref := os.Getenv("EMB_VEC_REF"); ref != "" {
		b, err := os.ReadFile(ref)
		if err != nil {
			t.Fatal(err)
		}
		var want [][]float32
		if err := json.Unmarshal(b, &want); err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if c := cosine(want[i], vecs[i]); c < 0.9999 {
				t.Errorf("bucket %d: cosine vs reference = %.6f", seqLenBuckets[i], c)
			}
		}
	}
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / math.Sqrt(na*nb)
}
