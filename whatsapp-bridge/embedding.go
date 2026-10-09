package main

import (
	"archive/tar"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	defaultEmbeddingModelID  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
	defaultEmbeddingOnnxFile = "onnx/model_qint8_arm64.onnx"

	// ONNX Runtime release that matches onnxruntime_go v1.27.0 C API headers.
	ortVersion = "1.24.1"
	ortBaseURL = "https://github.com/microsoft/onnxruntime/releases/download/v" + ortVersion + "/"
	ortTarball = "onnxruntime-linux-aarch64-" + ortVersion + ".tgz"
	ortLibName = "libonnxruntime.so." + ortVersion

	// embBatch is the number of texts per ONNX inference call. The persistent
	// session is pre-allocated for exactly this many inputs. Must match the
	// constant of the same name in search.go.
	embBatch = 8
)

// seqLenBuckets defines the allowed padded sequence lengths. padLen is rounded
// up to the first bucket that fits, which keeps the set of input shapes small
// and keeps embeddings identical to those already in the index (padding length
// affects the qint8 model's dynamic quantization slightly).
var seqLenBuckets = []int{16, 32, 64, 80, 96, 112, 128, 140, 160, 200, 220, 256}
var maxSequenceLen = seqLenBuckets[len(seqLenBuckets)-1]

// embeddingModelConfig holds relevant fields from config.json at the repo root.
type embeddingModelConfig struct {
	HiddenSize            int `json:"hidden_size"`
	Dim                   int `json:"dim"`
	MaxPositionEmbeddings int `json:"max_position_embeddings"`
}

// Embedder wraps an ONNX sentence-transformer model for generating text embeddings.
//
// It holds exactly one ORT session, with dynamic input shapes. An earlier version
// cached one fixed-shape session per seqLenBuckets entry, and every ORT session
// loads and optimizes its own copy of the model (~130-200 MB of C heap each), so
// the bridge grew to ~2 GB as longer messages arrived. That growth caused the
// 2026-10-08 OOM (see CLAUDE.md). Do not reintroduce per-shape sessions.
//
// EmbedBatch serialises on mu: the transcription worker embeds from its own
// goroutine while the event loop indexes new messages.
type Embedder struct {
	mu         sync.Mutex
	tok        tokenizers.Tokenizer
	session    *ort.DynamicAdvancedSession
	inputNames []string
	outName    string
	embDim     int
	maxSeqLen  int
	padTokenId int
	// seenSeqLens records which padded lengths have run, so the first use of
	// each is logged with process memory — a cheap regression signal.
	seenSeqLens map[int]bool
	// token stats for reporting
	totalTokens int64
	totalTexts  int64
}

// AvgTokens returns the average number of tokens per embedded text seen so far.
func (e *Embedder) AvgTokens() float64 {
	if e.totalTexts == 0 {
		return 0
	}
	return float64(e.totalTokens) / float64(e.totalTexts)
}

// NewEmbedder initialises the ONNX Runtime, downloads the model, and prepares
// the tokenizer and the session. Call Close() when done.
func NewEmbedder(modelID, onnxFile string) (*Embedder, error) {
	if modelID == "" {
		modelID = defaultEmbeddingModelID
	}
	if onnxFile == "" {
		onnxFile = defaultEmbeddingOnnxFile
	}

	// 1. Ensure ONNX Runtime shared library is available.
	ortLib, err := ensureONNXRuntime()
	if err != nil {
		return nil, fmt.Errorf("ONNX Runtime setup: %w", err)
	}
	ort.SetSharedLibraryPath(ortLib)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("initialize ORT environment: %w", err)
	}

	// 2. Download model files via go-huggingface hub.
	repo := hub.New(modelID).WithProgressBar(false)

	cfg, err := loadEmbeddingModelConfig(repo)
	if err != nil {
		ort.DestroyEnvironment()
		return nil, fmt.Errorf("load model config: %w", err)
	}
	embDim := cfg.HiddenSize
	maxSeqLen := cfg.MaxPositionEmbeddings

	onnxPath, err := repo.DownloadFile(onnxFile)
	if err != nil {
		ort.DestroyEnvironment()
		return nil, fmt.Errorf("download ONNX model: %w", err)
	}

	// Try to download external data file (some models split weights).
	if dataPath, err := repo.DownloadFile(onnxFile + "_data"); err == nil {
		_ = dataPath // just ensure it's cached alongside the model
	}

	// 3. Introspect model inputs/outputs.
	inputInfo, outputInfo, err := ort.GetInputOutputInfo(onnxPath)
	if err != nil {
		ort.DestroyEnvironment()
		return nil, fmt.Errorf("get ONNX input/output info: %w", err)
	}
	inputNames := make([]string, len(inputInfo))
	for i, info := range inputInfo {
		inputNames[i] = info.Name
	}
	for _, name := range inputNames {
		switch name {
		case "input_ids", "attention_mask", "token_type_ids":
		default:
			ort.DestroyEnvironment()
			return nil, fmt.Errorf("model requires unsupported input %q (inputs: %v)", name, inputNames)
		}
	}
	hasInputIDs := false
	for _, name := range inputNames {
		if name == "input_ids" {
			hasInputIDs = true
			break
		}
	}
	if !hasInputIDs {
		ort.DestroyEnvironment()
		return nil, fmt.Errorf("model does not expose 'input_ids' as an input; got: %v", inputNames)
	}

	// 4. Load tokenizer.
	tok, err := tokenizers.New(repo)
	if err != nil {
		ort.DestroyEnvironment()
		return nil, fmt.Errorf("create tokenizer: %w", err)
	}

	// 5. Get PadToken ID from tokenizer config, if available.
	padTokenId, err := tok.SpecialTokenID(api.TokPad)
	if err != nil {
		ort.DestroyEnvironment()
		return nil, fmt.Errorf("create tokenizer: %w", err)
	}

	outName := outputInfo[chooseOutputIndex(outputInfo)].Name

	// 6. Create the single session.
	session, err := newEmbeddingSession(onnxPath, inputNames, outName)
	if err != nil {
		ort.DestroyEnvironment()
		return nil, err
	}
	logger.Infof("Embedding session created: model=%s output=%s mem=%dMB", onnxFile, outName, procMemKB()/1024)

	return &Embedder{
		tok:         tok,
		session:     session,
		inputNames:  inputNames,
		outName:     outName,
		embDim:      embDim,
		maxSeqLen:   maxSeqLen,
		padTokenId:  padTokenId,
		seenSeqLens: make(map[int]bool),
	}, nil
}

// newEmbeddingSession creates the one dynamic-shape session the Embedder uses.
func newEmbeddingSession(onnxPath string, inputNames []string, outName string) (*ort.DynamicAdvancedSession, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("create ORT session options: %w", err)
	}
	defer opts.Destroy()
	// Embedding here is small batches on a low-core box, so single-threaded
	// sequential execution is plenty and keeps the OS thread count flat.
	if err := opts.SetIntraOpNumThreads(1); err != nil {
		return nil, fmt.Errorf("set intra-op threads: %w", err)
	}
	if err := opts.SetInterOpNumThreads(1); err != nil {
		return nil, fmt.Errorf("set inter-op threads: %w", err)
	}
	if err := opts.SetExecutionMode(ort.ExecutionModeSequential); err != nil {
		return nil, fmt.Errorf("set execution mode: %w", err)
	}
	// Input shapes vary per call. Without these, ORT's CPU arena and its
	// per-shape memory-pattern cache keep the high-water buffers of every shape
	// seen. Freeing activations after each run costs little at this batch size.
	if err := opts.SetCpuMemArena(false); err != nil {
		return nil, fmt.Errorf("disable CPU mem arena: %w", err)
	}
	if err := opts.SetMemPattern(false); err != nil {
		return nil, fmt.Errorf("disable mem pattern: %w", err)
	}
	session, err := ort.NewDynamicAdvancedSession(onnxPath, inputNames, []string{outName}, opts)
	if err != nil {
		return nil, fmt.Errorf("create ORT session: %w", err)
	}
	return session, nil
}

// EmbDim returns the embedding dimensionality.
func (e *Embedder) EmbDim() int {
	return e.embDim
}

// Embed generates a normalised embedding vector for a single text.
func (e *Embedder) Embed(text string) ([]float32, error) {
	vecs, err := e.EmbedBatch([]string{text})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

// EmbedBatch generates normalised embedding vectors for a batch of texts in a
// single ONNX inference call. len(texts) must be <= embBatch. Sequences are
// padded to the longest one in the batch, rounded up to a seqLenBuckets entry.
// The batch is padded to embBatch rows, as it always has been, so vectors stay
// bit-compatible with the existing index. Returns one vector per input text
// (same order).
func (e *Embedder) EmbedBatch(texts []string) ([][]float32, error) {
	batchSize := len(texts)
	if batchSize == 0 {
		return nil, nil
	}
	if batchSize > embBatch {
		return nil, fmt.Errorf("batch size %d exceeds session batch size %d", batchSize, embBatch)
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	// Tokenize all texts; find actual padLen for this batch.
	tokenized := make([][]int64, batchSize)
	padLen := 0
	efectiveMaxSeqLen := e.maxSeqLen
	if efectiveMaxSeqLen > maxSequenceLen {
		efectiveMaxSeqLen = maxSequenceLen
	}
	for i, text := range texts {
		rawIDs := e.tok.Encode(text)
		if len(rawIDs) > efectiveMaxSeqLen {
			rawIDs = rawIDs[:efectiveMaxSeqLen]
		}
		ids := make([]int64, len(rawIDs))
		for j, id := range rawIDs {
			ids[j] = int64(id)
		}
		tokenized[i] = ids
		if len(ids) > padLen {
			padLen = len(ids)
		}
		e.totalTokens += int64(len(ids))
		e.totalTexts++
	}
	if padLen == 0 {
		return nil, fmt.Errorf("all inputs produced empty token sequences")
	}

	// Round up to the next pre-defined bucket to bound the number of shapes.
	seqLen := seqLenBuckets[len(seqLenBuckets)-1]
	for _, b := range seqLenBuckets {
		if b >= padLen {
			seqLen = b
			break
		}
	}

	n := embBatch * seqLen
	flatIDs := make([]int64, n)
	flatMask := make([]int64, n)
	flatTypes := make([]int64, n)
	for i := range flatIDs {
		flatIDs[i] = int64(e.padTokenId)
	}
	for i, ids := range tokenized {
		base := i * seqLen
		for j, id := range ids {
			flatIDs[base+j] = id
			flatMask[base+j] = 1
		}
	}

	shape := ort.NewShape(int64(embBatch), int64(seqLen))
	buffers := map[string][]int64{
		"input_ids":      flatIDs,
		"attention_mask": flatMask,
		"token_type_ids": flatTypes,
	}
	inputs := make([]ort.Value, len(e.inputNames))
	defer func() {
		for _, v := range inputs {
			if v != nil {
				v.Destroy()
			}
		}
	}()
	for i, name := range e.inputNames {
		t, err := ort.NewTensor(shape, buffers[name])
		if err != nil {
			return nil, fmt.Errorf("create %s tensor: %w", name, err)
		}
		inputs[i] = t
	}

	outputs := []ort.Value{nil} // allocated by ORT to the shape it produces
	if err := e.session.Run(inputs, outputs); err != nil {
		return nil, fmt.Errorf("ORT inference: %w", err)
	}
	defer outputs[0].Destroy()
	out, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("ORT output %s is %T, want float32 tensor", e.outName, outputs[0])
	}

	pooled := meanPoolOutput(out.GetData(), out.GetShape(), flatMask, embBatch, seqLen, e.embDim)

	result := make([][]float32, batchSize)
	for i := range result {
		vec := make([]float32, e.embDim)
		copy(vec, pooled[i*e.embDim:(i+1)*e.embDim])
		l2Normalize(vec)
		result[i] = vec
	}

	if !e.seenSeqLens[seqLen] {
		e.seenSeqLens[seqLen] = true
		logger.Infof("Embedding: first batch at seqLen=%d mem=%dMB", seqLen, procMemKB()/1024)
	}
	return result, nil
}

// Close releases ONNX Runtime resources.
func (e *Embedder) Close() {
	e.session.Destroy()
	ort.DestroyEnvironment()
}

// procMemKB returns this process's anonymous memory, RssAnon+VmSwap, in kB
// (0 if unreadable). Embedding memory is C heap that Go's runtime stats don't
// see. File-backed pages (mmapped bleve segments, messages.db) are left out on
// purpose: they are reclaimable cache, so they would hide or fake a leak.
func procMemKB() int64 {
	b, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0
	}
	var total int64
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, "RssAnon:") || strings.HasPrefix(line, "VmSwap:") {
			if fields := strings.Fields(line); len(fields) >= 2 {
				n, _ := strconv.ParseInt(fields[1], 10, 64)
				total += n
			}
		}
	}
	return total
}

// --- helper functions ported from test-embedding ---

func loadEmbeddingModelConfig(repo *hub.Repo) (embeddingModelConfig, error) {
	path, err := repo.DownloadFile("config.json")
	if err != nil {
		return embeddingModelConfig{}, fmt.Errorf("download config.json: %w", err)
	}
	f, err := os.Open(path)
	if err != nil {
		return embeddingModelConfig{}, err
	}
	defer f.Close()
	var cfg embeddingModelConfig
	if err := json.NewDecoder(f).Decode(&cfg); err != nil {
		return embeddingModelConfig{}, fmt.Errorf("parse config.json: %w", err)
	}
	if cfg.HiddenSize == 0 && cfg.Dim != 0 {
		cfg.HiddenSize = cfg.Dim
	}
	if cfg.HiddenSize == 0 || cfg.MaxPositionEmbeddings == 0 {
		return embeddingModelConfig{}, fmt.Errorf("config.json missing hidden_size or max_position_embeddings")
	}
	return cfg, nil
}

func chooseOutputIndex(outputInfo []ort.InputOutputInfo) int {
	for i, info := range outputInfo {
		if len(info.Dimensions) == 2 {
			return i
		}
	}
	return 0
}

func meanPoolOutput(data []float32, shape ort.Shape, flatMask []int64, batchSize, seqLen, embDim int) []float32 {
	if len(shape) == 2 {
		return data
	}
	result := make([]float32, batchSize*embDim)
	for b := 0; b < batchSize; b++ {
		var count float32
		for s := 0; s < seqLen; s++ {
			if flatMask[b*seqLen+s] == 0 {
				continue
			}
			count++
			base := (b*seqLen + s) * embDim
			for j := 0; j < embDim; j++ {
				result[b*embDim+j] += data[base+j]
			}
		}
		if count > 0 {
			for j := 0; j < embDim; j++ {
				result[b*embDim+j] /= count
			}
		}
	}
	return result
}

func l2Normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if norm := float32(math.Sqrt(sum)); norm > 0 {
		for i := range v {
			v[i] /= norm
		}
	}
}

func ensureONNXRuntime() (string, error) {
	sysPaths := []string{
		"/usr/lib/aarch64-linux-gnu/" + ortLibName,
		"/usr/lib/aarch64-linux-gnu/libonnxruntime.so",
		"/usr/local/lib/" + ortLibName,
		"/usr/local/lib/libonnxruntime.so",
	}
	for _, p := range sysPaths {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}

	cacheDir, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("get user cache dir: %w", err)
	}
	ortDir := filepath.Join(cacheDir, "onnxruntime")
	libPath := filepath.Join(ortDir, ortLibName)

	if _, err := os.Stat(libPath); err == nil {
		return libPath, nil
	}

	// Download and extract.
	if err := os.MkdirAll(ortDir, 0o755); err != nil {
		return "", fmt.Errorf("create cache dir: %w", err)
	}
	url := ortBaseURL + ortTarball
	if err := downloadAndExtractORT(url, ortDir, ortLibName); err != nil {
		return "", fmt.Errorf("download ONNX Runtime: %w", err)
	}
	return libPath, nil
}

func downloadAndExtractORT(url, destDir, wantFile string) error {
	resp, err := http.Get(url) //nolint:gosec
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %s", resp.Status)
	}

	gz, err := gzip.NewReader(resp.Body)
	if err != nil {
		return err
	}
	defer gz.Close()

	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if filepath.Base(hdr.Name) != wantFile {
			continue
		}
		dest := filepath.Join(destDir, wantFile)
		f, err := os.OpenFile(dest, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755)
		if err != nil {
			return err
		}
		if _, err := io.Copy(f, tr); err != nil {
			f.Close()
			return err
		}
		return f.Close()
	}
	return fmt.Errorf("file %q not found in tarball", wantFile)
}

// sanitizeForEmbedding checks whether text is worth embedding.
func sanitizeForEmbedding(content string) string {
	content = strings.TrimSpace(content)
	if len(content) < 2 {
		return ""
	}
	return content
}
