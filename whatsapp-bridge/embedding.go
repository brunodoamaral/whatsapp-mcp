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
	"strings"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers"
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

// seqLenBuckets defines the allowed sequence lengths for cached sessions.
// padLen is rounded up to the first bucket that fits.
var seqLenBuckets = []int{16, 32, 48, 56, 64, 72, 80, 96, 128}

// embeddingModelConfig holds relevant fields from config.json at the repo root.
type embeddingModelConfig struct {
	HiddenSize            int `json:"hidden_size"`
	Dim                   int `json:"dim"`
	MaxPositionEmbeddings int `json:"max_position_embeddings"`
}

// sessionEntry holds a cached ONNX session and its pre-allocated tensors for a
// specific sequence length.
type sessionEntry struct {
	seqLen    int
	flatIDs   []int64
	flatMask  []int64
	flatTypes []int64
	inIDs     *ort.Tensor[int64]
	inMask    *ort.Tensor[int64]
	inTypes   *ort.Tensor[int64]
	outTensor *ort.Tensor[float32]
	session   *ort.AdvancedSession
}

// Embedder wraps an ONNX sentence-transformer model for generating text embeddings.
type Embedder struct {
	tok        tokenizers.Tokenizer
	onnxPath   string
	inputNames []string
	outputInfo []ort.InputOutputInfo
	outIdx     int
	embDim     int
	maxSeqLen  int
	// sessions is a cache of ONNX sessions keyed by sequence length.
	// Each unique padLen encountered gets its own session, created on first use.
	sessions map[int]*sessionEntry
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
// the tokenizer. Call Close() when done.
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

	outIdx := chooseOutputIndex(outputInfo)

	return &Embedder{
		tok:        tok,
		onnxPath:   onnxPath,
		inputNames: inputNames,
		outputInfo: outputInfo,
		outIdx:     outIdx,
		embDim:     embDim,
		maxSeqLen:  maxSeqLen,
		sessions:   make(map[int]*sessionEntry),
	}, nil
}

// getSession returns a cached session for the given seqLen, creating one if needed.
func (e *Embedder) getSession(seqLen int) (*sessionEntry, error) {
	if s, ok := e.sessions[seqLen]; ok {
		return s, nil
	}

	n := embBatch * seqLen
	s := &sessionEntry{
		seqLen:    seqLen,
		flatIDs:   make([]int64, n),
		flatMask:  make([]int64, n),
		flatTypes: make([]int64, n),
	}

	shape := ort.NewShape(int64(embBatch), int64(seqLen))

	var err error
	s.inIDs, err = ort.NewTensor(shape, s.flatIDs)
	if err != nil {
		return nil, fmt.Errorf("create input_ids tensor: %w", err)
	}
	s.inMask, err = ort.NewTensor(shape, s.flatMask)
	if err != nil {
		s.inIDs.Destroy()
		return nil, fmt.Errorf("create attention_mask tensor: %w", err)
	}
	s.inTypes, err = ort.NewTensor(shape, s.flatTypes)
	if err != nil {
		s.inIDs.Destroy()
		s.inMask.Destroy()
		return nil, fmt.Errorf("create token_type_ids tensor: %w", err)
	}

	knownInputs := map[string]*ort.Tensor[int64]{
		"input_ids":      s.inIDs,
		"attention_mask": s.inMask,
		"token_type_ids": s.inTypes,
	}
	sessionInputs := make([]ort.ArbitraryTensor, len(e.inputNames))
	for i, name := range e.inputNames {
		t, ok := knownInputs[name]
		if !ok {
			s.inIDs.Destroy()
			s.inMask.Destroy()
			s.inTypes.Destroy()
			return nil, fmt.Errorf("model requires input %q but no tensor was prepared", name)
		}
		sessionInputs[i] = t
	}

	chosenOutInfo := e.outputInfo[e.outIdx]
	outDims := make([]int64, len(chosenOutInfo.Dimensions))
	for j, d := range chosenOutInfo.Dimensions {
		switch {
		case d > 0:
			outDims[j] = d
		case j == 0:
			outDims[j] = int64(embBatch)
		case j == 1:
			outDims[j] = int64(seqLen)
		default:
			outDims[j] = int64(e.embDim)
		}
	}
	s.outTensor, err = ort.NewEmptyTensor[float32](ort.NewShape(outDims...))
	if err != nil {
		s.inIDs.Destroy()
		s.inMask.Destroy()
		s.inTypes.Destroy()
		return nil, fmt.Errorf("create output tensor: %w", err)
	}

	s.session, err = ort.NewAdvancedSession(
		e.onnxPath,
		e.inputNames,
		[]string{chosenOutInfo.Name},
		sessionInputs,
		[]ort.ArbitraryTensor{s.outTensor},
		nil,
	)
	if err != nil {
		s.inIDs.Destroy()
		s.inMask.Destroy()
		s.inTypes.Destroy()
		s.outTensor.Destroy()
		return nil, fmt.Errorf("create ORT session: %w", err)
	}

	e.sessions[seqLen] = s
	return s, nil
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
// padded to the longest one in the batch; a session is created for that
// sequence length on first use and reused on subsequent calls with the same
// length. Returns one vector per input text (same order).
func (e *Embedder) EmbedBatch(texts []string) ([][]float32, error) {
	batchSize := len(texts)
	if batchSize == 0 {
		return nil, nil
	}
	if batchSize > embBatch {
		return nil, fmt.Errorf("batch size %d exceeds session batch size %d", batchSize, embBatch)
	}

	// Tokenize all texts; find actual padLen for this batch.
	tokenized := make([][]int64, batchSize)
	padLen := 0
	for i, text := range texts {
		rawIDs := e.tok.Encode(text)
		maxLen := e.maxSeqLen
		if maxLen > 128 {
			maxLen = 128
		}
		if len(rawIDs) > maxLen {
			rawIDs = rawIDs[:maxLen]
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

	// Round up to the next pre-defined bucket to bound the number of cached sessions.
	seqLen := seqLenBuckets[len(seqLenBuckets)-1]
	for _, b := range seqLenBuckets {
		if b >= padLen {
			seqLen = b
			break
		}
	}

	s, err := e.getSession(seqLen)
	if err != nil {
		return nil, err
	}

	// Zero out buffers, then fill actual data.
	for i := range s.flatIDs {
		s.flatIDs[i] = 0
		s.flatMask[i] = 0
	}
	for i, ids := range tokenized {
		base := i * seqLen
		for j, id := range ids {
			s.flatIDs[base+j] = id
			s.flatMask[base+j] = 1
		}
	}

	if err := s.session.Run(); err != nil {
		return nil, fmt.Errorf("ORT inference: %w", err)
	}

	pooled := meanPoolOutput(s.outTensor.GetData(), s.outTensor.GetShape(), s.flatMask, embBatch, seqLen, e.embDim)

	result := make([][]float32, batchSize)
	for i := range result {
		vec := make([]float32, e.embDim)
		copy(vec, pooled[i*e.embDim:(i+1)*e.embDim])
		l2Normalize(vec)
		result[i] = vec
	}
	return result, nil
}

// Close releases ONNX Runtime resources.
func (e *Embedder) Close() {
	for _, s := range e.sessions {
		s.session.Destroy()
		s.outTensor.Destroy()
		s.inTypes.Destroy()
		s.inMask.Destroy()
		s.inIDs.Destroy()
	}
	ort.DestroyEnvironment()
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
