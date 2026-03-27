package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/blevesearch/bleve/v2"
	"github.com/blevesearch/bleve/v2/mapping"
	"github.com/blevesearch/bleve/v2/search/query"

	_ "github.com/blevesearch/bleve/v2/analysis/lang/pt"
)

// MessageDocument represents a message for bleve indexing.
type MessageDocument struct {
	ID        string    `json:"id"`
	ChatJID   string    `json:"chat_jid"`
	Sender    string    `json:"sender"`
	FullName  string    `json:"full_name"`
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
	IsFromMe  bool      `json:"is_from_me"`
	MediaType string    `json:"media_type"`
	Filename  string    `json:"filename"`
	Embedding []float32 `json:"embedding"`
	Score     float64   `json:"score,omitempty"`
}

const indexPath = "store/messages.bleve"

// openOrCreateIndex opens the bleve index at indexPath, or creates a new one
// with hybrid (text + vector) field mappings.
func openOrCreateIndex(embDim int) (bleve.Index, error) {
	index, err := bleve.Open(indexPath)
	if err == bleve.ErrorIndexPathDoesNotExist {
		m := buildIndexMapping(embDim)
		index, err = bleve.NewUsing(indexPath, m, "scorch", "scorch", nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create bleve index: %w", err)
		}
		return index, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to open bleve index: %w", err)
	}
	return index, nil
}

// buildIndexMapping creates the bleve index mapping with text fields and a
// vector field for hybrid search.
func buildIndexMapping(embDim int) mapping.IndexMapping {
	m := bleve.NewIndexMapping()
	m.ScoringModel = "bm25"

	textFieldMapping := bleve.NewTextFieldMapping()
	textFieldMapping.Analyzer = "pt"


	m.DefaultMapping.AddFieldMappingsAt("content", textFieldMapping)
	m.DefaultMapping.AddFieldMappingsAt("sender", textFieldMapping)
	m.DefaultMapping.AddFieldMappingsAt("filename", textFieldMapping)

	// Vector field for semantic search.
	vectorFieldMapping := mapping.NewVectorFieldMapping()
	vectorFieldMapping.Dims = embDim
	vectorFieldMapping.Similarity = "cosine"
	m.DefaultMapping.AddFieldMappingsAt("embedding", vectorFieldMapping)

	return m
}

// indexMessage indexes a single message document into bleve, generating an
// embedding if the embedder is available and the message has text content.
func indexMessage(index bleve.Index, embedder *Embedder, id, chatJID, sender, fullName, content string, timestamp time.Time, isFromMe bool, mediaType, filename string) {
	docID := chatJID + ":" + id
	doc := MessageDocument{
		ID:        id,
		ChatJID:   chatJID,
		Sender:    sender,
		FullName:  fullName,
		Content:   content,
		Timestamp: timestamp,
		IsFromMe:  isFromMe,
		MediaType: mediaType,
		Filename:  filename,
	}

	if embedder != nil {
		if text := sanitizeForEmbedding(content); text != "" {
			vec, err := embedder.Embed(text)
			if err != nil {
				logger.Warnf("Failed to embed message %s: %v", docID, err)
			} else {
				doc.Embedding = vec
			}
		}
	}

	if err := index.Index(docID, doc); err != nil {
		logger.Warnf("Failed to index message %s: %v", docID, err)
	}
}

// reindexRow holds a scanned DB row during batch re-indexing.
type reindexRow struct {
	doc  MessageDocument
	text string // sanitised text to embed, or "" if skipped
}

const (
	dbPageSize = 500 // rows fetched from SQLite per query
	// embBatch is defined in embedding.go
)

// reIndexAllMessages re-indexes every message from the database into bleve
// using batch ONNX inference (embBatch texts per call) and bleve batch inserts
// (one commit per DB page).
func reIndexAllMessages(store *MessageStore, maxRows int) error {
	logger.Infof("Starting re-indexing of all messages...")

	// Skip if index already populated.
	count, err := store.index.DocCount()
	if err == nil && count > 0 {
		logger.Infof("Index already has %d documents, skipping re-index", count)
		return nil
	}

	var total int
	if err := store.db.QueryRow(`SELECT COUNT(*) FROM messages WHERE content != '' OR media_type != ''`).Scan(&total); err != nil {
		logger.Warnf("Could not count messages: %v", err)
	}
	if maxRows > 0 && total > maxRows {
		total = maxRows
	}
	logger.Infof("Re-indexing %d messages...", total)

	offset := 0
	indexed := 0
	startTime := time.Now()
	lastReport := startTime
	lastReportIdx := 0

	for {
		pageSize := dbPageSize
		if maxRows > 0 && indexed+pageSize > maxRows {
			pageSize = maxRows - indexed
		}

		rows, err := store.db.Query(`
			SELECT id, chat_jid, sender, full_name, content, timestamp, is_from_me, media_type, filename
			FROM messages
			WHERE content != '' OR media_type != ''
			ORDER BY rowid
			LIMIT ? OFFSET ?
		`, pageSize, offset)
		if err != nil {
			return fmt.Errorf("failed to query messages: %v", err)
		}

		// Scan the full page into memory.
		page := make([]reindexRow, 0, dbPageSize)
		for rows.Next() {
			var r reindexRow
			err := rows.Scan(&r.doc.ID, &r.doc.ChatJID, &r.doc.Sender, &r.doc.FullName,
				&r.doc.Content, &r.doc.Timestamp, &r.doc.IsFromMe, &r.doc.MediaType, &r.doc.Filename)
			if err != nil {
				logger.Warnf("Error scanning message: %v", err)
				continue
			}
			r.text = sanitizeForEmbedding(r.doc.Content)
			page = append(page, r)
		}
		rows.Close()

		if len(page) == 0 {
			break
		}

		// --- Batch embedding ---
		// Collect indices of rows that have embeddable text.
		if store.embedder != nil {
			embIdxs := make([]int, 0, embBatch)
			embTexts := make([]string, 0, embBatch)

			flush := func() {
				if len(embTexts) == 0 {
					return
				}
				vecs, err := store.embedder.EmbedBatch(embTexts)
				if err != nil {
					logger.Warnf("EmbedBatch failed: %v", err)
				} else {
					for j, idx := range embIdxs {
						page[idx].doc.Embedding = vecs[j]
					}
				}
				embIdxs = embIdxs[:0]
				embTexts = embTexts[:0]
			}

			for i := range page {
				if page[i].text == "" {
					continue
				}
				embIdxs = append(embIdxs, i)
				embTexts = append(embTexts, page[i].text)
				if len(embTexts) == embBatch {
					flush()
				}
			}
			flush() // remaining partial batch
		}

		// --- Bleve batch insert ---
		batch := store.index.NewBatch()
		for i := range page {
			docID := page[i].doc.ChatJID + ":" + page[i].doc.ID
			if err := batch.Index(docID, page[i].doc); err != nil {
				logger.Warnf("Failed to stage doc %s in batch: %v", docID, err)
			}
		}
		if err := store.index.Batch(batch); err != nil {
			logger.Warnf("Failed to commit bleve batch at offset %d: %v", offset, err)
		}

		indexed += len(page)

		now := time.Now()
		pageElapsed := now.Sub(lastReport).Seconds()
		if pageElapsed > 0 {
			pageMsgsPerSec := float64(indexed-lastReportIdx) / pageElapsed
			totalElapsed := now.Sub(startTime).Seconds()
			totalMsgsPerSec := float64(indexed) / totalElapsed
			avgTok := 0.0
			if store.embedder != nil {
				avgTok = store.embedder.AvgTokens()
			}
			logger.Debugf("Re-indexed %d/%d messages | page: %.0f msg/s | overall: %.0f msg/s | elapsed: %.1fs | avg tokens/msg: %.1f",
				indexed, total, pageMsgsPerSec, totalMsgsPerSec, totalElapsed, avgTok)
		}
		lastReport = now
		lastReportIdx = indexed

		if len(page) < pageSize || (maxRows > 0 && indexed >= maxRows) {
			break
		}
		offset += pageSize
	}

	totalElapsed := time.Since(startTime).Seconds()
	overallRate := 0.0
	if totalElapsed > 0 {
		overallRate = float64(indexed) / totalElapsed
	}
	logger.Infof("Re-indexed %d messages in %.1fs (%.0f msg/s)", indexed, totalElapsed, overallRate)
	return nil
}

// searchMessages performs a hybrid text + vector search using bleve's score
// fusion, then applies custom rescoring (mute penalty, user-ratio boost).
func searchMessages(store *MessageStore, queryStr string, chatJID string, limit int, offset int) ([]Message, error) {
	logger.Debugf("Searching for \"%s\" (chatJID=%s, limit=%d, offset=%d)", queryStr, chatJID, limit, offset)
	// Build text query.
	var textQuery query.Query
	if chatJID != "" {
		chatTermQuery := bleve.NewTermQuery(chatJID)
		chatTermQuery.SetField("chat_jid")
		booleanQuery := bleve.NewBooleanQuery()
		booleanQuery.AddMust(bleve.NewMatchQuery(queryStr))
		booleanQuery.AddMust(chatTermQuery)
		textQuery = booleanQuery
	} else {
		textQuery = bleve.NewMatchQuery(queryStr)
	}

	searchRequest := bleve.NewSearchRequest(textQuery)
	searchRequest.Size = limit
	searchRequest.From = offset
	searchRequest.SortBy([]string{"-_score", "-timestamp"})

	// Add kNN vector query if embedder is available.
	if store.embedder != nil {
		if text := sanitizeForEmbedding(queryStr); text != "" {
			queryVec, err := store.embedder.Embed(text)
			if err != nil {
				logger.Warnf("Failed to embed search query, falling back to text-only: %v", err)
			} else {
				searchRequest.AddKNN("embedding", queryVec, int64(limit) * 10, 0.75)
				searchRequest.Score = "rrf"
			}
		}
	} else {
		logger.Debugf("Not using embedder for this query");
	}

	params := bleve.RequestParams{
		ScoreWindowSize: 150,                     // Window size (default: size)
	}
	searchRequest.AddParams(params)

	logger.Debugf("searchRequest = %+v", searchRequest)

	searchResult, err := store.index.Search(searchRequest)
	if err != nil {
		return nil, err
	}

	// Apply custom rescoring and fetch full messages from SQLite.
	var results []Message
	chatRatios := make(map[string]float64)

	for _, hit := range searchResult.Hits {
		idParts := strings.SplitN(hit.ID, ":", 2)
		if len(idParts) != 2 {
			continue
		}
		hitChatJID := idParts[0]
		messageID := idParts[1]
		doc, err := store.GetMessage(messageID, hitChatJID)
		if err != nil {
			continue
		}

		// Mute penalty.
		muted, err := store.IsChatMuted(hitChatJID)
		if err == nil && muted {
			hit.Score *= 0.1
		}

		// User message ratio boost.
		ratio, exists := chatRatios[hitChatJID]
		if !exists {
			ratio, err = store.calculateUserMessageRatio(hitChatJID)
			if err != nil {
				ratio = 0.5
			}
			chatRatios[hitChatJID] = ratio
		}
		hit.Score *= (1.0 + ratio)

		results = append(results, *doc)
	}

	return results, nil
}

// deleteIndex removes the bleve index directory so it can be recreated.
func deleteIndex() error {
	if _, err := os.Stat(indexPath); os.IsNotExist(err) {
		return nil
	}
	return os.RemoveAll(indexPath)
}

// CustomRescorer implements custom scoring for search results.
type CustomRescorer struct {
	store *MessageStore
}

// Rescore applies custom scoring logic.
func (r *CustomRescorer) Rescore(ctx context.Context, searchResult *bleve.SearchResult) error {
	chatRatios := make(map[string]float64)

	for _, hit := range searchResult.Hits {
		parts := strings.SplitN(hit.ID, ":", 2)
		if len(parts) != 2 {
			continue
		}
		chatJID := parts[0]

		muted, err := r.store.IsChatMuted(chatJID)
		if err == nil && muted {
			hit.Score *= 0.1
		}

		ratio, exists := chatRatios[chatJID]
		if !exists {
			ratio, err = r.calculateUserMessageRatio(chatJID)
			if err != nil {
				ratio = 0.5
			}
			chatRatios[chatJID] = ratio
		}
		hit.Score *= (1.0 + ratio)
	}

	return nil
}

func (r *CustomRescorer) calculateUserMessageRatio(chatJID string) (float64, error) {
	var totalMessages, userMessages int
	err := r.store.db.QueryRow(`
		SELECT COUNT(*) as total,
		       SUM(CASE WHEN is_from_me = 1 THEN 1 ELSE 0 END) as user
		FROM messages
		WHERE chat_jid = ?
	`, chatJID).Scan(&totalMessages, &userMessages)

	if err != nil {
		return 0, err
	}

	if totalMessages == 0 {
		return 0, nil
	}

	return float64(userMessages) / float64(totalMessages), nil
}
