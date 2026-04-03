package main

import (
	"database/sql"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/blevesearch/bleve/v2"
	"github.com/blevesearch/bleve/v2/analysis"
	"github.com/blevesearch/bleve/v2/mapping"
	"github.com/blevesearch/bleve/v2/registry"
	"github.com/blevesearch/bleve/v2/search/query"
	"github.com/schollz/progressbar/v3"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"

	_ "github.com/blevesearch/bleve/v2/analysis/lang/pt"
	_ "github.com/blevesearch/bleve/v2/analysis/token/unicodenorm"
)

// MessageDocument represents a context group document for bleve indexing.
// One document is created per group of contextNumMessages messages.
// ChatJID and group number are encoded in the bleve doc key (chatJID:group).
type MessageDocument struct {
	ChatJID        string    `json:"chat_jid"`
	Context        string    `json:"context"`
	TimestampFirst time.Time `json:"timestamp_first"`
	TimestampLast  time.Time `json:"timestamp_last"`
	Embedding      []float32 `json:"embedding"`
}

// contextMsg is a lightweight message used for building conversation context windows.
type contextMsg struct {
	Sender         string
	FullName       string
	Content        string
	ReplyToContent string
	Timestamp      time.Time
}

// SearchResult groups search hits by conversation context window.
type SearchResult struct {
	ChatJID  string    `json:"chat_jid"`
	ChatName string    `json:"chat_name"`
	Score    float64   `json:"score"`
	Messages []Message `json:"messages"`
}

const indexPath = "store/messages.bleve"

const asciiFoldingFilterName = "ascii_folding_custom"
const ptAsciiAnalyzerName = "pt_ascii"

// asciiFoldingTokenFilter strips diacritical marks via NFD decomposition.
type asciiFoldingTokenFilter struct{}

func (f *asciiFoldingTokenFilter) Filter(input analysis.TokenStream) analysis.TokenStream {
	t := transform.Chain(norm.NFD, transform.RemoveFunc(func(r rune) bool {
		return unicode.Is(unicode.Mn, r) // strip non-spacing marks
	}), norm.NFC)
	for _, token := range input {
		result, _, _ := transform.Bytes(t, token.Term)
		token.Term = result
	}
	return input
}

func init() {
	if err := registry.RegisterTokenFilter(asciiFoldingFilterName, func(_ map[string]interface{}, _ *registry.Cache) (analysis.TokenFilter, error) {
		return &asciiFoldingTokenFilter{}, nil
	}); err != nil {
		panic(err)
	}

	// Register pt_ascii globally so it is available both when creating a new
	// index and when reopening an existing one (bleve reconstructs analyzers
	// from the stored metadata using the global registry).
	if err := registry.RegisterAnalyzer(ptAsciiAnalyzerName, func(_ map[string]interface{}, cache *registry.Cache) (analysis.Analyzer, error) {
		tokenizer, err := cache.TokenizerNamed("unicode")
		if err != nil {
			return nil, err
		}
		toLower, err := cache.TokenFilterNamed("to_lower")
		if err != nil {
			return nil, err
		}
		asciiFilter, err := cache.TokenFilterNamed(asciiFoldingFilterName)
		if err != nil {
			return nil, err
		}
		stopPt, err := cache.TokenFilterNamed("stop_pt")
		if err != nil {
			return nil, err
		}
		stemPt, err := cache.TokenFilterNamed("stemmer_pt_light")
		if err != nil {
			return nil, err
		}
		return &analysis.DefaultAnalyzer{
			Tokenizer: tokenizer,
			TokenFilters: []analysis.TokenFilter{
				toLower,
				asciiFilter,
				stopPt,
				stemPt,
			},
		}, nil
	}); err != nil {
		panic(err)
	}
}

const contextNumMessages = 16 // messages per indexed context group

// formatContextWindow builds a context string from a slice of messages.
func formatContextWindow(msgs []contextMsg) string {
	var b strings.Builder
	for i, m := range msgs {
		if m.Content == "" {
			continue
		}
		name := m.FullName
		if name == "" {
			name = m.Sender
		}
		if i > 0 {
			b.WriteByte('\n')
		}
		b.WriteString("[")
		b.WriteString(name)
		b.WriteString("]: ")
		if m.ReplyToContent != "" {
			b.WriteString("> ")
			// Replace \n in the replied-to content with spaces to keep it one line, and truncate to 100 chars for context.
			replyFormated := strings.ReplaceAll(m.ReplyToContent, "\n", " ")
			if len(replyFormated) > 100 {
				replyFormated = replyFormated[:100] + "..."
			}
			b.WriteString(replyFormated) // include a snippet of the replied-to message for context
			b.WriteString("\n")
		}
		b.WriteString(m.Content)
	}
	return b.String()
}

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
	m.DefaultAnalyzer = ptAsciiAnalyzerName

	textFieldMapping := bleve.NewTextFieldMapping()
	textFieldMapping.Analyzer = ptAsciiAnalyzerName

	m.DefaultMapping.AddFieldMappingsAt("context", textFieldMapping)

	// Vector field for semantic search.
	vectorFieldMapping := mapping.NewVectorFieldMapping()
	vectorFieldMapping.Dims = embDim
	vectorFieldMapping.Similarity = "cosine"
	m.DefaultMapping.AddFieldMappingsAt("embedding", vectorFieldMapping)

	return m
}

// indexMessage indexes or updates the context group document for a newly stored
// message. The doc ID is chatJID:groupNumber, so appending a message to an
// existing group is a simple upsert (bleve Index is idempotent by key).
func indexMessage(index bleve.Index, embedder *Embedder, db *sql.DB, id, chatJID, sender, fullName, content string, timestamp time.Time, isFromMe bool, mediaType, filename string) {
	group := 0
	var groupMsgs []contextMsg

	if db != nil {
		// Count indexed messages for this chat (the new message is already stored).
		var count int
		if err := db.QueryRow(
			`SELECT COUNT(*) FROM messages WHERE chat_jid = ? AND (content != '' OR media_type != '')`,
			chatJID,
		).Scan(&count); err != nil {
			logger.Warnf("Failed to count messages for %s: %v", chatJID, err)
			count = 1
		}

		messageJIDOrder := count - 1 // 0-indexed position of this message
		group = messageJIDOrder / contextNumMessages
		groupStart := group * contextNumMessages
		groupSize := messageJIDOrder - groupStart + 1

		// Fetch all messages in the current group from SQLite.
		groupRows, err := db.Query(`
			SELECT m.sender, m.full_name, m.content, m.timestamp, r.content
			FROM messages m
			LEFT JOIN messages r ON m.reply_to_id = r.id
			WHERE m.chat_jid = ? AND (m.content != '' OR m.media_type != '')
			ORDER BY m.timestamp LIMIT ? OFFSET ?`,
			chatJID, groupSize, groupStart,
		)
		if err != nil {
			logger.Warnf("Failed to fetch group messages for %s group %d: %v", chatJID, group, err)
		} else {
			for groupRows.Next() {
				var m contextMsg
				if scanErr := groupRows.Scan(&m.Sender, &m.FullName, &m.Content, &m.Timestamp, &m.ReplyToContent); scanErr == nil {
					groupMsgs = append(groupMsgs, m)
				}
			}
			groupRows.Close()
		}
	}

	ctxStr := formatContextWindow(groupMsgs)
	embText := ctxStr
	if embText == "" {
		embText = content
	}

	firstTimestamp := timestamp
	if len(groupMsgs) > 0 {
		firstTimestamp = groupMsgs[0].Timestamp
	}

	doc := MessageDocument{
		ChatJID:        chatJID,
		Context:        ctxStr,
		TimestampFirst: firstTimestamp,
		TimestampLast:  timestamp,
	}

	if embedder != nil {
		if text := sanitizeForEmbedding(embText); text != "" {
			vec, err := embedder.Embed(text)
			if err != nil {
				logger.Warnf("Failed to embed message %s group %d: %v", chatJID, group, err)
			} else {
				doc.Embedding = vec
			}
		}
	}

	docID := chatJID + ":" + strconv.Itoa(group)
	if err := index.Index(docID, doc); err != nil {
		logger.Warnf("Failed to index %s: %v", docID, err)
	}
}

// reindexRow holds a group document during batch re-indexing.
type reindexRow struct {
	doc   MessageDocument
	text  string // sanitised text to embed, or "" if skipped
	group int    // group number, used as doc ID suffix
}

const (
// embBatch is defined in embedding.go
)

// reIndexAllMessages re-indexes every message from the database into bleve
// using batch ONNX inference (embBatch texts per call) and bleve batch inserts.
// Messages are processed per chat_jid and grouped into fixed-size windows of
// contextNumMessages, producing one bleve document per group (doc ID: chatJID:groupNumber).
func reIndexAllMessages(store *MessageStore, maxRows int) error {
	logger.Infof("Starting re-indexing of all messages...")

	// Skip if index already populated.
	count, err := store.index.DocCount()
	if err == nil && count > 0 {
		logger.Infof("Index already has %d documents, skipping re-index", count)
		return nil
	}

	var total int
	if err := store.db.QueryRow(`SELECT COUNT(*) FROM messages WHERE (content != '' OR media_type != '')`).Scan(&total); err != nil {
		logger.Warnf("Could not count messages: %v", err)
	}
	if maxRows > 0 && total > maxRows {
		total = maxRows
	}
	logger.Infof("Re-indexing %d messages...", total)

	bar := progressbar.NewOptions(total,
		progressbar.OptionSetDescription("indexing"),
		progressbar.OptionSetWidth(20),
		progressbar.OptionShowCount(),
		progressbar.OptionShowIts(),
		progressbar.OptionSetItsString("msg"),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer:        "=",
			SaucerHead:    ">",
			SaucerPadding: " ",
			BarStart:      "[",
			BarEnd:        "]",
		}),
	)

	indexed := 0 // raw messages processed
	groups := 0  // group docs written to bleve
	startTime := time.Now()

	// emitDocs embeds a slice of group docs and writes them to bleve.
	emitDocs := func(docs []reindexRow) {
		if len(docs) == 0 {
			return
		}
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
						docs[idx].doc.Embedding = vecs[j]
					}
				}
				embIdxs = embIdxs[:0]
				embTexts = embTexts[:0]
			}

			for i := range docs {
				if docs[i].text == "" {
					continue
				}
				embIdxs = append(embIdxs, i)
				embTexts = append(embTexts, docs[i].text)
				if len(embTexts) == embBatch {
					flush()
				}
			}
			flush()
		}

		batch := store.index.NewBatch()
		for i := range docs {
			docID := docs[i].doc.ChatJID + ":" + strconv.Itoa(docs[i].group)
			if err := batch.Index(docID, docs[i].doc); err != nil {
				logger.Warnf("Failed to stage doc %s in batch: %v", docID, err)
			}
		}
		if err := store.index.Batch(batch); err != nil {
			logger.Warnf("Failed to commit bleve batch: %v", err)
		}

		groups += len(docs)
	}

	// Get all chat_jids with indexable messages.
	chatRows, err := store.db.Query(
		`SELECT DISTINCT chat_jid FROM messages WHERE (content != '' OR media_type != '')`,
	)
	if err != nil {
		return fmt.Errorf("failed to query chat_jids: %v", err)
	}
	var chatJIDs []string
	for chatRows.Next() {
		var jid string
		if err := chatRows.Scan(&jid); err == nil {
			chatJIDs = append(chatJIDs, jid)
		}
	}
	chatRows.Close()

	for _, jid := range chatJIDs {
		if maxRows > 0 && indexed >= maxRows {
			break
		}

		// Load all messages for this chat.
		msgRows, err := store.db.Query(`
				SELECT m.id, m.sender, m.full_name, m.content, m.timestamp, m.is_from_me, m.media_type, m.filename, r.content
				FROM messages m
				LEFT JOIN messages r ON m.reply_to_id = r.id
				WHERE m.chat_jid = ? AND (m.content != '' OR m.media_type != '')
				ORDER BY m.timestamp
			`, jid)
		if err != nil {
			logger.Warnf("Failed to query messages for %s: %v", jid, err)
			continue
		}

		type rawMsg struct {
			id, sender, fullName string
			content              sql.NullString
			replyToContent       sql.NullString
			timestamp            time.Time
			isFromMe             bool
			mediaType            sql.NullString
			filename             sql.NullString
		}
		var msgs []rawMsg
		for msgRows.Next() {
			var m rawMsg
			if err := msgRows.Scan(&m.id, &m.sender, &m.fullName, &m.content,
				&m.timestamp, &m.isFromMe, &m.mediaType, &m.filename, &m.replyToContent); err != nil {
				logger.Warnf("Error scanning message for %s: %v", jid, err)
				continue
			}
			msgs = append(msgs, m)
		}
		msgRows.Close()

		// Trim to maxRows if needed.
		if maxRows > 0 && indexed+len(msgs) > maxRows {
			msgs = msgs[:maxRows-indexed]
		}
		if len(msgs) == 0 {
			continue
		}

		// Build group docs: one per chunk of contextNumMessages.
		var chatDocs []reindexRow
		for i := 0; i < len(msgs); i += contextNumMessages {
			end := i + contextNumMessages
			if end > len(msgs) {
				end = len(msgs)
			}
			chunk := msgs[i:end]

			ctxMsgs := make([]contextMsg, len(chunk))
			for k, m := range chunk {
				ctxMsgs[k] = contextMsg{
					Sender:         m.sender,
					FullName:       m.fullName,
					Content:        m.content.String,
					Timestamp:      m.timestamp,
					ReplyToContent: m.replyToContent.String,
				}
			}
			ctxStr := formatContextWindow(ctxMsgs)

			r := reindexRow{
				group: i / contextNumMessages,
				text:  sanitizeForEmbedding(ctxStr),
				doc: MessageDocument{
					ChatJID:        jid,
					Context:        ctxStr,
					TimestampFirst: chunk[0].timestamp,
					TimestampLast:  chunk[len(chunk)-1].timestamp,
				},
			}
			chatDocs = append(chatDocs, r)
		}

		emitDocs(chatDocs)
		indexed += len(msgs)
		_ = bar.Add(len(msgs))
	}
	_ = bar.Finish()

	totalElapsed := time.Since(startTime).Seconds()
	overallRate := 0.0
	if totalElapsed > 0 {
		overallRate = float64(indexed) / totalElapsed
	}
	logger.Infof("Re-indexed %d messages into %d context groups in %.1fs (%.0f msg/s)", indexed, groups, totalElapsed, overallRate)
	return nil
}

// rescoredHit holds a bleve hit after rescoring, with parsed fields.
type rescoredHit struct {
	chatJID string
	group   int
	score   float64
}

// searchMessages performs a hybrid text + vector search using bleve's score
// fusion, then applies custom rescoring (mute penalty, user-ratio boost) and
// returns one SearchResult per matched context group.
func searchMessages(store *MessageStore, queryStr string, chatJIDs []string, limit int, semanticWeight float64, daysSince int) ([]SearchResult, error) {
	logger.Debugf("Searching for \"%s\" (chatJID=%v, limit=%d, offset=%d)", queryStr, chatJIDs, limit)

	// Main query — target the context field so bleve uses pt_ascii to analyze
	// the query, matching the analyzer used at index time.
	matchQuery := bleve.NewMatchQuery(queryStr)
	matchQuery.SetField("context")
	matchQuery.SetBoost(1.0 - semanticWeight)

	// Build text query.
	var searchQuery query.Query
	var fetchSize int
	if len(chatJIDs) > 0 {
		// Match chat JIDs
		booleanQuery := bleve.NewBooleanQuery()
		for _, jid := range chatJIDs {
			chatTermQuery := bleve.NewTermQuery(jid)
			chatTermQuery.SetField("chat_jid")
			booleanQuery.AddMust(chatTermQuery)
		}
		// Text query
		booleanQuery.AddMust(matchQuery)
		searchQuery = booleanQuery
		expandFactor := len(chatJIDs)
		if expandFactor > 5 {
			expandFactor = 5
		}
		fetchSize = limit * expandFactor
	} else {
		searchQuery = matchQuery
		fetchSize = limit * 5 // Increase fecth size for open search to compensate for deduplication and filtering
	}

	// Add filter for daysSince if specified, by creating a new bleve.NewDateRangeQuery()
	if daysSince > 0 {
		today := time.Now()
		sinceTime := today.Add(-time.Duration(daysSince) * 24 * time.Hour)
		dateQuery := bleve.NewDateRangeQuery(sinceTime, today)
		dateQuery.SetField("timestamp_last")
		booleanQuery := bleve.NewBooleanQuery()
		booleanQuery.AddMust(searchQuery)
		booleanQuery.AddMust(dateQuery)
		searchQuery = booleanQuery
	}

	// Over-fetch to compensate for deduplication.
	searchRequest := bleve.NewSearchRequest(searchQuery)
	searchRequest.Size = fetchSize
	searchRequest.From = 0
	searchRequest.SortBy([]string{"-_score"})
	//searchRequest.Fields = []string{"title", "content"}

	// Add kNN vector query if embedder is available.
	if store.embedder != nil && semanticWeight > 0.0 {
		if text := sanitizeForEmbedding(queryStr); text != "" {
			queryVec, err := store.embedder.Embed(text)
			if err != nil {
				logger.Warnf("Failed to embed search query, falling back to text-only: %v", err)
			} else {
				searchRequest.AddKNN("embedding", queryVec, int64(fetchSize), semanticWeight)
				searchRequest.Score = bleve.ScoreRSF
			}
		}
	} else {
		logger.Debugf("Not using embedder for this query")
	}

	params := bleve.RequestParams{
		ScoreWindowSize: fetchSize * 5,
	}
	searchRequest.AddParams(params)

	logger.Debugf("searchRequest = %+v", searchRequest)

	searchResult, err := store.index.Search(searchRequest)

	logger.Debugf("searchResult = %+v (err=%v)", searchResult, err)

	if err != nil {
		return nil, err
	}

	// Apply custom rescoring and deduplicate by (chatJID, group).
	chatRatios := make(map[string]float64)
	seen := make(map[string]bool) // key: "chatJID:group"
	var hits []rescoredHit

	for _, hit := range searchResult.Hits {
		idParts := strings.SplitN(hit.ID, ":", 2)
		if len(idParts) != 2 {
			continue
		}
		hitChatJID := idParts[0]
		group, err := strconv.Atoi(idParts[1])
		if err != nil {
			continue
		}

		if seen[hit.ID] {
			continue
		}
		seen[hit.ID] = true

		// Mute penalty.
		muted, err := store.IsChatMuted(hitChatJID)
		if err == nil && muted {
			//	hit.Score *= 0.1
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
		// hit.Score *= (1.0 + ratio)

		hits = append(hits, rescoredHit{
			chatJID: hitChatJID,
			group:   group,
			score:   hit.Score,
		})
	}

	// Sort hits by rescored value and timestamp (descending).
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].score == hits[j].score {
			return hits[i].chatJID > hits[j].chatJID // tie-breaker: newer chats first
		}
		return hits[i].score > hits[j].score
	})

	// Build SearchResults: one per hit, fetching group messages directly from SQLite.
	var results []SearchResult
	for _, h := range hits {
		msgRows, err := store.db.Query(
			`SELECT sender, full_name, COALESCE(content, ''), timestamp, is_from_me, COALESCE(media_type, ''), COALESCE(filename, '')
			 FROM messages
			 WHERE chat_jid = ? AND (content != '' OR media_type != '')
			 ORDER BY timestamp LIMIT ? OFFSET ?`,
			h.chatJID, contextNumMessages, h.group*contextNumMessages,
		)
		if err != nil {
			logger.Warnf("Failed to fetch group messages for search result: %v", err)
			continue
		}

		var messages []Message
		for msgRows.Next() {
			var m Message
			if scanErr := msgRows.Scan(&m.Sender, &m.FullName, &m.Content, &m.Time, &m.IsFromMe, &m.MediaType, &m.Filename); scanErr == nil {
				messages = append(messages, m)
			}
		}
		msgRows.Close()

		if len(messages) == 0 {
			continue
		}

		results = append(results, SearchResult{
			ChatJID:  h.chatJID,
			ChatName: store.GetChatNameByJID(h.chatJID),
			Score:    h.score,
			Messages: messages,
		})

		if len(results) >= limit {
			break
		}
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
