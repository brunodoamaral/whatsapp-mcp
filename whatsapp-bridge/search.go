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
	"github.com/blevesearch/bleve/v2/search/searcher"
	index "github.com/blevesearch/bleve_index_api"
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

// fuzzinessAuto picks the edit distance per term from the term's length
// (>5 chars -> 2, 3-5 -> 1, <=2 -> 0), delegating to searcher.GetAutoFuzziness
// so the thresholds stay in step with bleve's own.
const fuzzinessAuto = -1

// maxFuzziness mirrors bleve's searcher.MaxFuzziness — anything above it is a
// hard query error ("fuzziness exceeds max (2)"), so callers get clamped.
const maxFuzziness = 2

// searchPrefixLength requires the first character of each term to match
// exactly. Fuzzy expansion walks the term FST, so an unbounded prefix means a
// full dictionary scan per query term on a multi-hundred-MB index.
const searchPrefixLength = 1

// maxFuzzyExpansion caps how many index terms a single query token may expand
// to. This corpus is WhatsApp messages and so is dense with misspellings: a
// 3-letter stem at edit distance 2 pulls in five hundred terms, each of which
// becomes its own term searcher.
const maxFuzzyExpansion = 256

// fuzzyTerm is one index term within edit distance of a query token.
type fuzzyTerm struct {
	term string
	dist uint8
	freq uint64
}

// fuzzyToken is one analyzed query token together with the index terms it
// expanded to. The token is kept because its length decides how much weight
// its near-misses deserve — see fuzzyLengthScale.
type fuzzyToken struct {
	token    string
	variants []fuzzyTerm
}

// analyzeQueryTokens runs queryStr through the same analyzer the field uses at
// index time, so the resulting tokens are directly comparable to index terms.
// Hand-tokenizing would skip to_lower/ascii_folding/stop_pt/stemmer_pt_light
// and produce terms that are not in the dictionary at all.
func analyzeQueryTokens(index bleve.Index, queryStr string) []string {
	analyzer := index.Mapping().AnalyzerNamed(ptAsciiAnalyzerName)
	if analyzer == nil {
		return nil
	}
	tokens := analyzer.Analyze([]byte(queryStr))
	terms := make([]string, 0, len(tokens))
	for _, t := range tokens {
		terms = append(terms, string(t.Term))
	}
	return terms
}

// expandFuzzyTerms resolves each query token to the index terms within edit
// distance of it, returning one slice per token plus the total across tokens.
//
// This exists because bleve's own MatchQuery fuzziness cannot be used for
// scoring here: it rewrites each token into a FuzzyQuery whose searcher is a
// disjunction over the whole expansion, and the disjunction scorer multiplies
// by coord = matchedTerms/expansionSize. The expansion size varies by two
// orders of magnitude between tokens in this corpus ("aniversario" ~8 terms,
// "bolo" ~108 at distance 1), so tokens with large expansions get silently
// zeroed out relative to tokens with small ones. Expanding here lets the
// caller put every variant in a single flat disjunction with one shared
// denominator, which cancels.
func expandFuzzyTerms(idx bleve.Index, field string, tokens []string, fuzziness int) ([]fuzzyToken, int, error) {
	adv, err := idx.Advanced()
	if err != nil {
		return nil, 0, fmt.Errorf("advanced index unavailable: %w", err)
	}
	reader, err := adv.Reader()
	if err != nil {
		return nil, 0, fmt.Errorf("index reader unavailable: %w", err)
	}
	defer reader.Close()

	fuzzyReader, ok := reader.(index.IndexReaderFuzzy)
	if !ok {
		return nil, 0, fmt.Errorf("index reader does not support fuzzy dictionaries")
	}

	expanded := make([]fuzzyToken, 0, len(tokens))
	total := 0
	for _, token := range tokens {
		dist := fuzziness
		if dist == fuzzinessAuto {
			dist = searcher.GetAutoFuzziness(token)
		}
		if dist <= 0 {
			// Too short to be worth an edit; the token itself is the only variant.
			expanded = append(expanded, fuzzyToken{token: token, variants: []fuzzyTerm{{term: token}}})
			total++
			continue
		}

		// FieldDictFuzzy takes a literal prefix, not a length. Tokens are
		// ascii-folded by the analyzer, so slicing bytes is safe here.
		prefix := token
		if len(prefix) > searchPrefixLength {
			prefix = prefix[:searchPrefixLength]
		}
		variants, err := fuzzyDictTerms(fuzzyReader, field, token, dist, prefix)
		if err != nil {
			return nil, 0, err
		}
		if len(variants) == 0 {
			// Not in the index at all — keep it as an exact term so the result
			// matches what fuzziness=0 would have returned (no hits).
			variants = []fuzzyTerm{{term: token}}
		}
		expanded = append(expanded, fuzzyToken{token: token, variants: variants})
		total += len(variants)
	}
	return expanded, total, nil
}

// fuzzyDictTerms walks the fuzzy field dictionary for one token, keeping at
// most maxFuzzyExpansion terms, closest first.
func fuzzyDictTerms(reader index.IndexReaderFuzzy, field, token string, dist int, prefix string) ([]fuzzyTerm, error) {
	dict, err := reader.FieldDictFuzzy(field, token, dist, prefix)
	if err != nil {
		return nil, fmt.Errorf("fuzzy dictionary for %q: %w", token, err)
	}
	defer dict.Close()

	var variants []fuzzyTerm
	for {
		entry, err := dict.Next()
		if err != nil {
			return nil, fmt.Errorf("reading fuzzy dictionary for %q: %w", token, err)
		}
		if entry == nil {
			break
		}
		variants = append(variants, fuzzyTerm{term: entry.Term, dist: entry.EditDistance, freq: entry.Count})
	}

	if len(variants) > maxFuzzyExpansion {
		// Closest edits first, then the ones actually seen in the corpus, so
		// the cap drops the noise rather than the real spellings.
		sort.Slice(variants, func(i, j int) bool {
			if variants[i].dist != variants[j].dist {
				return variants[i].dist < variants[j].dist
			}
			return variants[i].freq > variants[j].freq
		})
		variants = variants[:maxFuzzyExpansion]
	}
	return variants, nil
}

// fuzzyDistanceBoost weights a variant by its edit distance from the query
// term. Variants are deliberately near-weightless next to an exact match:
// fuzzy matching is a fallback for when the spelling the user typed is not in
// the index, not a competitor to the spelling that is.
//
// Two things make a gentle decay like 1/(dist+1) actively wrong here.
//
// First, bleve's disjunction scorer multiplies each document's summed score by
// coord = matchedTerms/clauseCount, so a document is *rewarded* for holding
// several different misspellings of the query word. coord is per-document, so
// no choice of boosts can cancel it. At 1/(dist+1), searching "predisin"
// returned a document holding "predsim" and "predsin" above the nine
// documents that actually contain "predisin".
//
// Second, an edit-distance-1 neighbour is often a different word rather than
// a typo — "bolo"/"bola", "aline"/"alien" — and nothing about the distance
// distinguishes the two cases. Weighting distance-1 highly pulled "bola"
// (a ball) to rank 2 for a query about cake.
//
// Measured against this corpus (see CLAUDE.md), recall is *unaffected* by
// these weights: when the query's spelling is absent from the index every
// candidate is fuzzy, so the shared factor cancels and the best true spelling
// still lands at rank 1. The weights only govern how much fuzzy hits disturb
// exact ones. Above these values they start to: at 0.25 a fuzzy hit displaces
// a real "predisin" match out of the top ten.
//
// These are the weights for a *long* token; fuzzyLengthScale cuts them down
// for short ones, which is what keeps "bola" away from a query for "bolo".
//
// Overridable at runtime with FUZZY_BOOST.
var fuzzyDistanceBoost = [maxFuzziness + 1]float64{1.0, 0.12, 0.04}

// fuzzyLengthScale scales a near-miss by the length of the query token it came
// from, indexed by that length and clamped at the final entry.
//
// The longer the word, the likelier a near-miss is a misspelling rather than a
// different word. At four characters the edit-distance-1 neighbourhood is
// mostly real vocabulary — bolo/bola/bolo/polo/bobo — so a neighbour there
// says almost nothing. At nine or ten characters practically nothing but typos
// lands within one edit, so the neighbour is worth taking seriously. This is a
// separate axis from bleve's auto-fuzziness, which uses length to pick how far
// to search; this decides how much to trust what it finds.
//
// The lengths are of the *stemmed* token, which is what actually gets
// expanded. Portuguese light stemming shortens aggressively ("obrigado" ->
// "obrig"), and the stem is the right unit because it is the stem's
// neighbourhood being walked.
//
// Overridable at runtime with FUZZY_LENGTH_SCALE.
var fuzzyLengthScale = []float64{
	// 0    1    2    3     4     5     6     7     8+
	0, 0, 0, 0.10, 0.20, 0.40, 0.70, 0.90, 1.0,
}

// loadFuzzyBoostOverride lets FUZZY_BOOST="1,0.12,0.04" retune the decay
// without a rebuild. What the right weights are depends on the corpus — how
// often a one-edit neighbour is a real typo rather than a different word — so
// this is a dial worth having on a running bridge.
func loadFuzzyBoostOverride() {
	raw := os.Getenv("FUZZY_BOOST")
	if raw == "" {
		return
	}
	parts := strings.Split(raw, ",")
	if len(parts) != len(fuzzyDistanceBoost) {
		logger.Warnf("Ignoring FUZZY_BOOST=%q: want %d comma-separated weights", raw, len(fuzzyDistanceBoost))
		return
	}
	var parsed [maxFuzziness + 1]float64
	for i, part := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil || v < 0 {
			logger.Warnf("Ignoring FUZZY_BOOST=%q: bad weight %q", raw, part)
			return
		}
		parsed[i] = v
	}
	fuzzyDistanceBoost = parsed
	logger.Infof("Fuzzy distance boosts overridden: %v", fuzzyDistanceBoost)
}

// loadFuzzyLengthScaleOverride lets FUZZY_LENGTH_SCALE="0,0,0,0.1,..." retune
// the length curve without a rebuild. Entry i is the weight for a token of
// length i; the last entry covers everything longer.
func loadFuzzyLengthScaleOverride() {
	raw := os.Getenv("FUZZY_LENGTH_SCALE")
	if raw == "" {
		return
	}
	parts := strings.Split(raw, ",")
	if len(parts) < 2 {
		logger.Warnf("Ignoring FUZZY_LENGTH_SCALE=%q: need at least 2 weights", raw)
		return
	}
	parsed := make([]float64, len(parts))
	for i, part := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil || v < 0 {
			logger.Warnf("Ignoring FUZZY_LENGTH_SCALE=%q: bad weight %q", raw, part)
			return
		}
		parsed[i] = v
	}
	fuzzyLengthScale = parsed
	logger.Infof("Fuzzy length scale overridden: %v", fuzzyLengthScale)
}

// buildFuzzyTextQuery turns the expanded terms into one flat disjunction.
//
// Flat is the whole point. When bleve expands fuzziness itself it nests a
// disjunction per token, and each one divides by its own clause count — which
// differs by two orders of magnitude between tokens in this corpus. One level
// means one shared denominator, so tokens keep their relative weight.
//
// Boosts here are only meaningful relative to each other: a uniform factor
// across every clause cancels out exactly, because queryNorm is 1/sqrt(sum of
// squared clause weights) and each weight is (boost*idf)^2. Multiplying every
// boost by the clause count to undo the coord division was tried and provably
// does nothing. Absolute scale is handled in normalizeHitScores instead.
func buildFuzzyTextQuery(expanded []fuzzyToken, field string) query.Query {
	disjunction := bleve.NewDisjunctionQuery()
	disjunction.SetMin(1)
	for _, ft := range expanded {
		scale := fuzzyLengthScaleFor(len(ft.token))
		for _, v := range ft.variants {
			tq := bleve.NewTermQuery(v.term)
			tq.SetField(field)
			tq.SetBoost(fuzzyVariantBoost(v.dist, scale))
			disjunction.AddQuery(tq)
		}
	}
	return disjunction
}

// fuzzyVariantBoost combines the edit-distance weight with the length scale of
// the token the variant came from. An exact term (distance 0) is never scaled
// down — the length rule is about how much to trust a *near*-miss.
func fuzzyVariantBoost(dist uint8, lengthScale float64) float64 {
	base := fuzzyDistanceBoost[len(fuzzyDistanceBoost)-1]
	if int(dist) < len(fuzzyDistanceBoost) {
		base = fuzzyDistanceBoost[dist]
	}
	if dist == 0 {
		return base
	}
	return base * lengthScale
}

// fuzzyLengthScaleFor indexes fuzzyLengthScale by token length, clamping to
// the last entry. Tokens are ascii-folded by the analyzer, so byte length is
// character length here.
func fuzzyLengthScaleFor(n int) float64 {
	if n < 0 {
		n = 0
	}
	if n >= len(fuzzyLengthScale) {
		n = len(fuzzyLengthScale) - 1
	}
	return fuzzyLengthScale[n]
}

// debugLogContext logs a context group at DEBUG level (no-op when logger is above DEBUG).
func debugLogContext(chatJID string, group int, ctxStr string) {
	logger.Debugf("--- %s group %d ---\n%s", chatJID, group, ctxStr)
}

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

	// chat_jid must be indexed as a keyword (no analysis) so TermQuery can
	// match the full JID string like "120363313357391553@g.us" as a single token.
	keywordFieldMapping := bleve.NewTextFieldMapping()
	keywordFieldMapping.Analyzer = "keyword"
	m.DefaultMapping.AddFieldMappingsAt("chat_jid", keywordFieldMapping)

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
		// CTEs materialise contact names and best stored sender names once,
		// then the main query joins them — avoiding per-row correlated subqueries.
		groupRows, err := db.Query(`
			WITH contact_names AS (
			    SELECT their_jid,
			           COALESCE(NULLIF(full_name,''), NULLIF(first_name,''), NULLIF(push_name,'')) AS display_name
			    FROM wdb.whatsmeow_contacts
			),
			sender_names AS (
			    SELECT sender, full_name AS display_name
			    FROM messages
			    WHERE full_name != '' AND full_name != sender
			    GROUP BY sender
			)
			SELECT m.sender,
			       COALESCE(cn.display_name, sn.display_name, m.full_name, ''),
			       COALESCE(m.content, ''), m.timestamp, COALESCE(r.content, '')
			FROM messages m
			LEFT JOIN messages r ON m.reply_to_id = r.id AND r.chat_jid = m.chat_jid
			LEFT JOIN contact_names cn ON cn.their_jid = CASE WHEN m.sender LIKE '%@%' THEN m.sender ELSE m.sender||'@s.whatsapp.net' END
			LEFT JOIN sender_names sn ON sn.sender = m.sender
			WHERE m.chat_jid = ? AND (m.content != '' OR m.media_type != '')
			ORDER BY m.timestamp, m.id LIMIT ? OFFSET ?`,
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
	debugLogContext(chatJID, group, ctxStr)
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

// groupIndexForMessage returns the 0-based context group a message belongs to,
// using the same ordering as reIndexAllMessages (timestamp ascending over the
// indexable messages of the chat).
//
// indexMessage derives the group from a plain COUNT(*), which only gives the
// right answer for the newest message in a chat. Transcripts arrive minutes
// after the voice note, by which time newer messages may exist, so the position
// has to be computed from the message's own rank instead.
// The target row's timestamp is read back from the database rather than bound
// from Go: timestamps are stored as text with their original offset
// ("2026-07-25 13:31:17-03:00"), and a bound time.Time serialises to a
// different representation, so comparing the two lexically gives a wrong rank.
// The (timestamp, id) tiebreak mirrors the ORDER BY used when building groups.
func groupIndexForMessage(db *sql.DB, chatJID, id string) (int, error) {
	var rank int
	err := db.QueryRow(`
		WITH target AS (
		    SELECT timestamp AS ts, id AS tid FROM messages
		    WHERE id = ? AND chat_jid = ?
		)
		SELECT COUNT(*) FROM messages, target
		WHERE chat_jid = ? AND (content != '' OR media_type != '')
		  AND (timestamp < ts OR (timestamp = ts AND id < tid))`,
		id, chatJID, chatJID,
	).Scan(&rank)
	if err != nil {
		return 0, err
	}
	return rank / contextNumMessages, nil
}

// rebuildGroupDoc re-reads one context group from SQLite and rewrites its bleve
// document, re-embedding the group text. Used when a message's content changes
// after it was first indexed (voice-note transcription).
func rebuildGroupDoc(store *MessageStore, chatJID string, group int) error {
	rows, err := store.db.Query(`
		WITH contact_names AS (
		    SELECT their_jid,
		           COALESCE(NULLIF(full_name,''), NULLIF(first_name,''), NULLIF(push_name,'')) AS display_name
		    FROM wdb.whatsmeow_contacts
		),
		sender_names AS (
		    SELECT sender, full_name AS display_name
		    FROM messages
		    WHERE full_name != '' AND full_name != sender
		    GROUP BY sender
		)
		SELECT m.sender,
		       COALESCE(cn.display_name, sn.display_name, m.full_name, ''),
		       COALESCE(m.content, ''), m.timestamp, COALESCE(r.content, '')
		FROM messages m
		LEFT JOIN messages r ON m.reply_to_id = r.id AND r.chat_jid = m.chat_jid
		LEFT JOIN contact_names cn ON cn.their_jid = CASE WHEN m.sender LIKE '%@%' THEN m.sender ELSE m.sender||'@s.whatsapp.net' END
		LEFT JOIN sender_names sn ON sn.sender = m.sender
		WHERE m.chat_jid = ? AND (m.content != '' OR m.media_type != '')
		ORDER BY m.timestamp, m.id LIMIT ? OFFSET ?`,
		chatJID, contextNumMessages, group*contextNumMessages,
	)
	if err != nil {
		return fmt.Errorf("fetch group %d of %s: %w", group, chatJID, err)
	}

	var groupMsgs []contextMsg
	for rows.Next() {
		var m contextMsg
		if scanErr := rows.Scan(&m.Sender, &m.FullName, &m.Content, &m.Timestamp, &m.ReplyToContent); scanErr == nil {
			groupMsgs = append(groupMsgs, m)
		}
	}
	rows.Close()

	if len(groupMsgs) == 0 {
		return nil
	}

	ctxStr := formatContextWindow(groupMsgs)
	doc := MessageDocument{
		ChatJID:        chatJID,
		Context:        ctxStr,
		TimestampFirst: groupMsgs[0].Timestamp,
		TimestampLast:  groupMsgs[len(groupMsgs)-1].Timestamp,
	}

	if store.embedder != nil {
		if text := sanitizeForEmbedding(ctxStr); text != "" {
			vec, err := store.embedder.Embed(text)
			if err != nil {
				logger.Warnf("Failed to embed %s group %d: %v", chatJID, group, err)
			} else {
				doc.Embedding = vec
			}
		}
	}

	docID := chatJID + ":" + strconv.Itoa(group)
	if err := store.index.Index(docID, doc); err != nil {
		return fmt.Errorf("index %s: %w", docID, err)
	}
	return nil
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

// reIndexAllMessages re-indexes messages from the database into bleve.
// When chatFilter is non-empty, only chats whose JID contains the filter string
// (case-insensitive LIKE match) are processed, existing bleve docs for those
// chats are deleted first, and the global "already populated" skip is bypassed.
func reIndexAllMessages(store *MessageStore, maxRows int, chatFilter string) error {
	if chatFilter != "" {
		logger.Infof("Starting re-indexing for chats matching %q...", chatFilter)
	} else {
		logger.Infof("Starting re-indexing of all messages...")
	}

	// Skip if index already populated — only for full reindex.
	if chatFilter == "" {
		count, err := store.index.DocCount()
		if err == nil && count > 0 {
			logger.Infof("Index already has %d documents, skipping re-index", count)
			return nil
		}
	}

	// When a plain phone number is given as filter, anchor it before the '@' so
	// that legacy group JIDs like "5521992125269-1462706974@g.us" (where the
	// creator's number is embedded in the JID) do not match.
	likeFilter := ""
	if chatFilter != "" {
		if strings.Contains(chatFilter, "@") {
			likeFilter = "%" + chatFilter + "%"
		} else {
			likeFilter = "%" + chatFilter + "@%"
		}
	}

	countQuery := `SELECT COUNT(*) FROM messages WHERE (content != '' OR media_type != '')`
	var countArgs []interface{}
	if likeFilter != "" {
		countQuery = `SELECT COUNT(*) FROM messages WHERE (content != '' OR media_type != '') AND chat_jid LIKE ?`
		countArgs = append(countArgs, likeFilter)
	}
	var total int
	if err := store.db.QueryRow(countQuery, countArgs...).Scan(&total); err != nil {
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

	// Get chat_jids with indexable messages, optionally filtered.
	chatQuery := `SELECT DISTINCT chat_jid FROM messages WHERE (content != '' OR media_type != '')`
	var chatArgs []interface{}
	if likeFilter != "" {
		chatQuery += ` AND chat_jid LIKE ?`
		chatArgs = append(chatArgs, likeFilter)
	}
	chatRows, err := store.db.Query(chatQuery, chatArgs...)
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
		// CTEs materialise contact names and best stored sender names once,
		// then the main query joins them — avoiding per-row correlated subqueries.
		msgRows, err := store.db.Query(`
				WITH contact_names AS (
				    SELECT their_jid,
				           COALESCE(NULLIF(full_name,''), NULLIF(first_name,''), NULLIF(push_name,'')) AS display_name
				    FROM wdb.whatsmeow_contacts
				),
				sender_names AS (
				    SELECT sender, full_name AS display_name
				    FROM messages
				    WHERE full_name != '' AND full_name != sender
				    GROUP BY sender
				)
				SELECT m.id, m.sender,
				       COALESCE(cn.display_name, sn.display_name, m.full_name),
				       m.content, m.timestamp, m.is_from_me, m.media_type, m.filename, r.content
				FROM messages m
				LEFT JOIN messages r ON m.reply_to_id = r.id AND r.chat_jid = m.chat_jid
				LEFT JOIN contact_names cn ON cn.their_jid = CASE WHEN m.sender LIKE '%@%' THEN m.sender ELSE m.sender||'@s.whatsapp.net' END
				LEFT JOIN sender_names sn ON sn.sender = m.sender
				WHERE m.chat_jid = ? AND (m.content != '' OR m.media_type != '')
				ORDER BY m.timestamp, m.id
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

			if chatFilter != "" {
				debugLogContext(jid, i/contextNumMessages, ctxStr)
			}

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

// buildTextQuery builds the text side of the search.
//
// With fuzziness off this is bleve's plain MatchQuery. With it on we expand
// the terms ourselves and build a flat disjunction (see expandFuzzyTerms for
// why bleve's own MatchQuery fuzziness cannot be used for scoring). If
// anything about the expansion fails we fall back to MatchQuery fuzziness —
// its scoring is skewed, but a skewed search beats no search.
func buildTextQuery(idx bleve.Index, queryStr string, fuzziness int) query.Query {
	matchQuery := bleve.NewMatchQuery(queryStr)
	matchQuery.SetField("context")

	if fuzziness == 0 {
		return matchQuery
	}

	if tokens := analyzeQueryTokens(idx, queryStr); len(tokens) > 0 {
		expanded, total, err := expandFuzzyTerms(idx, "context", tokens, fuzziness)
		if err == nil && total > 0 {
			logger.Debugf("Fuzzy expansion: %d token(s) -> %d index terms", len(tokens), total)
			return buildFuzzyTextQuery(expanded, "context")
		}
		if err != nil {
			logger.Warnf("Fuzzy term expansion failed, falling back to match-query fuzziness: %v", err)
		}
	}

	if fuzziness == fuzzinessAuto {
		matchQuery.SetAutoFuzziness(true)
	} else {
		matchQuery.SetFuzziness(fuzziness)
	}
	matchQuery.SetPrefix(searchPrefixLength)
	return matchQuery
}

// normalizeHitScores rescales scores so the best hit is 1.0.
//
// Raw bleve scores are not comparable across queries, and with fuzzy matching
// they are not even comparable across fuzziness settings: a disjunction
// divides every score by coord (matchedTerms/clauseCount) and again by
// queryNorm (1/sqrt of the summed squared clause weights), both of which grow
// with the size of the term expansion. Boosting the clauses cannot undo this —
// queryNorm is computed from the boosts, so a uniform boost cancels itself
// out. Since both factors are constants for a given query, they leave the
// ordering intact and only distort the magnitude, which is exactly what
// dividing through by the maximum removes.
//
// This also makes the text-only path consistent with hybrid search, where
// bleve's RSF already min-max normalizes the text leg before fusing it.
//
// hits must already be sorted descending.
func normalizeHitScores(hits []rescoredHit) {
	if len(hits) == 0 || hits[0].score <= 0 {
		return
	}
	max := hits[0].score
	for i := range hits {
		hits[i].score /= max
	}
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
func searchMessages(store *MessageStore, queryStr string, chatJIDs []string, limit int, semanticWeight float64, daysSince int, fuzziness int) ([]SearchResult, error) {
	logger.Debugf("Searching for \"%s\" (chatJID=%v, limit=%d, fuzziness=%d)", queryStr, chatJIDs, limit, fuzziness)

	// Main query — target the context field so the same pt_ascii analysis is
	// applied to the query as at index time.
	textQuery := buildTextQuery(store.index, queryStr, fuzziness)

	// Build text query.
	var searchQuery query.Query
	var fetchSize int
	if len(chatJIDs) > 0 {
		booleanQuery := bleve.NewBooleanQuery()
		// Multiple JIDs must be OR'd (disjunction), then AND'd with the text query.
		jidDisjunction := bleve.NewDisjunctionQuery()
		for _, jid := range chatJIDs {
			chatTermQuery := bleve.NewTermQuery(jid)
			chatTermQuery.SetField("chat_jid")
			jidDisjunction.AddQuery(chatTermQuery)
		}
		booleanQuery.AddMust(jidDisjunction)
		// Text query
		booleanQuery.AddMust(textQuery)
		searchQuery = booleanQuery
		expandFactor := len(chatJIDs)
		if expandFactor > 5 {
			expandFactor = 5
		}
		fetchSize = limit * expandFactor
	} else {
		searchQuery = textQuery
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

	// The semantic/text balance has to be set on whatever ends up as the
	// top-level query: bleve's RSF rescorer reads its fusion weight from
	// req.Query.Boost() and neutralizes the live boost during the search. Set
	// on an inner query it would be ignored whenever chat_jid or days_since
	// wraps it in a BooleanQuery, silently pinning the text weight to 1.0.
	// Container queries ignore their own boost when scoring, so this only ever
	// acts as the fusion weight.
	if boostable, ok := searchQuery.(query.BoostableQuery); ok {
		boostable.SetBoost(1.0 - semanticWeight)
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

	// ScoreWindowSize is only needed for KNN/hybrid RSF scoring.
	// Applying it for pure FTS forces bleve to scan a huge candidate window
	// and makes queries an order of magnitude slower.
	if store.embedder != nil && semanticWeight > 0.0 {
		params := bleve.RequestParams{
			ScoreWindowSize: fetchSize * 5,
		}
		searchRequest.AddParams(params)
	}

	logger.Debugf("searchRequest = %+v", searchRequest)

	searchResult, err := store.index.Search(searchRequest)
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

		// Post-filter by chat_jid. This is the authoritative guard: KNN results
		// bypass bleve's boolean-query MUST filters, so we enforce the constraint
		// here using the JID embedded in the doc ID.
		if len(chatJIDs) > 0 {
			matched := false
			for _, jid := range chatJIDs {
				if hitChatJID == jid {
					matched = true
					break
				}
			}
			if !matched {
				continue
			}
		}

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

	normalizeHitScores(hits)

	// Build SearchResults: one per hit, fetching group messages directly from SQLite.
	var results []SearchResult
	for _, h := range hits {
		msgRows, err := store.db.Query(
			`SELECT m.sender,
			        COALESCE(
			            (SELECT COALESCE(NULLIF(c.full_name,''), NULLIF(c.first_name,''), NULLIF(c.push_name,''))
			             FROM wdb.whatsmeow_contacts c
			             WHERE c.their_jid = CASE WHEN m.sender LIKE '%@%' THEN m.sender ELSE m.sender||'@s.whatsapp.net' END),
			            m.full_name
			        ),
			        COALESCE(m.content, ''), m.timestamp, m.is_from_me, COALESCE(m.media_type, ''), COALESCE(m.filename, '')
			 FROM messages m
			 WHERE m.chat_jid = ? AND (m.content != '' OR m.media_type != '')
			 ORDER BY m.timestamp LIMIT ? OFFSET ?`,
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
