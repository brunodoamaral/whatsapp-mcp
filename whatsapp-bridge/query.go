package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// queryDB is a dedicated, read-only handle onto store/messages.db, separate from
// MessageStore's read-write connection. mode=ro and _query_only=1 make write
// statements fail at the SQLite driver level regardless of what the SQL parsing
// below catches, so this is safe even if the string checks below have a gap.
var queryDB *sql.DB

func initQueryDB() error {
	db, err := sql.Open("sqlite3", "file:store/messages.db?mode=ro&_query_only=1&_busy_timeout=5000")
	if err != nil {
		return fmt.Errorf("failed to open read-only query database: %v", err)
	}
	if err := db.Ping(); err != nil {
		return fmt.Errorf("failed to ping read-only query database: %v", err)
	}
	// ATTACH is per-connection state, so the pool must never open a second
	// physical connection without it — mirrors the discipline in store.go.
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)
	db.SetConnMaxLifetime(0)

	if _, err := db.Exec(`ATTACH DATABASE 'file:store/whatsapp.db?mode=ro' AS wdb`); err != nil {
		fmt.Printf("Warning: could not attach whatsapp.db to query database: %v\n", err)
	}

	queryDB = db
	return nil
}

const (
	defaultQueryLimit = 500
	maxQueryLimit     = 5000
	queryTimeout      = 5 * time.Second
)

// leadingKeywordRE matches the first SQL keyword, ignoring leading whitespace and
// SQL comments, so `-- comment\nSELECT ...` and `  select ...` are both accepted.
var leadingKeywordRE = regexp.MustCompile(`^(?is)\s*(?:--[^\n]*\n\s*|/\*.*?\*/\s*)*(select|with)\b`)

type queryRequest struct {
	SQL   string        `json:"sql"`
	Args  []interface{} `json:"args"`
	Limit int           `json:"limit"`
}

type queryResponse struct {
	Columns   []string        `json:"columns"`
	Rows      [][]interface{} `json:"rows"`
	Truncated bool            `json:"truncated"`
}

// isReadOnlyQuery rejects anything that doesn't start with SELECT/WITH, and
// rejects multiple statements (a bare ';' outside of a quoted string literal).
// This is a pre-filter for a clearer error message; initQueryDB's mode=ro
// connection is what actually enforces read-only access.
func isReadOnlyQuery(sqlText string) error {
	if !leadingKeywordRE.MatchString(sqlText) {
		return fmt.Errorf("only SELECT/WITH statements are allowed")
	}

	inSingle, inDouble := false, false
	for i := 0; i < len(sqlText); i++ {
		c := sqlText[i]
		switch {
		case c == '\'' && !inDouble:
			inSingle = !inSingle
		case c == '"' && !inSingle:
			inDouble = !inDouble
		case c == ';' && !inSingle && !inDouble:
			rest := strings.TrimSpace(sqlText[i+1:])
			if rest != "" {
				return fmt.Errorf("multiple statements are not allowed")
			}
		}
	}
	return nil
}

// makeQueryHandler exposes read-only, ad-hoc SQL access to messages.db for
// trusted local tooling (e.g. the todo-bruno companion app's candidate-chat
// prefilter). Not intended to be exposed beyond localhost.
func makeQueryHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if queryDB == nil {
			http.Error(w, "Query database not initialised", http.StatusServiceUnavailable)
			return
		}

		var req queryRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request format", http.StatusBadRequest)
			return
		}
		if strings.TrimSpace(req.SQL) == "" {
			http.Error(w, "sql is required", http.StatusBadRequest)
			return
		}
		if err := isReadOnlyQuery(req.SQL); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		limit := defaultQueryLimit
		if req.Limit > 0 {
			limit = req.Limit
		}
		if limit > maxQueryLimit {
			limit = maxQueryLimit
		}

		ctx, cancel := context.WithTimeout(r.Context(), queryTimeout)
		defer cancel()

		rows, err := queryDB.QueryContext(ctx, req.SQL, req.Args...)
		if err != nil {
			http.Error(w, fmt.Sprintf("Query failed: %v", err), http.StatusBadRequest)
			return
		}
		defer rows.Close()

		cols, err := rows.Columns()
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read columns: %v", err), http.StatusInternalServerError)
			return
		}

		resp := queryResponse{Columns: cols, Rows: [][]interface{}{}}
		for rows.Next() {
			if len(resp.Rows) >= limit {
				resp.Truncated = true
				break
			}
			vals := make([]interface{}, len(cols))
			ptrs := make([]interface{}, len(cols))
			for i := range vals {
				ptrs[i] = &vals[i]
			}
			if err := rows.Scan(ptrs...); err != nil {
				http.Error(w, fmt.Sprintf("Failed to scan row: %v", err), http.StatusInternalServerError)
				return
			}
			for i, v := range vals {
				if b, ok := v.([]byte); ok {
					vals[i] = string(b)
				}
			}
			resp.Rows = append(resp.Rows, vals)
		}
		if err := rows.Err(); err != nil {
			http.Error(w, fmt.Sprintf("Query error: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}
