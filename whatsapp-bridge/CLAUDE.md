# whatsapp-bridge

Go REST/WebSocket bridge over a whatsmeow-backed WhatsApp session, storing
messages in `store/messages.db` and auth/contacts in `store/whatsapp.db`.
API surface is documented in `API.md` — this file is about decisions behind
one addition to it, `POST /api/query`, made while building a companion app
(`../../todo-bruno`) that needed ad-hoc read access this bridge didn't expose.

## `POST /api/query` (`query.go`)

**Why a generic SQL endpoint instead of a narrow one.** The original need was
just "list chats with their last message," but the actual consumer
(`todo-bruno`'s daily scan) needed several different aggregate queries over
`chats`/`messages` as its filtering logic evolved (last-message-not-from-me,
per-group activity windows, reply-cadence stats for context). A narrow
endpoint would mean a Go change + rebuild + service restart per filter tweak;
GraphQL would mean a schema/resolver layer for what is a 3-table SQLite DB.
A read-only SQL endpoint, scoped tightly enough to be safe, covers all of
that with zero further bridge changes.

**Read-only is enforced in layers, not just by string-checking the SQL.**
`initQueryDB` opens a **separate** `*sql.DB` from `MessageStore`'s
read-write one, as `file:store/messages.db?mode=ro&_query_only=1` — SQLite
refuses writes at the driver level on that connection regardless of what the
query text says. `isReadOnlyQuery` (reject anything not starting with
`SELECT`/`WITH`, reject multiple statements) is defense-in-depth on top of
that, not the primary control. This matters because the query text comes
from a trusted-but-not-audited local caller, not from user input filtered
elsewhere.

**`whatsapp.db` is attached a second time, on this connection specifically.**
`MessageStore.newMessageStore` (`store.go`) already does
`ATTACH DATABASE 'file:store/whatsapp.db?mode=ro' AS wdb` — but only on its
*own* connection. `queryDB` is a distinct `*sql.DB`, so it needed its own
`ATTACH` to let `/api/query` callers join against `wdb.whatsmeow_contacts` /
`wdb.whatsmeow_lid_map` (e.g. resolving a `@lid` chat to a real phone number
— needed for `todo-bruno`'s "open in WhatsApp" links). This was added in a
*second* pass after the endpoint already existed and worked for
`chats`/`messages`-only queries — the gap only became visible once a
consumer actually needed `wdb.*`.

**`SetMaxOpenConns(1)` / `SetMaxIdleConns(1)` are load-bearing, not
incidental.** `ATTACH` is per-connection SQLite state. If the pool opened a
second physical connection under load, queries against `wdb.*` would
intermittently fail with "no such table" on whichever connection didn't run
the `ATTACH`. `store.go` already does this for the same reason (see its own
comment); `query.go` mirrors it deliberately, not by copy-paste accident —
if you ever "simplify" this to let the pool scale, the `wdb` join breaks
silently under concurrency.

## `typing=true` on `/ws/messages` (`broadcaster.go`, `handlers.go`, `main.go`)

**Why `SendPresence(Available)` had to move from "never called" to
"unconditional on every connect."** whatsmeow only emits `events.ChatPresence`
(the typing/paused notification) after the client has sent an "available"
presence at least once — this bridge never did, so chat-presence events
never fired regardless of what the WS layer did with them. Making that call
is a real behavior change, not incidental plumbing: it also makes the
account show as online to contacts and switches on active read-receipt
sending (`sendActiveReceipts` in whatsmeow). Doing it unconditionally at
`*events.Connected` (rather than lazily, only when some client first asks
for `typing=true`) was a deliberate choice confirmed with the user — the
alternative (lazy) would make the account's online/read-receipt behavior
depend on which WS clients happen to be connected, which is a much harder
thing to reason about than "always on once the bridge is running."

**Own-device typing is not a separate feature — it's the same event.**
`events.ChatPresence.IsFromMe` is true when the composing/paused update
came from one of the account's *own* other linked devices (phone/web/desktop)
syncing its typing state in a chat, per whatsmeow's `parseMessageSource`
matching `from.User == clientID.User`. This bridge passes `IsFromMe` straight
through in the `typing` WS payload rather than filtering it out or treating
it specially — a consumer that only cares about "someone typing at me" can
just check `is_from_me == false` client-side.

**Typing events bypass catch-up/persistence entirely.** Unlike
`BroadcastMessage`, `TypingMessage` is never written to SQLite and has no
per-client last-seen cursor in `ClientRegistry` — a missed typing event
while disconnected is meaningless to replay (the state has almost certainly
already changed by the time a reconnecting client would see it), so there
was no reason to pay the same catch-up complexity `messages` needs for
message history.

**Per-subscriber `typingCh` is nil unless requested, not filtered post-hoc.**
`subscriber.typingCh` is left `nil` for connections that didn't pass
`typing=true`. `BroadcastTyping` skips those subscribers outright, and the
WS handler's `select` never fires on a nil channel — so a client that didn't
ask for typing events pays no cost (no allocation, no wasted sends) rather
than receiving-and-discarding.

## `GET /api/contacts/{jid}/avatar` (`avatar.go`)

**Why a second endpoint instead of changing `/profile-picture`.** The
existing JSON endpoint (`handlers.go::makeGetProfilePictureHandler`) calls
`client.GetProfilePictureInfo` fresh on every request and returns a
short-lived signed WhatsApp CDN `url` — a different URL each time even when
the photo hasn't changed, so a downstream browser can never cache it by URL,
and some JIDs are observed to hang on that full fetch, stalling whoever
called synchronously. A downstream consumer (a Python/FastAPI app rendering
`<img>` tags) needed a URL that is itself stable and standard-HTTP-cacheable.
Changing `/profile-picture`'s response shape would break whatever already
depends on its documented JSON contract (grepped for other callers — none in
this repo, but the contract is public API per `API.md`), so it stays exactly
as-is and the new behavior lives at its own path, serving raw image bytes
instead of JSON.

**Cache table (`store/avatars.db`, its own SQLite file, not a table bolted
onto `messages.db`/`whatsapp.db`).** Everything in it is disposable/derived
— re-fetchable from WhatsApp at any time — so it doesn't belong in either of
the two databases whose loss would actually be a problem. Schema:

```sql
CREATE TABLE avatars (
    jid            TEXT PRIMARY KEY,
    picture_id     TEXT NOT NULL DEFAULT '',  -- whatsmeow's stable id; '' when has_picture is false
    content_type   TEXT NOT NULL DEFAULT '',
    data           BLOB,
    has_picture    BOOLEAN NOT NULL DEFAULT 0,
    fetched_at     TIMESTAMP NOT NULL,        -- when the bytes/absence were last actually established
    revalidated_at TIMESTAMP NOT NULL         -- when we last confirmed that's still current
);
```

A JID with no photo (`ErrProfilePictureNotSet`) — or whose owner has hidden
it from us (`ErrProfilePictureUnauthorized`) — is cached the same way, with
`has_picture = 0` and no bytes, specifically so repeatedly requesting a
contact with no photo doesn't hit WhatsApp on every request. This doesn't
need the `SetMaxOpenConns(1)`/`ATTACH` discipline `store.go`/`query.go` rely
on — there's no cross-database join here — but it's set anyway since the
table is low-traffic and a single writer costs nothing.

**Freshness window: 6 hours before even a cheap revalidation call, 1 hour of
browser-side `Cache-Control`.** Profile pictures change far less often than
messages, so `avatarFreshness` (6h) means at most 4 round-trips to WhatsApp
per contact per day under continuous traffic, while still catching a changed
photo the same day. Past that window the bridge still doesn't re-download
anything by default — it revalidates via whatsmeow's existing
`ExistingID`/`known_id` mechanism (already wired into `/profile-picture` but
previously unused by any caller), which returns "unchanged" without
resending image bytes; a full re-download only happens when whatsmeow
reports the photo actually changed, or there's no cache yet for that JID.
`avatarBrowserMaxAge` (1h) is deliberately shorter than the 6h server-side
window: once a browser's cached copy expires, its next request carries
`If-None-Match`, which lands well inside the 6h window and gets answered
`304` straight from `avatars.db` — the common case never reaches WhatsApp.

**The whatsmeow call and the CDN download both run under a 10s timeout
(`avatarFetchTimeout`), and a timeout falls back to serving stale cache
rather than failing the request.** This was explicitly called out as an
observed failure mode (some JIDs hang on the full profile-picture fetch) —
without a bound, a slow/hung JID would stall the HTTP handler (and whatever
downstream `<img>` load is waiting on it) indefinitely. Since an avatar a few
hours stale is a non-issue for a downstream UI, "serve what we already have"
beats "5xx and make the browser show a broken image" whenever there's
something in the cache to fall back to; only a JID with *no* cache at all and
a failing/timing-out WhatsApp call gets a real error (502).

**ETag / If-None-Match / 304 / Cache-Control, the standard HTTP way.** The
`ETag` is the picture's whatsmeow `id` (the same value the JSON endpoint
calls `id`, and the same thing `known_id`/`ExistingID` round-trips) quoted
per RFC. A matching `If-None-Match` gets a bodyless `304`. `Cache-Control` is
`private` — this is single-consumer image data, not something a shared/CDN
cache should hold — with `max-age` set to `avatarBrowserMaxAge`. The 404
("no photo") response also carries the same `Cache-Control`, so a browser
stops asking for a contact with no photo for the same window rather than
retrying on every page load.

**Route registration mirrors `/profile-picture`** —
`r.Get("/api/contacts/{jid}/avatar", makeGetAvatarHandler(client))` in
`handlers.go`'s `startRESTServer`, using the same `urlParamJID` percent-decode
helper. `initAvatarDB()` is called from `main.go` right after `initQueryDB()`,
logging (not fataling) on failure — consistent with how the query DB's init
failure is handled, since neither endpoint is required for the bridge's core
job of relaying messages.

## `fuzziness` on `/api/search` (`search.go`, `handlers.go`)

**The engine always supported this — it was simply never switched on.** The
bridge built one bleve `MatchQuery` on `context` and never called
`SetFuzziness`/`SetPrefix`, so bleve's default `Fuzziness = 0` applied and
terms had to match the index exactly after analysis. Nothing about the index or the mapping had to change to enable
typo tolerance — it is purely query-time, so turning it on (or off again, or
retuning it) never requires a reindex of `store/messages.bleve`.

**Auto rather than a fixed edit distance.** The distance is picked per term
from the term's length via bleve's own `searcher.GetAutoFuzziness` (>5 chars →
2, 3–5 → 1, ≤2 → 0) — called directly rather than reimplemented, so the
thresholds cannot drift from bleve's. A single fixed
distance is wrong at both ends of that range: distance 1 on a 3-letter stem
matches most of the dictionary, while distance 1 on a long word misses the
two-character slips people actually make. The `fuzziness` param still allows
`0`/`1`/`2` for callers that want to pin it; anything higher is clamped,
because bleve's `MaxFuzziness` is 2 and a larger value is a hard query error
(`fuzziness exceeds max (2)`), not a silently-degraded search.

**Bleve's own MatchQuery fuzziness is not usable for scoring here, so the
terms are expanded by hand.** `MatchQuery` with fuzziness rewrites each token
into a `FuzzyQuery`, whose searcher is a disjunction over *every index term
within edit distance*. The disjunction scorer then multiplies by
`coord = matchedTerms/clauseCount`, and the term scorer multiplies again by
`queryNorm = 1/sqrt(sum of squared clause weights)` — both of which grow with
the size of the expansion. This corpus is WhatsApp messages and is therefore
dense with misspellings, so expansions are huge *and wildly uneven*: measured
on the live index, `aniversario` expands to ~8 terms at edit distance 1 while
`bolo` expands to ~108 (and ~526 at distance 2). Since each token is divided
by its *own* expansion size, a short token is silenced relative to a long one
and multi-word ranking breaks — `q=aniversario bolo` started returning
documents matching only `aniversario`, above documents matching both.

`expandFuzzyTerms` therefore walks the dictionary itself via
`FieldDictFuzzy` (`DictEntry` already carries `EditDistance` and `Count`, so
no `FuzzyAutomaton` is needed) and `buildFuzzyTextQuery` puts every variant of
every token into **one flat disjunction**. The denominator is then a single
constant shared by all tokens, so the per-token distortion cancels and each
variant is weighted only by its own `1/(editDistance+1)` boost. If the index
reader does not implement `IndexReaderFuzzy`, or the walk fails, the code
falls back to `MatchQuery` fuzziness and logs a warning — skewed scoring beats
no search.

**Boosts cannot restore the lost magnitude; normalization does.** The obvious
fix for the collapse — boosting every clause by the clause count so `coord`
cancels — does not work, and it is worth recording why so nobody tries it
again: `TermQueryScorer.Weight()` is `(boost·idf)²` and `queryNorm` is
`1/sqrt(Σ Weight)`, so scaling every boost by the same factor scales
`queryNorm` down by exactly that factor. A uniform boost inside a disjunction
always cancels itself out. Boost only ever expresses *relative* weight, which
is why `1/(editDistance+1)` still does its job. Absolute magnitude is instead
restored in `normalizeHitScores`, which divides through by the top score after
the final sort: `coord` and `queryNorm` are constants for a given query, so
they never affected ordering, only scale. This also makes the text-only path
consistent with hybrid search, where bleve's RSF already min-max normalizes
the text leg before fusing it.

**`prefix_length = 1` is a cost guard, not a correctness one.** Fuzzy
expansion walks the term FST; with no prefix constraint, every query term at
distance 2 scans the whole term dictionary of a multi-hundred-MB index. This
build also has bleve's `DisjunctionMaxClauseCount` at its default `0`
(unlimited), so a pathological expansion degrades latency rather than
erroring. `maxFuzzyExpansion` is the second bound, capping each token's
variant list and keeping the closest edits (then the most frequent) when it
has to cut. The tradeoff of the prefix is that a typo in the *first*
character isn't caught; raise it to 2 only if latency demands it, since that
rejects noticeably more real typos.

**Fuzziness applies to stemmed terms, which is why the analyzer matters
here.** `pt_ascii` (`to_lower` → `ascii_folding_custom` → `stop_pt` →
`stemmer_pt_light`) runs on both the index and the query side, so accent
variants (`remedio`/`remédio`) and inflections (`comprimido`/`comprimidos`)
already matched before this change — the edit distance only has to absorb
genuine misspellings. The flip side is that auto's length thresholds are
measured on *stems*, not on what the user typed, so a stem of 6 characters
gets distance 2. If recall ever turns out noisy in practice, `fuzziness=1` is
the dial.

**Side effect on `semantic_weight=1`.** At weight 1 the text leg's fusion
weight is zeroed but the text query is still the base of the search request,
so it acts as a hard filter on which docs the KNN leg can score at all.
Fuzziness widens that
filter, which makes "pure semantic" searches less term-dependent than they
were — an improvement, but a real behavior change for callers that had tuned
around the old narrowness.

**`semantic_weight` has to be set on the top-level query, not the text
query.** Bleve's RSF rescorer reads the text leg's fusion weight from
`req.Query.Boost()` (`rescorer.go`: `origBoosts[0] = bQuery.Boost()`) and
forces the live boost to 1.0 for the search itself. The original code set
`SetBoost(1 - semanticWeight)` on the inner `MatchQuery`, which is the
top-level query *only* when no filter is applied — pass `chat_jid` or
`days_since` and the top level becomes a `BooleanQuery` whose boost defaults
to 1.0 (a nil `*Boost` reads as 1.0), silently pinning the text weight to 1.0
and taking `semantic_weight` out of the loop for every filtered hybrid
search. The boost is now applied to whatever `searchQuery` ends up being,
immediately before the request is built. This is safe because `BooleanQuery`
and `DisjunctionQuery` both ignore their own `BoostVal` when constructing
searchers — on a container query the boost is fusion weight and nothing else.

## Not done, on purpose

Row-level or query-level auth/audit beyond "not intended to be reachable
beyond localhost" (per `API.md`) — this bridge has no auth model at all for
any endpoint, so `/api/query` isn't a new category of exposure, just a wider
one on an already-trusted local surface.
