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

## Not done, on purpose

Row-level or query-level auth/audit beyond "not intended to be reachable
beyond localhost" (per `API.md`) — this bridge has no auth model at all for
any endpoint, so `/api/query` isn't a new category of exposure, just a wider
one on an already-trusted local surface.
