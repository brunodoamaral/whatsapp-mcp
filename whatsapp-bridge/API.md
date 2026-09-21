# WhatsApp Bridge API

Base URL: `http://localhost:8080`

---

## Send message

```
POST /api/send
```

```json
{
  "recipient": "5511999999999@s.whatsapp.net",
  "message": "Hello!",
  "media_path": "/path/to/file.jpg"
}
```

`media_path` is optional. For groups use `@g.us` JIDs.

Response:
```json
{
  "success": true,
  "message": "Message sent to 5511999999999@s.whatsapp.net"
}
```

---

## Download media

```
POST /api/download
```

```json
{
  "message_id": "ABCDEF123",
  "chat_jid": "5511999999999@s.whatsapp.net"
}
```

Response:
```json
{
  "success": true,
  "message": "Successfully downloaded image media",
  "filename": "image_20260101_120000.jpg",
  "path": "/absolute/path/to/store/5511999999999_s.whatsapp.net/image_20260101_120000.jpg"
}
```

Files are saved under `store/{chat_jid_colons_replaced}/` and cached (re-downloading the same message returns the existing file).

---

## Search messages

```
GET /api/search?q=hello&limit=10&days_since=30&chat_jid=...&semantic_weight=0.5&fuzziness=auto
```

| Param | Default | Description |
|---|---|---|
| `q` | required | Search query |
| `limit` | 10 | Max results (max 100) |
| `days_since` | — | Restrict to last N days |
| `chat_jid` | — | Comma-separated JIDs to filter |
| `semantic_weight` | 0.5 | 0 = text only, 1 = semantic only |
| `fuzziness` | `auto` | Typo tolerance: `auto` (edit distance per term by length), `0` (exact), `1`, `2` |

Results are grouped into context windows of up to 16 consecutive messages.

Fuzziness matches against *analyzed* terms — already lowercased, accent-folded
and Portuguese-light-stemmed — so accent and inflection variants (`remedio` /
`remédio`, `comprimido` / `comprimidos`) match at `fuzziness=0` already, and the
edit distance only has to absorb real misspellings. `auto` means edit distance
2 for terms longer than 5 characters, 1 for 3–5, and 0 for 2 or shorter. The
first character of a term must always match. Fuzzy hits score
`1/(editDistance+1)` of an exact hit, so exact matches still rank on top.
Values above 2 are clamped.

Scores are normalized so the best hit in a response is always `1.0` and the
rest are relative to it. Raw BM25 scores are not comparable between queries,
and fuzzy matching scales them by the size of the term expansion, so the
absolute number never carried meaning — do not threshold on it across
different queries.

Note: messages within a search result use Go's default (capitalized, no `omitempty`) field names, unlike every other endpoint below — this differs from the `snake_case` used elsewhere in this API.

Response:
```json
{
  "query": "hello",
  "fuzziness": "auto",
  "total": 2,
  "results": [
    {
      "chat_jid": "5511999999999@s.whatsapp.net",
      "chat_name": "John",
      "score": 1.0,
      "messages": [
        {
          "Time": "2026-01-01T12:00:00Z",
          "Sender": "5511999999999",
          "FullName": "John",
          "Content": "Hello there!",
          "IsFromMe": false,
          "MediaType": "",
          "Filename": "",
          "ReplyToID": ""
        }
      ]
    }
  ]
}
```

---

## Message history

```
GET /api/chats/{jid}/messages?limit=50&offset=0&start=...&end=...
```

| Param | Default | Description |
|---|---|---|
| `limit` | 50 | Max results (max 200) |
| `offset` | 0 | Pagination offset |
| `start` | — | RFC3339 timestamp, inclusive |
| `end` | — | RFC3339 timestamp, inclusive |

Results are ordered newest-first.

Example:
```
GET /api/chats/5511999999999@s.whatsapp.net/messages?limit=20&start=2026-01-01T00:00:00Z
```

Response:
```json
{
  "chat_jid": "5511999999999@s.whatsapp.net",
  "count": 2,
  "messages": [
    {
      "id": "ABCDEF123",
      "time": "2026-03-28T20:00:00Z",
      "sender": "5511999999999",
      "full_name": "John",
      "content": "Hello",
      "is_from_me": false,
      "media_type": "image",
      "filename": "photo.jpg",
      "reply_to_id": "XYZ789"
    }
  ]
}
```

`media_type`, `filename`, and `reply_to_id` are omitted when empty.

---

## Profile picture

```
GET /api/contacts/{jid}/profile-picture
```

| Param | Default | Description |
|---|---|---|
| `preview` | false | Return low-res thumbnail instead of full picture |
| `is_community` | false | Required for community group photos (avoids 401) |
| `known_id` | — | Last known picture ID; if unchanged, returns `changed: false` |

Example:
```
GET /api/contacts/5511999999999@s.whatsapp.net/profile-picture
GET /api/contacts/5511999999999@s.whatsapp.net/profile-picture?known_id=abc123&preview=true
```

Response (picture available or changed):
```json
{
  "changed": true,
  "id": "abc123",
  "url": "https://pps.whatsapp.net/v/...",
  "type": "image"
}
```

Response (unchanged, `known_id` matched the current picture):
```json
{ "changed": false }
```

---

## Profile picture (cacheable image)

```
GET /api/contacts/{jid}/avatar
```

Serves the actual image bytes (not JSON) from a small server-side cache, so
it can be used directly as an `<img src>` and rely on standard HTTP caching
instead of re-fetching WhatsApp's short-lived signed CDN URL on every view.
Unlike `/profile-picture`, this URL is stable per `jid` and supports
conditional requests.

Example:
```
GET /api/contacts/5511999999999@s.whatsapp.net/avatar
```

Response headers (picture available):
```
200 OK
ETag: "abc123"
Cache-Control: private, max-age=3600
Content-Type: image/jpeg

<binary image data>
```

Send the `ETag` value back as `If-None-Match` to revalidate without
re-downloading:
```
GET /api/contacts/5511999999999@s.whatsapp.net/avatar
If-None-Match: "abc123"
```
```
304 Not Modified
```

If the contact has no profile picture (or has hidden it), responds `404`
with a `Cache-Control` header too, so repeat requests don't need to ask
WhatsApp again within the cache window.

The server refreshes its cache in the background on access: a cached picture
younger than a few hours is served with no call to WhatsApp at all; older
entries are revalidated cheaply (no image re-download unless the picture
actually changed) rather than re-fetched in full. See this repo's `CLAUDE.md`
for the exact freshness windows and why.

---

## Read-only SQL query

```
POST /api/query
```

```json
{
  "sql": "SELECT jid, name FROM chats WHERE jid = ?",
  "args": ["5511999999999@s.whatsapp.net"],
  "limit": 500
}
```

For trusted local tooling that needs ad-hoc access to `chats`/`messages` beyond
what the fixed endpoints above expose (e.g. cross-chat aggregate prefiltering).
Not intended to be reachable beyond localhost.

`whatsapp.db` is attached read-only as `wdb`, so queries can also join against
`wdb.whatsmeow_contacts` and `wdb.whatsmeow_lid_map` (e.g. to resolve an `@lid`
chat JID to the underlying phone-number JID).

| Field | Default | Description |
|---|---|---|
| `sql` | required | Must start with `SELECT` or `WITH`. Single statement only |
| `args` | `[]` | Positional `?` parameters, bound (not interpolated) |
| `limit` | 500 | Max rows returned (max 5000) |

Enforcement is layered: the connection itself is opened `mode=ro&_query_only=1`
(SQLite refuses writes at the driver level regardless of the SQL text), and the
handler additionally rejects anything not starting with `SELECT`/`WITH` and any
multi-statement input. Queries time out after 5s.

Response:
```json
{
  "columns": ["jid", "name"],
  "rows": [["5511999999999@s.whatsapp.net", "John"]],
  "truncated": false
}
```

`truncated: true` means more rows matched than `limit` allowed.

---

## Mute chat

```
POST /api/chats/{jid}/mute
```

```json
{ "muted": true }
```

Response:
```json
{
  "success": true,
  "message": "Chat 5511999999999@s.whatsapp.net muted status updated"
}
```

---

## Live messages (WebSocket)

```
GET /ws/messages?client_name=my-app&jids=5511999999999@s.whatsapp.net,123456789@g.us
```

| Param | Default | Description |
|---|---|---|
| `client_name` | required | Unique name for this client (used for catch-up tracking) |
| `jids` | — | Comma-separated JIDs to filter (omit to receive all messages) |
| `typing` | `false` | Set to `true` to also receive typing/paused chat-presence events |
| `groupinfo` | `false` | Set to `true` to also receive group metadata change events (rename, topic, membership, settings) |
| `pushname` | `false` | Set to `true` to also receive contact display-name change events |

Connect with any WebSocket client. Each incoming WhatsApp message is pushed immediately as JSON:

```json
{
  "chat_jid": "5511999999999@s.whatsapp.net",
  "chat_name": "John",
  "message": {
    "id": "ABCDEF123",
    "time": "2026-03-28T20:00:00Z",
    "sender": "5511999999999",
    "full_name": "John",
    "content": "Hello",
    "is_from_me": false,
    "media_type": "image",
    "filename": "photo.jpg",
    "reply_to_id": ""
  }
}
```

`media_type`, `filename`, and `reply_to_id` are omitted when empty.

**Catch-up:** On connect, the server replays all messages missed since this client's last disconnect (tracked by `client_name`). If `jids` is set, only messages matching those JIDs are replayed. This ensures clients never miss messages across restarts.

**Typing events (`typing=true`):** When enabled, a `typing` payload is pushed whenever someone starts or stops typing in a chat the connection is subscribed to (or any chat, if `jids` is omitted):

```json
{
  "typing": {
    "chat_jid": "5511999999999@s.whatsapp.net",
    "jid": "5511999999999@s.whatsapp.net",
    "is_from_me": false,
    "state": "composing"
  }
}
```

`state` is `composing` (started typing) or `paused` (stopped typing) — WhatsApp doesn't guarantee a `paused` for every `composing` (e.g. the message may just be sent instead), so don't assume the two always pair up. `is_from_me: true` means `jid` is one of your *own* other linked devices composing/pausing in that chat (WhatsApp's multi-device sync), not another person — the account's own typing indicator isn't shown to itself in the app UI, but the bridge, as a companion device, does receive the underlying protocol event. Typing events are not persisted and are not replayed by catch-up.

For typing events to arrive at all, the bridge sends an "available" presence to WhatsApp on every connect (unconditionally, regardless of whether any client has `typing=true`) — this also makes the account show as online to contacts and enables active read receipts.

**Group info events (`groupinfo=true`):** When enabled, a `groupinfo` payload is pushed whenever a group's metadata changes — rename, topic/description, membership (join/leave/promote/demote), or settings (locked/announce/disappearing-messages/membership-approval/invite-link/deletion) — for a group the connection is subscribed to (or any group, if `jids` is omitted):

```json
{
  "groupinfo": {
    "chat_jid": "123456789@g.us",
    "sender": "5511999999999@s.whatsapp.net",
    "timestamp": "2026-03-28T20:00:00Z",
    "name": {
      "name": "New Group Name",
      "set_by": "5511999999999@s.whatsapp.net"
    }
  }
}
```

Only the field(s) that actually changed in a given event are present; every other field on the payload is omitted. The possible fields are:

| Field | Present when |
|---|---|
| `name` | Group renamed — `{name, set_by}` |
| `topic` | Topic/description changed or cleared — `{topic, deleted, set_by}` |
| `locked` | "Only admins can edit group info" toggled — boolean |
| `announce` | "Only admins can send messages" toggled — boolean |
| `ephemeral` | Disappearing messages toggled/changed — `{enabled, disappearing_timer_seconds}` |
| `membership_approval_required` | Membership approval mode toggled — boolean |
| `deleted` | Group deleted — `{deleted, reason}` |
| `new_invite_link` | Invite link regenerated |
| `join` / `leave` / `promote` / `demote` | JIDs who joined, left, were promoted to admin, or were demoted |
| `suspended` / `unsuspended` | Group suspended/unsuspended |

`sender` is omitted when WhatsApp doesn't report who made the change (e.g. `notify=invite`). Group info events are not persisted and are not replayed by catch-up.

**Push name events (`pushname=true`):** When enabled, a `pushname` payload is pushed whenever a contact's WhatsApp display name changes (detected from an incoming message), regardless of `jids` filtering — a contact's name isn't scoped to one chat:

```json
{
  "pushname": {
    "jid": "5511999999999@s.whatsapp.net",
    "old_push_name": "John",
    "new_push_name": "Johnny"
  }
}
```

Push name events are not persisted and are not replayed by catch-up.

---

## JID formats

- Individual: `5511999999999@s.whatsapp.net`
- Group: `123456789-1234567890@g.us`
