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
GET /api/search?q=hello&limit=10&days_since=30&chat_jid=...&semantic_weight=0.5
```

| Param | Default | Description |
|---|---|---|
| `q` | required | Search query |
| `limit` | 10 | Max results (max 100) |
| `days_since` | — | Restrict to last N days |
| `chat_jid` | — | Comma-separated JIDs to filter |
| `semantic_weight` | 0.5 | 0 = text only, 1 = semantic only |

Results are grouped into context windows of up to 16 consecutive messages.

Note: messages within a search result use Go's default (capitalized, no `omitempty`) field names, unlike every other endpoint below — this differs from the `snake_case` used elsewhere in this API.

Response:
```json
{
  "query": "hello",
  "total": 2,
  "results": [
    {
      "chat_jid": "5511999999999@s.whatsapp.net",
      "chat_name": "John",
      "score": 0.95,
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

---

## JID formats

- Individual: `5511999999999@s.whatsapp.net`
- Group: `123456789-1234567890@g.us`
