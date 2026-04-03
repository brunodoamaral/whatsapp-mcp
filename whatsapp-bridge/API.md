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
      "is_from_me": false
    }
  ]
}
```

---

## Mute chat

```
POST /api/chats/{jid}/mute
```

```json
{ "muted": true }
```

---

## Live messages (WebSocket)

```
GET /ws/messages
```

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

---

## JID formats

- Individual: `5511999999999@s.whatsapp.net`
- Group: `123456789-1234567890@g.us`
