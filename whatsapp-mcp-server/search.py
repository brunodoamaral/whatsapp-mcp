"""
Full-text search module for WhatsApp messages using Whoosh.

Features:
- Index WhatsApp messages with custom scoring
- Ignore muted groups
- Weight chats by message ratio (my messages / total messages)
- Track database changes and rebuild index when needed
"""

import os
import sqlite3
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from whoosh.filedb.filestore import FileStorage
from whoosh.fields import Schema, TEXT, KEYWORD, DATETIME, NUMERIC, ID
from whoosh.index import create_in, open_dir, exists_in
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.scoring import WeightingModel
from whoosh.analysis import StemmingAnalyzer

# Database paths
MESSAGES_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    '..', 
    'whatsapp-bridge', 
    'store', 
    'messages.db'
)

WHATSAPP_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    '..', 
    'whatsapp-bridge', 
    'store', 
    'whatsapp.db'
)

# Index storage path
INDEX_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    '.whoosh_index'
)

INDEX_METADATA_FILE = os.path.join(INDEX_DIR, '.index_metadata')


class CustomWeightingModel(WeightingModel):
    """
    Custom weighting model that:
    - Ignores muted groups
    - Weights chats by message ratio (my messages / total messages)
    - Boosts chats where the user is more active
    """
    
    def __init__(self, muted_chats: set = None, chat_weights: Dict[str, float] = None):
        """
        Args:
            muted_chats: Set of chat JIDs that are muted
            chat_weights: Dict mapping chat JID to weight factor (0-1 range, 1.0 = normal weight)
        """
        self.muted_chats = muted_chats or set()
        self.chat_weights = chat_weights or {}
    
    def score(self, searcher, fieldnum: int, text: str, matcher):
        """
        Score matches based on chat weight and muting status.
        """
        # Get the document being scored
        doc = matcher.docnum()
        try:
            stored = searcher.stored_fields(doc)
            chat_jid = stored.get('chat_jid', '')
            
            # Ignore muted chats completely (score = 0)
            if chat_jid in self.muted_chats:
                return 0.0
            
            # Apply custom weight for active chats
            weight = self.chat_weights.get(chat_jid, 1.0)
            
            # Base TF-IDF-like scoring
            # This is a simplified approach - Whoosh handles the actual scoring
            # We just apply a multiplier based on our custom weights
            base_score = super().score(searcher, fieldnum, text, matcher)
            return base_score * weight
        except Exception:
            # If we can't get stored fields, return a base score
            return super().score(searcher, fieldnum, text, matcher)


def create_schema() -> Schema:
    """Create the Whoosh index schema for WhatsApp messages."""
    analyzer = StemmingAnalyzer()
    
    return Schema(
        message_id=ID(stored=True, unique=True),
        chat_jid=KEYWORD(stored=True),
        chat_name=TEXT(stored=True),
        sender=KEYWORD(stored=True),
        sender_name=TEXT(stored=True),
        content=TEXT(stored=True, analyzer=analyzer),
        timestamp=DATETIME(stored=True, sortable=True),
        is_from_me=ID(stored=True),
        media_type=KEYWORD(stored=True),
    )


def get_muted_chats() -> set:
    """
    Get the set of muted chat JIDs from whatsapp.db.
    
    A chat is muted if muted_until is in the future (not null/0).
    """
    muted_chats = set()
    
    if not os.path.exists(WHATSAPP_DB_PATH):
        return muted_chats
    
    try:
        conn = sqlite3.connect(WHATSAPP_DB_PATH)
        cursor = conn.cursor()
        
        # Get muted chats - ones where muted_until is in the future
        cursor.execute("""
            SELECT DISTINCT chat_jid 
            FROM whatsmeow_chat_settings
            WHERE muted_until IS NOT NULL 
            AND muted_until > datetime('now')
        """)
        
        for row in cursor.fetchall():
            muted_chats.add(row[0])
        
        conn.close()
    except sqlite3.Error as e:
        print(f"Error reading muted chats: {e}")
    
    return muted_chats


def calculate_chat_weights() -> Dict[str, float]:
    """
    Calculate weight for each chat based on message ratio.
    Weight = (messages_from_me / total_messages) * 2  (capped at 2.0)
    
    Chats where user is more active get higher weight.
    """
    chat_weights = {}
    
    if not os.path.exists(MESSAGES_DB_PATH):
        return chat_weights
    
    try:
        conn = sqlite3.connect(MESSAGES_DB_PATH)
        cursor = conn.cursor()
        
        # Get message counts per chat
        cursor.execute("""
            SELECT 
                chat_jid,
                SUM(CASE WHEN is_from_me = 1 THEN 1 ELSE 0 END) as my_messages,
                COUNT(*) as total_messages
            FROM messages
            GROUP BY chat_jid
        """)
        
        for row in cursor.fetchall():
            chat_jid, my_messages, total_messages = row
            if total_messages > 0:
                ratio = my_messages / total_messages
                # Weight from 0.5 (user sends few messages) to 2.0 (user sends many)
                weight = 0.5 + (ratio * 1.5)  # Maps [0, 1] -> [0.5, 2.0]
                chat_weights[chat_jid] = weight
        
        conn.close()
    except sqlite3.Error as e:
        print(f"Error calculating chat weights: {e}")
    
    return chat_weights


def get_index_metadata() -> Dict[str, Any]:
    """Load index metadata (last update timestamp, db modification time)."""
    if not os.path.exists(INDEX_METADATA_FILE):
        return {
            'last_indexed': 0,
            'last_db_mtime': 0,
        }
    
    try:
        import json
        with open(INDEX_METADATA_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {
            'last_indexed': 0,
            'last_db_mtime': 0,
        }


def save_index_metadata(metadata: Dict[str, Any]) -> None:
    """Save index metadata."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    try:
        import json
        with open(INDEX_METADATA_FILE, 'w') as f:
            json.dump(metadata, f)
    except Exception as e:
        print(f"Error saving index metadata: {e}")


def should_rebuild_index() -> bool:
    """
    Check if the index should be rebuilt.
    Rebuilds if:
    - Index doesn't exist
    - Database file has been modified since last index
    """
    # Check if index exists
    if not exists_in(INDEX_DIR):
        return True
    
    # Check if database has been modified
    if not os.path.exists(MESSAGES_DB_PATH):
        return False
    
    metadata = get_index_metadata()
    current_mtime = os.path.getmtime(MESSAGES_DB_PATH)
    last_db_mtime = metadata.get('last_db_mtime', 0)
    
    return current_mtime > last_db_mtime


def rebuild_index() -> bool:
    """
    Rebuild the entire search index from the database.
    Returns True if successful, False otherwise.
    """
    if not os.path.exists(MESSAGES_DB_PATH):
        print(f"Database not found at {MESSAGES_DB_PATH}")
        return False
    
    print("Rebuilding search index...")
    
    try:
        # Create or clear index directory
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
        os.makedirs(INDEX_DIR, exist_ok=True)
        
        # Create index
        storage = FileStorage(INDEX_DIR)
        schema = create_schema()
        index = storage.create_index(schema)
        writer = index.writer()
        
        # Get metadata for weighting
        muted_chats = get_muted_chats()
        chat_weights = calculate_chat_weights()
        
        # Read messages from database
        conn = sqlite3.connect(MESSAGES_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                messages.id,
                messages.chat_jid,
                chats.name,
                messages.sender,
                messages.content,
                messages.timestamp,
                messages.is_from_me,
                messages.media_type
            FROM messages
            JOIN chats ON messages.chat_jid = chats.jid
            ORDER BY messages.timestamp DESC
        """)
        
        message_count = 0
        for row in cursor.fetchall():
            msg_id, chat_jid, chat_name, sender, content, timestamp, is_from_me, media_type = row
            
            # Skip empty content
            if not content or not content.strip():
                continue
            
            # Convert timestamp string to datetime if needed
            try:
                if isinstance(timestamp, str):
                    ts = datetime.fromisoformat(timestamp)
                else:
                    ts = timestamp
            except (ValueError, TypeError):
                ts = datetime.now()
            
            # Get sender name
            sender_name = get_sender_name_for_index(sender, is_from_me)
            
            writer.add_document(
                message_id=msg_id,
                chat_jid=chat_jid,
                chat_name=chat_name or '',
                sender=sender,
                sender_name=sender_name,
                content=content,
                timestamp=ts,
                is_from_me=str(is_from_me),
                media_type=media_type or '',
            )
            
            message_count += 1
        
        writer.commit()
        conn.close()
        
        # Save metadata
        save_index_metadata({
            'last_indexed': datetime.now().isoformat(),
            'last_db_mtime': os.path.getmtime(MESSAGES_DB_PATH),
            'message_count': message_count,
        })
        
        print(f"Index rebuilt successfully with {message_count} messages")
        return True
        
    except Exception as e:
        print(f"Error rebuilding index: {e}")
        return False


def get_sender_name_for_index(sender_jid: str, is_from_me: bool) -> str:
    """Get sender name for indexing."""
    if is_from_me:
        return "Me"
    
    try:
        conn = sqlite3.connect(MESSAGES_DB_PATH)
        cursor = conn.cursor()
        
        contact_jid = f"{sender_jid}@s.whatsapp.net"
        cursor.execute("""
            SELECT name FROM chats
            WHERE jid = ? AND jid NOT LIKE '%@g.us'
            LIMIT 1
        """, (contact_jid,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            return result[0]
    except sqlite3.Error:
        pass
    
    return sender_jid


def search_messages(
    query: str,
    limit: int = 20,
    page: int = 0,
    sort_by: str = 'timestamp',
    reverse: bool = True
) -> List[Dict[str, Any]]:
    """
    Search for messages using full-text search.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return
        page: Page number for pagination
        sort_by: Field to sort by ('timestamp', 'relevance')
        reverse: Whether to reverse sort order
    
    Returns:
        List of matching messages with metadata
    """
    # Rebuild index if needed
    if should_rebuild_index():
        rebuild_index()
    
    # Check if index exists
    if not exists_in(INDEX_DIR):
        print("Search index not found and could not be created")
        return []
    
    results = []
    
    try:
        storage = FileStorage(INDEX_DIR)
        index = open_dir(INDEX_DIR)
        
        # Get muted chats and weights for custom scoring
        muted_chats = get_muted_chats()
        chat_weights = calculate_chat_weights()
        
        # Create searcher with custom weighting
        searcher = index.searcher()
        
        # Create multi-field parser for searching across content, chat_name, and sender_name
        parser = MultifieldParser(
            ['content', 'chat_name', 'sender_name'],
            schema=index.schema
        )
        
        parsed_query = parser.parse(query)
        
        # Search with custom weighting
        # Note: Whoosh's default BM25F weighting is used, but we filter muted chats
        hit_count = 0
        offset = page * limit
        
        results_list = []
        for hit in searcher.search(parsed_query, limit=None):
            chat_jid = hit['chat_jid']
            
            # Skip muted chats
            if chat_jid in muted_chats:
                continue
            
            results_list.append({
                'message_id': hit['message_id'],
                'chat_jid': chat_jid,
                'chat_name': hit['chat_name'],
                'sender': hit['sender'],
                'sender_name': hit['sender_name'],
                'content': hit['content'],
                'timestamp': hit['timestamp'].isoformat() if hit['timestamp'] else None,
                'is_from_me': hit['is_from_me'] == '1',
                'media_type': hit['media_type'],
                'weight': chat_weights.get(chat_jid, 1.0),
            })
        
        # Sort by weight for active chats if requested
        if sort_by == 'relevance':
            results_list.sort(key=lambda x: x['weight'], reverse=True)
        elif sort_by == 'timestamp':
            results_list.sort(
                key=lambda x: x['timestamp'] or '',
                reverse=reverse
            )
        
        # Apply pagination
        results = results_list[offset:offset + limit]
        
        searcher.close()
        
    except Exception as e:
        print(f"Error searching messages: {e}")
    
    return results


def get_search_stats() -> Dict[str, Any]:
    """Get statistics about the search index."""
    stats = {
        'index_exists': exists_in(INDEX_DIR) if os.path.exists(INDEX_DIR) else False,
        'index_path': INDEX_DIR,
        'metadata': get_index_metadata(),
    }
    
    if stats['index_exists']:
        try:
            index = open_dir(INDEX_DIR)
            stats['document_count'] = index.doc_count_all()
            index.close()
        except Exception as e:
            print(f"Error getting index stats: {e}")
    
    return stats


def clear_index() -> bool:
    """Clear the search index."""
    try:
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
        print("Index cleared successfully")
        return True
    except Exception as e:
        print(f"Error clearing index: {e}")
        return False
