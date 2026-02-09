"""
Telegram client using Telethon for direct API access.
Supports: date range queries, search, fast message loading, media download.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, AsyncIterator

from telethon import TelegramClient
from telethon.tl.types import Message, MessageMediaPhoto, MessageMediaDocument
from telethon.tl.functions.messages import SearchRequest
from telethon.tl.types import InputMessagesFilterEmpty

# Session and config paths
SESSION_PATH = Path(__file__).parent / "telegram_session"
CONFIG_PATH = Path(__file__).parent / "telegram_credentials.txt"


@dataclass
class TelegramMessage:
    """Represents a Telegram message with metadata."""
    id: int
    text: str
    timestamp: datetime
    sender_name: str
    media_ids: tuple[str, ...]  # File IDs for media
    media_type: str  # "photo", "video", "document", or ""
    channel_name: str
    channel_username: str
    url: str

    @property
    def timestamp_iso(self) -> str:
        return self.timestamp.isoformat() if self.timestamp else ""

    @property
    def timestamp_display(self) -> str:
        if not self.timestamp:
            return ""
        return self.timestamp.strftime("%Y-%m-%d %H:%M")


def load_credentials() -> tuple[int, str] | None:
    """Load API credentials from config file."""
    if not CONFIG_PATH.exists():
        return None
    try:
        lines = CONFIG_PATH.read_text().strip().split("\n")
        api_id = int(lines[0].split("=")[1].strip())
        api_hash = lines[1].split("=")[1].strip()
        return api_id, api_hash
    except Exception:
        return None


def save_credentials(api_id: int, api_hash: str) -> None:
    """Save API credentials to config file."""
    CONFIG_PATH.write_text(f"api_id={api_id}\napi_hash={api_hash}\n")


class TelegramAPI:
    """Async Telegram client wrapper."""

    def __init__(self, api_id: int, api_hash: str):
        self.api_id = api_id
        self.api_hash = api_hash
        self.client: Optional[TelegramClient] = None
        self._media_cache: dict[str, bytes] = {}

    async def connect(self) -> bool:
        """Connect and authenticate if needed. Returns True if connected."""
        self.client = TelegramClient(
            str(SESSION_PATH),
            self.api_id,
            self.api_hash
        )
        await self.client.connect()
        return await self.client.is_user_authorized()

    async def send_code(self, phone: str) -> str:
        """Send auth code to phone. Returns phone_code_hash."""
        result = await self.client.send_code_request(phone)
        return result.phone_code_hash

    async def sign_in(self, phone: str, code: str, phone_code_hash: str) -> bool:
        """Complete sign in with code. Returns True on success."""
        try:
            await self.client.sign_in(phone, code, phone_code_hash=phone_code_hash)
            return True
        except Exception as e:
            print(f"Sign in error: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect the client."""
        if self.client:
            await self.client.disconnect()

    async def get_channel_messages(
        self,
        channel: str,
        limit: int = 100,
        offset_date: Optional[datetime] = None,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
        search_query: Optional[str] = None,
    ) -> list[TelegramMessage]:
        """
        Fetch messages from a channel.

        Args:
            channel: Channel username or URL
            limit: Max messages to fetch
            offset_date: Start fetching from this date (going backward)
            min_date: Only include messages after this date
            max_date: Only include messages before this date
            search_query: Search for specific text
        """
        if not self.client:
            raise RuntimeError("Client not connected")

        # Normalize channel input
        channel = self._normalize_channel(channel)

        try:
            entity = await self.client.get_entity(channel)
        except Exception as e:
            raise ValueError(f"Could not find channel: {channel}") from e

        channel_name = getattr(entity, 'title', channel)
        channel_username = getattr(entity, 'username', '') or channel

        messages: list[TelegramMessage] = []

        if search_query:
            # Use search API for text search
            async for msg in self.client.iter_messages(
                entity,
                limit=limit,
                search=search_query,
                offset_date=offset_date,
            ):
                if not isinstance(msg, Message):
                    continue
                tm = self._convert_message(msg, channel_name, channel_username)
                if self._in_date_range(tm, min_date, max_date):
                    messages.append(tm)
        else:
            # Regular message iteration
            # When date range is specified, don't limit - fetch all in range
            effective_limit = limit if not (min_date and max_date) else None
            fetched = 0
            max_fetch = limit * 10 if effective_limit is None else limit  # Safety cap

            async for msg in self.client.iter_messages(
                entity,
                limit=effective_limit,
                offset_date=offset_date,
            ):
                if not isinstance(msg, Message):
                    continue

                fetched += 1
                if fetched > max_fetch:
                    break  # Safety cap to prevent infinite fetching

                tm = self._convert_message(msg, channel_name, channel_username)

                # Check date range
                if max_date and tm.timestamp and tm.timestamp > max_date:
                    continue  # Skip messages newer than max_date
                if min_date and tm.timestamp and tm.timestamp < min_date:
                    break  # Stop when we've gone past min_date

                messages.append(tm)

                # Stop if we have enough messages
                if len(messages) >= limit:
                    break

        return messages

    async def download_media(self, message_or_id, max_size_mb: float = 10) -> bytes | None:
        """Download media from a message. Returns bytes or None."""
        if not self.client:
            return None
        try:
            # Check if it's a photo (usually small)
            data = await self.client.download_media(message_or_id, bytes)
            if data and len(data) <= max_size_mb * 1024 * 1024:
                return data
            return None
        except Exception:
            return None

    async def download_media_for_messages(
        self,
        channel: str,
        message_ids: list[int],
        max_size_mb: float = 10,
    ) -> dict[int, bytes]:
        """Download media for multiple messages. Returns {msg_id: bytes}."""
        if not self.client:
            return {}

        channel = self._normalize_channel(channel)
        entity = await self.client.get_entity(channel)

        result: dict[int, bytes] = {}

        # Fetch messages with media
        async for msg in self.client.iter_messages(entity, ids=message_ids):
            if not isinstance(msg, Message) or not msg.media:
                continue

            # Download photos
            if isinstance(msg.media, MessageMediaPhoto):
                data = await self.download_media(msg, max_size_mb)
                if data:
                    result[msg.id] = data
            elif isinstance(msg.media, MessageMediaDocument):
                doc = msg.media.document
                if doc and hasattr(doc, 'mime_type'):
                    mime = doc.mime_type or ""
                    # Download image documents
                    if mime.startswith('image/'):
                        if doc.size <= max_size_mb * 1024 * 1024:
                            data = await self.download_media(msg, max_size_mb)
                            if data:
                                result[msg.id] = data
                    # Download video thumbnails (not full videos - too large)
                    elif mime.startswith('video/'):
                        # Try to get video thumbnail
                        if hasattr(doc, 'thumbs') and doc.thumbs:
                            try:
                                # Download the largest thumbnail
                                data = await self.client.download_media(msg, bytes, thumb=-1)
                                if data:
                                    result[msg.id] = data
                            except Exception:
                                pass

        return result

    def _normalize_channel(self, channel: str) -> str:
        """Normalize channel input to username."""
        channel = channel.strip()
        # Remove URL prefixes
        for prefix in ['https://t.me/', 'http://t.me/', 't.me/', 'https://t.me/s/', 't.me/s/']:
            if channel.lower().startswith(prefix):
                channel = channel[len(prefix):]
                break
        # Remove @ prefix
        if channel.startswith('@'):
            channel = channel[1:]
        # Remove trailing slashes and query params
        channel = channel.split('/')[0].split('?')[0]
        return channel

    def _convert_message(self, msg: Message, channel_name: str, channel_username: str) -> TelegramMessage:
        """Convert Telethon Message to our TelegramMessage."""
        text = msg.message or ""
        timestamp = msg.date
        if timestamp and timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        sender_name = channel_name
        if msg.sender:
            sender_name = getattr(msg.sender, 'first_name', '') or channel_name

        # Track if message has downloadable media and its type
        media_ids: list[str] = []
        media_type = ""
        if msg.media:
            if isinstance(msg.media, MessageMediaPhoto):
                media_ids.append(f"msg:{msg.id}")
                media_type = "photo"
            elif isinstance(msg.media, MessageMediaDocument):
                doc = msg.media.document
                if doc and hasattr(doc, 'mime_type'):
                    mime = doc.mime_type or ""
                    if mime.startswith('image/'):
                        media_ids.append(f"msg:{msg.id}")
                        media_type = "photo"
                    elif mime.startswith('video/'):
                        media_ids.append(f"msg:{msg.id}")
                        media_type = "video"

        url = f"https://t.me/{channel_username}/{msg.id}" if channel_username else ""

        return TelegramMessage(
            id=msg.id,
            text=text,
            timestamp=timestamp,
            sender_name=sender_name,
            media_ids=tuple(media_ids),
            media_type=media_type,
            channel_name=channel_name,
            channel_username=channel_username,
            url=url,
        )

    def _in_date_range(
        self,
        msg: TelegramMessage,
        min_date: Optional[datetime],
        max_date: Optional[datetime],
    ) -> bool:
        """Check if message is within date range."""
        if not msg.timestamp:
            return True
        if min_date and msg.timestamp < min_date:
            return False
        if max_date and msg.timestamp > max_date:
            return False
        return True


# Synchronous wrapper for use in the GUI
class TelegramClientSync:
    """Synchronous wrapper around the async Telegram API."""

    def __init__(self):
        self.api: Optional[TelegramAPI] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _run(self, coro):
        loop = self._get_loop()
        return loop.run_until_complete(coro)

    def is_configured(self) -> bool:
        """Check if credentials are configured."""
        return load_credentials() is not None

    def connect(self) -> bool:
        """Connect to Telegram. Returns True if authenticated."""
        creds = load_credentials()
        if not creds:
            return False
        api_id, api_hash = creds
        self.api = TelegramAPI(api_id, api_hash)
        return self._run(self.api.connect())

    def send_code(self, phone: str) -> str:
        """Send verification code. Returns phone_code_hash."""
        if not self.api:
            raise RuntimeError("Not connected")
        return self._run(self.api.send_code(phone))

    def sign_in(self, phone: str, code: str, phone_code_hash: str) -> bool:
        """Complete sign in."""
        if not self.api:
            raise RuntimeError("Not connected")
        return self._run(self.api.sign_in(phone, code, phone_code_hash))

    def disconnect(self) -> None:
        """Disconnect."""
        if self.api:
            self._run(self.api.disconnect())

    def get_messages(
        self,
        channel: str,
        limit: int = 100,
        offset_date: Optional[datetime] = None,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
        search_query: Optional[str] = None,
    ) -> list[TelegramMessage]:
        """Fetch messages from channel."""
        if not self.api:
            raise RuntimeError("Not connected")
        return self._run(self.api.get_channel_messages(
            channel, limit, offset_date, min_date, max_date, search_query
        ))

    def download_media(self, channel: str, message_ids: list[int]) -> dict[int, bytes]:
        """Download media for messages."""
        if not self.api:
            return {}
        return self._run(self.api.download_media_for_messages(channel, message_ids))


# Global client instance
_client: Optional[TelegramClientSync] = None


def get_client() -> TelegramClientSync:
    """Get or create the global Telegram client."""
    global _client
    if _client is None:
        _client = TelegramClientSync()
    return _client
