"""WebSocket connection manager for /api/ws.

Broadcasts JSON messages ({type: event_new|event_update|event_end|doorbell,
event: {...}} and {type: camera_status, ...}) to all connected clients.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from fastapi import WebSocket

log = logging.getLogger(__name__)


class WSManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket, subprotocol: Optional[str] = None) -> None:
        # `subprotocol` MUST be echoed when the client offered one (the bearer
        # token rides Sec-WebSocket-Protocol), or the browser drops the
        # connection right after the handshake. None = client offered none.
        await ws.accept(subprotocol=subprotocol)
        async with self._lock:
            self._connections.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(ws)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send to all clients concurrently with a per-send timeout, so one
        slow/stalled client can't serially block the broadcast (this runs on
        the inference worker's hot path). A client that times out or errors is
        reaped."""
        if not self._connections:
            return
        text = json.dumps(message)
        async with self._lock:
            targets = list(self._connections)

        async def _send(ws: WebSocket) -> Optional[WebSocket]:
            try:
                await asyncio.wait_for(ws.send_text(text), timeout=2.0)
                return None
            except Exception:  # noqa: BLE001 — timeout or send error => reap
                return ws

        # return_exceptions so an unexpected raise can't propagate into the
        # broadcasting worker; treat any exception result as already handled.
        results = await asyncio.gather(
            *(_send(ws) for ws in targets), return_exceptions=True
        )
        dead = [r for r in results if isinstance(r, WebSocket)]
        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.discard(ws)

    @property
    def count(self) -> int:
        return len(self._connections)
