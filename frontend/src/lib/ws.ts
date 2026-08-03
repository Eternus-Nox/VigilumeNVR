/**
 * Live event WebSocket (/api/ws; JWT via Sec-WebSocket-Protocol) with
 * exponential backoff reconnect.
 * Reconnects on close/error, resets backoff on a successful open, and
 * fast-tracks a reconnect when the browser regains network or visibility.
 */
import { getToken, type ServerMessage } from './api';

export type SocketStatus = 'connecting' | 'open' | 'closed';

type MessageListener = (msg: ServerMessage) => void;
type StatusListener = (status: SocketStatus) => void;

const MIN_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 30_000;

export class LiveSocket {
  private ws: WebSocket | null = null;
  private attempts = 0;
  private stopped = true;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private messageListeners = new Set<MessageListener>();
  private statusListeners = new Set<StatusListener>();
  status: SocketStatus = 'closed';

  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    this.attempts = 0;
    window.addEventListener('online', this.onWake);
    document.addEventListener('visibilitychange', this.onVisibility);
    this.connect();
  }

  stop(): void {
    this.stopped = true;
    window.removeEventListener('online', this.onWake);
    document.removeEventListener('visibilitychange', this.onVisibility);
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
    this.setStatus('closed');
  }

  onMessage(fn: MessageListener): () => void {
    this.messageListeners.add(fn);
    return () => this.messageListeners.delete(fn);
  }

  onStatus(fn: StatusListener): () => void {
    this.statusListeners.add(fn);
    return () => this.statusListeners.delete(fn);
  }

  private onWake = (): void => {
    if (!this.stopped && this.status !== 'open') this.reconnectNow();
  };

  private onVisibility = (): void => {
    if (document.visibilityState === 'visible' && !this.stopped && this.status !== 'open') {
      this.reconnectNow();
    }
  };

  private reconnectNow(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.attempts = 0;
    this.connect();
  }

  private setStatus(status: SocketStatus): void {
    if (this.status === status) return;
    this.status = status;
    for (const fn of this.statusListeners) fn(status);
  }

  private connect(): void {
    if (this.stopped || this.ws) return;
    const token = getToken();
    if (!token) return; // logged out — AppState stops the socket on logout
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    // Token rides Sec-WebSocket-Protocol, NOT the query string: nginx logs the
    // full request line (query included) into its error log, where `log_format`
    // does not apply — that is how a live 30-day admin JWT leaked once. A
    // browser cannot set an Authorization header on a WS handshake, but the
    // subprotocol list is a header. The server echoes "bearer" back on accept.
    const url = `${proto}://${window.location.host}/api/ws`;
    this.setStatus('connecting');
    let ws: WebSocket;
    try {
      ws = new WebSocket(url, ['bearer', token]);
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      this.attempts = 0;
      this.setStatus('open');
    };
    ws.onmessage = (ev: MessageEvent) => {
      if (typeof ev.data !== 'string') return;
      let msg: ServerMessage;
      try {
        msg = JSON.parse(ev.data) as ServerMessage;
      } catch {
        return;
      }
      if (!msg || typeof msg !== 'object' || typeof msg.type !== 'string') return;
      for (const fn of this.messageListeners) fn(msg);
    };
    ws.onclose = () => {
      this.ws = null;
      this.setStatus('closed');
      this.scheduleReconnect();
    };
    ws.onerror = () => {
      // onclose always follows; nothing to do here.
    };
  }

  private scheduleReconnect(): void {
    if (this.stopped || this.reconnectTimer) return;
    const base = Math.min(MAX_BACKOFF_MS, MIN_BACKOFF_MS * 2 ** this.attempts);
    const delay = base / 2 + Math.random() * (base / 2); // jitter
    this.attempts = Math.min(this.attempts + 1, 6);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }
}
