/**
 * Push-to-talk engine for camera two-way audio.
 *
 * Pipeline: getUserMedia (echo-cancelled mic) -> AudioWorklet capture node
 * (ScriptProcessorNode fallback) -> averaging decimator to 8 kHz Int16 mono
 * PCM -> ~40 ms little-endian binary frames over the /talk WebSocket. The
 * backend transcodes to G.711 A-law and streams it to the camera speaker.
 *
 * One TalkSession per press: start() on press, stop() on release. stop() is
 * idempotent and always releases the mic. Server close codes 4003 (no
 * speaker), 4009 (busy) and 4502 (camera rejected audio) surface as faults
 * via the onFault callback; state transitions drive the button visuals.
 */

import { wsSubprotocols } from './api';

export type TalkState = 'idle' | 'connecting' | 'live' | 'error';

export interface TalkFault {
  /** WS close code when the server ended the session (4003/4009/4502). */
  code?: number;
  title: string;
  body: string;
}

export interface TalkCallbacks {
  onState: (state: TalkState) => void;
  onFault: (fault: TalkFault) => void;
}

const TARGET_RATE = 8000;
/** Samples per binary frame: 320 = 40 ms at 8 kHz (spec allows 20–60 ms). */
const FRAME_SAMPLES = 320;

/** Inline AudioWorklet module: forwards raw Float32 capture chunks to main. */
const WORKLET_SOURCE = `
class SentinelPttCapture extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch && ch.length) {
      const copy = new Float32Array(ch);
      this.port.postMessage(copy, [copy.buffer]);
    }
    return true;
  }
}
registerProcessor('sentinel-ptt-capture', SentinelPttCapture);
`;

function faultForClose(code: number): TalkFault | null {
  switch (code) {
    case 4003:
      return {
        code,
        title: 'No speaker',
        body: 'This camera does not support two-way talk.',
      };
    case 4009:
      return {
        code,
        title: 'Speaker busy',
        body: 'Someone else is already talking through this camera.',
      };
    case 4502:
      return {
        code,
        title: 'Camera rejected audio',
        body: 'The camera refused the audio stream.',
      };
    default:
      return null;
  }
}

/**
 * Stateful averaging decimator: arbitrary input rate -> 8 kHz Int16 mono.
 * Carries fractional read position and leftover samples across chunks so
 * non-integer ratios (e.g. 44100/8000) stay drift-free.
 */
class Downsampler {
  private readonly ratio: number;
  private carry = new Float32Array(0);
  private pos = 0;

  constructor(inputRate: number) {
    this.ratio = inputRate / TARGET_RATE;
  }

  process(input: Float32Array): Int16Array {
    const buf = new Float32Array(this.carry.length + input.length);
    buf.set(this.carry, 0);
    buf.set(input, this.carry.length);
    const out = new Int16Array(Math.ceil(buf.length / this.ratio) + 1);
    let n = 0;
    let pos = this.pos;
    while (pos + this.ratio <= buf.length) {
      const start = Math.floor(pos);
      const end = Math.max(start + 1, Math.floor(pos + this.ratio));
      let sum = 0;
      for (let i = start; i < end; i++) sum += buf[i];
      const v = sum / (end - start);
      out[n++] = v >= 1 ? 32767 : v <= -1 ? -32768 : Math.round(v * 32767);
      pos += this.ratio;
    }
    const keep = Math.floor(pos);
    this.carry = buf.slice(keep);
    this.pos = pos - keep;
    return out.subarray(0, n);
  }
}

export class TalkSession {
  private readonly url: string;
  private readonly cb: TalkCallbacks;
  private stopped = false;
  private opened = false;
  private ws: WebSocket | null = null;
  private stream: MediaStream | null = null;
  private ctx: AudioContext | null = null;
  private nodes: AudioNode[] = [];
  private down: Downsampler | null = null;
  private readonly pending = new Int16Array(FRAME_SAMPLES);
  private pendingLen = 0;

  constructor(url: string, cb: TalkCallbacks) {
    this.url = url;
    this.cb = cb;
  }

  async start(): Promise<void> {
    this.cb.onState('connecting');

    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      this.fail({
        title: 'Microphone unavailable',
        body: 'Two-way talk requires HTTPS (see docs/remote-access.md).',
      });
      return;
    }

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      });
    } catch {
      this.fail({
        title: 'Microphone blocked',
        body: 'Allow microphone access in the browser to use two-way talk.',
      });
      return;
    }
    if (this.stopped) {
      // Released before the permission prompt resolved.
      for (const t of stream.getTracks()) t.stop();
      return;
    }
    this.stream = stream;

    // Token via subprotocol, never the query string — see api.wsSubprotocols.
    const ws = new WebSocket(this.url, wsSubprotocols());
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    ws.onopen = () => {
      this.opened = true;
    };
    ws.onclose = (ev) => this.handleClose(ev);
    await new Promise<void>((resolve) => {
      ws.addEventListener('open', () => resolve(), { once: true });
      ws.addEventListener('close', () => resolve(), { once: true });
    });
    // stop() during connect, or handleClose already faulted.
    if (this.stopped || ws.readyState !== WebSocket.OPEN) return;

    try {
      await this.startAudio(stream);
    } catch {
      this.fail({
        title: 'Audio setup failed',
        body: 'Could not start microphone capture in this browser.',
      });
      return;
    }
    if (this.stopped) {
      // Server hung up (busy/refused) while the pipeline was starting.
      this.releaseMedia();
      return;
    }
    this.cb.onState('live');
  }

  /** Release the mic, tell the server to stop, close the socket. Idempotent. */
  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    const ws = this.ws;
    this.ws = null;
    if (ws) {
      ws.onclose = null; // our own close is not a fault
      if (ws.readyState === WebSocket.OPEN) {
        try {
          if (this.pendingLen > 0) sendFrame(ws, this.pending, this.pendingLen);
          ws.send(JSON.stringify({ type: 'stop' }));
        } catch {
          /* socket already dying */
        }
      }
      try {
        ws.close(1000);
      } catch {
        /* ignore */
      }
    }
    this.releaseMedia();
    this.cb.onState('idle');
  }

  private fail(fault: TalkFault): void {
    if (this.stopped) return;
    this.stopped = true;
    const ws = this.ws;
    this.ws = null;
    if (ws) {
      ws.onclose = null;
      try {
        ws.close(1000);
      } catch {
        /* ignore */
      }
    }
    this.releaseMedia();
    this.cb.onFault(fault);
    this.cb.onState('error');
  }

  /** Server-initiated close (4003/4009/4502, 120 s cap, or network drop). */
  private handleClose(ev: CloseEvent): void {
    if (this.stopped) return;
    this.stopped = true;
    this.ws = null;
    this.releaseMedia();
    const fault =
      faultForClose(ev.code) ??
      (this.opened
        ? { title: 'Talk ended', body: ev.reason || 'The connection was closed.' }
        : { title: 'Talk failed', body: 'Could not reach the NVR.' });
    this.cb.onFault(fault);
    this.cb.onState('error');
  }

  private async startAudio(stream: MediaStream): Promise<void> {
    const ctx = new AudioContext();
    this.ctx = ctx;
    if (ctx.state === 'suspended') {
      try {
        await ctx.resume();
      } catch {
        /* autoplay policy — we're inside a user gesture, so this is unlikely */
      }
    }
    this.down = new Downsampler(ctx.sampleRate);
    const source = ctx.createMediaStreamSource(stream);

    let capture: AudioNode | null = null;
    if (ctx.audioWorklet && typeof AudioWorkletNode !== 'undefined') {
      try {
        const blobUrl = URL.createObjectURL(
          new Blob([WORKLET_SOURCE], { type: 'text/javascript' }),
        );
        try {
          await ctx.audioWorklet.addModule(blobUrl);
        } finally {
          URL.revokeObjectURL(blobUrl);
        }
        const node = new AudioWorkletNode(ctx, 'sentinel-ptt-capture', {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          channelCount: 1,
        });
        node.port.onmessage = (ev: MessageEvent) => {
          if (ev.data instanceof Float32Array) this.handleChunk(ev.data);
        };
        capture = node;
      } catch {
        capture = null; // CSP/browser quirk — fall back below
      }
    }
    if (!capture) {
      // Deprecated but universally supported fallback.
      const node = ctx.createScriptProcessor(4096, 1, 1);
      node.onaudioprocess = (ev: AudioProcessingEvent) => {
        this.handleChunk(ev.inputBuffer.getChannelData(0));
      };
      capture = node;
    }

    // Keep the graph pulled without feeding mic audio back to the speakers.
    const mute = ctx.createGain();
    mute.gain.value = 0;
    source.connect(capture);
    capture.connect(mute);
    mute.connect(ctx.destination);
    this.nodes = [source, capture, mute];
  }

  private handleChunk(input: Float32Array): void {
    if (this.stopped || !this.down) return;
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const pcm = this.down.process(input);
    let offset = 0;
    while (offset < pcm.length) {
      const take = Math.min(FRAME_SAMPLES - this.pendingLen, pcm.length - offset);
      this.pending.set(pcm.subarray(offset, offset + take), this.pendingLen);
      this.pendingLen += take;
      offset += take;
      if (this.pendingLen === FRAME_SAMPLES) {
        sendFrame(ws, this.pending, FRAME_SAMPLES);
        this.pendingLen = 0;
      }
    }
  }

  private releaseMedia(): void {
    for (const node of this.nodes) {
      try {
        node.disconnect();
      } catch {
        /* already disconnected */
      }
    }
    this.nodes = [];
    if (this.stream) {
      for (const t of this.stream.getTracks()) t.stop();
      this.stream = null;
    }
    if (this.ctx) {
      void this.ctx.close().catch(() => undefined);
      this.ctx = null;
    }
    this.down = null;
    this.pendingLen = 0;
  }
}

/** Explicit little-endian Int16 PCM frame (contract: raw LE Int16, mono, 8 kHz). */
function sendFrame(ws: WebSocket, samples: Int16Array, length: number): void {
  if (length <= 0) return;
  const buf = new ArrayBuffer(length * 2);
  const view = new DataView(buf);
  for (let i = 0; i < length; i++) view.setInt16(i * 2, samples[i], true);
  try {
    ws.send(buf);
  } catch {
    /* close handler will surface the failure */
  }
}
