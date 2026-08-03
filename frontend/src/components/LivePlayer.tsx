/**
 * Live player backed by go2rtc's vendored video-rtc.js web component,
 * negotiating over the nginx-proxied websocket at /go2rtc/api/ws?src=<camera>.
 * WebRTC is tried first (sub-second latency when go2rtc's WebRTC port is
 * reachable) with automatic MSE fallback when ICE fails.
 *
 * Tiles lazy-connect: the websocket is only opened once the tile scrolls
 * into view (IntersectionObserver), and the component's own lifecycle
 * disconnects on unmount / hidden tab (video-rtc handles both). TV mode
 * passes `eager` to connect every tile immediately instead.
 */
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { VideoRTC } from '../vendor/video-rtc.js';
import { api } from '../lib/api';
import AuthImage from './AuthImage';

if (!customElements.get('video-rtc')) {
  customElements.define('video-rtc', VideoRTC);
}

// Normal viewing: WebRTC-first with MSE/HLS/MJPEG fallbacks, receive-only.
// The live player is always recvonly — two-way talk is handled entirely by the
// /talk WebSocket (lib/talk.ts), which the backend delivers to the camera
// (RTSP backchannel for backchannel-capable cameras, CGI postAudio otherwise).
// The live WebRTC connection must never add a mic transceiver.
const MODE_DEFAULT = 'webrtc,mse,hls,mjpeg';

// On-screen transport readout, enabled with ?debug=live.
//
// Exists because the ONLY place remote streaming can be diagnosed is the phone
// that is actually on cellular — and opening a devtools console there needs a
// tethered Mac (iOS) or USB debugging (Android). This puts the same numbers on
// the page so they can just be read or screenshotted. Off unless asked for, so
// it costs nothing in normal use.
const DEBUG_LIVE =
  typeof window !== 'undefined' &&
  new URLSearchParams(window.location.search).get('debug') === 'live';
const MEDIA_DEFAULT = 'video,audio';

interface LivePlayerProps {
  camera: string;
  /** muted tile mode (dashboard) vs full view (camera detail) */
  controls?: boolean;
  className?: string;
  /** Connect immediately instead of waiting for visibility (TV wall). */
  eager?: boolean;
  /**
   * Use the camera's low-bitrate SUB stream (`<name>_sub`) instead of the
   * full-res MAIN stream. Multi-camera walls (dashboard grid, TV wall) pass
   * `sub` so N tiles don't each pull a full-res feed; a single/focused view
   * leaves it off for full quality. Both go2rtc streams are browser-serveable.
   */
  sub?: boolean;
}

// memo: all props are primitives, so a parent tile re-rendering on live churn
// (last-event overlay, online flip) skips the player subtree entirely — the
// stream element is only touched when camera/sub/controls actually change.
function LivePlayer({
  camera,
  controls = false,
  className,
  eager = false,
  sub = false,
}: LivePlayerProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const playerRef = useRef<VideoRTC | null>(null);
  const [visible, setVisible] = useState(false);
  const [state, setState] = useState<'idle' | 'connecting' | 'playing' | 'error'>('idle');
  // Some cameras (notably the AD410 doorbell) serve no usable `_sub` substream,
  // so a `sub` tile would show a permanent black/offline state. When a sub
  // stream errors or never yields a frame, downgrade ONCE to the main stream
  // before giving up. `downgraded` latches so we never loop main<->sub: after
  // one downgrade a further failure shows the offline state.
  const [downgraded, setDowngraded] = useState(false);
  // Fullscreen + sound. BOTH are deliberately absent from the element effect's
  // dependency array — toggling either must never recreate <video-rtc>, which
  // would drop the negotiated stream.
  const [fs, setFs] = useState(false);
  const [muted, setMuted] = useState(!controls);
  // Read by the element effect WITHOUT being a dependency: a mute toggle must
  // not tear the stream down and rebuild it.
  const mutedRef = useRef(muted);
  mutedRef.current = muted;

  // Lazy-connect: wait until the tile is (nearly) on screen.
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    if (eager || !('IntersectionObserver' in window)) {
      setVisible(true);
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setVisible(true);
          io.disconnect();
        }
      },
      { rootMargin: '100px' },
    );
    io.observe(host);
    return () => io.disconnect();
  }, [eager]);

  useEffect(() => {
    if (!visible) return;
    const host = hostRef.current;
    if (!host) return;

    const el = document.createElement('video-rtc') as VideoRTC;
    // Contract (groups/WebRTC addendum): WebRTC first — video-rtc negotiates
    // over the same /go2rtc/ WS proxy and falls back to MSE automatically
    // when ICE fails (i.e. when no configured WebRTC candidate is
    // reachable); hls/mjpeg remain as last-resort same-path fallbacks. The
    // player is always receive-only; two-way talk rides the /talk WebSocket,
    // never this peer connection, so no mic transceiver is ever added here.
    el.mode = MODE_DEFAULT;
    el.media = MEDIA_DEFAULT;
    el.background = false;
    el.visibilityCheck = true;
    el.style.width = '100%';
    el.style.height = '100%';
    playerRef.current = el;
    host.appendChild(el);

    // Multi-camera views pass `sub` to negotiate the low-bitrate substream
    // (`<name>_sub`); single/focused views use the full-res main stream. Once
    // a sub stream has failed we drop `_sub` and use the main stream instead.
    const usingSub = sub && !downgraded;
    const stream = usingSub ? `${camera}_sub` : camera;
    setState('connecting');
    // NO token. This used to call mediaUrl(), which appends the SESSION JWT —
    // written for an nginx auth_request gate that was REVERTED after it broke
    // streaming. The token therefore bought nothing here, while nginx writes
    // the full request line into its error log (where log_format does not
    // apply), so every live tile leaked an admin token into a plaintext log.
    // That is the same defect task #41 fixed on the backend WebSocket.
    //
    // /go2rtc/ is instead protected by a deny-by-default path+query allowlist
    // in nginx.conf.template — no credential, so nothing can expire mid-stream
    // or lock a client out. Keep this a bare path.
    el.src = `/go2rtc/api/ws?src=${encodeURIComponent(stream)}`;

    // video element exists after the element connects (oninit ran in appendChild)
    const video = el.video;
    let cancelled = false;
    // A sub stream that fails downgrades once to main (re-runs this effect via
    // `downgraded`); after that, or when already on main, failures show error.
    const fail = () => {
      if (cancelled) return;
      if (usingSub) setDowngraded(true);
      else setState('error');
    };
    const onPlaying = () => !cancelled && setState('playing');
    const onWaiting = () => !cancelled && setState('connecting');
    const onError = fail;
    if (video) {
      // Native controls are NEVER used. On iPhone their fullscreen button opens
      // a presentation layer that cannot render a MediaStream (frozen frame,
      // audio still playing), and mixing the browser's control bar with our own
      // buttons was simply confusing. We draw every control ourselves.
      video.muted = mutedRef.current;
      video.controls = false;
      video.playsInline = true;
      video.addEventListener('playing', onPlaying);
      video.addEventListener('waiting', onWaiting);
      video.addEventListener('error', onError);
    }
    // If nothing arrives for a while, downgrade (sub) or surface an offline
    // hint (main); the underlying stream keeps retrying regardless.
    const stallTimer = setTimeout(() => {
      if (!cancelled && video && video.readyState < 2) fail();
    }, 12_000);

    return () => {
      cancelled = true;
      clearTimeout(stallTimer);
      if (video) {
        video.removeEventListener('playing', onPlaying);
        video.removeEventListener('waiting', onWaiting);
        video.removeEventListener('error', onError);
      }
      // Removing from DOM triggers disconnectedCallback -> ondisconnect after
      // DISCONNECT_TIMEOUT; force it immediately so tiles free the socket now.
      host.removeChild(el);
      el.background = false;
      try {
        el.ondisconnect();
      } catch {
        /* already closed */
      }
      playerRef.current = null;
    };
  }, [visible, camera, controls, sub, downgraded]);

  // Reset the one-shot sub->main downgrade whenever the target changes, so a
  // new camera (or re-enabling sub) gets a fresh attempt at its `_sub` stream.
  useEffect(() => {
    setDowngraded(false);
  }, [camera, sub]);

  const [dbg, setDbg] = useState('');
  useEffect(() => {
    if (!DEBUG_LIVE) return;
    const id = setInterval(() => {
      const el = playerRef.current;
      const v = el?.video;
      if (!el || !v) {
        setDbg('no player');
        return;
      }
      // ws/pc use the WebSocket constants: 0 CONNECTING, 1 OPEN, 2 CLOSING,
      // 3 CLOSED. A ws that keeps flipping 0<->1 means the socket is being
      // torn down and rebuilt; buf stuck at 0 with ws=1 means the socket is
      // open but no media is arriving.
      const buf = v.buffered.length
        ? (v.buffered.end(v.buffered.length - 1) - v.buffered.start(0)).toFixed(1)
        : '0';
      setDbg(
        `ws${el.wsState} pc${el.pcState} rs${v.readyState} ` +
          `buf${buf}s t${v.currentTime.toFixed(1)} ${el.mseCodecs ? 'mse' : 'no-mse'}`,
      );
    }, 1000);
    return () => clearInterval(id);
  }, [visible]);

  // Push mute changes straight onto the live <video>. Separate effect, so the
  // element effect above never re-runs for a sound toggle.
  useEffect(() => {
    const v = playerRef.current?.video;
    if (v) v.muted = muted;
  }, [muted]);

  // Real fullscreen where the API exists, CSS pseudo-fullscreen where it does
  // not. Feature-tested rather than UA-sniffed: on iPhone
  // Element.requestFullscreen is simply undefined (WebKit bug 206854, still
  // open), so the test lands on the fallback by itself.
  //
  // WHY A FALLBACK IS NEEDED AT ALL: a <video> on iPhone has exactly one
  // fullscreen, the OS player, and it cannot present a MediaStream — which is
  // what video-rtc assigns via srcObject on the WebRTC path (always the winner
  // here, since the `ffmpeg:<name>#audio=aac` producer never runs on this
  // go2rtc image, so MSE can never out-score it). The symptom is the last
  // decoded frame held while audio keeps playing. Confirmed on the fleet.
  // Restyling the host IN PLACE avoids it: the <video> is never reparented,
  // and moving a MediaStream-backed element drops the stream on WebKit.
  const toggleFullscreen = useCallback(() => {
    const host = hostRef.current;
    if (!host) return;
    if (document.fullscreenElement) {
      void document.exitFullscreen?.().catch(() => {});
      return;
    }
    if (!fs && typeof host.requestFullscreen === 'function') {
      host
        .requestFullscreen()
        .then(() => setFs(true))
        .catch(() => setFs(true)); // API present but refused -> CSS fallback
      return;
    }
    setFs((v) => !v);
  }, [fs]);

  // Keep our state honest when fullscreen is left by Esc or a system gesture
  // rather than by our button.
  useEffect(() => {
    const onChange = () => {
      if (!document.fullscreenElement) setFs(false);
    };
    document.addEventListener('fullscreenchange', onChange);
    return () => document.removeEventListener('fullscreenchange', onChange);
  }, []);

  const hostClass = [
    'live-player',
    fs ? 'is-fs' : '',
    controls ? 'has-controls' : '',
    className ?? '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div ref={hostRef} className={hostClass} data-state={state}>
      {DEBUG_LIVE && dbg && <span className="live-player-debug">{dbg}</span>}
      {controls && (
        <div className="live-player-controls">
          <button
            type="button"
            className="live-player-btn"
            aria-label={muted ? 'Unmute' : 'Mute'}
            aria-pressed={!muted}
            onClick={() => setMuted((v) => !v)}
          >
            {muted ? '\u{1F507}' : '\u{1F50A}'}
          </button>
          <button
            type="button"
            className="live-player-btn"
            aria-label={fs ? 'Exit full screen' : 'Full screen'}
            aria-pressed={fs}
            onClick={toggleFullscreen}
          >
            {fs ? '\u2715' : '\u26f6'}
          </button>
        </div>
      )}
      {/* Instant first paint. The backend answers this from the detector's
          in-memory frame cache (no camera round-trip), so the tile shows an
          image almost immediately instead of a black box for the 0.5-3 s the
          stream takes to negotiate + wait for the camera's next keyframe.
          Unmounted the moment real video plays, so it can never mask a live
          feed. A private camera 403s here and AuthImage simply renders
          nothing — which is correct, Privacy Mode must not show a stale frame. */}
      {state !== 'playing' && (
        <AuthImage
          className="live-player-poster"
          src={api.cameraSnapshotPath(camera)}
          alt=""
          aria-hidden="true"
        />
      )}
      {state !== 'playing' && (
        <div className="live-player-overlay">
          {state === 'error' ? (
            <span className="live-player-msg">stream unavailable</span>
          ) : (
            <span className="live-player-spinner" aria-label="connecting" />
          )}
        </div>
      )}
    </div>
  );
}

export default memo(LivePlayer);
