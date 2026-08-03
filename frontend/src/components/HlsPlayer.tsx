/**
 * HLS <video> for continuous recordings. Plays a VOD playlist URL that already
 * carries its media `?token=` (segment URIs inherit it server-side).
 *
 * Playback path:
 * - Safari / iOS have native HLS — we set `video.src` directly and skip hls.js.
 * - Everyone else (Chrome, Firefox, Android) gets hls.js, dynamically imported
 *   so it only ships to the Timeline route and never bloats the main bundle.
 *
 * The parent owns seeking: it holds `videoRef` and maps wall-clock → media time
 * itself, applying the seek once `onReady` fires (manifest parsed / metadata
 * loaded). We keep this component about transport + recovery only.
 */
import { useEffect, useRef, useState } from 'react';
import type HlsType from 'hls.js';

interface HlsPlayerProps {
  /** Playlist URL (token inline). null → nothing loaded (renders a placeholder). */
  src: string | null;
  /** Owned by the parent; the underlying <video> for seek + timeupdate. */
  videoRef: React.RefObject<HTMLVideoElement | null>;
  /** Fires when the media is seekable (manifest parsed / metadata loaded). */
  onReady?: () => void;
  /** Fires when playback reaches the end of the loaded window. */
  onEnded?: () => void;
  /**
   * Show the browser's native video controls (default true). Synchronized
   * multi-camera tiles pass false — a native per-tile pause/seek would silently
   * desync that tile from the shared transport driving the group.
   */
  controls?: boolean;
}

export default function HlsPlayer({
  src,
  videoRef,
  onReady,
  onEnded,
  controls = true,
}: HlsPlayerProps) {
  const hlsRef = useRef<HlsType | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  // Keep the latest callbacks without re-running the load effect on every render.
  const onReadyRef = useRef(onReady);
  const onEndedRef = useRef(onEnded);
  onReadyRef.current = onReady;
  onEndedRef.current = onEnded;

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (!src) {
      setStatus('idle');
      if (hlsRef.current) {
        hlsRef.current.destroy();
        hlsRef.current = null;
      }
      video.removeAttribute('src');
      video.load();
      return;
    }

    let cancelled = false;
    setStatus('loading');

    const markReady = () => {
      if (cancelled) return;
      setStatus('ready');
      onReadyRef.current?.();
    };
    const onEnd = () => onEndedRef.current?.();
    video.addEventListener('ended', onEnd);

    // Prefer native HLS when the browser has it (Safari/iOS) — lighter, and it
    // handles the token-bearing segment URIs the same way.
    const nativeHls =
      video.canPlayType('application/vnd.apple.mpegurl') !== '' ||
      video.canPlayType('application/x-mpegURL') !== '';

    if (nativeHls) {
      const onError = () => !cancelled && setStatus('error');
      video.addEventListener('loadedmetadata', markReady, { once: true });
      video.addEventListener('error', onError, { once: true });
      video.src = src;
      video.load();
      return () => {
        cancelled = true;
        video.removeEventListener('ended', onEnd);
        video.removeEventListener('loadedmetadata', markReady);
        video.removeEventListener('error', onError);
      };
    }

    // hls.js path (dynamic import keeps it off the main bundle).
    void import('hls.js').then(({ default: Hls }) => {
      if (cancelled) return;
      if (!Hls.isSupported()) {
        // Last resort: hand the URL to the element and hope for a plugin.
        video.addEventListener('loadedmetadata', markReady, { once: true });
        video.src = src;
        return;
      }
      const hls = new Hls({ enableWorker: true, lowLatencyMode: false, backBufferLength: 90 });
      hlsRef.current = hls;
      hls.on(Hls.Events.MANIFEST_PARSED, markReady);
      // Bounded self-heal for the transient faults that otherwise leave a blank
      // tile: a fatal NETWORK_ERROR (segment/manifest load hiccup — e.g. the
      // on-demand HEVC→H.264 transcode not cached yet) or MEDIA_ERROR (buffer
      // append/demux glitch). Each fatal fault retries a few times; a healthy
      // fragment buffering resets the budget so a later, unrelated blip still
      // gets its own retries instead of tipping straight into the error state.
      let recover = 0;
      const MAX_RECOVER = 3;
      hls.on(Hls.Events.FRAG_BUFFERED, () => {
        recover = 0;
      });
      hls.on(Hls.Events.ERROR, (_evt, data) => {
        if (!data.fatal) return;
        if (data.type === Hls.ErrorTypes.NETWORK_ERROR) {
          if (recover++ < MAX_RECOVER) hls.startLoad();
          else if (!cancelled) setStatus('error');
        } else if (data.type === Hls.ErrorTypes.MEDIA_ERROR) {
          if (recover++ < MAX_RECOVER) hls.recoverMediaError();
          else if (!cancelled) setStatus('error');
        } else if (!cancelled) {
          setStatus('error');
        }
      });
      hls.loadSource(src);
      hls.attachMedia(video);
    });

    return () => {
      cancelled = true;
      video.removeEventListener('ended', onEnd);
      video.removeEventListener('loadedmetadata', markReady);
      if (hlsRef.current) {
        hlsRef.current.destroy();
        hlsRef.current = null;
      }
    };
  }, [src, videoRef]);

  return (
    <div className="hls-player">
      <video ref={videoRef} controls={controls} playsInline muted preload="auto" />
      {status === 'loading' && (
        <div className="hls-overlay">
          <span className="live-player-spinner" aria-label="loading recording" />
        </div>
      )}
      {status === 'idle' && !src && (
        <div className="hls-overlay hls-overlay-text">No recording at this time</div>
      )}
      {status === 'error' && (
        <div className="hls-overlay hls-overlay-text">Playback error — try another time</div>
      )}
    </div>
  );
}
