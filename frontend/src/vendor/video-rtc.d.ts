/**
 * Type surface for the vendored go2rtc video-rtc.js web component
 * (https://github.com/AlexxIT/go2rtc, www/video-rtc.js, MIT).
 * Only the members the app touches are declared.
 */
export class VideoRTC extends HTMLElement {
  /** Supported modes, comma separated: webrtc, webrtc/tcp, mse, hls, mp4, mjpeg */
  mode: string;
  /** Requested media kinds, e.g. "video,audio" */
  media: string;
  /** Keep the connection open while the element is hidden / detached */
  background: boolean;
  /** Auto-disconnect when the tab or element is not visible */
  visibilityCheck: boolean;
  visibilityThreshold: number;
  /** ms before a detached element drops its connection (default 5000) */
  DISCONNECT_TIMEOUT: number;
  RECONNECT_TIMEOUT: number;
  /** The inner <video> element (created on first DOM connection) */
  video: HTMLVideoElement;
  wsState: number;
  pcState: number;
  /** Negotiated MSE codec string; '' until the server answers the mse offer. */
  mseCodecs: string;
  /** Assigning connects: accepts ws(s)://, http(s):// or an absolute path */
  set src(value: string | URL);
  play(): void;
  ondisconnect(): void;
}
