/**
 * <img> for Bearer-protected endpoints: fetches with the Authorization header
 * (plain <img src> cannot send it) and renders via an object URL.
 * Fetch is deferred until the element nears the viewport, so long event grids
 * do not stampede the backend.
 */
import { useEffect, useRef, useState, type ImgHTMLAttributes } from 'react';
import { fetchBlobUrl } from '../lib/api';

interface AuthImageProps extends Omit<ImgHTMLAttributes<HTMLImageElement>, 'src'> {
  src: string;
  /** re-fetch interval in ms (e.g. live snapshot refresh); 0 = never */
  refreshMs?: number;
  /** start fetching immediately instead of waiting for viewport proximity */
  eager?: boolean;
  fallback?: React.ReactNode;
}

export default function AuthImage({
  src,
  refreshMs = 0,
  eager = false,
  fallback,
  alt = '',
  ...rest
}: AuthImageProps) {
  const holderRef = useRef<HTMLDivElement | null>(null);
  const [ready, setReady] = useState(eager);
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  // Defer the fetch until the placeholder is near the viewport.
  useEffect(() => {
    if (ready) return;
    const el = holderRef.current;
    if (!el || !('IntersectionObserver' in window)) {
      setReady(true);
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setReady(true);
          io.disconnect();
        }
      },
      { rootMargin: '300px' },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [ready]);

  useEffect(() => {
    if (!ready) return;
    let active = true;
    let current: string | null = null;
    const controller = new AbortController();

    const load = async () => {
      try {
        const next = await fetchBlobUrl(src, controller.signal);
        if (!active) {
          URL.revokeObjectURL(next);
          return;
        }
        if (current) URL.revokeObjectURL(current);
        current = next;
        setUrl(next);
        setFailed(false);
      } catch {
        if (active && !current) setFailed(true);
      }
    };

    void load();
    const timer = refreshMs > 0 ? setInterval(() => void load(), refreshMs) : null;
    return () => {
      active = false;
      controller.abort();
      if (timer) clearInterval(timer);
      if (current) URL.revokeObjectURL(current);
    };
  }, [ready, src, refreshMs]);

  if (failed) return <>{fallback ?? <div className="img-fallback" aria-hidden="true" />}</>;
  if (!url) return <div className="img-skeleton" aria-hidden="true" ref={holderRef} />;
  return <img src={url} alt={alt} {...rest} />;
}
