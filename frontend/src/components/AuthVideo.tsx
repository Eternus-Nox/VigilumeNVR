/**
 * <video> for Bearer-protected clips: fetched as a blob with the auth header
 * (media elements cannot attach Authorization), then played from an object URL.
 * Event clips are short, so buffering the whole file is acceptable and seeking
 * stays instant.
 */
import { useEffect, useState, type VideoHTMLAttributes } from 'react';
import { fetchBlobUrl } from '../lib/api';

interface AuthVideoProps extends Omit<VideoHTMLAttributes<HTMLVideoElement>, 'src'> {
  src: string;
}

export default function AuthVideo({ src, ...rest }: AuthVideoProps) {
  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let active = true;
    let objectUrl: string | null = null;
    const controller = new AbortController();
    setUrl(null);
    setError(false);
    fetchBlobUrl(src, controller.signal)
      .then((u) => {
        if (!active) {
          URL.revokeObjectURL(u);
          return;
        }
        objectUrl = u;
        setUrl(u);
      })
      .catch(() => {
        if (active) setError(true);
      });
    return () => {
      active = false;
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [src]);

  if (error) {
    return (
      <div className="video-fallback">
        <p>Clip unavailable.</p>
      </div>
    );
  }
  if (!url) {
    return (
      <div className="video-fallback">
        <span className="live-player-spinner" aria-label="loading clip" />
      </div>
    );
  }
  return <video src={url} controls playsInline {...rest} />;
}
