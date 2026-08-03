/** Toast stack for live WS events (event_new / doorbell) and errors. */
import { useNavigate } from 'react-router-dom';
import { useAppState, useUiLive } from '../state/AppState';

export default function Toasts() {
  const { toasts } = useUiLive();
  const { dismissToast } = useAppState();
  const navigate = useNavigate();

  if (toasts.length === 0) return null;
  return (
    <div className="toast-stack" role="status" aria-live="polite">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`toast toast-${t.kind}`}
          onClick={() => {
            dismissToast(t.id);
            if (t.url) navigate(t.url);
          }}
        >
          <div className="toast-icon" aria-hidden="true">
            {t.kind === 'doorbell' ? '🔔' : t.kind === 'error' ? '⚠' : '◉'}
          </div>
          <div className="toast-text">
            <strong>{t.title}</strong>
            {t.body && <span>{t.body}</span>}
          </div>
          <button
            type="button"
            className="icon-btn"
            aria-label="Dismiss"
            onClick={(e) => {
              e.stopPropagation();
              dismissToast(t.id);
            }}
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}
