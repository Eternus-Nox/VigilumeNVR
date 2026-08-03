/**
 * Dashboard: live camera grid with status + last-event overlays, a group
 * selector bar ("All cameras" + each group, persisted in localStorage) and
 * a TV-mode button (fullscreen tiles-only wall of the current selection).
 */
import { memo, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import LivePlayer from '../components/LivePlayer';
import TvMode from '../components/TvMode';
import { useAppState, useCameraLive, useLastEvents } from '../state/AppState';
import { readStoredGroup, selectGroupCameras, storeGroup, useGroups } from '../lib/groups';
import { formatRelative, titleCase } from '../lib/format';
import type { Camera } from '../lib/api';

// memo: Dashboard-level re-renders (group switch, cameras refetch) skip tiles
// whose props are unchanged. Live churn (status/last-event WS pushes) enters
// each tile directly via the per-slice live hooks instead of re-rendering the
// whole grid through the Dashboard parent; model_status ticks touch neither.
const CameraTile = memo(function CameraTile({ cam, sub }: { cam: Camera; sub: boolean }) {
  const { isOnline, ingestHealth } = useCameraLive();
  const { lastEvents } = useLastEvents();
  const online = isOnline(cam);
  const last = lastEvents[cam.name];
  // Live detector-ingest health from camera_status WS messages; undefined
  // until the backend reports it (then a tiny, subtle dot appears).
  const ingest = cam.detect.enabled ? ingestHealth[cam.name] : undefined;

  return (
    <Link to={`/cameras/${encodeURIComponent(cam.name)}`} className="camera-tile">
      <div className="camera-tile-video">
        {/* Privacy Mode wins over every other tile state, INCLUDING offline: the
            backend has removed this camera's go2rtc streams, so mounting the
            player would only spin on a stream that no longer exists — and
            showing "offline" would misreport a deliberate choice as a fault. */}
        {cam.private ? (
          <div className="camera-private">
            <span className="camera-private-title">Privacy Mode</span>
            <span className="camera-private-sub">nothing is being captured</span>
          </div>
        ) : online ? (
          <LivePlayer camera={cam.name} sub={sub} />
        ) : (
          <div className="camera-offline">
            <span>offline</span>
          </div>
        )}
      </div>
      <div className="camera-tile-bar">
        <span className={`status-dot ${online ? 'ok' : 'down'}`} aria-hidden="true" />
        <span className="camera-tile-name">{cam.friendly_name || titleCase(cam.name)}</span>
        {ingest !== undefined && (
          <span
            className={`ingest-dot ${ingest ? 'ok' : 'stall'}`}
            title={ingest ? 'Detector ingest: receiving frames' : 'Detector ingest: stalled'}
          />
        )}
        {/* No detect-mode pill here: WHERE detection runs is configuration, not
            live camera state — it lives in Settings → Cameras, where it can
            actually be changed. Keeps the tile about the camera itself. */}
        {last && (
          <span className="camera-tile-event" title={`last event: ${last.label}`}>
            {titleCase(last.label)}
            {last.count > 1 ? ` ×${last.count}` : ''} · {formatRelative(last.start_time)}
          </span>
        )}
      </div>
    </Link>
  );
});

export default function Dashboard() {
  const { cameras, camerasError, refreshCameras, isAdmin } = useAppState();
  const groups = useGroups();
  const [selected, setSelected] = useState<string>(readStoredGroup);
  const [tv, setTv] = useState(false);

  // The persisted selection may reference a since-deleted group.
  useEffect(() => {
    if (groups && selected !== 'all' && !groups.some((g) => String(g.id) === selected)) {
      setSelected('all');
      storeGroup('all');
    }
  }, [groups, selected]);

  const select = (value: string) => {
    setSelected(value);
    storeGroup(value);
  };

  const shown = useMemo(
    () => (cameras ? selectGroupCameras(cameras, groups ?? [], selected) : []),
    [cameras, groups, selected],
  );

  if (cameras === null) {
    return <div className="page-loading">Loading cameras…</div>;
  }

  return (
    <div className="page">
      {camerasError && (
        <div className="banner banner-error">
          <span>{camerasError}</span>
          <button type="button" className="btn btn-sm" onClick={() => void refreshCameras()}>
            Retry
          </button>
        </div>
      )}
      {cameras.length === 0 && !camerasError ? (
        <div className="empty-state">
          <h2>No cameras yet</h2>
          {isAdmin ? (
            <>
              <p className="muted">Add your first camera to start monitoring.</p>
              <Link to="/settings/cameras" className="btn btn-primary">
                Add a camera
              </Link>
            </>
          ) : (
            <p className="muted">No cameras have been configured yet.</p>
          )}
        </div>
      ) : (
        <>
          <div className="dash-bar">
            <div className="chips dash-groups" role="tablist" aria-label="Camera groups">
              <button
                type="button"
                role="tab"
                aria-selected={selected === 'all'}
                className={`chip ${selected === 'all' ? 'chip-on' : ''}`}
                onClick={() => select('all')}
              >
                All cameras
              </button>
              {(groups ?? []).map((g) => (
                <button
                  key={g.id}
                  type="button"
                  role="tab"
                  aria-selected={selected === String(g.id)}
                  className={`chip ${selected === String(g.id) ? 'chip-on' : ''}`}
                  onClick={() => select(String(g.id))}
                >
                  {g.name}
                </button>
              ))}
            </div>
            <button
              type="button"
              className="btn btn-sm"
              title="TV mode — fullscreen, tiles only"
              disabled={shown.length === 0}
              onClick={() => setTv(true)}
            >
              <svg
                viewBox="0 0 24 24"
                width="15"
                height="15"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />
              </svg>
              TV
            </button>
          </div>
          {shown.length === 0 ? (
            // Camera-less with an error: the banner above already explains.
            cameras.length === 0 ? null : (
              <p className="muted">
                This group has no cameras — its members may have been deleted. Edit it in{' '}
                <Link to="/settings/groups" className="link">
                  Settings → Groups
                </Link>
                .
              </p>
            )
          ) : (
            <div className="camera-grid">
              {shown.map((cam) => (
                // >1 tile on screen → each uses the low-bitrate substream; a
                // lone tile gets the full-res main stream.
                <CameraTile key={cam.name} cam={cam} sub={shown.length > 1} />
              ))}
            </div>
          )}
        </>
      )}

      {tv && <TvMode cameras={shown} autoFullscreen onExit={() => setTv(false)} />}
    </div>
  );
}
