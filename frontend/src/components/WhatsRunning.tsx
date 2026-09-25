/**
 * "Did my change actually deploy?" — in one card.
 *
 * The web app and the backend are separate containers, deployed by rsync plus
 * `docker compose up -d --build`, and they can be rebuilt independently. When
 * they disagree the symptom is never "the versions disagree" — it is "the
 * thing I changed didn't work", which sends you looking at the code instead of
 * at the deploy. This card is the place that says so plainly.
 *
 * Three facts, each answering a failure this project has actually hit:
 *
 *   backend version   — the backend image was not rebuilt
 *   schema version    — the database did not take its migration, which
 *                       otherwise surfaces much later as unrelated 500s
 *   this page's build — nginx is serving an old bundle, or the browser is
 *                       holding one (see lib/bundle.ts)
 *
 * The schema row is the one worth reading twice. `schema_version` is read back
 * from the database file; `expects_schema` is the constant this build ships.
 * Equal is silence. Different is a migration that did not run, and it is
 * called out loudly because nothing else in the UI would ever mention it.
 */
import { useEffect, useState } from 'react';
import { api, type HealthStatus } from '../lib/api';
import { checkBundle, runningBundleName, type BundleState } from '../lib/bundle';

function since(startedAt: number): string {
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - startedAt));
  if (secs < 90) return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 90) return `${mins} min ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 48) return `${hours}h ago`;
  return `${Math.floor(hours / 24)} days ago`;
}

export default function WhatsRunning() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [failed, setFailed] = useState(false);
  const [bundle, setBundle] = useState<BundleState>('unknown');

  useEffect(() => {
    const controller = new AbortController();
    let alive = true;
    void api
      .health()
      .then((h) => {
        if (alive) setHealth(h);
      })
      .catch(() => {
        if (alive) setFailed(true);
      });
    void checkBundle(controller.signal).then((b) => {
      if (alive) setBundle(b);
    });
    return () => {
      alive = false;
      controller.abort();
    };
  }, []);

  if (failed) {
    return (
      <p className="empty-state">
        The backend did not answer. That is itself the answer to “is it running?”.
      </p>
    );
  }
  if (!health) return <p className="empty-state">Checking…</p>;

  // Only a definite disagreement is worth shouting about. A backend that
  // predates these fields sends neither, and must read as "nothing to say"
  // rather than as a fault.
  const dbVersion = health.schema_version ?? null;
  const wants = health.expects_schema ?? null;
  const schemaMismatch = dbVersion !== null && wants !== null && dbVersion !== wants;

  return (
    <>
      <dl className="whats-running">
        <div>
          <dt>Backend</dt>
          <dd>
            <span className="mono">{health.version}</span>
            {health.started_at ? (
              <span className="muted small"> · started {since(health.started_at)}</span>
            ) : null}
          </dd>
        </div>

        <div>
          <dt>Database</dt>
          <dd>
            {dbVersion === null ? (
              <span className="muted small">not reported by this backend</span>
            ) : (
              <>
                <span className="mono">schema v{dbVersion}</span>
                {schemaMismatch ? (
                  <span className="whats-running-bad">
                    {' '}
                    — this build expects v{wants}
                  </span>
                ) : (
                  <span className="muted small"> · up to date</span>
                )}
              </>
            )}
          </dd>
        </div>

        <div>
          <dt>This page</dt>
          <dd>
            <span className="mono">{runningBundleName()}</span>{' '}
            {bundle === 'stale' ? (
              <span className="whats-running-bad">— the server has a newer build</span>
            ) : bundle === 'current' ? (
              <span className="muted small">· current</span>
            ) : (
              <span className="muted small">· build not identifiable</span>
            )}
          </dd>
        </div>
      </dl>

      {schemaMismatch && (
        <p className="control-hint whats-running-bad">
          <strong>The database is on a different schema than this backend expects.</strong>{' '}
          A migration did not run. Recognition, camera settings and event history can all
          fail in unrelated-looking ways until it does — restarting the backend re-runs
          migrations, and the backend log says which step failed.
        </p>
      )}

      {bundle === 'stale' && (
        <p className="control-hint">
          Reload the page to pick up the newer build. If it keeps saying this after a hard
          reload, the web container is serving an older bundle than you think and needs
          rebuilding.
        </p>
      )}
    </>
  );
}
