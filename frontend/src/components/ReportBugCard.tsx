/**
 * "Report a bug": opens a PRE-FILLED GitHub issue on the public tracker.
 *
 * WHY NO BACKEND. Vigilume is self-hosted — a report posted to *your own* NVR
 * reaches nobody. There is deliberately no endpoint, no queue and no phone-home
 * here: the link builds a `github.com/.../issues/new` URL from data the browser
 * already has, and the user's own GitHub session does the posting. That also
 * means nothing leaves the box unless the reporter presses Submit, and they see
 * (and can edit) every diagnostic first.
 *
 * Viewer-reachable by design: the admin Settings → System tab renders this, and
 * so does the shell for non-admins (see pages/Settings.tsx). Anyone can hit a
 * bug, so gating it behind admin would be silly. `/api/system/health` is
 * unauthenticated, so the self-fetch below works for every role.
 */
import { useEffect, useState } from 'react';
import { api, type HealthStatus } from '../lib/api';
import { describeDetector } from '../lib/detector';

/**
 * The public repository issues are filed against. ONE literal, in ONE place:
 * a fork search/replaces `YOUR-USERNAME` here and nowhere else, so never inline
 * this URL at a call site.
 */
const REPO_URL = 'https://github.com/YOUR-USERNAME/vigilume';

/** Prefix so the tracker can tell a web report from an iOS one at a glance. */
const ISSUE_TITLE = '[Web] ';

/**
 * Build the issue URL.
 *
 * Every diagnostic line is optional: anything we cannot actually read is
 * OMITTED rather than printed as "undefined", which would read like a real
 * answer to whoever triages it. The web bundle carries no version constant
 * (nothing is injected at build time), so there is no "App:" line here — the
 * `[Web]` title prefix and the user agent carry that instead.
 */
function issueUrl(health: HealthStatus | null): string {
  const lines = [
    '### What happened?',
    '<!-- describe the bug -->',
    '',
    '### Steps to reproduce',
    '1.',
    '2.',
    '',
    '### Environment (auto-filled, please keep)',
    `- Device: ${navigator.userAgent}`,
    `- Server version: ${health?.version ?? 'unknown'}`,
  ];
  if (health?.detector) {
    const d = health.detector;
    lines.push(
      `- Detector: ${d.ready ? 'ready' : 'not ready'} · ` +
        `${describeDetector(d.device, d.kind)} · ${d.model}`,
    );
  }
  // encodeURIComponent (not encodeURI) — it escapes `&` and `+`, which would
  // otherwise truncate the body at the first ampersand and turn every plus
  // into a space.
  return (
    `${REPO_URL}/issues/new` +
    `?title=${encodeURIComponent(ISSUE_TITLE)}` +
    `&body=${encodeURIComponent(lines.join('\n'))}`
  );
}

interface Props {
  /**
   * Health the host page already has in hand, so we don't duplicate its poll.
   * Pass `null` while it is loading/failed; leave it OFF entirely and this card
   * fetches health once for itself.
   */
  health?: HealthStatus | null;
}

export default function ReportBugCard({ health }: Props) {
  const [fetched, setFetched] = useState<HealthStatus | null>(null);
  const selfFetch = health === undefined;

  useEffect(() => {
    if (!selfFetch) return;
    let active = true;
    api
      .health()
      .then((h) => {
        if (active) setFetched(h);
      })
      // A missing server version is not worth an error state — the report just
      // says "unknown" and is still perfectly filable.
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, [selfFetch]);

  const effective = selfFetch ? fetched : (health ?? null);

  return (
    <section className="card">
      <h2>Report a bug</h2>
      <p className="muted small">
        Opens a new GitHub issue with the report template and this device&rsquo;s
        diagnostics already filled in — your browser, and this server&rsquo;s
        version and detector. Nothing is sent anywhere until you press Submit on
        GitHub, and you can edit or delete anything you&rsquo;d rather not share
        before you do.
      </p>
      <div className="row-inline wrap">
        <a
          className="btn"
          href={issueUrl(effective)}
          target="_blank"
          rel="noopener noreferrer"
        >
          Report a bug on GitHub
        </a>
      </div>
      <span className="control-hint">
        Opens github.com in a new tab. A GitHub account is needed to post.
      </span>
    </section>
  );
}
