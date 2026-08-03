import { type ReactNode } from 'react';

/**
 * A settings card that collapses, with its STATE always visible in the summary.
 *
 * Why this exists: the Integrations tab stacked four full-height cards — this
 * browser's push, notification rules, phone push (itself a two-channel tabbed
 * panel), and Home Assistant/MQTT. Finding the one setting you came for meant
 * scrolling past three you didn't. Collapsing them turns that into a single
 * screen you can scan.
 *
 * THE RULE THAT MAKES COLLAPSING SAFE: a closed section must still tell you
 * whether it is on. Hiding "phone push is off" behind a triangle on a security
 * system is how someone discovers their alerts were disabled a week too late.
 * So `badge` is not decoration — it is the reason this is allowed to be closed,
 * and every caller passes one.
 *
 * Matches the Privacy Mode card's disclosure language (same triangle, same
 * badge shape) so the two read as one system.
 */
export default function SettingsDisclosure({
  title,
  badge,
  tone = 'muted',
  open = false,
  children,
}: {
  title: string;
  /** Always-visible state, e.g. "Relay" / "Off" / "3 labels". Required. */
  badge: string;
  /** `on` = active/enabled (accent). `muted` = inactive. `warn` = needs attention. */
  tone?: 'on' | 'muted' | 'warn';
  /** Start expanded — for the section that is the point of the tab. */
  open?: boolean;
  children: ReactNode;
}) {
  return (
    <section className="card settings-disclosure-card">
      <details className="settings-disclosure" open={open}>
        <summary>
          <span className="settings-disclosure-title">{title}</span>
          <span className={`settings-disclosure-badge settings-disclosure-badge-${tone}`}>
            {badge}
          </span>
        </summary>
        <div className="settings-disclosure-body">{children}</div>
      </details>
    </section>
  );
}
