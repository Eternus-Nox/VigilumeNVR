/**
 * Web Push enable/disable control for the current device — the one
 * notification capability a viewer has. Reused by Settings → Notifications
 * (admins, who also get `allowTest` for the admin-only test send) and by the
 * viewer's minimal notifications view. Uses only no-auth / any-authenticated
 * endpoints (vapid key, subscribe, unsubscribe), so it is safe for viewers;
 * the "Send test" action (admin-only endpoint) is gated behind `allowTest`.
 */
import { useEffect, useState } from 'react';
import { api } from '../lib/api';
import {
  getPushState,
  subscribeThisDevice,
  unsubscribeThisDevice,
  type PushState,
} from '../lib/push';
import { useAppState } from '../state/AppState';

export default function DevicePushCard({
  allowTest = false,
  inline = false,
}: {
  allowTest?: boolean;
  /** Render bare, for embedding inside a section that already supplies the
      card chrome and heading (the Integrations tab's disclosure). */
  inline?: boolean;
}) {
  const { pushToast } = useAppState();
  const [pushState, setPushState] = useState<PushState>('unsupported');
  const [pushBusy, setPushBusy] = useState(false);
  const [testBusy, setTestBusy] = useState(false);

  useEffect(() => {
    void getPushState().then(setPushState);
  }, []);

  const enablePush = async () => {
    setPushBusy(true);
    try {
      await subscribeThisDevice();
      setPushState('subscribed');
      pushToast({ kind: 'info', title: 'Push enabled on this device', body: '' });
    } catch (e) {
      setPushState(await getPushState());
      pushToast({
        kind: 'error',
        title: 'Push subscription failed',
        body: e instanceof Error ? e.message : '',
      });
    } finally {
      setPushBusy(false);
    }
  };

  const disablePush = async () => {
    setPushBusy(true);
    try {
      await unsubscribeThisDevice();
      setPushState('unsubscribed');
      pushToast({ kind: 'info', title: 'Push disabled on this device', body: '' });
    } finally {
      setPushBusy(false);
    }
  };

  const sendTest = async () => {
    setTestBusy(true);
    try {
      const result = await api.testNotification();
      pushToast({
        kind: 'info',
        title: 'Test sent',
        body: `Delivered to ${result.push_sent} device${result.push_sent === 1 ? '' : 's'}`,
      });
    } catch (e) {
      pushToast({
        kind: 'error',
        title: 'Test failed',
        body: e instanceof Error ? e.message : '',
      });
    } finally {
      setTestBusy(false);
    }
  };

  const body = (
    <>
      {pushState === 'insecure' && (
        <p className="muted">
          Web push needs HTTPS. Open the console via its secure URL (e.g. the Caddy
          <code> :8443</code> endpoint) to enable notifications here.
        </p>
      )}
      {pushState === 'unsupported' && (
        <p className="muted">This browser does not support Web Push.</p>
      )}
      {pushState === 'denied' && (
        <p className="muted">
          Notifications are blocked for this site. Allow them in the browser’s site settings,
          then return here.
        </p>
      )}
      {pushState === 'unsubscribed' && (
        <button
          type="button"
          className="btn btn-primary"
          disabled={pushBusy}
          onClick={() => void enablePush()}
        >
          {pushBusy ? 'Enabling…' : 'Enable push on this device'}
        </button>
      )}
      {pushState === 'subscribed' && (
        <div className="row-inline">
          <span className="pill pill-ok">Push active on this device</span>
          <button
            type="button"
            className="btn btn-sm"
            disabled={pushBusy}
            onClick={() => void disablePush()}
          >
            Disable
          </button>
          {allowTest && (
            <button
              type="button"
              className="btn btn-sm"
              disabled={testBusy}
              onClick={() => void sendTest()}
            >
              {testBusy ? 'Sending…' : 'Send test'}
            </button>
          )}
        </div>
      )}
    </>
  );

  if (inline) return body;
  return (
    <section className="card">
      <h2>This device</h2>
      {body}
    </section>
  );
}
