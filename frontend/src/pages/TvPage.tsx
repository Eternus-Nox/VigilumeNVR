/**
 * /tv?group=<id|all> — deep-linkable TV wall for a TV browser / kiosk
 * bookmark. Renders the TV layout directly (no app chrome); true fullscreen
 * still needs one tap due to browser gesture rules, so TvMode shows a
 * subtle hint until then. Without a ?group param the dashboard's persisted
 * selection is used; an unknown group id falls back to all cameras.
 */
import { useMemo } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import TvMode from '../components/TvMode';
import { useAppState } from '../state/AppState';
import { readStoredGroup, selectGroupCameras, useGroups } from '../lib/groups';

export default function TvPage() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const { cameras } = useAppState();
  const groups = useGroups();

  const selection = params.get('group') ?? readStoredGroup();

  const shown = useMemo(
    () => (cameras && groups ? selectGroupCameras(cameras, groups, selection) : null),
    [cameras, groups, selection],
  );

  if (!shown) {
    return (
      <div className="tv-root">
        <div className="page-loading">Loading cameras…</div>
      </div>
    );
  }

  return <TvMode cameras={shown} standalone onExit={() => navigate('/')} />;
}
