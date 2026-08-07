/**
 * Camera-group helpers shared by the Dashboard selector bar and TV mode
 * (normal + /tv deep link). The dashboard selection ("all" or a group id as
 * a string) persists in localStorage under a namespaced key.
 */
import { useEffect, useState } from 'react';
import { api, type Camera, type CameraGroup } from './api';

import { migrateKey } from './legacyStorage';

export const GROUP_STORAGE_KEY = 'vigilume.dashboard.group';
/** Pre-rename key (the app shipped as "Sentinel"), migrated on first read. */
const LEGACY_GROUP_STORAGE_KEY = 'sentinel.dashboard.group';

export function readStoredGroup(): string {
  try {
    return migrateKey(GROUP_STORAGE_KEY, LEGACY_GROUP_STORAGE_KEY) ?? 'all';
  } catch {
    return 'all';
  }
}

export function storeGroup(selection: string): void {
  try {
    localStorage.setItem(GROUP_STORAGE_KEY, selection);
  } catch {
    /* private mode — selection just won't persist */
  }
}

/**
 * Fetch the group list once on mount. `null` while loading, `[]` on failure
 * (the selector then simply shows only "All cameras").
 */
export function useGroups(): CameraGroup[] | null {
  const [groups, setGroups] = useState<CameraGroup[] | null>(null);
  useEffect(() => {
    let active = true;
    api
      .groups()
      .then((g) => {
        if (active) setGroups(g);
      })
      .catch(() => {
        if (active) setGroups([]);
      });
    return () => {
      active = false;
    };
  }, []);
  return groups;
}

/**
 * Resolve a selection ("all" | group id string) to an ordered camera list.
 * Group member names that no longer exist are filtered out; an unknown or
 * deleted group id falls back to all cameras.
 */
export function selectGroupCameras(
  cameras: Camera[],
  groups: CameraGroup[],
  selection: string,
): Camera[] {
  if (selection !== 'all') {
    const group = groups.find((g) => String(g.id) === selection);
    if (group) {
      const byName = new Map(cameras.map((c) => [c.name, c]));
      return group.cameras
        .map((n) => byName.get(n))
        .filter((c): c is Camera => c !== undefined);
    }
  }
  return cameras;
}
