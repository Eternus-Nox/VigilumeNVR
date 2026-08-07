/**
 * One-time localStorage key migrations from the app's pre-rename name.
 *
 * The product shipped as "Sentinel" before becoming Vigilume, and several
 * localStorage keys carried that name — including the session token. Renaming
 * them outright would have signed every existing user out and dropped their
 * saved dashboard/timeline preferences, so each key is migrated on first read
 * instead: copy the legacy value to the new key, then remove the legacy one.
 *
 * Idempotent and failure-tolerant by design — a private-mode browser that
 * throws on localStorage access must not break the app, so every path is
 * wrapped and simply yields "nothing to migrate".
 */

/**
 * Copy `legacyKey` to `key` if `key` is unset, then delete the legacy entry.
 * Returns the effective value (new, else migrated, else null).
 */
export function migrateKey(key: string, legacyKey: string): string | null {
  try {
    const current = localStorage.getItem(key);
    if (current !== null) {
      // Already migrated (or set fresh). Clear any stale legacy leftover so a
      // later read can never resurrect it.
      localStorage.removeItem(legacyKey);
      return current;
    }
    const legacy = localStorage.getItem(legacyKey);
    if (legacy === null) return null;
    localStorage.setItem(key, legacy);
    localStorage.removeItem(legacyKey);
    return legacy;
  } catch {
    // Private mode / storage disabled — treat as "no stored value".
    return null;
  }
}
