#!/usr/bin/env bash
# Sentinel NVR — sync the source tree to the run location (Unraid share).
#
# Usage:  ./deploy.sh [dest]     (default dest: /Volumes/appdata/Setinel)
#
# Copies source only. NEVER touches runtime state on the destination:
#   .env, data/, media/, go2rtc/config/   (live server state)
#   secrets/                              (the APNs .p8 — Apple lets you
#                                          download it exactly ONCE, and this
#                                          script runs --delete against a Mac
#                                          that has no secrets/ dir)
#   frigate/, mosquitto/                  (pre-standalone leftovers on the
#                                          share — left alone; remove them
#                                          manually if/when you want)
# Dev clutter (node_modules, dist, __pycache__, Xcode build dirs) is skipped.
#
# After syncing, on the Unraid terminal:
#   cd /mnt/user/appdata/Setinel && docker compose up -d --build
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-/Volumes/appdata/Setinel}"

# Prefer real rsync (brew) — macOS's bundled openrsync is flaky over SMB.
RSYNC="rsync"
for cand in /opt/homebrew/bin/rsync /usr/local/bin/rsync; do
  [ -x "$cand" ] && RSYNC="$cand" && break
done

if [ ! -d "$DEST" ]; then
  echo "error: destination $DEST not found — is the share mounted?" >&2
  exit 1
fi

# --inplace: no dot-prefixed temp files — Unraid SMB vetoes `._*` names
# (macOS-interop AppleDouble filtering), which breaks temp files for any
# `__*.py`. Writing in place avoids the veto entirely.
"$RSYNC" -a --inplace --info=stats1 --delete \
  --exclude '.env' \
  --exclude 'secrets/' \
  --exclude 'data/' \
  --exclude 'media/' \
  --exclude 'go2rtc/' \
  --exclude 'frigate/' \
  --exclude 'mosquitto/' \
  --exclude 'node_modules/' \
  --exclude 'dist/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.venv/' \
  --exclude '.claude/' \
  --exclude '.DS_Store' \
  --exclude 'ios/build/' \
  --exclude 'DerivedData/' \
  --exclude 'deploy.sh.log' \
  "$SRC/" "$DEST/"

echo
echo "synced -> $DEST"
echo "next, on the Unraid terminal:"
echo "  cd /mnt/user/appdata/Setinel && docker compose up -d --build"
