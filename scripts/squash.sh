#!/usr/bin/env bash
set -euo pipefail

# Move checkpoint binaries out of git into /checkpoints/ (gitignored),
# replace with relative symlinks, then squash all history into a single
# commit and force push.
#
# Usage:
#   ./scripts/squash.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "Moving checkpoint binaries to /checkpoints/..."
for d in experiments/*/artifacts_*/; do
    [[ -f "$d/checkpoint.tar.gz" ]] || continue
    [[ -L "$d/checkpoint.tar.gz" ]] && continue  # already a symlink
    EXPERIMENT=$(echo "$d" | cut -d/ -f2)
    ARTIFACT=$(basename "$d")
    DEST="checkpoints/$EXPERIMENT/$ARTIFACT"
    mkdir -p "$DEST"
    mv "$d/checkpoint.tar.gz" "$DEST/checkpoint.tar.gz"
    ln -s "../../../checkpoints/$EXPERIMENT/$ARTIFACT/checkpoint.tar.gz" "$d/checkpoint.tar.gz"
    echo "  $d -> $DEST/"
done

# Ensure /checkpoints/ is gitignored
if ! grep -q '/checkpoints/' .gitignore 2>/dev/null; then
    echo '/checkpoints/' >> .gitignore
fi

# Stage all tracked files, new symlinks, and gitignore
git add -u
git add .gitignore
git add experiments/*/artifacts_*/checkpoint.tar.gz 2>/dev/null || true

# Squash
echo "Squashing history..."
git checkout --orphan squashed
git commit -m "Golf research: experiment framework, quantization pipeline, baseline results"
git branch -M main
git push --force origin main

echo "Done. Repository squashed to single commit."
