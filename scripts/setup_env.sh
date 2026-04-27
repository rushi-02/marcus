#!/bin/bash
# Setup / repair the Marcus development environment.
#
# Run this:
#   - After cloning the repo for the first time
#   - After deleting .venv to start fresh
#   - If `uv run marcus` ever fails with "No module named 'marcus'"
#     (caused by macOS UF_HIDDEN flag making site.py skip .pth files;
#     see scripts/sitecustomize.py for details).
#
# Usage:
#   bash scripts/setup_env.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "==> Marcus environment setup"
echo "    Project dir: $PROJECT_DIR"

# 1. Ensure uv is installed
if ! command -v uv >/dev/null 2>&1; then
    echo "==> Installing uv..."
    if command -v brew >/dev/null 2>&1; then
        brew install uv
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
fi

# 2. Sync dependencies (creates .venv if missing)
echo "==> Syncing dependencies..."
uv sync

# 3. Install sitecustomize.py to make `import marcus` robust against
#    macOS UF_HIDDEN flag on .pth files. This is the permanent fix
#    that survives `uv sync` and Time Machine background activity.
SITECUSTOMIZE_SRC="$PROJECT_DIR/scripts/sitecustomize.py"
SITECUSTOMIZE_DST=$(uv run python -c \
    "import site; print(site.getsitepackages()[0])")"/sitecustomize.py"
echo "==> Installing sitecustomize.py → $SITECUSTOMIZE_DST"
cp "$SITECUSTOMIZE_SRC" "$SITECUSTOMIZE_DST"

# 4. Clear any UF_HIDDEN flags currently set on .venv (defensive belt-and-
#    suspenders, even though sitecustomize.py makes it unnecessary).
if [[ "$(uname -s)" == "Darwin" ]] && [[ -d .venv ]]; then
    chflags -R nohidden .venv 2>/dev/null || true
fi

# 5. Verify the import works
echo "==> Verifying marcus import..."
if uv run python -c "import marcus" 2>/dev/null; then
    echo "    ✓ marcus imports correctly"
else
    echo "    ✗ marcus import still failing — see scripts/sitecustomize.py"
    exit 1
fi

echo
echo "==> Done. Try it:"
echo "    uv run marcus chat       # voice chat"
echo "    uv run marcus text       # text-only chat"
echo "    ./marcus chat            # wrapper (also clears UF_HIDDEN)"
