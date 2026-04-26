#!/bin/bash
# Setup / repair the Marcus development environment.
#
# Run this:
#   - After cloning the repo for the first time
#   - If `uv run marcus` ever fails with "No module named 'marcus'"
#     (caused by macOS UF_HIDDEN flag making site.py skip .pth files;
#     a known issue with some uv venv states on macOS).
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

# 3. macOS-specific: clear UF_HIDDEN flag from .venv if set.
#    See pyproject.toml [tool.uv] notes — some venv states acquire a
#    UF_HIDDEN flag that makes Python's site.py silently skip .pth files,
#    which breaks the editable install of `marcus`.
if [[ "$(uname -s)" == "Darwin" ]] && [[ -d .venv ]]; then
    if find .venv -maxdepth 5 -flags +hidden 2>/dev/null | grep -q .; then
        echo "==> Clearing UF_HIDDEN flags from .venv (macOS workaround)..."
        chflags -R nohidden .venv
    fi
fi

# 4. Verify the import works
echo "==> Verifying marcus import..."
if uv run python -c "import marcus" 2>/dev/null; then
    echo "    ✓ marcus imports correctly"
else
    echo "    ✗ marcus import still failing — recreating .venv from scratch..."
    rm -rf .venv
    uv sync
    chflags -R nohidden .venv 2>/dev/null || true
    uv run python -c "import marcus" && echo "    ✓ Fixed after clean recreate"
fi

echo
echo "==> Done. Try it:"
echo "    uv run marcus --help"
echo "    uv run marcus chat       # voice chat"
echo "    uv run marcus text       # text-only chat"
