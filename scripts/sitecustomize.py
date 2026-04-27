"""Self-heal Marcus's editable install against macOS UF_HIDDEN issue.

Python's site.py silently skips .pth files marked UF_HIDDEN (a flag
macOS background services like Time Machine apply to .venv to mark it
as a backup-skip cache). When the editable install's .pth file is
hidden, `marcus` becomes unimportable.

sitecustomize.py is loaded automatically by Python on startup via a
regular import (NOT through addpackage()), so it bypasses the
hidden-file check. We use it to add the project's src/ directory to
sys.path unconditionally, making the marcus package importable
regardless of .pth file state.

Installation: copy this file to
    .venv/lib/python3.11/site-packages/sitecustomize.py
This is automated by scripts/setup_env.sh.
"""

from __future__ import annotations

import os
import sys

# Resolve the project root from this file's location:
# .venv/lib/python3.11/site-packages/sitecustomize.py → ../../../../
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")

if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
