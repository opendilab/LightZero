"""
Utility module to ensure local LightZero is used across all PriorZero modules.

This ensures PriorZero uses the local LightZero installation at:
/mnt/nfs/zhangjinouwen/puyuan/LightZero

Usage:
    Import this at the beginning of any PriorZero module:

    from ensure_local_lightzero import ensure_local_lightzero
    ensure_local_lightzero()
"""

import sys
from pathlib import Path


def ensure_local_lightzero():
    """
    Ensures the local LightZero path is first in sys.path.

    This allows PriorZero to use a LightZero version that has been
    specifically adapted for PriorZero, rather than a globally installed version.

    Also adds the PriorZero directory to sys.path to ensure PriorZero modules
    can be imported.
    """
    LIGHTZERO_ROOT = Path("/mnt/afs/wanzunian/niuyazhe/xiongjyu/jericho/LightZero").resolve()
    PRIORZERO_DIR = Path(__file__).parent.resolve()

    if not LIGHTZERO_ROOT.exists():
        print(f"⚠️  Warning: LightZero root not found at {LIGHTZERO_ROOT}")
        return False

    lightzero_str = str(LIGHTZERO_ROOT)
    priorzero_str = str(PRIORZERO_DIR)

    # Remove any existing LightZero paths from sys.path
    sys.path = [p for p in sys.path if 'LightZero' not in p or p == lightzero_str]

    # Insert local LightZero at the beginning
    if lightzero_str not in sys.path:
        sys.path.insert(0, lightzero_str)

    # Also ensure PriorZero directory is in sys.path for module imports
    if priorzero_str not in sys.path:
        sys.path.insert(0, priorzero_str)

    # Verify
    try:
        import lzero
        lzero_path = Path(lzero.__file__).parent.parent

        if lzero_path == LIGHTZERO_ROOT:
            print(f"✓ Using local LightZero: {lzero_path}")
            print(f"✓ PriorZero modules path: {priorzero_str}")
            return True
        else:
            print(f"⚠️  Warning: Using LightZero from {lzero_path}")
            print(f"   Expected: {LIGHTZERO_ROOT}")
            return False
    except ImportError as e:
        print(f"⚠️  Warning: Could not import lzero: {e}")
        return False


# Auto-ensure on import
ensure_local_lightzero()
