from __future__ import annotations

import sys


def main():
    src = "/Users/abhijita/Pinak_Package/src"
    if src not in sys.path:
        sys.path.insert(0, src)
    from pinak.menubar.app import main as run

    return run()


if __name__ == "__main__":
    raise SystemExit(main())
