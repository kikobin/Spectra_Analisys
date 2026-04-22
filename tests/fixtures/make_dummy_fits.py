from __future__ import annotations

from pathlib import Path

from demo_data import make_dummy_fits


def main() -> None:
    out = make_dummy_fits(Path("dummy.fits"))
    print(f"Created {out}")


if __name__ == "__main__":
    main()
