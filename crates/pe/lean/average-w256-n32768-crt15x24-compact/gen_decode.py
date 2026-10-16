#!/usr/bin/env python3
"""Regenerate the streamed stage proof from its date-row templates."""

from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    (HERE / 'PeDecodeRoot.lean').write_text((HERE / 'PeDecodeRoot.lean.in').read_text())


if __name__ == '__main__':
    main()
