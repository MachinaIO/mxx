#!/usr/bin/env python3
"""Regenerate the streamed stage proof from its date-row templates."""

from pathlib import Path
from export_format import write_lean

HERE = Path(__file__).resolve().parent


def main():
    write_lean(HERE / 'PeDecodeRoot.lean', (HERE / 'PeDecodeRoot.lean.in').read_text())


if __name__ == '__main__':
    main()
