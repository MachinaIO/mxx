#!/usr/bin/env python3
"""Regenerate the streamed stage proof from its date-row templates."""

from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    (HERE / 'PeKeygenScatterBase.lean').write_text((HERE / 'PeKeygenScatterBase.lean.in').read_text())
    (HERE / 'PeKeygenScatter.lean').write_text((HERE / 'PeKeygenScatter.lean.in').read_text())
    (HERE / 'PeKeygenRoot.lean').write_text((HERE / 'PeKeygenRoot.lean.in').read_text())


if __name__ == '__main__':
    main()
