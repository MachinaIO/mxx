#!/usr/bin/env python3
"""Regenerate the streamed stage proof from its date-row templates."""

from pathlib import Path
import re
from export_format import write_lean

HERE = Path(__file__).resolve().parent


def main():
    source = (HERE / 'generated/Stage_keygen.lean').read_text()
    scatter = '[' + max(re.findall(r'\[([0-9(), -]+)\]', source), key=len) + ']'
    if len(re.findall(r'\d+', scatter)) != 11520:
        raise ValueError('Unexpected helper-scatter table size')
    for module in ['PeKeygenScatterBase', 'PeKeygenScatter', 'PeKeygenRoot']:
        text = (HERE / f'{module}.lean.in').read_text().replace('⟦SCATTER_LIST⟧', scatter)
        write_lean(HERE / f'{module}.lean', text)


if __name__ == '__main__':
    main()
