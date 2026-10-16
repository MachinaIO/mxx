#!/usr/bin/env python3
"""Regenerate the date proof from its export and the read-only W128 reference."""

import os
import runpy
from pathlib import Path

import date_parameters
from prepare_export import normalize_keygen


def main():
    os.chdir(Path(__file__).resolve().parent)
    keygen = Path('generated/Stage_keygen.lean')
    normalized = normalize_keygen(keygen.read_text())
    if normalized != keygen.read_text():
        keygen.write_text(normalized)
    date_parameters.generate_math()
    date_parameters.generate_requirement()
    date_parameters.generate_numeric()
    date_parameters.generate_certificate()
    for generator in ['gen_steps.py', 'gen_pre.py', 'gen_pre_root.py',
                      'gen_online_root.py', 'gen_keygen.py', 'gen_decode.py', 'gen_chain.py']:
        runpy.run_path(generator, run_name='__main__')


if __name__ == '__main__':
    main()
