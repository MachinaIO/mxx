#!/usr/bin/env python3
"""Regenerate the date proof from its export and the read-only W128 reference."""

import os
import runpy
from pathlib import Path

import date_parameters
from export_format import translate


def main():
    os.chdir(Path(__file__).resolve().parent)
    date_parameters.generate_math()
    date_parameters.generate_requirement()
    date_parameters.generate_numeric()
    date_parameters.generate_certificate()
    for generator in ['gen_steps.py', 'gen_pre.py', 'gen_pre_root.py',
                      'gen_online_root.py', 'gen_keygen.py', 'gen_decode.py', 'gen_chain.py']:
        runpy.run_path(generator, run_name='__main__')
    for proof in Path('.').glob('Pe*.lean'):
        source = proof.read_text()
        rendered = translate(source)
        if rendered != source:
            proof.write_text(rendered)


if __name__ == '__main__':
    main()
