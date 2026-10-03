#!/usr/bin/env python3
"""Generate the Lean proof of a recorded sparse-LWR average-case parameter set.

Usage: gen.py <parameter dir> [<module>...]

`<parameter dir>` is `lean/lwr_prf/<parameter id>`, whose row is in `lean/lwr_prf/parameters.csv` and
whose claim `tests/lwr_prf.rs` exported to `<parameter dir>/generated`. Without modules, the script
writes every proof module and the Lake package files; with modules, only those. The cells, the
block sums, and the readout LUT of `average-n32-crt7x40` are the templates of every parameter set,
whose generated stages have the same structure up to the counts and the moduli.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gen_blocks  # noqa: E402
import gen_cert  # noqa: E402
import gen_encrypt  # noqa: E402
import gen_inputs  # noqa: E402
import gen_lut  # noqa: E402
import gen_package  # noqa: E402
import gen_preprocess  # noqa: E402
import gen_setup  # noqa: E402
from info import Info  # noqa: E402


def generators(I):
    gen = {'LwrSetup': gen_setup.setup, 'LwrInputs': gen_inputs.inputs, 'LwrEncrypt': gen_encrypt.encrypt,
           'LwrPreprocess': gen_preprocess.preprocess, 'LwrCells': gen_blocks.cells_module,
           'LwrLut': gen_lut.lut, 'LwrCertificate': gen_cert.cert}
    for b in range(I.B):
        gen[f'LwrBlock{b}'] = (lambda b: lambda I: gen_blocks.block_module(I, b))(b)
    return gen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory')
    parser.add_argument('modules', nargs='*')
    args = parser.parse_args()
    I = Info(args.directory)
    gen = generators(I)
    files = {f'{m}.lean': gen[m] for m in (args.modules or gen_package.modules(I))}
    if not args.modules:
        files.update({'lakefile.toml': gen_package.lakefile, 'lake-manifest.json': gen_package.manifest,
                      'lean-toolchain': gen_package.toolchain})
    # Every module is generated before any is written: the template parameter set reads its own
    # modules.
    contents = {name: write(I) for name, write in files.items()}
    for name, text in contents.items():
        with open(os.path.join(args.directory, name), 'w') as f:
            f.write(text)
        print('wrote', name)


if __name__ == '__main__':
    main()
