#!/usr/bin/env python3
"""Generate the certificate under H2 of a parameter set.

Usage: gen_h2.py <parameter dir> [<stage>...]

Writes, after the base proof of `gen.py`, the modules of the `LwrH2` library in dependency order:
the errors named, the model of a run, the chain of the run to its model, and the decomposition of
the decoded coefficient with its bounds and the certificate. Each stage reads the modules written
before it. With stages, only those run.
"""
import os
import subprocess
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from info import Info  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def balanced_digits(value, base, count):
    out = []
    for _ in range(count):
        r = value % base
        if r > base // 2 or (r == base // 2 and (value // base) % 2 == 1):
            r -= base
        out.append(r)
        value = (value - r) // base
    assert value == 0
    return out


def final_l1(I):
    """The column `l1` bound of the finalization digits used by the bounds: the exact digits of
    the scale `k` for a set under H2, the size bound `rows · β` otherwise."""
    if not I.h2:
        return Fraction(I.m(1) * I.digit)
    import re
    k = int(re.search(r'abbrev outScale : Int := (\d+)', open(os.path.join(I.d, 'LwrLut.lean')).read()).group(1))
    return Fraction(sum(abs(x) for x in balanced_digits(k, I.base, I.m(1))))


def run(script, *args):
    subprocess.run([sys.executable, os.path.join(HERE, script), *args], check=True)


STAGES = ['encrypt_e', 'preprocess_e', 'common', 'blocks_e', 'lut_e', 'model_base', 'model', 'chain',
          'chain_final', 'h2_model', 'h2_bounds', 'h2_agree', 'h2_cert']


def main():
    d = sys.argv[1]
    stages = sys.argv[2:] or STAGES
    I = Info(d)
    for st in stages:
        if st == 'encrypt_e':
            run('gen_encrypt_e.py', d)
        elif st == 'preprocess_e':
            run('gen_preprocess_e.py', d)
        elif st == 'common':
            run('gen_model_lut.py', d)
        elif st == 'blocks_e':
            run('gen_blocks_e.py', d)
        elif st == 'lut_e':
            run('gen_lut_e.py', d)
        elif st == 'model_base':
            run('gen_model_base.py', d)
        elif st == 'model':
            run('gen_model.py', d)
        elif st == 'chain':
            run('gen_chain.py', d)
        elif st == 'chain_final':
            run('gen_chain_final.py', d)
        elif st == 'h2_model':
            run('gen_h2_model.py', d)
        elif st == 'h2_bounds':
            run('gen_h2_bounds.py', d, str(final_l1(I)))
        elif st == 'h2_agree':
            run('gen_h2_cert.py', d)
        elif st == 'h2_cert':
            run('gen_h2_cert.py', 'cert', d)
        else:
            raise SystemExit(f'unknown stage {st}')


if __name__ == '__main__':
    main()
