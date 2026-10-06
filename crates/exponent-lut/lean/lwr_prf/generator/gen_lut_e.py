#!/usr/bin/env python3
"""LwrLutE.lean: the readout LUT with its error and keys named.

Usage: gen_lut_e.py <parameter dir>

`LwrLutE` of `average-n32-crt7x40` is the template, transformed like `LwrLut` by `gen_lut`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from info import Info  # noqa: E402
from gen_lut import transform  # noqa: E402
from gen_blocks_e import basis  # noqa: E402


def module(I):
    m1, B = I.m(1), I.B
    extra = [
        ('ErrorMatrix N 1 2', f'ErrorMatrix N 1 {m1}'), ('reduceMatrix Q1 N 1 2', f'reduceMatrix Q1 N 1 {m1}'), ('ErrorMatrix N 2 1', f'ErrorMatrix N {m1} 1'),
        ('ExactMatrix Q1 N 2 1', f'ExactMatrix Q1 N {m1} 1'), ('ExactMatrix Q2 N (2 + 2) 2', f'ExactMatrix Q2 N ({m1} + {m1}) {m1}'),
        ('gadgetDecomposeRuns Backend.backend 1048576 2)', f'gadgetDecomposeRuns Backend.backend 1048576 {m1})'),
        ('stackDigits Q2 2 524288 1048576 2', f'stackDigits Q2 {m1} 524288 1048576 {m1}'),
        ('LoopState Q2 Q1 2', f'LoopState Q2 Q1 {m1}'), ('reduceMatrix Q1 N 2 1', f'reduceMatrix Q1 N {m1} 1'),
        ('import LwrBlockE4', f'import LwrBlockE{B - 1}'), ('encrypt_helper5', 'encrypt_helper'),
        ('[1099511626049, 1099511625409]', basis(I, 2)),
        (r're:\b15\b', str(I.W - 1)),
    ]
    return transform(I, 'LwrLutE',
                     'set_option maxHeartbeats 64000000 in\nset_option maxRecDepth 100000 in\n/-- `lutRound`',
                     extra)


def main():
    d = sys.argv[1]
    I = Info(d)
    path = os.path.join(d, 'LwrLutE.lean')
    text = module(I)
    with open(path, 'w') as f:
        f.write(text)
    print('wrote', path)


if __name__ == '__main__':
    main()
