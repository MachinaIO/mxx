#!/usr/bin/env python3
"""LwrErrorCommon.lean and LwrModelLut.lean from the templates of `average-n32-crt7x40`.

Usage: gen_model_lut.py <parameter dir>
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from info import Info  # noqa: E402

TEMPLATES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'average-n32-crt7x40')


def small(name):
    return open(os.path.join(TEMPLATES, f'{name}.lean')).read()


def error_common(I):
    t = small('LwrErrorCommon')
    t = t.replace('gadgetMatrixRuns Backend.backend 1048576 m G', f'gadgetMatrixRuns Backend.backend {I.base} m G')
    t = t.replace('of base `2^20`', f'of base `2^{I.base.bit_length() - 1}`')
    return t


def model_lut(I):
    B, W, m1 = I.B, I.W, I.m(1)
    t = small('LwrModelLut')
    reps = [('import LwrModel4', f'import LwrModel{B - 1}'), ('Fin 16', f'Fin {W}'),
            ('ExactMatrix Q1 N 1 2', f'ExactMatrix Q1 N 1 {m1}'), ('ExactMatrix Q2 N 1 2', f'ExactMatrix Q2 N 1 {m1}'),
            ('ErrorMatrix N 1 2', f'ErrorMatrix N 1 {m1}'), ('ErrorMatrix N 1 1', 'ErrorMatrix N 1 1'),
            ('gadgetOf Q1 2', f'gadgetOf Q1 {m1}'), ('reduceMatrix Q1 N 1 2', f'reduceMatrix Q1 N 1 {m1}'),
            ('key5 ', f'key{B} '), ('err5 ', f'err{B} '), ('muOf ext 5', f'muOf ext {B}'),
            ('sixteen branches', f'{W} branches')]
    for x, y in reps:
        t = t.replace(x, y)
    left = re.findall(r'Fin 16\b|\bkey5\b|\berr5\b|N 1 2\b', t)
    assert (B, W, m1) == (5, 16, 2) or not left, left
    return t


def main():
    d = sys.argv[1]
    I = Info(d)
    for name, f in (('LwrErrorCommon', error_common), ('LwrModelLut', model_lut)):
        path = os.path.join(d, f'{name}.lean')
        with open(path, 'w') as fh:
            fh.write(f(I))
        print('wrote', path)


if __name__ == '__main__':
    main()
