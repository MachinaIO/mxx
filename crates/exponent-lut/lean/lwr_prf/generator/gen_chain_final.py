#!/usr/bin/env python3
"""LwrChainFinal.lean: the finalized readout of a run is its model.

Usage: gen_chain_final.py <parameter dir>

`LwrChainFinal` of `average-n32-crt7x40` is the template; the run's statement head is the one of
the set's `chain` (`LwrChainAll`), and the names of the readout round are renumbered as in `gen_lut`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from info import Info, wrap  # noqa: E402

TEMPLATES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'average-n32-crt7x40')


def module(I):
    B, W, s, m1 = I.B, I.W, I.stride, I.m(1)
    t = open(os.path.join(TEMPLATES, 'LwrChainFinal.lean')).read()
    chain = open(os.path.join(I.d, 'LwrChainAll.lean')).read()
    a = chain.index('theorem chain ')
    head = chain[chain.index('    ∃ w, Stage_online_eval', a):chain.index('execution.«stage_4» w ∧', a)]
    ta = t.index('    ∃ w, Stage_online_eval')
    tb = t.index('execution.«stage_4» w ∧', ta)
    t = t[:ta] + head + t[tb:]
    shift = 22 * (B - 5)
    t = re.sub(r'\bw\.w_(\d+)_', lambda m: f'w.w_{int(m.group(1)) + shift}_' if int(m.group(1)) >= 130 else m.group(0), t)
    divs = wrap([f'div{b} model external.input_0 external tape' for b in range(B)] +
                ['divLut model external.input_0 external tape'], ' ∧ ', 6, 6)
    old = t[t.index('      div0 model external.input_0 external tape ∧'):t.index('      (∃ d : ErrorMatrix N 2 1')]
    t = t.replace(old, '      ' + divs + ' ∧\n')
    hd = ', '.join(f'hdiv{b}' for b in range(B))
    t = t.replace('⟨w, hb, hdiv0, hdiv1, hdiv2, hdiv3, hdiv4, hk, hrel⟩', f'⟨w, hb, {hd}, hk, hrel⟩')
    t = t.replace('⟨w, hb, hdiv0, hdiv1, hdiv2, hdiv3, hdiv4, hdivLut, hfd, ?_⟩', f'⟨w, hb, {hd}, hdivLut, hfd, ?_⟩')
    # The selector lists name every block; `hsN` keeps them from the renaming of block `4` below.
    sels = ', '.join(f'hsN{b}, -' for b in range(B))
    t = t.replace('⟨-, -, -, ha, -, -, hs0, -, hs1, -, hs2, -, hs3, -, hs4, -⟩', f'⟨-, -, -, ha, -, -, {sels}⟩')
    last = B - 1
    t = t.replace('⟨-, -, -, -, -, hsum4⟩', '⟨-, ' + '-, ' * (B - 1) + f'hsum{last}⟩')
    t = t.replace('exacts [(hs0 i).1, (hs1 i).1, (hs2 i).1, (hs3 i).1, (hs4 i).1]',
                  'exacts [' + wrap([f'(hsN{b} i).1' for b in range(B)], ', ', 6, 12) + ']')
    reps = [(r'\bsel4\b', f'sel{last}'), (r'\bhsel4\b', f'hsel{last}'), (r'\bhs4\b', f'hs{last}'),
            (r'\bhsum4\b', f'hsum{last}'), (r'\bhstep4\b', f'hstep{last}'),
            (r'muOf external 5\b', f'muOf external {B}'), (r'muOf external 4\b', f'muOf external {last}'),
            (r'muOf_step external 4\b', f'muOf_step external {last}'), (r'nuOf external 4\b', f'nuOf external {last}'),
            (r'\bcoords 4\b', f'coords {last}'), (r'\bcoords_lt 4\b', f'coords_lt {last}'),
            (r'zOf external 5\b', f'zOf external {B}'), (r'block-4 target', f'block-{last} target'),
            (r'\bhg2u\b', f'hg{m1}u'), (r'\bhg2\b', f'hg{m1}'), (r'\bg2\b', f'g{m1}'), (r'\(m := 2\)', f'(m := {m1})'),
            (r'ExactMatrix Q2 N 1 2\b', f'ExactMatrix Q2 N 1 {m1}'), (r'ErrorMatrix N 2 1\b', f'ErrorMatrix N {m1} 1'),
            (r'CoeffBound d 524288', f'CoeffBound d {I.digit}'), (r'Fin 16\b', f'Fin {W}'),
            (r'ExactMatrix Q1 N 1 2\b', f'ExactMatrix Q1 N 1 {m1}'), (r'reduceMatrix Q1 N 2 1\b', f'reduceMatrix Q1 N {m1} 1'),
            (r'gadgetMatrixRuns Backend\.backend 1048576 2 G', f'gadgetMatrixRuns Backend.backend {I.base} {m1} G'),
            (r'Xe \(n := N\) \(2 \* ', f'Xe (n := N) ({s} * '), (r'encrypt_helper5E', 'encrypt_helperE')]
    for x, y in reps:
        t = re.sub(x, y, t)
    t = re.sub(r'\bhsN(\d+)', r'hs\1', t)
    t = t.replace('\nopen Mxx.Primitives MxxRuntime MxxExponentLut\n',
                  '\nopen Mxx.Primitives MxxRuntime MxxExponentLut\n\nset_option maxRecDepth 100000\n', 1)
    return t


def main():
    d = sys.argv[1]
    I = Info(d)
    path = os.path.join(d, 'LwrChainFinal.lean')
    if os.path.samefile(os.path.dirname(path) or '.', TEMPLATES):
        raise SystemExit(f'{path} is the template')
    text = module(I)
    with open(path, 'w') as f:
        f.write(text)
    print('wrote', path)


if __name__ == '__main__':
    main()
