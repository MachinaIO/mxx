#!/usr/bin/env python3
"""LwrBlockE<b>.lean: every block with its error and keys named.

Usage: gen_blocks_e.py <parameter dir> [<block>...]

The cells of `LwrBlockE0` and `LwrBlockE1` of `average-n32-crt7x40` are the templates: block `0`
multiplies the state by its member, every later block first aligns the member from the top modulus.
The block sums are `gen_blocks.block_lemma` rewritten to the named error.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen_blocks import block_lemma  # noqa: E402
from info import Info  # noqa: E402

TEMPLATES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'average-n32-crt7x40')


def small(name):
    return open(os.path.join(TEMPLATES, f'{name}.lean')).read()


def basis(I, k):
    """The CRT primes of `Q k`, in the order of the exported ModDowns."""
    return '[' + ', '.join(str(p) for p in [I.Q[1]] + [I.P[j] for j in range(2, k + 1)]) + ']'


def cell_template(b):
    """The definitions and the cell lemma of the template block `b` (0 or 1), with placeholders."""
    s = small(f'LwrBlockE{b}')
    s = s[s.index('/-- The digits ') if b == 0 else s.index('/-- The digits `H`'):
          s.index('set_option maxHeartbeats 64000000 in\nset_option maxRecDepth 100000 in\n/-- Block')]
    n32 = Info(TEMPLATES)
    if b == 0:
        rep = [(re.escape(basis(n32, 7)), '«BAS_B»'),
               (r'\bq7_eq\b', '«QEB»'), (r'\bBackend\.layout6 layout7\b', '«LB»'),
               (r'\bBackend\.layout5 layout6\b', '«LN»'),
               (r'\bhQ7\b', '«HQB»'), (r'\bhQ6\b', '«HQN»'), (r'\bQ7\b', '«QB»'), (r'\bQ6\b', '«QN»'),
               (r'\bP7\b', '«PB»'), (r'\bP6\b', '«PR»'),
               (r'\b14\b', '«MA»'), (r'\b12\b', '«MS»'), (r'\b10\b', '«MR»'), (r'\b17\b', '«M»'),
               (r'\brhsPrefix0\b', '«RP»'), (r'\bcoords 0\b', '«CO»'), (r'\bcoords_lt 0\b', '«CL»'),
               (r'\bselectInt_coords0\b', '«SC»'), (r'parallel_generatedRoot_23\b', '«PAR»'),
               (r'\b(pubDigits|rhsDigits|cellKeyUp|cell)0(E?)\b', r'\1«B»\2'),
               (r'block `0`', 'block `«B»`'), (r'block-0', 'block-«B»')]
    else:
        rep = [(re.escape(basis(n32, 7)), '«BAS_T»'), (re.escape(basis(n32, 6)), '«BAS_B»'),
               (r'\bq7_eq6\b', '«QEA»'), (r'\bq6_eq\b', '«QEB»'), (r'\bBackend\.layout6 layout7\b', '«LA»'),
               (r'\bBackend\.layout5 layout6\b', '«LB»'), (r'\bBackend\.layout4 layout5\b', '«LN»'),
               (r'\bhQ7\b', '«HQT»'), (r'\bhQ6\b', '«HQB»'), (r'\bhQ5\b', '«HQN»'), (r'\bQ7\b', '«QT»'),
               (r'\bQ6\b', '«QB»'), (r'\bQ5\b', '«QN»'), (r'\bA6\b', '«A»'), (r'\bP6\b', '«PB»'),
               (r'\bP5\b', '«PR»'),
               (r'\b14\b', '«MT»'), (r'\b12\b', '«MA»'), (r'\b10\b', '«MS»'), (r'\b8\b', '«MR»'),
               (r'\b17\b', '«M»'),
               (r'\brhsPrefix1\b', '«RP»'), (r'\bcoords 1\b', '«CO»'), (r'\bcoords_lt 1\b', '«CL»'),
               (r'\bselectInt_coords1\b', '«SC»'), (r'parallel_generatedRoot_45\b', '«PAR»'),
               (r'\b(alignDigits|alignKey|pubDigits|rhsDigits|cellKeyUp|cell)1(E?)\b', r'\1«B»\2'),
               (r'block `1`', 'block `«B»`'), (r'block-1', 'block-«B»'), (r'`Q7` to `Q6`', '`«QT»` to `«QB»`')]
    rep = [(r'Fin 14\b', '«FINC»'), (r'\bfourteen\b', '«NC»')] + rep + \
        [(r'\b1048576\b', '«BASE»'), (r'\b524288\b', '«DIGIT»'), (r'2\^20', '«BASEPOW»')]
    for a, c in rep:
        s = re.sub(a, c, s)
    return s


def fill(s, d):
    for a, c in d.items():
        s = s.replace(a, c)
    assert '«' not in s, re.findall(r'«\w+»', s)[:5]
    return s


def placeholders(I, b):
    L = I.L
    qb, qn = L - b, L - b - 1
    d = {'«QB»': f'Q{qb}', '«QN»': f'Q{qn}', '«PB»': f'P{qb}', '«PR»': f'P{qn}', '«M»': str(I.M),
         '«MS»': str(I.m(qn)), '«MR»': str(I.m(qn - 1)), '«RP»': f'rhsPrefix{b}', '«CO»': f'coords {b}',
         '«CL»': f'coords_lt {b}', '«SC»': f'selectInt_coords{b}',
         '«PAR»': f'parallel_generatedRoot_{23 + 22 * b}', '«B»': str(b),
         '«FINC»': f'Fin {I.C}', '«NC»': str(I.C), '«BASE»': str(I.base), '«DIGIT»': str(I.digit),
         '«BASEPOW»': f'2^{I.base.bit_length() - 1}',
         '«QEB»': f'q{qb}_eq', '«LB»': f'Backend.layout{qb - 1} layout{qb}',
         '«LN»': f'Backend.layout{qn - 1} layout{qn}', '«HQB»': f'hQ{qb}', '«HQN»': f'hQ{qn}',
         '«BAS_B»': basis(I, qb)}
    if b == 0:
        d['«MA»'] = str(I.m(L))
    else:
        d.update({'«QEA»': f'q{L}_eq{qb}', '«LA»': f'Backend.layout{L - 1} layout{L}', '«HQT»': f'hQ{L}',
                  '«QT»': f'Q{L}', '«A»': f'A{qb}', '«MT»': str(I.m(L)), '«MA»': str(I.m(qb)),
                  '«BAS_T»': basis(I, L)})
    return d


def block_lemma_e(I, b):
    """`gen_blocks.block_lemma` with the named error of `cellE`."""
    L, C = I.L, I.C
    qb, qn = L - b, L - b - 1
    par, seq, prev = 23 + 22 * b, 42 + 22 * b, 42 + 22 * (b - 1)
    s = block_lemma(I, b)

    def rep(old, new):
        nonlocal s
        assert old in s, old[:120]
        s = s.replace(old, new)

    rep('set_option maxHeartbeats 64000000 in\n', 'set_option maxHeartbeats 64000000 in\nset_option maxRecDepth 100000 in\n')
    rep(s[s.index('/-- Block `'):s.index('\ntheorem block')],
        f'/-- Block `{b}` with its error and key named: the sum of the cells of `cell{b}E`. -/')
    rep(f'theorem block{b} ', f'theorem block{b}E ')
    # Input types are inferred from the generated root.
    s = re.sub(r'\{(s0 s1 s2 s3|k0 k1|ca0 ca1|cs0 cs1|apay|r\d0 r\d1|r\d\d0 r\d\d1) : [^}]*\}', r'{\1}', s)
    rep(f'''    (hsm : polyNorm sm ≤ 1)
    (hαμ : ∀ i : Fin {C}, SmulGain (α ⟨coords {b} i, coords_lt {b} i⟩ * μ) 1)
    (hα : ∀ k, SmulGain (α k) 1)
''', '')
    rep(f'''    (ha : ∀ k : Fin {I.M}, Rel sm (memberKey model key k)
      (fun j ↦ reducePoly Q{L} N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] {I.cutoff})''',
        f'''    {{ea : Fin {I.M} → Fin 2 → ErrorMatrix N 1 {I.m(L)}}}
    (ha : ∀ k : Fin {I.M}, RelE sm (memberKey model key k)
      (fun j ↦ reducePoly Q{L} N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] (ea k))''')
    ms_, mr_ = I.m(qn), I.m(qn - 1)
    if b == 0:
        rep(f'''(hs : Rel sm (stateKey model key)
      (fun j ↦ reducePoly Q{L} N (((P{L} : Nat) : Int) * μ * unitPair sp j) • g) ![cs0, cs1] {I.cutoff})''',
            f'''{{es : Fin 2 → ErrorMatrix N 1 {I.m(L - 1)}}}
    (hs : RelE sm (stateKey model key)
      (fun j ↦ reducePoly Q{L} N (((P{L} : Nat) : Int) * μ * unitPair sp j) • g) ![cs0, cs1] es)''')
        state, rows = 'stateKey model key', '![cs0, cs1]'
    else:
        rep(f'''{{Bs : Nat}} (hs : Rel sm ![![w.w_{prev}_0, w.w_{prev}_1], ![w.w_{prev}_2, w.w_{prev}_3]]
      (fun j ↦ reducePoly Q{qb} N (((P{qb} : Nat) : Int) * μ * unitPair sp j) • g)
      ![w.w_{prev}_4, w.w_{prev}_5] Bs)''',
            f'''{{es : Fin 2 → ErrorMatrix N 1 {I.m(qb - 1) if False else ms_}}}
    (hs : RelE sm ![![w.w_{prev}_0, w.w_{prev}_1], ![w.w_{prev}_2, w.w_{prev}_3]]
      (fun j ↦ reducePoly Q{qb} N (((P{qb} : Nat) : Int) * μ * unitPair sp j) • g)
      ![w.w_{prev}_4, w.w_{prev}_5] es)''')
        state, rows = f'![![w.w_{prev}_0, w.w_{prev}_1], ![w.w_{prev}_2, w.w_{prev}_3]]', f'![w.w_{prev}_4, w.w_{prev}_5]'
    rep(f'''    (hrhs : ∀ i : Fin {C}, ∃ e : Fin 2 → ErrorMatrix N 1 {mr_}, (∀ j, CoeffBound (e j) {I.cutoff}) ∧ ∀ j,''',
        f'''    {{eB : Fin {C} → Fin 2 → ErrorMatrix N 1 {mr_}}}
    (hrhs : ∀ i : Fin {C}, ∀ j,''')
    rep(f'reduceMatrix Q{qn} N 1 {mr_} (e j)) :', f'reduceMatrix Q{qn} N 1 {mr_} (eB i j)) :')
    head = f'''    Rel sm ![![w.w_{seq}_0, w.w_{seq}_1], ![w.w_{seq}_2, w.w_{seq}_3]]'''
    tail_start = s.index(head)
    tail_end = s.index(':= by\n', tail_start)
    c = f'⟨coords {b} i, coords_lt {b} i⟩'
    if b == 0:
        aligned = ''
        crow = f'![ca0 c, ca1 c]'
        pay = '(apay c)'
        cellerr = 'ea c'
        va_intro = ''
        refine_va = ''
    else:
        aligned = f'''    ∃ va : Fin {C} → Fin 2 → ErrorMatrix N 1 {I.m(qb)},
    (∀ i j row column,
      let c : Fin {I.M} := {c}
      let mk := memberKey model key c
      (ea c j * liftMatrix alignDigits{b}) row column -
          downResidue A{qb} (![ca0 c, ca1 c] j * alignDigits{b}) row column +
          sm * downResidue A{qb} (mk 0 j * alignDigits{b}) row column +
          downResidue A{qb} (mk 1 j * alignDigits{b}) row column =
        ((A{qb} : Int) : ErrorPoly N) * va i j row column) ∧
'''
        crow = 'crow'
        pay = f'(fun _ _ ↦ reducePoly Q{qb} N (α c))'
        cellerr = 'va i'
        va_intro = f'''        let crow : Fin 2 → ExactMatrix Q{qb} N 1 {I.m(qb)} := fun j ↦
          theOutput (blockModSwitchRuns {basis(I, L)} 1) (![ca0 c, ca1 c] j * alignDigits{b})
'''
        refine_va = 'va, fun i j row column ↦ hva i j row column, '
    mk = f'memberKey model key (({c[1:-1].split(",")[0]} : Nat) : Int)'
    new = f'''    let As := {state}
    (![![w.w_{seq}_0, w.w_{seq}_1], ![w.w_{seq}_2, w.w_{seq}_3]] = ∑ i : Fin {C}, fun k j ↦
      theOutput (blockModSwitchRuns {basis(I, qb)} 1)
        (cellKeyUp{b} (memberKey model key ((coords {b} i : Nat) : Int)) As
          (rhsTop model key Q{qn} {mr_} rhsPrefix{b} i) ![r{b}0 i, r{b}1 i] k j)) ∧
{aligned}    ∃ v : Fin {C} → Fin 2 → ErrorMatrix N 1 {mr_},
      RelE sm ![![w.w_{seq}_0, w.w_{seq}_1], ![w.w_{seq}_2, w.w_{seq}_3]]
        (∑ i : Fin {C}, fun j ↦ reducePoly Q{qn} N (((P{qn} : Nat) : Int) *
          (α {c} * μ * ν i) * unitPair sp j) • h)
        ![w.w_{seq}_4, w.w_{seq}_5]
        (∑ i : Fin {C}, fun j ↦ v i j - (α {c} * μ) • eB i j) ∧
      ∀ i j row column,
        let c : Fin {I.M} := {c}
        let top := rhsTop model key Q{qn} {mr_} rhsPrefix{b} i
        let bot := ![r{b}0 i, r{b}1 i]
        let mk := memberKey model key c
{va_intro}        pairMul (fun j' ↦ pairMul ({cellerr}) (liftMatrix (pubDigits{b} As j')) + α c • es j')
            (liftMatrix (rhsDigits{b} top bot j)) row column -
          downResidue P{qb} (pairMul (fun j' ↦ pairMul {crow} (pubDigits{b} As j') +
            matrixMulScalarRight ({rows} j') {pay})
            (rhsDigits{b} top bot j)) row column +
          sm * downResidue P{qb} (cellKeyUp{b} mk As top bot 0 j) row column +
          downResidue P{qb} (cellKeyUp{b} mk As top bot 1 j) row column =
        ((P{qb} : Int) : ErrorPoly N) * v i j row column '''
    s = s[:tail_start] + new + s[tail_end:]
    rep(':= by\n  unfold Stage_online_eval.generatedRoot.body at hb', ':= by\n  intro As\n  unfold Stage_online_eval.generatedRoot.body at hb')
    if b == 0:
        rep('  rw [hkey] at hs\n', '  have hs\' := hs\n  rw [hkey] at hs\'\n')
    i = s.index('  have hcells : ∀ i')
    j = s.index('  have hsum := Rel.sum Finset.univ (fun i _ ↦ hcells i)\n')
    hs_name = "hs'" if b == 0 else 'hs'
    pattern = 'hkeys v hv hvp' if b == 0 else 'hkeys va hva v hv hvp'
    s = s[:i] + f'''  have hcells := fun i : Fin {C} ↦ cell{b}E (h{par}_5 i) hG hg' hg ha hpay {hs_name} (hrhs i)
  choose {pattern} using hcells
  have hsum := RelE.sum Finset.univ (fun i _ ↦ hv i)
''' + s[j + len('  have hsum := Rel.sum Finset.univ (fun i _ ↦ hcells i)\n'):]
    if b == 0:
        fin = '''  refine ⟨?_, v, hsum, fun i j row column ↦ ?_⟩
  · rw [← hA, show As = _ from hkey]
    exact Finset.sum_congr rfl fun i _ ↦ hkeys i
  · rw [show As = _ from hkey]
    exact hvp i j row column
'''
    else:
        fin = f'''  refine ⟨?_, {refine_va}v, hsum, fun i j row column ↦ hvp i j row column⟩
  rw [← hA]
  exact Finset.sum_congr rfl fun i _ ↦ hkeys i
'''
    rep('  rw [hA, hc] at hsum\n  exact hsum\n', '  rw [hA, hc] at hsum\n' + fin)
    del mk
    return s


def module(I, b):
    t = 0 if b == 0 else 1
    cell = fill(cell_template(t), placeholders(I, b))
    prev = 'LwrErrorCommon' if b == 0 else f'LwrBlockE{b - 1}'
    what = ('multiplies the state by its member,\nthen by the selector, with every error and key named'
            if b == 0 else 'aligns its member, then\nmultiplies as in block `0`')
    return f'''import {prev}
import LwrBlock{b}

/-!
# Block {b} with explicit errors and keys

Generated by `generator/gen_blocks_e.py`. Cell `i` of block `{b}` {what}.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {{tape : SampleTape}} {{path cellPath : List Nat}}

''' + cell + block_lemma_e(I, b).lstrip('\n') + '\nend LwrProof\n'


def deepen(text):
    """The aligned cells of the large sets elaborate moduli of hundreds of digits."""
    return text.replace('set_option maxRecDepth 100000 in\n', 'set_option maxRecDepth 1000000 in\n')


def main():
    d = sys.argv[1]
    I = Info(d)
    blocks = [int(x) for x in sys.argv[2:]] or list(range(I.B))
    contents = {b: deepen(module(I, b)) for b in blocks}
    for b, text in contents.items():
        path = os.path.join(d, f'LwrBlockE{b}.lean')
        with open(path, 'w') as f:
            f.write(text)
        print('wrote', path)


if __name__ == '__main__':
    main()
