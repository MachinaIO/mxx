#!/usr/bin/env python3
"""LwrModel<b>.lean: the model of block `b`.

Usage: gen_model.py <parameter dir> [<block>...]

The numerators and the model functions of `LwrModel1` of `average-n32-crt7x40` are the template of
every block `b ≥ 1` (block `0`, without alignment, has `LwrModel0`). The lemma `block<b>M`, which
rewrites the cell and alignment errors of `block<b>E` as divisions of their numerators, is built
from the statement of the parameter set's own `LwrBlockE<b>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from info import Info  # noqa: E402
from gen_blocks_e import basis  # noqa: E402

TEMPLATES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'average-n32-crt7x40')


def small(name):
    return open(os.path.join(TEMPLATES, f'{name}.lean')).read()


def template(b):
    s = small(f'LwrModel{b}')
    body = s[s.index('variable {tape : SampleTape} {path : List Nat}\n') + len('variable {tape : SampleTape} {path : List Nat}\n'):
             s.rindex('end LwrProof')]
    # The block lemma is generated from the set's own `LwrBlockE<b>`.
    i = body.index('set_option maxHeartbeats 16000000 in\nset_option maxRecDepth 100000 in\n/-- `block')
    j = body.index('section Model')
    numer, model = body[:i], body[j:]
    n32 = Info(TEMPLATES)
    if b == 0:
        rep = [(re.escape(basis(n32, 7)), '«BAS_B»'),
               (r'\bQ7\b', '«QB»'), (r'\bQ6\b', '«QN»'), (r'\bQ5\b', '«QM»'),
               (r'\bP7\b', '«PB»'), (r'\bP6\b', '«PR»'),
               (r'\b14\b', '«MA»'), (r'\b12\b', '«MS»'), (r'\b10\b', '«MR»'), (r'\b17\b', '«M»'),
               (r'\brhsPrefix0\b', '«RP»'), (r'\bcoords 0\b', '«CO»'), (r'\bcoords_lt 0\b', '«CL»'),
               (r'\b(pubDigits|rhsDigits|cellKeyUp|cellNumer|numerM|div|bot|stateTarget|stateRow|rhsError)0\b', r'\1«B»'),
               (r'\bkey1\b', 'key«B1»'), (r'\berr1\b', 'err«B1»'), (r'\bModel0\b', 'Model«B»'),
               (r'\bstateTarget1\b', 'stateTarget«B1»'),
               (r'\bmuOf ext 1\b', 'muOf ext «B1»'),
               (r'\bmuOf ext 0\b', 'muOf ext «B»'), (r'\bnuOf ext 0\b', 'nuOf ext «B»'),
               (r'block `1`', 'block `«B1»`'), (r'block `0`', 'block `«B»`')]
    else:
        rep = [(re.escape(basis(n32, 7)), '«BAS_T»'), (re.escape(basis(n32, 6)), '«BAS_B»'),
               (r'\bQ7\b', '«QT»'), (r'\bQ6\b', '«QB»'), (r'\bQ5\b', '«QN»'), (r'\bQ4\b', '«QM»'),
               (r'\bA6\b', '«A»'), (r'\bP6\b', '«PB»'), (r'\bP5\b', '«PR»'),
               (r'\b14\b', '«MT»'), (r'\b12\b', '«MA»'), (r'\b10\b', '«MS»'), (r'\b8\b', '«MR»'),
               (r'\b17\b', '«M»'),
               (r'\brhsPrefix1\b', '«RP»'), (r'\bcoords 1\b', '«CO»'), (r'\bcoords_lt 1\b', '«CL»'),
               (r'\b(alignDigits|alignNumer|alignM|numerM|div|pubDigits|rhsDigits|cellKeyUp|cellNumer|bot|stateRow|rhsError)1\b',
                r'\1«B»'),
               (r'\bstateTarget2\b', 'stateTarget«B1»'), (r'\bstateTarget1\b', 'stateTarget«B»'),
               (r'\bkey2\b', 'key«B1»'), (r'\berr2\b', 'err«B1»'), (r'\bkey1\b', 'key«B»'), (r'\berr1\b', 'err«B»'),
               (r'\bModel1\b', 'Model«B»'),
               (r'\bmuOf ext 2\b', 'muOf ext «B1»'), (r'\bmuOf ext 1\b', 'muOf ext «B»'),
               (r'\bnuOf ext 1\b', 'nuOf ext «B»'),
               (r'block `2`', 'block `«B1»`'), (r'block `1`', 'block `«B»`')]
    rep = [(r'Fin 14\b', '«FINC»')] + rep
    out = []
    for text in (numer, model):
        for a, c in rep:
            text = re.sub(a, c, text)
        out.append(text)
    return out


def placeholders(I, b):
    L = I.L
    qb, qn = L - b, L - b - 1
    d = {'«QB»': f'Q{qb}', '«QN»': f'Q{qn}', '«QM»': f'Q{qn - 1}', '«PB»': f'P{qb}', '«PR»': f'P{qn}',
         '«M»': str(I.M), '«MS»': str(I.m(qn)), '«MR»': str(I.m(qn - 1)), '«RP»': f'rhsPrefix{b}',
         '«CO»': f'coords {b}', '«CL»': f'coords_lt {b}', '«B»': str(b), '«B1»': str(b + 1),
         '«FINC»': f'Fin {I.C}', '«BAS_B»': basis(I, qb)}
    if b == 0:
        d['«MA»'] = str(I.m(L))
    else:
        d.update({'«QT»': f'Q{L}', '«A»': f'A{qb}', '«MT»': str(I.m(L)), '«MA»': str(I.m(qb)),
                  '«BAS_T»': basis(I, L)})
    return d


def fill(s, d):
    for a, c in d.items():
        s = s.replace(a, c)
    assert '«' not in s, re.findall(r'«\w+»', s)[:5]
    return s


def block_m(I, b, d):
    """`block<b>M` from the statement of the set's `block<b>E`."""
    src = open(os.path.join(I.d, f'LwrBlockE{b}.lean')).read()
    a = src.index(f'theorem block{b}E ')
    stmt = src[a:src.index(':= by', a)]
    C = I.C
    seq, prev = 42 + 22 * b, 42 + 22 * (b - 1)
    c = f'⟨coords {b} i, coords_lt {b} i⟩'
    if b == 0:
        k0 = stmt.index(f'    (![![w.w_{seq}_0, w.w_{seq}_1], ![w.w_{seq}_2, w.w_{seq}_3]] = ∑ i : Fin {C}, fun k j ↦')
        hyps = stmt[:k0]
        keyeq = stmt[k0:stmt.index(f'    ∃ v : Fin {C} → Fin 2 → ErrorMatrix N 1')]
        state, rows, va, crow, pay = '(stateKey model key)', '![cs0, cs1]', f'(ea {c})', f'![ca0 {c}, ca1 {c}]', f'(apay {c})'
        intro = ''
    else:
        k0 = stmt.index(f'    let As := ![![w.w_{prev}_0, w.w_{prev}_1], ![w.w_{prev}_2, w.w_{prev}_3]]')
        hyps = stmt[:k0]
        keyeq = stmt[k0:stmt.index(f'    ∃ va : Fin {C} → Fin 2 → ErrorMatrix N 1')]
        keyeq = keyeq.replace(f'    let As := ![![w.w_{prev}_0, w.w_{prev}_1], ![w.w_{prev}_2, w.w_{prev}_3]]\n', '')
        state = f'![![w.w_{prev}_0, w.w_{prev}_1], ![w.w_{prev}_2, w.w_{prev}_3]]'
        keyeq = keyeq.replace(' As\n', f' {state}\n')
        rows = f'![w.w_{prev}_4, w.w_{prev}_5]'
        va = (f"(fun j' ↦ divInt {d['«A»']} (alignNumer{b} sm (memberKey model key ((coords {b} i : Nat) : Int))\n"
              f"              (ea {c}) ![ca0 {c}, ca1 {c}] j'))")
        crow = (f"(fun j' ↦ theOutput (blockModSwitchRuns {d['«BAS_T»']} 1) (![ca0 {c}, ca1 {c}] j' * "
                f"alignDigits{b}))")
        pay = f"(fun _ _ ↦ reducePoly {d['«QB»']} N (α {c}))"
    qn, P, mr = d['«QN»'], d['«PB»'], d['«MR»']
    numer = (f"divInt {P} (cellNumer{b} sm (memberKey model key ((coords {b} i : Nat) : Int))\n"
             f"            {va} es\n"
             f"            (α {c}) {state}\n"
             f"            (rhsTop model key {qn} {mr} rhsPrefix{b} i) ![r{b}0 i, r{b}1 i]\n"
             f"            {crow}\n"
             f"            {rows} {pay} j)")
    cellid = (f"      (∀ i j row column, {numer[len('divInt ' + P + ' '):]} row column =\n"
              f"        (({P} : Int) : ErrorPoly N) * {numer} row column) ∧\n")
    if b > 0:
        alig = (f"alignNumer{b} sm (memberKey model key ((coords {b} i : Nat) : Int))\n"
                f"          (ea {c}) ![ca0 {c}, ca1 {c}] j")
        cellid = (f"      (∀ i j row column, ({alig}) row column =\n"
                  f"        (({d['«A»']} : Int) : ErrorPoly N) * divInt {d['«A»']} ({alig}) row column) ∧\n") + cellid
    concl = cellid + (f"      RelE sm ![![w.w_{seq}_0, w.w_{seq}_1], ![w.w_{seq}_2, w.w_{seq}_3]]\n"
             f"        (∑ i : Fin {C}, fun j ↦ reducePoly {qn} N ((({d['«PR»']} : Nat) : Int) *\n"
             f"          (α {c} * μ * ν i) * unitPair sp j) • h)\n"
             f"        ![w.w_{seq}_4, w.w_{seq}_5]\n"
             f"        (∑ i : Fin {C}, fun j ↦ {numer} - (α {c} * μ) • eB i j) ")
    if b == 0:
        proof = f''':= by
  obtain ⟨hk, v, hv, hid⟩ := block0E hb hG hg' hg ha hpay hs hrhs
  have hvM : ∀ i, v i = fun j ↦ {numer} := by
    intro i
    funext j
    exact div_of_identity (by norm_num) (fun row column ↦ hid i j row column)
  refine ⟨hk, fun i j row column ↦ ?_, ?_⟩
  · have h := hid i j row column
    rw [hvM i] at h
    exact h
  · simp only [hvM] at hv
    exact hv
'''
    else:
        proof = f''':= by
  obtain ⟨hk, va, hva, v, hv, hid⟩ := block{b}E hb hG hg' hg ha hpay hs hrhs
  have hvaM : ∀ i, va i = fun j ↦ divInt {d['«A»']} (alignNumer{b} sm (memberKey model key ((coords {b} i : Nat) : Int))
      (ea {c}) ![ca0 {c}, ca1 {c}] j) := by
    intro i
    funext j
    exact div_of_identity (by norm_num) (fun row column ↦ hva i j row column)
  have hvM : ∀ i, v i = fun j ↦ {numer} := by
    intro i
    funext j
    have h := fun row column ↦ hid i j row column
    simp only [hvaM] at h
    exact div_of_identity (by norm_num) h
  refine ⟨hk, fun i j row column ↦ ?_, fun i j row column ↦ ?_, ?_⟩
  · have h := hva i j row column
    rw [hvaM i] at h
    exact h
  · have h := hid i j row column
    simp only [hvaM] at h
    rw [hvM i] at h
    exact h
  · simp only [hvM] at hv
    exact hv
'''
    what = 'cell errors' if b == 0 else 'cell and alignment errors'
    return (f'set_option maxHeartbeats 16000000 in\nset_option maxRecDepth 100000 in\n'
            f'/-- `block{b}E` with the {what} as divisions of their numerators. -/\n'
            + hyps.replace(f'theorem block{b}E ', f'theorem block{b}M ') + keyeq + concl + proof + '\n')


def module(I, b):
    t = 0 if b == 0 else 1
    numer, model = template(t)
    d = placeholders(I, b)
    prev = 'LwrModelBase' if b == 0 else f'LwrModel{b - 1}'
    return (f'''import {prev}
import LwrBlockE{b}

/-!
# The model of block {b}

Generated by `generator/gen_model.py`: the numerators of block `{b}`, `block{b}E` with its errors as
their divisions, and the keys, rows and errors of the model that block `{b}` produces.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {{tape : SampleTape}} {{path : List Nat}}
''' + fill(numer, d) + block_m(I, b, d) + fill(model, d) + '\nend LwrProof\n')


def main():
    d = sys.argv[1]
    I = Info(d)
    blocks = [int(x) for x in sys.argv[2:]] or list(range(I.B))
    # Every module is generated before any is written: the template parameter set reads its own
    # modules.
    contents = {b: module(I, b) for b in blocks}
    for b, text in contents.items():
        path = os.path.join(d, f'LwrModel{b}.lean')
        with open(path, 'w') as f:
            f.write(text)
        print('wrote', path)


if __name__ == '__main__':
    main()
