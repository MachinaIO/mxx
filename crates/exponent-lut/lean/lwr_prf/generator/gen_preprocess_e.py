#!/usr/bin/env python3
"""LwrPreprocessE.lean: the selector RHS of every block with its error named.

Usage: gen_preprocess_e.py <parameter dir>

Every `preprocess_selector<b>` of the parameter set's `LwrPreprocess.lean` is rewritten so that its
error is the pair of Gaussian tape samples of the cell, read off the samples the proof bounds.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from info import Info  # noqa: E402


def selectors(text):
    out = []
    for m in re.finditer(r'theorem preprocess_selector(\d+) ', text):
        b = int(m.group(1))
        start = text.rindex('set_option maxHeartbeats', 0, m.start())
        end = text.index('\nset_option', m.end()) + 1 if '\nset_option' in text[m.end():] else text.index('end LwrProof')
        out.append((b, text[start:end]))
    return out


def rewrite(I, b, sec):
    par = re.findall(r'parallel_generatedRoot_(\d+) backend hashModel tape \(path \+\+ \[\1, i\.val\]\)', I.pre)
    par = sorted({int(p) for p in par})
    p = [x for x in par if x not in (3,)][b]
    s0, s1 = re.findall(r'gaussian_bound_at hN h(\d+)_0', sec)
    m = re.search(r'∃ e : Fin 2 → ErrorMatrix N 1 (\d+), \(∀ jj, CoeffBound \(e jj\) (\d+)\) ∧ ∀ jj,', sec)
    cols, cut = m.group(1), m.group(2)
    sec = sec.replace(m.group(0), f'(∀ jj, CoeffBound (rhsError{b} tape path i jj) {cut}) ∧ ∀ jj,')
    sec = re.sub(r'reduceMatrix (Q\d+) N 1 (\d+) \(e jj\) := by', rf'reduceMatrix \1 N 1 \2 (rhsError{b} tape path i jj) := by', sec)
    sec = sec.replace(f'theorem preprocess_selector{b} ', f'theorem preprocess_selector{b}E ')
    sec = re.sub(r'/-- The selector RHS of member `i` of block `\d+`[^\n]*-/',
                 f'/-- `preprocess_selector{b}` with the RHS error named: the tape samples of cell `i`. -/', sec)
    sec = sec.replace(f'obtain ⟨e0\', hb0, he0⟩ := gaussian_bound_at hN h{s0}_0',
                      f'obtain ⟨hb0, he0⟩ := gaussianSampleAt_explicit hN h{s0}_0\n'
                      f'  rw [← clampedWitness_eq_of_bound hN (c := {cut}) he0] at hb0')
    sec = sec.replace(f'obtain ⟨e1\', hb1, he1⟩ := gaussian_bound_at hN h{s1}_0',
                      f'obtain ⟨hb1, he1⟩ := gaussianSampleAt_explicit hN h{s1}_0\n'
                      f'  rw [← clampedWitness_eq_of_bound hN (c := {cut}) he1] at hb1')
    old = "  refine ⟨![e0', e1'], fun jj ↦ by fin_cases jj <;> assumption, fun jj ↦ ?_⟩"
    assert old in sec
    sec = sec.replace(old, '''  refine ⟨fun jj ↦ ?_, fun jj ↦ ?_⟩
  · fin_cases jj
    · exact clampedWitness_bound hN _ _ _
    · exact clampedWitness_bound hN _ _ _''')
    sec = sec.replace('simp (config := { zetaDelta := true }) [ho', f'simp (config := {{ zetaDelta := true }}) [rhsError{b}, ho')
    law = f'(.gaussian (4/1) {cut})'
    d = f'''/-- The RHS error of cell `i` of block `{b}`: the Gaussian tape samples of its two components. -/
noncomputable def rhsError{b} (tape : SampleTape) (path : List Nat) (i : Fin {I.C}) :
    Fin 2 → ErrorMatrix N 1 {cols} :=
  ![clampedWitness tape (path ++ [{p}, i.val] ++ [{s0}]) {law} (-(({cut} : Nat) : Int)) (({cut} : Nat) : Int),
    clampedWitness tape (path ++ [{p}, i.val] ++ [{s1}]) {law} (-(({cut} : Nat) : Int)) (({cut} : Nat) : Int)]

'''
    return d + sec


def switch(I, text):
    """`preprocess_switch` with the switching error named."""
    m = re.search(r'theorem preprocess_switch ', text)
    # The `set_option ... in` lines right above the docstring of the theorem.
    doc = text.rindex('/--', 0, m.start())
    start = doc
    while True:
        prev = text.rfind('\n', 0, start - 1) + 1
        if text[prev:start].startswith('set_option '):
            start = prev
        else:
            break
    end = text.index('end LwrProof', m.end())
    sec = text[start:end]
    s0, s1 = re.findall(r'gaussian_bound_at hN h(\d+)_0', sec)
    m2 = re.search(r'∃ e : Fin 2 → ErrorMatrix N 1 (\d+), \(∀ j, CoeffBound \(e j\) (\d+)\) ∧ ∀ j,', sec)
    cols, cut = m2.group(1), m2.group(2)
    sec = sec.replace(m2.group(0), f'(∀ j, CoeffBound (switchError tape path i j) {cut}) ∧ ∀ j,')
    sec = re.sub(r'reduceMatrix Q1 N 1 (\d+) \(e j\) := by', r'reduceMatrix Q1 N 1 \1 (switchError tape path i j) := by', sec)
    sec = sec.replace('theorem preprocess_switch ', 'theorem preprocess_switchE ')
    sec = re.sub(r"/-- The readout LUT's switching ciphertexts[^-]*-/",
                 '/-- `preprocess_switch` with the switching error named: the tape samples of branch `i`. -/', sec, flags=re.S)
    sec = sec.replace(f"obtain ⟨e0', hb0, he0⟩ := gaussian_bound_at hN h{s0}_0",
                      f'obtain ⟨hb0, he0⟩ := gaussianSampleAt_explicit hN h{s0}_0\n'
                      f'  rw [← clampedWitness_eq_of_bound hN (c := {cut}) he0] at hb0')
    sec = sec.replace(f"obtain ⟨e1', hb1, he1⟩ := gaussian_bound_at hN h{s1}_0",
                      f'obtain ⟨hb1, he1⟩ := gaussianSampleAt_explicit hN h{s1}_0\n'
                      f'  rw [← clampedWitness_eq_of_bound hN (c := {cut}) he1] at hb1')
    old = "  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩"
    assert old in sec
    sec = sec.replace(old, """  refine ⟨fun j ↦ ?_, fun j ↦ ?_⟩
  · fin_cases j
    · exact clampedWitness_bound hN _ _ _
    · exact clampedWitness_bound hN _ _ _""")
    sec = sec.replace('    simp [ho0, ho1, ho2, ho3,', '    simp [switchError, ho0, ho1, ho2, ho3,')
    sec = sec.replace('simp (config := { zetaDelta := true }) only [', 'simp (config := { zetaDelta := true }) only [switchError, ')
    law = f'(.gaussian (4/1) {cut})'
    d = f'''/-- The error of the switching ciphertext of branch `i`: the Gaussian tape samples of its two
components. -/
noncomputable def switchError (tape : SampleTape) (path : List Nat) (i : Fin {I.W}) :
    Fin 2 → ErrorMatrix N 1 {cols} :=
  ![clampedWitness tape (path ++ [3, i.val] ++ [{s0}]) {law} (-(({cut} : Nat) : Int)) (({cut} : Nat) : Int),
    clampedWitness tape (path ++ [3, i.val] ++ [{s1}]) {law} (-(({cut} : Nat) : Int)) (({cut} : Nat) : Int)]

'''
    return d + sec


def main():
    d = sys.argv[1]
    I = Info(d)
    text = open(os.path.join(d, 'LwrPreprocess.lean')).read()
    secs = [rewrite(I, b, sec) for b, sec in selectors(text)] + [switch(I, text)]
    out = '''import LwrPreprocess
import ExponentFunctional

/-!
# The selector RHS with its errors named

Generated by `generator/gen_preprocess_e.py` from `LwrPreprocess`: the error of the selector RHS
of every cell is the pair of Gaussian tape samples of the cell.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxRecDepth 100000

variable {tape : SampleTape} {path : List Nat}

''' + '\n'.join(secs) + 'end LwrProof\n'
    with open(os.path.join(d, 'LwrPreprocessE.lean'), 'w') as f:
        f.write(out)
    print('wrote', os.path.join(d, 'LwrPreprocessE.lean'), len(secs), 'selectors')


if __name__ == '__main__':
    main()
