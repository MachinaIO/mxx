#!/usr/bin/env python3
"""LwrChain.lean: a run is its model, block by block.

Usage: gen_chain.py <parameter dir>

The theorem `chain1` of `average-n32-crt7x40` is the template: its block-0 part (the secrets, the
encryptions, the selector RHS of block `0`, and `block0M`) and its block-1 part (the collapse of the
previous target, the selector RHS, and `block1M`) are repeated for every block of the parameter set.
The generated theorem `chain` ends at the input of the readout LUT.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from info import Info, wrap  # noqa: E402

TEMPLATES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'average-n32-crt7x40')


def small(name):
    return open(os.path.join(TEMPLATES, f'{name}.lean')).read()


def pieces():
    s = small('LwrChain')
    a = s.index('theorem chain1 ')
    body = s[s.index(':= by\n', a) + len(':= by\n'):s.index('end LwrProof', a)]
    pre_end = body.index('  have hspM := payload_model hpay\n')
    b0_end = body.index('  -- Block 1: the block-0 target collapses to the state target of block `1`.\n')
    b1_end = body.index('  -- End of block 1.\n')
    helpers = s[s.index('theorem muOf_zero'):s.index('set_option maxRecDepth 100000 in\nset_option maxHeartbeats 64000000 in\n/-- The keys and the error leaving block `0`')]
    return helpers, body[pre_end:b0_end], body[b0_end:b1_end]


def preamble(I):
    B = I.B
    hs = ', '.join(f'hs{b}, hx{b}' for b in range(B))
    sums = ', '.join(f'hsum{b}' for b in range(B))
    sets = ''.join(f'  set s{b} := external.input_{6 + 2 * b}\n' for b in range(B))
    sets += ''.join(f'  set x{b} := external.input_{7 + 2 * b}\n' for b in range(B))
    sels = ''.join(f'  obtain ⟨sel{b}, hsel{b}⟩ := one_hot hs{b} hsum{b}\n' for b in range(B))
    nus = ''.join(f'''  have hν{b} : ∀ i, message (n := N) (s{b} i) (x{b} i) = (s{b} i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx{b} i; omega)
''' for b in range(B))
    return f'''  obtain ⟨hvalid, hpay, hmask, hpre, henc, honlineEval, hdec, hreq, hideal, hreqtrue⟩ := hruns
  obtain ⟨sp, hsp, hspn⟩ := secret_ternary hpay
  obtain ⟨sm, hsm, hsmn⟩ := mask_ternary hmask
  obtain ⟨-, ⟨hc1, hc2⟩, ⟨he1, he2⟩, ha, hac, hae, {hs}⟩ :=
    hvalid
  obtain ⟨hmem, {sums}⟩ := requirement_holds hreq hreqtrue
  have hid := ideal_value hideal
  set a := external.input_3
{sets}  -- The selected cells and the plaintexts.
{sels}  let α : Fin {I.M} → ErrorPoly N := fun k ↦ Xe ({I.stride} * (a k).toNat)
  have hα : ∀ k, message (n := N) (external.input_4 k) (external.input_5 k) = α k := fun k ↦
    message_member (hmem k).1 (hmem k).2 (hae k)
{nus}  have hμ0 : message (n := N) external.input_1 external.input_2 = Xe 0 :=
    message_one (k := 0) (by omega) (by simp; omega) (by simp only [N]; omega)
'''


def online_tuple(I):
    """The arguments of the online evaluation in the run relation of the set's claim."""
    claim = open(os.path.join(I.d, 'generated', 'Claim.lean')).read()
    head = 'Stage_online_eval.generatedRoot Backend.backend hashModel tape [4] stage_4_params (('
    a = claim.index(head) + len(head)
    b = claim.index(')) execution.«stage_4»', a)
    items = [x.strip() for x in claim[a:b].split(',')]
    assert items[-1] == '()' and len(items) == 12 + 2 * I.B + 1, len(items)
    return items[:-1]


def rhs_proj(I, b, j):
    """The run's RHS bottoms `r{b}{j}` of block `b`."""
    return online_tuple(I)[12 + 2 * b + j]


def proj2(k):
    return 'execution.«stage_2»' + '.2' * k + '.1'


def block0(I, text):
    L = I.L
    rep = [(r'\bG14\b', f'G{I.m(L)}'), (r'\bhG14\b', f'hG{I.m(L)}'),
           (r'\bg12\b', f'g{I.m(L - 1)}'), (r'\bhg12\b', f'hg{I.m(L - 1)}'), (r'\bhg12u\b', f'hg{I.m(L - 1)}u'),
           (r'\bg10\b', f'g{I.m(L - 2)}'), (r'\bhg10\b', f'hg{I.m(L - 2)}'), (r'\bhg10u\b', f'hg{I.m(L - 2)}u'),
           (r'\(m := 14\)', f'(m := {I.m(L)})'), (r'\(m := 12\)', f'(m := {I.m(L - 1)})'),
           (r'\(m := 10\)', f'(m := {I.m(L - 2)})'),
           (r'\blayout7\b', f'layout{L}'), (r'\blayout6\b', f'layout{L - 1}'), (r'\blayout5\b', f'layout{L - 2}'),
           (r'\bhQ7\b', f'hQ{L}'), (r'\bhQ6\b', f'hQ{L - 1}'), (r'\bQ7\b', f'Q{L}'), (r'\bQ6\b', f'Q{L - 1}'),
           (r'\bQ5\b', f'Q{L - 2}'), (r'\bP6\b', f'P{L - 1}'),
           (r'ExactMatrix (Q\d+) N 1 12\b', rf'ExactMatrix \1 N 1 {I.m(L - 1)}'),
           (r'ExactMatrix (Q\d+) N 1 10\b', rf'ExactMatrix \1 N 1 {I.m(L - 2)}'),
           (r'gReb (Q\d+) (Q\d+) 10\b', rf'gReb \1 \2 {I.m(L - 2)}'),
           (r'Fin 17\b', f'Fin {I.M}'), (r'Fin 14\b', f'Fin {I.C}'),
           (re.escape(proj2(8)) + r' i', rhs_proj(I, 0, 0) + ' i'), (re.escape(proj2(9)) + r' i', rhs_proj(I, 0, 1) + ' i')]
    for a, c in rep:
        text = re.sub(a, c, text)
    return text


def selector_signs(I, text):
    """The sign facts of every block's selector, listed for `fin_cases` over the blocks."""
    old = 'exacts [(hs0 i).1, (hs1 i).1, (hs2 i).1, (hs3 i).1, (hs4 i).1]'
    new = 'exacts [' + ', '.join(f'(hs{b} i).1' for b in range(I.B)) + ']'
    return text.replace(old, new)


def block(I, b, text):
    L, C, M = I.L, I.C, I.M
    qb, qn = L - b, L - b - 1
    seq, prev = 42 + 22 * b, 42 + 22 * (b - 1)
    win, wout = I.m(qn), I.m(qn - 1)
    direct = [(r'ExactMatrix (Q\d+) N 1 8\b', rf'ExactMatrix \1 N 1 {wout}'),
              (r'gReb (Q\d+) (Q\d+) 10\b', rf'gReb \1 \2 {win}'),
              (r'gReb (Q\d+) (Q\d+) 8\b', rf'gReb \1 \2 {wout}')]
    for a, c in direct:
        text = re.sub(a, c, text)
    rep = [(r'\bG14\b', f'G{I.m(L)}'), (r'\bhG14\b', f'hG{I.m(L)}'), (r'\(m := 14\)', f'(m := {I.m(L)})'),
           (r'\bhdiv1M\b', f'hdiv{b}M'), (r'\bhadiv1\b', f'hadiv{b}'), (r'\bhdiv1\b', f'hdiv{b}'),
           (r'\bdiv1\b', f'div{b}'),
           (r'\bhk0M\b', f'hk{b - 1}M'), (r'\bhrel0M\b', f'hrel{b - 1}M'),
           (r'\bhstep0\b', f'hstep{b - 1}'), (r'\bsel0\b', f'sel{b - 1}'), (r'\bhsel0\b', f'hsel{b - 1}'),
           (r'\bhk1M\b', f'hk{b}M'), (r'\bhrel1M\b', f'hrel{b}M'), (r'\bhin1\b', f'hin{b}'),
           (r'\bhsrow1\b', f'hsrow{b}'), (r'\bhk1\b', f'hk{b}'), (r'\bhrel1\b', f'hrel{b}'),
           (r'\bhbot1\b', f'hbot{b}'), (r'\bhrhs1\b', f'hrhs{b}'), (r'\bhν1\b', f'hν{b}'),
           (r'\bs1\b', f's{b}'), (r'\bx1\b', f'x{b}'),
           (r'\bblock1M\b', f'block{b}M'), (r'\bpreprocess_selector1E\b', f'preprocess_selector{b}E'),
           (r'\bkey2\b', f'key{b + 1}'), (r'\berr2\b', f'err{b + 1}'), (r'\bkey1\b', f'key{b}'),
           (r'\berr1\b', f'err{b}'), (r'\bstateRow1\b', f'stateRow{b}'), (r'\bstateTarget1\b', f'stateTarget{b}'),
           (r'\bbot1\b', f'bot{b}'),
           (r'external 1\b', f'external {b}'), (r'external 0\b', f'external {b - 1}'),
           (r'\bcoords 1\b', f'coords {b}'), (r'\bcoords_lt 1\b', f'coords_lt {b}'),
           (r'\bcoords 0\b', f'coords {b - 1}'), (r'\bcoords_lt 0\b', f'coords_lt {b - 1}'),
           (re.escape(proj2(10)) + r' i', rhs_proj(I, b, 0) + ' i'),
           (re.escape(proj2(11)) + r' i', rhs_proj(I, b, 1) + ' i'),
           (r'\bg10\b', f'g{win}'), (r'\bhg10\b', f'hg{win}'), (r'\bhg10u\b', f'hg{win}u'),
           (r'\bg8\b', f'g{wout}'), (r'\bhg8\b', f'hg{wout}'), (r'\bhg8u\b', f'hg{wout}u'),
           (r'\(m := 8\)', f'(m := {wout})'), (r'\blayout4\b', f'layout{qn - 1}'),
           (r'\bhQ5\b', f'hQ{qn}'), (r'\bQ6\b', f'Q{qb}'), (r'\bQ5\b', f'Q{qn}'), (r'\bQ4\b', f'Q{qn - 1}'),
           (r'\bP6\b', f'P{qb}'), (r'\bP5\b', f'P{qn}'),
           (r'\bw\.w_64_', f'w.w_{seq}_'), (r'\bw\.w_42_', f'w.w_{prev}_'),
           (r'Fin 17\b', f'Fin {M}'), (r'Fin 14\b', f'Fin {C}'),
           (r'block-0 target collapses to the state target of block `1`',
            f'block-{b - 1} target collapses to the state target of block `{b}`')]
    # Protect the earlier substitutions from the later ones by staged placeholders.
    marks = {}
    for k, (a, c) in enumerate(rep):
        mark = f'«{k}»'
        text = re.sub(a, mark, text)
        marks[mark] = c
    for mark, c in marks.items():
        text = text.replace(mark, c)
    return text


def chain(I):
    B, L, C = I.B, I.L, I.C
    helpers, b0, b1 = pieces()
    seq = 42 + 22 * (B - 1)
    qn = L - B
    body = selector_signs(I, preamble(I) + block0(I, b0) + ''.join(block(I, b, b1) for b in range(1, B)))
    divs = ' ∧ '.join(f'div{b} model external.input_0 external tape' for b in range(B))
    concl = f'''      {divs} ∧
      ![![w.w_{seq}_0, w.w_{seq}_1], ![w.w_{seq}_2, w.w_{seq}_3]] = key{B} model external.input_0 external tape ∧
      RelE (maskM tape) ![![w.w_{seq}_0, w.w_{seq}_1], ![w.w_{seq}_2, w.w_{seq}_3]]
        (∑ i : Fin {C}, fun j ↦ reducePoly Q{qn} N (((P{qn} : Nat) : Int) *
          (alphaOf external ⟨coords {B - 1} i, coords_lt {B - 1} i⟩ * muOf external {B - 1} * nuOf external {B - 1} i) *
            unitPair (payloadM tape) j) • gReb Q{qn} Q{qn - 1} {I.m(qn - 1)})
        ![w.w_{seq}_4, w.w_{seq}_5] (err{B} model external.input_0 external tape)'''
    s = small('LwrChain')
    a = s.index('theorem chain1 ')
    head = s[s.rindex('set_option maxRecDepth 100000 in', 0, a):a]
    head = head.replace('set_option maxRecDepth 100000 in', 'set_option maxRecDepth 1000000 in')
    stmt = s[a:s.index(':= by\n', a)]
    stmt = stmt.replace('theorem chain1 ', 'theorem chain ')
    ta = stmt.index('        (execution.«stage_2».1,')
    tb = stmt.index('()) execution.«stage_4» w', ta)
    stmt = stmt[:ta] + '        (' + wrap(online_tuple(I) + ['()'], ', ', 10, 9) + stmt[tb + 2:]
    k = stmt.index('      div0 model external.input_0 external tape ∧')
    stmt = stmt[:k] + concl + ' '
    old = ('/-- The keys and the error leaving block `1` are those of the model, and the numerators of blocks\n'
           '`0` and `1` are divisible by their dropped factors. -/')
    assert old in head
    head = head.replace(old, f'/-- The keys and the error leaving block `{B - 1}` are those of the model, and the numerators of\n'
                        'every block are divisible by their dropped factors. -/')
    hdivs = ''.join(f'hdiv{b}M, ' for b in range(B))
    return head + stmt + ':= by\n' + body + f'  exact ⟨w, hb, {hdivs}hk{B - 1}M, hrel{B - 1}M⟩\n'


def module(I):
    imports = f'import LwrModel{I.B - 1}\nimport LwrCertificate\n'
    helpers, _, _ = pieces()
    helpers = (helpers.replace('Fin 5', f'Fin {I.B}').replace('Fin 14', f'Fin {I.C}')
               .replace('1048576', str(I.base)))
    return imports + '''
/-!
# A run is its model

Generated by `generator/gen_chain.py`: every key, row and error of a run satisfying the generated
claim is the corresponding function of the model at its tape, up to the input of the readout LUT.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {tape : SampleTape}

''' + helpers + chain(I) + '\nend LwrProof\n'


def main():
    d = sys.argv[1]
    I = Info(d)
    path = os.path.join(d, 'LwrChainAll.lean')
    with open(path, 'w') as f:
        f.write(module(I))
    print('wrote', path)


if __name__ == '__main__':
    main()
