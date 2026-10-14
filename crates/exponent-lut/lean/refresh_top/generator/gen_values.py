#!/usr/bin/env python3
"""Generate `Gen_Online_Values.lean` (WP G): the instance certificates of the LUT member values of
the fresh noise, for `GenFinal.fresh_value_gen` (the mask is the caller's encoding of `σ_0 G`,
evaluated in its own stages).

Recognition (data flow, through `gen_final`): for each of the fresh carrier and the mask sampler,
the preprocess key member (`lut.key-member.carrier` / `.sampler`, the sampling LUT) and the
`lut.key-member.readout` feeding it, and their key-table loops; from each table scope its scale `a`
and inverse `b` (the two leading `matrixPolynomial` constants), its automorphism list `σ` and its
width `W`, the table modulus (the table output type).

Certificates (namespace `RefreshTopProof.GenOnlineValues`, `interval_cases i0 <;> decide` or
`decide`): `<lut>_inv : (b.toNat * W) % q = 1` (with `GenFinal.cast_inverse` the `hinverse` of
the value lemmas), `<lut>_odd : ∀ k, Odd σ_k`, `<lut>_perm` (an explicit equivalence by
`Equiv.ofBijective … (by decide)`) and `<lut>_index : σ_k mod 2W = 1 + 2 perm k`.

Usage: gen_values.py <param dir> --roles roles.json [--out FILE]
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xmodel  # noqa: E402
import gen_final as G  # noqa: E402
from xmodel import strip_parens  # noqa: E402


def table_of(pre, roles, member_role, count, which):
    rp = roles['Stage_preprocess']
    mem = [a for a in pre.inst.atoms if a.loop and a.loop.count == count and rp.get(a.loop.child, {}).get('role') == member_role]
    if len(mem) != 1:
        raise G.Fail(f'{member_role} x{count}: {len(mem)}')
    if which == 'readout':
        # the readout member feeding this member: a par loop whose outputs are arguments of `mem`
        need = {G.bare(x) for x in mem[0].loop.args} - {None}
        rd = [a for a in pre.inst.atoms if a.loop and G.outputs_of(a) & need and
              rp.get(a.loop.child, {}).get('role') == 'lut.key-member.readout']
        if len(rd) != 1:
            raise G.Fail('readout member')
        mem = rd
    need = {G.bare(x) for x in mem[0].loop.args} - {None}
    tabs = [a for a in pre.inst.atoms if a.loop and a.loop.kind == 'par' and G.outputs_of(a) & need and
            rp.get(a.loop.child, {}).get('role') not in ('lut.key-member.readout',)]
    tabs = [t for t in tabs if re.findall(r'matrixPolynomial \[', G.scope_text(pre.sc, t.loop.child))]
    if len(tabs) != 1:
        raise G.Fail(f'{which} table: {[t.loop.child for t in tabs]}')
    t = tabs[0]
    txt = G.scope_text(pre.sc, t.loop.child)
    polys = [strip_parens(x) for x in re.findall(r'matrixPolynomial \[([^\]]*)\]', txt)[:2]]
    sig = [strip_parens(x) for x in xmodel.split_top(
        re.search(r'ringAutomorphismRuns \(\(MxxIR\.selectInt \([^)]*\)\) \[([^\]]*)\]\)', txt).group(1), ', ')]
    tw = pre.sc[t.loop.child]
    otype = [w for w in tw.output][0] if tw.output else None
    mod = re.match(r'^Mxx\.Primitives\.ExactMatrix (.*) \d+ \d+ \d+$', otype.strip()).group(1)
    return dict(scope=t.loop.child, a=polys[0], b=polys[1], sigma=sig, W=len(sig), mod=strip_parens(mod))


def labels(gdir, R, roles, req_path):
    """The label payload constants (online `label.payload-const`: signs, exponents) and the
    requirement's label-exponent table (`famAt x0 n = selectInt n [...]`)."""
    _, on, pre = G.load_all(gdir)
    ro = roles['Stage_online_eval']
    sc = [a for a in on.inst.atoms if a.loop and ro.get(a.loop.child, {}).get('role') in ('label.payload', 'label.payload-const')]
    txt = G.scope_text(on.sc, sc[0].loop.child)
    lists = [[strip_parens(x) for x in xmodel.split_top(m.group(1), ', ')]
             for m in re.finditer(r'selectInt \(\(Int\.ofNat i_\d\)\) \[([^\]]*)\]', txt)]
    signs, exps = lists[0], lists[1]
    rt = open(req_path).read()
    m = re.search(r'\(famAt x\d+ n\d+\) = \(MxxIR\.selectInt \(Int\.ofNat n\d+\) \[([^\]]*)\]\)', rt)
    a = [x.strip() for x in m.group(1).split(',')]
    per = re.search(r'\) % (\d+)\) / (\d+)\)', rt)
    nprime = int(per.group(2))
    return dict(signs=signs, exps=exps, a=a, s0=R['n'] // nprime, K=len(signs))


def emit(R, luts, lab=None):
    q = G.q
    SS = R['S']
    lines = []
    for name, L in luts.items():
        qp = lambda t: '(' + q(t, 'Stage_preprocess') + ')'
        W = L['W']
        sig = '[' + ', '.join(L['sigma']) + ']'
        b, mod = qp(L['b']), qp(L['mod'])
        dep = 'i0' in b or 'i0' in mod
        I = f'(i0 : Nat) [h : Fact (i0 < {SS})] ' if dep else ''
        pre = '  have := h.out\n  interval_cases i0 <;> decide' if dep else '  decide'
        lines.append(f"""set_option maxRecDepth 65536 in
set_option maxHeartbeats 4000000 in
/-- `{L['scope']}`: the inverse of its width. -/
theorem {name}_inv {I}: (({b} : Int)).toNat * {W} % {mod} = 1 := by
{pre}

/-- `{L['scope']}`: its automorphisms are odd. -/
theorem {name}_odd : ∀ k : Fin {W}, Odd (MxxIR.selectInt (Int.ofNat (k : Nat)) ({sig} : List Int)).toNat := by
  decide

/-- `{L['scope']}`: the automorphism of branch `k` reads bin `perm k`. -/
def {name}_permFun (k : Fin {W}) : Fin {W} :=
  ⟨((MxxIR.selectInt (Int.ofNat (k : Nat)) ({sig} : List Int)).toNat % (2 * {W}) - 1) / 2, by omega⟩

set_option maxRecDepth 65536 in
theorem {name}_perm_bij : Function.Bijective {name}_permFun := by
  decide

noncomputable def {name}_perm : Fin {W} ≃ Fin {W} := Equiv.ofBijective _ {name}_perm_bij

theorem {name}_index : ∀ k : Fin {W}, (MxxIR.selectInt (Int.ofNat (k : Nat)) ({sig} : List Int)).toNat % (2 * {W}) =
    1 + 2 * 1 * ({name}_perm k).val := by
  intro k
  show _ = 1 + 2 * 1 * ({name}_permFun k).val
  revert k
  decide
""")
    if lab is not None:
        K, n, s0 = lab['K'], R['n'], lab['s0']
        # each entry typed (`(x : Int)`): an untyped 1025-entry literal with `-1`s spends ~25 s in
        # typeclass inference; the same terms elaborate in ~0.2 s
        L = lambda l: '[' + ', '.join(f'({x} : Int)' for x in l) + ']'
        lines.append(f"""set_option maxRecDepth 65536 in
/-- The label payloads `c_k X^(e_k)` are `X^(s₀ a_k)` (`GenFinal.message_Xe`), `a` the requirement's
label exponents; checked by the linear `GenFinal.labelOk` (`GenFinal.label_rel_of`), not by
enumerating `Fin K` (out of memory at `K = 1025`). -/
theorem label_rel : ∀ k : Fin {K},
    (MxxIR.selectInt (Int.ofNat (k : Nat)) ({L(lab['signs'])} : List Int) = 1 ∧
      MxxIR.selectInt (Int.ofNat (k : Nat)) ({L(lab['exps'])} : List Int) = {s0} * MxxIR.selectInt (Int.ofNat (k : Nat)) ({L(lab['a'])} : List Int)) ∨
    (MxxIR.selectInt (Int.ofNat (k : Nat)) ({L(lab['signs'])} : List Int) = -1 ∧
      MxxIR.selectInt (Int.ofNat (k : Nat)) ({L(lab['exps'])} : List Int) + {n} = {s0} * MxxIR.selectInt (Int.ofNat (k : Nat)) ({L(lab['a'])} : List Int)) :=
  RefreshTopProof.GenFinal.label_rel_of {K} (by decide) (by decide) (by decide) (by decide)
""")
    return """import Stage_preprocess
import GenFinalValues
import Mathlib.Tactic.IntervalCases

/-! Generated by `generator/gen_values.py`: the instance certificates of the LUT member values
(fresh noise and mask: readout and sampling LUTs) for `GenFinal.fresh_value_gen` /
`GenFinal.mask_value_gen`; tables """ + ', '.join(f"`{k}` = `{v['scope']}`" for k, v in luts.items()) + """. -/

namespace RefreshTopProof.GenOnlineValues

""" + '\n'.join(lines) + '\nend RefreshTopProof.GenOnlineValues\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('d')
    ap.add_argument('--roles', required=True)
    ap.add_argument('--out')
    ap.add_argument('--requirement')
    a = ap.parse_args()
    gdir = os.path.join(a.d, 'generated')
    R = G.recognize(gdir)
    roles = json.load(open(a.roles))
    _, on, pre = G.load_all(gdir)
    n, M = R['n'], R['n'] * R['m']
    luts = {
        'fresh_sampler': table_of(pre, roles, 'lut.key-member.carrier', M, 'sampler'),
        'fresh_readout': table_of(pre, roles, 'lut.key-member.carrier', M, 'readout'),
    }
    out = a.out or os.path.join(a.d, 'Gen_Online_Values.lean')
    lab = labels(gdir, R, roles, a.requirement or os.path.join(a.d, 'Gen_Requirement.lean'))
    print('labels', lab)
    open(out, 'w').write(emit(R, luts, lab))
    for k, v in luts.items():
        print(k, v['scope'], 'W', v['W'], 'b', v['b'][:40], 'mod', v['mod'][:40])
    print('wrote', out)


if __name__ == '__main__':
    try:
        main()
    except G.Fail as e:
        sys.exit(f'gen_values.py: recognition failed: {e}')
