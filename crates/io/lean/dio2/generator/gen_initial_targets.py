#!/usr/bin/env python3
"""Write `Dio2InitialTargets.lean`: the targets of every root row of the initial step input.

`gen_initial.py` proves every root row of `Stage_initial` a `RootRow` with the targets `T_0, T_1`
stated as the stage's wires. This script evaluates those wires from the stage's `let` chain and
constraint atoms (constants, inputs, `matrixMulScalarLeft`, `multiplyMonomial`, ring
automorphisms of the mask, family reads of the label exponents) and writes, per root object member,

    initial_target_<object>_<member> : T_0 = fun _ _ ↦ reducePoly Q_0 8 (α * unitPair μ 0) ∧
                                       T_1 = fun _ _ ↦ reducePoly Q_0 8 (α * unitPair μ 1)
    initial_rel_<object>_<member>    : gadgetMatrixRuns … G ∧
      Rel sm (rootKey …) (fun j ↦ reducePoly Q_0 8 (α * unitPair μ j) • G) ![c_0, c_1] 1

with the plaintext `α` and the payload secret `μ` of the object:

| object | `α` | `μ` |
| --- | --- | --- |
| `Aut(k)`, `ExpLayout` member of tag `k` | `1` | `σ_k(s̄)` (`substitution k`) |
| label member with exponent `e` | `message 1 e` (`X^e`) | `s̄` |
| `label-0` member 1 | `1` | `s̄` |
| `one-private` | `1` | `t̄` |
| `state` | `message 1 (z_0 * 2)` | `t̄` |

Usage: gen_initial_targets.py <generated dir> <out file>
"""
import json
import re
import sys

gdir, out_file = sys.argv[1], sys.argv[2]
stage = open(f'{gdir}/Stage_initial.lean').read()
names = json.load(open(f'{gdir}/stages.json'))['initial']['outputs']

lets = {}
for m in re.finditer(r'^  let (w_\d+_\d+)(?: : [^:=]+)? := (.+)$', stage, re.M):
    lets[m.group(1)] = m.group(2).strip()
outputs = re.search(r'^  outputs = \((.*)\)$', stage, re.M).group(1)
out_wires = re.findall(r'w_\d+_\d+', outputs)
assert len(out_wires) == len(names), (len(out_wires), len(names))
packs = {}
for m in re.finditer(r'MxxRuntime\.familyPack \((\d+)\) \[([^\]]*)\] (w_\d+_\d+)', stage):
    packs[m.group(3)] = [w.strip() for w in m.group(2).split(',')]
hashes = {}
for m in re.finditer(r'MxxRuntime\.hashSample \(hashModel\) \(\[([^\]]*)\]\) \(\[(.*?)\]\) \((w_\d+_\d+)\) (w_\d+_\d+)', stage):
    hashes[m.group(4)] = (m.group(1), m.group(2), m.group(3))
gadgets = set(re.findall(r'MxxRuntime\.gadgetMatrixRuns backend 4096 25 (w_\d+_\d+)', stage))
auts = {m.group(3): (int(m.group(1)), m.group(2)) for m in re.finditer(
    r'MxxRuntime\.ringAutomorphismRuns \((\d+)\) (w_\d+_\d+) (w_\d+_\d+)', stage)}
gets = {m.group(3): (m.group(1), int(m.group(2))) for m in re.finditer(
    r'MxxRuntime\.familyGetStatic (w_\d+_\d+) \((\d+)\) (w_\d+_\d+)', stage)}
fragment = {}
for m in re.finditer(r'^abbrev generatedRoot\.constraints_(\d+) .*?:=\n(.*?)(?=^abbrev |^def |^structure |^set_option )',
                     stage, re.M | re.S):
    for w in re.findall(r'(w_\d+_\d+) ∧$|(w_\d+_\d+)$', m.group(2), re.M):
        fragment[w[0] or w[1]] = int(m.group(1))
# The stage inputs (`inputs.1`, `inputs.2.1`, ...).
inputs = {m.group(1): m.group(2) for m in re.finditer(r'^  let (w_\d+_\d+) : [^:=]+ := (inputs[.0-9]*)$', stage, re.M)}
MASK, PAYLOAD, LABELS, Z = 'inputs.1', 'inputs.2.2.2.2.1', 'inputs.2.2.2.1', 'inputs.2.2.2.2.2.1'


def fact(wire):
    return f'F.c{fragment[wire]}.h{wire[2:]}'


def app(term, head):
    m = re.fullmatch(rf'MxxRuntime\.{head} (w_\d+_\d+) (w_\d+_\d+)', term)
    if not m:
        raise ValueError(f'{term!r} is not {head}')
    return m.group(1), m.group(2)


def row_parts(c):
    a, b = app(lets[c], 'matrixAdd')
    p, q = app(lets[a], 'matrixSub')
    k0s, k1s = app(lets[p], 'matrixAdd')
    _, k0 = app(lets[k0s], 'matrixMulScalarLeft')
    _, k1 = app(lets[k1s], 'matrixMulScalarLeft')
    t, g = app(lets[q], 'matrixMulScalarLeft')
    assert g in gadgets, c
    return k0, k1, t, g


def strip_component(tag_components):
    i = tag_components.rfind('.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116]')
    return tag_components[:i].rstrip(', ')


def W(x):
    return f'(generatedRoot.{x} ctx)'


class Proof:
    """`have` lines evaluating wires: an integer wire to an integer term, a `1 × 1` matrix wire
    to `reducePoly Q_0 8 x` at its entry."""

    def __init__(self):
        self.lines, self.ints, self.polys = [], {}, {}

    def int(self, w):
        if w in self.ints:
            return self.ints[w]
        term = lets.get(w)
        if w in inputs:
            val = inputs[w]
            prf = 'rfl'
        elif term is not None and re.fullmatch(r'\(?-?\d+\)?', term):
            val = term.strip('()')
            prf = 'rfl'
        elif term is not None and re.fullmatch(r'\((w_\d+_\d+) \* (w_\d+_\d+)\)', term):
            a, b = re.fullmatch(r'\((w_\d+_\d+) \* (w_\d+_\d+)\)', term).groups()
            va, vb = self.int(a), self.int(b)
            val = f'({va} * {vb})'
            prf = f'congrArg₂ (· * ·) I_{a} I_{b}'
        elif w in gets:
            fam, k = gets[w]
            assert inputs.get(fam) == LABELS, (w, fam)
            val = f'{LABELS} {k}'
            prf = f'getStatic_eq (k := {k}) (by decide) {fact(w)}'
        else:
            raise ValueError(f'int wire {w}: {term!r}')
        self.lines.append(f'  have I_{w} : {W(w)} = {val} := {prf}')
        self.ints[w] = val
        return val

    def poly(self, w):
        if w in self.polys:
            return self.polys[w]
        term = lets.get(w)
        if w in inputs:
            if inputs[w] == MASK:
                val, prf = 'sm', 'hmask'
            elif inputs[w] == PAYLOAD:
                val, prf = 'tp', 'hpay'
            else:
                raise ValueError(w)
        elif term == '(1 : mxxMat3)':
            val, prf = '1', 'one_entry'
        elif term is not None and term.startswith('MxxRuntime.matrixMulScalarLeft'):
            a, b = app(term, 'matrixMulScalarLeft')
            va, vb = self.poly(a), self.poly(b)
            val, prf = f'({va} * {vb})', f'msl_entry P_{a} P_{b}'
        elif term is not None and term.startswith('MxxRuntime.multiplyMonomial'):
            a, e = app(term, 'multiplyMonomial')
            va, ve = self.poly(a), self.int(e)
            val = f'({va} * message 1 ({ve}))'
            prf = f'(mono_entry P_{a} _).trans (by rw [I_{e}])'
        elif w in auts:
            k, src = auts[w]
            vs = self.poly(src)
            val = f'(substitution {k} (⟨{(k - 1) // 2}, rfl⟩ : Odd {k}) {vs})'
            prf = f'aut_entry (k := {k}) _ {fact(w)} P_{src}'
        else:
            raise ValueError(f'poly wire {w}: {term!r}')
        self.lines.append(f'  have P_{w} : {W(w)} 0 0 = reducePoly modulus_0 8 {val} := {prf}')
        self.polys[w] = val
        return val


objects = {}
for name, wire in zip(names, out_wires):
    m = re.fullmatch(r'rns-bgg/root-(row|helper)/(.+)\.([01])', name)
    if not m:
        continue
    obj, port = f'{m.group(1)}:{m.group(2)}', int(m.group(3))
    for i, w in enumerate(packs.get(wire, [wire])):
        objects.setdefault((obj, i), [None, None])[port] = w


def classify(t0, t1):
    """`(α, μ, payload?)` of the targets, read off the stage's terms."""
    # T_1 = α · 1 (times 1 again for the private rows); T_0 = α · μ or σ_k(s̄) for helpers.
    if t0 in auts:
        k, src = auts[t0]
        assert inputs.get(src) == 'inputs.1' and lets.get(t1) == '(1 : mxxMat3)'
        return '1', f'(substitution {k} (⟨{(k - 1) // 2}, rfl⟩ : Odd {k}) sm)', False
    a, x = app(lets[t0], 'matrixMulScalarLeft')
    a1, one = app(lets[t1], 'matrixMulScalarLeft')
    assert a1 == a, (t0, t1)
    secret = inputs[x]
    mu, payload = ('sm', False) if secret == MASK else ('tp', True)
    if lets[a] == '(1 : mxxMat3)':
        return '1', mu, payload
    base, e = app(lets[a], 'multiplyMonomial')
    assert lets[base] == '(1 : mxxMat3)'
    m = re.fullmatch(r'\((w_\d+_\d+) \* (w_\d+_\d+)\)', lets[e])
    x0, factor = m.groups()
    factor = lets[factor].strip('()')
    if x0 in gets:
        exp = f'{LABELS} {gets[x0][1]}'
    else:
        exp = inputs[x0]
    exp = exp if factor == '1' else f'{exp} * {factor}'
    return f'message 1 ({exp})', mu, payload


out = ['''import W_Stage_initial_Facts
import Dio2InitialRows
import ExponentRuntime

/-! Generated by `crates/io/lean/dio2/generator/gen_initial_targets.py`: the targets of every root
row of the initial step input (`Dio2InitialRows`) in closed form.

Every root row encodes `(α μ_0, α)` against the gadget: the target of port `j` is
`reducePoly Q_0 8 (α * unitPair μ j)`, with

* `Aut(k)` and the `ExpLayout` members (the `Aut(k)` rows of their tags `k`): `α = 1`,
  `μ = σ_k(s̄)` (`substitution k`; the stage computes the automorphism of the mask);
* the members of `label` and member 0 of `label-0`: `α = message 1 e = X^e` with `e` the label
  exponent (the stage's fourth input), `μ = s̄`; member 1 of `label-0`: `α = 1`, `μ = s̄`;
* `one-private`: `α = 1`, `μ = t̄` (the stage's fifth input, `hpay`);
* `state`: `α = message 1 (z_0 * 2) = X^(2 z_0)` (`mono(z_0)`, the sixth input), `μ = t̄`.

`initial_rel_*` restates the `RootRow` of `Dio2InitialRows` in the form
`Rel s̄ K (fun j ↦ reducePoly Q_0 8 (α * unitPair μ j) • G) c 1` of `mul_public`/`cell_flat`. -/

set_option maxHeartbeats 4000000
set_option maxRecDepth 16384
set_option linter.unusedVariables false

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut W_Stage_initial

section Entries

variable {q n : Nat}

/-- A `1 × 1` matrix is the constant of its entry. -/
theorem entry_ext {M : ExactMatrix q n 1 1} {p : ExactPoly q n} (h : M 0 0 = p) :
    M = fun _ _ ↦ p := by
  funext i j
  rw [Subsingleton.elim i 0, Subsingleton.elim j 0, h]

theorem one_entry : (1 : ExactMatrix q n 1 1) 0 0 = reducePoly q n 1 := by
  rw [map_one]
  rfl

theorem msl_entry {M N : ExactMatrix q n 1 1} {a b : ErrorPoly n}
    (hM : M 0 0 = reducePoly q n a) (hN : N 0 0 = reducePoly q n b) :
    matrixMulScalarLeft M N 0 0 = reducePoly q n (a * b) := by
  simp only [matrixMulScalarLeft, hM, hN, map_mul]

theorem mono_entry {M : ExactMatrix q n 1 1} {a : ErrorPoly n} (hM : M 0 0 = reducePoly q n a)
    (e : Int) : multiplyMonomial M e 0 0 = reducePoly q n (a * message 1 e) := by
  have h := congrFun (congrFun (multiplyMonomial_liftInteger (q := q) (n := n) 1 e) 0) 0
  simp only [multiplyMonomial, liftInteger, Int.cast_one, one_mul] at h
  simp only [multiplyMonomial, hM, map_mul, h]

theorem substitution_congr {R : Type} [CommRing R] {a b : Nat} (h : a = b) (ha : Odd a)
    (hb : Odd b) (x : Negacyclic n R) : substitution a ha x = substitution b hb x := by
  subst h
  rfl

theorem aut_entry (hq : 1 < q) (hn : 0 < n) {k : Nat} (hk : Odd k) {M out : ExactMatrix q n 1 1}
    {a : ErrorPoly n} (h : ringAutomorphismRuns (k : Int) M out) (hM : M 0 0 = reducePoly q n a) :
    out 0 0 = reducePoly q n (substitution k hk a) := by
  obtain ⟨hodd, hout⟩ := ringAutomorphismRuns_eq hq hn h
  rw [hout]
  simp only [hM, ← reducePoly_substitution]
  rw [substitution_congr (Int.toNat_natCast k) hodd hk]

theorem getStatic_eq {α : Type} {c k : Nat} {f : Fin c → α} {out : α} (hk : k < c)
    (h : familyGetStatic f (k : Int) out) : out = f ⟨k, hk⟩ := by
  obtain ⟨hk', hout⟩ := familyGet_eq h
  exact hout

theorem msl_const {m : Nat} (p : ExactPoly q n) (G : ExactMatrix q n 1 m) :
    matrixMulScalarLeft (fun _ _ ↦ p) G = p • G := by
  funext i j
  rfl

end Entries

theorem one_lt_modulus_0 : 1 < modulus_0 := by
  unfold modulus_0
  norm_num
''']

for (obj, member), (c0, c1) in sorted(objects.items()):
    parts = [row_parts(c0), row_parts(c1)]
    keys = {}
    for j, (k0, k1, *_r) in enumerate(parts):
        for k, kw in enumerate((k0, k1)):
            pre, comps, key = hashes[kw]
            keys[(k, j)] = (pre, strip_component(comps))
    (pre, comps), = set(keys.values())
    comps = re.sub(r'w_\d+_\d+', lambda m: lets[m.group(0)], comps)
    assert not re.search(r'w_\d+_\d+', comps), comps
    ident = re.sub(r'[^A-Za-z0-9]+', '_', obj).strip('_') + f'_{member}'
    (_, _, t0, g), (_, _, t1, _) = parts
    alpha, mu, payload = classify(t0, t1)
    pf = Proof()
    v0, v1 = pf.poly(t0), pf.poly(t1)
    pay_hyp = ('\n    {tp : ErrorPoly 8} (hpay : inputs.2.2.2.2.1 0 0 = reducePoly modulus_0 8 tp)'
               if payload else '')
    pay_arg = ' hpay' if payload else ''
    hyps = f'''{{backend : BackendContext}} {{model : HashModel}} {{params : Params}}
    {{inputs : _}} {{witness : generatedRoot.Witness}} {{outputs : _}}
    (hbody : generatedRoot.body backend model params inputs outputs witness)
    {{sm : ErrorPoly 8}} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm){pay_hyp}'''
    target = lambda j: f'(fun _ _ ↦ reducePoly modulus_0 8 ({alpha} * unitPair {mu} {j}))'
    body = '\n'.join(pf.lines).replace('aut_entry (k', 'aut_entry one_lt_modulus_0 (by norm_num) (k')
    out.append(f'''
/-- The targets of `{obj}` member {member}. -/
theorem initial_target_{ident} {hyps} :
    let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
    {W(t0)} = {target(0)} ∧
      {W(t1)} = {target(1)} := by
  intro ctx
  have F := generatedRoot.facts ctx hbody
{body}
  refine ⟨entry_ext (P_{t0}.trans ?_), entry_ext (P_{t1}.trans ?_)⟩ <;>
    simp [unitPair]

/-- `{obj}` member {member} as an encoding of `α (μ, 1) G`. -/
theorem initial_rel_{ident} {hyps} :
    let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
    gadgetMatrixRuns backend 4096 25 {W(g)} ∧
      Rel sm (rootKey model inputs.2.2.1 [{pre}] [{comps}] modulus_0 25)
        (fun j ↦ reducePoly modulus_0 8 ({alpha} * unitPair {mu} j) • {W(g)})
        ![{W(c0)}, {W(c1)}] 1 := by
  intro ctx
  obtain ⟨hG, hrow⟩ := initial_row_{ident} hbody hmask
  obtain ⟨h0, h1⟩ := initial_target_{ident} hbody hmask{pay_arg}
  refine ⟨hG, hrow.target_eq ?_⟩
  funext j
  fin_cases j
  · exact (congrArg (matrixMulScalarLeft · _) h0).trans (msl_const _ _)
  · exact (congrArg (matrixMulScalarLeft · _) h1).trans (msl_const _ _)
''')
out.append('\nend Dio2Proof\n')
open(out_file, 'w').write(''.join(out))
print(f'{len(objects)} root objects')
