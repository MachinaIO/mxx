#!/usr/bin/env python3
"""Write `Dio2InitialRows.lean`: every root row of the initial step input is a `RootRow`.

The initial stage (`Stage_initial`) encrypts every root row as
`c_j = (s K_(0j) + 1 K_(1j)) - T_j G + (u_j - u'_j)` (`RnsBggRoot::encode`). For every root object
member (a scalar row output, or a member of a family output assembled by `familyPack`), this
script finds, from the stage's `let` chain and constraint atoms, the key blocks (hash samples of
tag component `2 k + j`), the target wires `T_j`, the gadget `G` and the uniform bits `u_j, u'_j`
of both ports, and writes the theorem

    initial_row_<object>_<member> : ∃ G, gadgetMatrixRuns … G ∧ T_0 = … ∧ T_1 = … ∧
      RootRow model key prefix comps sm G ![T_0, T_1] ![c_0, c_1]

whose proof is the one of `Dio2Initial.initial_private`, with the targets stated as the stage's
own wires (`generatedRoot.w_<t> ⟨inputs, witness⟩` is not available outside the proof, so a
target is the term the wire's `let` names, expanded down to inputs, constants and witnesses that
the theorem quantifies).

Usage: gen_initial.py <generated dir> <out file>
"""
import json
import re
import sys

gdir, out_file = sys.argv[1], sys.argv[2]
stage = open(f'{gdir}/Stage_initial.lean').read()
names = json.load(open(f'{gdir}/stages.json'))['initial']['outputs']

# The root body's `let` definitions and the constraint atoms.
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
uniform = set(re.findall(r'MxxRuntime\.uniformIntervalSample \(0\) \(1\) (w_\d+_\d+)', stage))
gadgets = set(re.findall(r'MxxRuntime\.gadgetMatrixRuns backend 4096 25 (w_\d+_\d+)', stage))
# The constraint fragment (`generatedRoot.constraints_<k>`, the facts' `c<k>`) of every atom output.
fragment = {}
for m in re.finditer(r'^abbrev generatedRoot\.constraints_(\d+) .*?:=\n(.*?)(?=^abbrev |^def |^structure |^set_option )',
                     stage, re.M | re.S):
    for w in re.findall(r'(w_\d+_\d+) ∧$|(w_\d+_\d+)$', m.group(2), re.M):
        fragment[w[0] or w[1]] = int(m.group(1))


def fact(wire):
    """The facts field of the atom whose output is `wire`."""
    return f'F.c{fragment[wire]}.h{wire[2:]}'


def app(term, head):
    """The two arguments of `MxxRuntime.<head> a b`."""
    m = re.fullmatch(rf'MxxRuntime\.{head} (w_\d+_\d+) (w_\d+_\d+)', term)
    if not m:
        raise ValueError(f'{term!r} is not {head}')
    return m.group(1), m.group(2)


def row_parts(c):
    """`(K0, K1, T, G, u, u')` of a row wire `c`."""
    a, b = app(lets[c], 'matrixAdd')
    p, q = app(lets[a], 'matrixSub')
    k0s, k1s = app(lets[p], 'matrixAdd')
    mask, k0 = app(lets[k0s], 'matrixMulScalarLeft')
    one, k1 = app(lets[k1s], 'matrixMulScalarLeft')
    assert mask == 'w_0_0' and one == 'w_1_0', (c, mask, one)
    t, g = app(lets[q], 'matrixMulScalarLeft')
    u, u2 = app(lets[b], 'matrixSub')
    assert g in gadgets and u in uniform and u2 in uniform, c
    return k0, k1, t, g, u, u2


def component(tag_components):
    m = re.search(r'\.u64Le \((\d+)\)\]?$', tag_components.strip())
    return int(m.group(1))


def strip_component(tag_components):
    """The components before `"component", index`."""
    i = tag_components.rfind('.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116]')
    return tag_components[:i].rstrip(', ')


# Root objects: (object name, member, row wire of port 0, of port 1).
objects = {}
for name, wire in zip(names, out_wires):
    m = re.fullmatch(r'rns-bgg/root-(row|helper)/(.+)\.([01])', name)
    if not m:
        continue
    obj, port = f'{m.group(1)}:{m.group(2)}', int(m.group(3))
    members = packs.get(wire, [wire])
    for i, w in enumerate(members):
        objects.setdefault((obj, i), [None, None])[port] = w

theorems = []
for (obj, member), (c0, c1) in sorted(objects.items()):
    parts = [row_parts(c0), row_parts(c1)]
    keys = {}
    for j, (k0, k1, *_rest) in enumerate(parts):
        for k, kw in enumerate((k0, k1)):
            pre, comps, key = hashes[kw]
            assert key == 'w_9_0', kw
            assert component(comps) == 2 * k + j, (obj, member, kw, comps)
            keys[(k, j)] = (pre, strip_component(comps), kw)
    pres = {(p, c) for p, c, _ in keys.values()}
    assert len(pres) == 1, (obj, member, pres)
    pre, comps = pres.pop()
    # Integer components name `let` constants of the stage (an automorphism index).
    comps = re.sub(r'w_\d+_\d+', lambda m: re.fullmatch(r'\(?-?\d+\)?', lets[m.group(0)]) and lets[m.group(0)] or m.group(0), comps)
    assert not re.search(r'w_\d+_\d+', comps), comps
    ident = re.sub(r'[^A-Za-z0-9]+', '_', obj).strip('_') + f'_{member}'
    theorems.append((ident, obj, member, c0, c1, pre, comps, keys, parts))

out = ['''import W_Stage_initial_Facts
import Dio2Inv

/-! Generated by `crates/io/lean/dio2/generator/gen_initial.py`: every root row of the initial step
input is a `RootRow` (see `Dio2Initial`). -/

set_option maxHeartbeats 4000000
set_option maxRecDepth 16384

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut W_Stage_initial
''']
for ident, obj, member, c0, c1, pre, comps, keys, parts in theorems:
    (_, _, t0, g0, u0, v0), (_, _, t1, g1, u1, v1) = parts
    w = lambda x: f'(generatedRoot.{x} ctx)'
    out.append(f'''
/-- `{obj}` member {member}. -/
theorem initial_row_{ident} {{backend : BackendContext}} {{model : HashModel}} {{params : Params}}
    {{inputs : _}} {{witness : generatedRoot.Witness}} {{outputs : _}}
    (hbody : generatedRoot.body backend model params inputs outputs witness)
    {{sm : ErrorPoly 8}} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm) :
    let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
    gadgetMatrixRuns backend 4096 25 {w(g0)} ∧
      RootRow model inputs.2.2.1 [{pre}] [{comps}] sm {w(g0)}
        ![{w(t0)}, {w(t1)}] ![{w(c0)}, {w(c1)}] := by
  intro ctx
  have F := generatedRoot.facts ctx hbody
  have e00 : rootKey model inputs.2.2.1 [{pre}] [{comps}] modulus_0 25 0 0 = {w(keys[(0, 0)][2])} := by
    rw [hashSample_eq {fact(keys[(0, 0)][2])}]; rfl
  have e01 : rootKey model inputs.2.2.1 [{pre}] [{comps}] modulus_0 25 0 1 = {w(keys[(0, 1)][2])} := by
    rw [hashSample_eq {fact(keys[(0, 1)][2])}]; rfl
  have e10 : rootKey model inputs.2.2.1 [{pre}] [{comps}] modulus_0 25 1 0 = {w(keys[(1, 0)][2])} := by
    rw [hashSample_eq {fact(keys[(1, 0)][2])}]; rfl
  have e11 : rootKey model inputs.2.2.1 [{pre}] [{comps}] modulus_0 25 1 1 = {w(keys[(1, 1)][2])} := by
    rw [hashSample_eq {fact(keys[(1, 1)][2])}]; rfl
  have hG : gadgetMatrixRuns backend 4096 25 {w(g0)} := by
    exact {fact(g0)}
  refine ⟨hG, ?_⟩
  apply encode_rel hmask (u := ![{w(u0)}, {w(u1)}]) (u' := ![{w(v0)}, {w(v1)}])
  · intro j; fin_cases j
    · exact {fact(u0)}
    · exact {fact(u1)}
  · intro j; fin_cases j
    · exact {fact(v0)}
    · exact {fact(v1)}
  · intro j; fin_cases j
    · simp only [Fin.zero_eta, Fin.isValue, Matrix.cons_val_zero, e00, e10]
      rfl
    · simp only [Fin.mk_one, Fin.isValue, Matrix.cons_val_one, e01, e11]
      rfl
''')
out.append('\nend Dio2Proof\n')
open(out_file, 'w').write(''.join(out))
print(f'{len(theorems)} root rows')
