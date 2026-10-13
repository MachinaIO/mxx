"""Write the decode-stage theorem of parameter directory `d` from the generic lemmas of
`TopCrtGeneric.lean`: the decode stage of the refresh of every CRT slot outputs zero when, for each
row, the residue of every slot `q_c` is that of the expected row plus one small error row (shared by
the slots), modulo the decoding modulus `Q`.

Unlike `gen_decode_crt.py`, nothing is fixed to the toy parameter set: the number of rows (hash
samples), CRT slots, columns, decoded coefficients and σ coefficients, the slot primes, the CRT
basis, the decoding modulus and the gadget are all read from `generated/Stage_decode.lean`. The
proof instantiates `crt_matrix_gen` (the slots' recombination), `sigmaFold` (σ) and
`column_zero_gen` (each decode); the instance's CRT data is checked by `decide` once.

The statement names the instance's moduli (`--Q`, default `Q0`; `--gadget-Q`, default `Q1`; `--N`,
default `N`) and the definitions `topValue`, `expectedRow` and `topTagI` of the parameter
directory; it checks that the names denote the stage's numbers. The residue hypotheses use
`ResidueAt`, which a parameter set's own `Residue` unfolds to.

Usage: `python3 gen_decode_crt_generic.py <parameter directory> [--out FILE] [--name THEOREM]
[--import MODULE ...] [--Q NAME] [--gadget-Q NAME] [--N NAME]`. The default output is
`<d>/TopDecodeCrtGeneric.lean` with theorem `decode_zero_generic`, importing `TopCrtGeneric` and
`TopAssembly`.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from patterns import destruct, parse

ap = argparse.ArgumentParser()
ap.add_argument('d')
ap.add_argument('--out')
ap.add_argument('--name', default='decode_zero_generic')
ap.add_argument('--import', dest='imports', action='append')
ap.add_argument('--Q', default='Q0')
ap.add_argument('--gadget-Q', dest='gadget_q', default='Q1')
ap.add_argument('--N', default='N')
a = ap.parse_args()
d = a.d
out_path = a.out or f'{d}/TopDecodeCrtGeneric.lean'
imports = a.imports or ['TopCrtGeneric', 'TopAssembly']
thm = a.name
prefix = re.sub(r'\W', '_', thm)

path = f'{d}/generated/Stage_decode.lean'
text = open(path).read()
lets, names, atoms = parse(path, 'generatedRoot')
atom = dict(zip(names, atoms))
body = text[text.index('abbrev generatedRoot.body'):]
defs = dict(re.findall(r'^  let (w_\S+) : [^\n]*?:= (.*)$', body, re.M))
defs.update(re.findall(r'^  let (w_\S+) := (witness\.\S+)$', body, re.M))
types = dict(re.findall(r'\((w_\d+_\d+) : (Mxx\.Primitives\.ExactMatrix \d+ \d+ \d+ \d+)\)', text))
types.update(re.findall(r'^  let (w_\S+) : ([^\n]*?) :=', body, re.M))
w = lambda h: 'w_' + h[1:]
named = [n for n in names if n != '-']
kind = lambda prefix: [n for n in named if atom[n].startswith('MxxRuntime.' + prefix)]
MAT = r'Mxx\.Primitives\.ExactMatrix (\d+) (\d+) (\d+) (\d+)'


def deps(x, stop):
    """The let-bound wires `x` is computed from (`x` included), down to the witnesses, inputs and
    `stop`."""
    out, todo = [], [x]
    while todo:
        y = todo.pop()
        if y in out or y in stop or y not in defs or 'witness' in defs[y] or 'inputs' in defs[y]:
            continue
        out.append(y)
        todo += re.findall(r'\bw_\d+_\d+\b', defs[y])
    return out


def op(x, name):
    """The arguments of the let `x := MxxRuntime.name a b`, or `None`."""
    m = re.match(rf'MxxRuntime\.{name} (w_\S+) (w_\S+)$', defs.get(x, ''))
    return (m.group(1), m.group(2)) if m else None


def find(name, k, y):
    """The lets `x := MxxRuntime.name a b` whose argument `k` is `y`."""
    return [x for x in defs if op(x, name) and op(x, name)[k] == y]


def one(xs, what):
    assert len(xs) == 1, (what, xs)
    return xs[0]


def mat(x):
    """`(q, n, rows, columns)` of a matrix-typed let."""
    return tuple(int(v) for v in re.match(MAT, types[x]).groups())


# The gadget, its centered rebase and the slot lifts.
gadget = one(kind('gadgetMatrixRuns'), 'gadget')
gbase, gcols = re.match(r'MxxRuntime\.gadgetMatrixRuns backend (\d+) (\d+) ', atom[gadget]).groups()
rebases = kind('centeredRebaseRuns w_')
rebase_args = {r: re.match(r'MxxRuntime\.centeredRebaseRuns (w_\S+) (w_\S+)$', atom[r].rstrip(' ∧')).groups()
               for r in rebases}
grebase = one([r for r in rebases if rebase_args[r][0] == w(gadget)], 'gadget rebase')
lifts = {rebase_args[r][1]: r for r in rebases if r != grebase}
hashes = kind('hashSample')
slices = kind('sliceMatrix')
decodes = kind('thresholdDecode')
slice_args = {s: re.match(r'MxxRuntime\.sliceMatrix (w_\S+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (w_\S+)',
                          atom[s]).groups() for s in slices}
inputs = [n for n in lets if 'inputs' in defs.get(n, '')]
key = one([n for n in inputs if types[n].strip() == 'ByteArray'], 'key')

# The σ family and its fold.
fams = sorted((int(re.match(r'MxxRuntime\.familyGetStatic \S+ \((\d+)\)', atom[n]).group(1)), n)
              for n in named if atom[n].startswith('MxxRuntime.familyGetStatic '))
sigma = one(sorted({re.match(r'MxxRuntime\.familyGetStatic (\S+) ', atom[n]).group(1) for _, n in fams}), 'σ')
count = int(re.match(r'Fin (\d+) → Int$', types[sigma].strip()).group(1))
assert [k for k, _ in fams] == list(range(count)), 'σ reads every coefficient once'

rows = []
for h in hashes:
    tag = int(re.findall(r'\.u64Le \((\d+)\)', atom[h])[-1])
    a_rows = sorted((int(slice_args[s][1]), s) for s in slices if slice_args[s][0] == w(h))
    assert [r for r, _ in a_rows] == [0, 1], (h, a_rows)
    r0, r1 = (s for _, s in a_rows)
    scaled = one(find('matrixMulScalarLeft', 1, w(r0)), 'σ A₀')
    sigma_wire = op(scaled, 'matrixMulScalarLeft')[0]
    added = one([x for x in find('matrixAdd', 0, scaled) if op(x, 'matrixAdd')[1] == w(r1)], 'σ A₀ + A₁')
    expected = one(find('matrixSub', 0, added), 'expected row')
    diff = one(find('matrixSub', 1, expected), 'difference')
    combined = op(diff, 'matrixSub')[0]
    # `Δ r X^e G'`: `((Δ r) (c X^e)) G'`.
    dr, mono = op(op(op(expected, 'matrixSub')[1], 'matrixMulScalarLeft')[0], 'matrixMulScalarLeft')
    r_wire = op(dr, 'matrixMulScalarLeft')[1]
    # `r` is an input, or a literal constant matrix.
    r = r_wire if 'inputs' in defs[r_wire] else re.match(r'\((-?\d+) :', defs[r_wire]).group(1)
    lifted, expo = re.match(r'MxxRuntime\.multiplyMonomial (w_\S+) (w_\S+)$', defs[mono]).groups()
    coef = re.match(r'MxxRuntime\.liftInteger (w_\S+)$', defs[lifted]).group(1)
    # The recombination `((l_0 K_0 + l_1 K_1) + …) + l_k K_k`, in slot order.
    terms, x = [], combined
    while op(x, 'matrixAdd'):
        x, t = op(x, 'matrixAdd')
        terms.append(t)
    terms.append(x)
    terms = terms[::-1]
    slot_lifts = [op(t, 'matrixMulScalarRight')[0] for t in terms]
    consts = [int(re.match(r'\(MxxRuntime\.matrixPolynomial \[\((-?\d+)\)\]', defs[op(t, 'matrixMulScalarRight')[1]])
                  .group(1)) for t in terms]
    residues = [rebase_args[lifts[l]][0] for l in slot_lifts]
    cols = sorted((int(slice_args[s][3]), s) for s in slices if slice_args[s][0] == diff)
    rows.append(dict(hash=h, tag=tag, r0=r0, r1=r1, sigma_wire=sigma_wire, expected=expected,
                     diff=diff, combined=combined, r=r, coef=coef, expo=expo,
                     lifts=slot_lifts, consts=consts, residues=residues, cols=[s for _, s in cols]))

R = len(rows)
k = len(rows[0]['lifts'])
primes = [mat(x)[0] for x in rows[0]['residues']]
Q, n, _, m = mat(rows[0]['combined'])
K = rows[0]['consts']
G_q = mat(w(gadget))[0]
assert all(len(rw['lifts']) == k and rw['consts'] == K and [mat(x)[0] for x in rw['residues']] == primes
           for rw in rows), 'every row recombines the same slots'
assert len({rw['sigma_wire'] for rw in rows}) == 1, 'one σ'
assert all(mat(x)[3] == m for rw in rows for x in rw['residues']) and int(gcols) == m
assert all(len(rw['cols']) == m for rw in rows), 'one column slice per column'
col_row = {s: j for j, rw in enumerate(rows) for s in rw['cols']}
dec_src = {dn: re.match(r'MxxRuntime\.thresholdDecode \(2\) \((\d+)\) \d+ (w_\S+)', atom[dn]).group(2)
           for dn in decodes}
assert all(any(w(s) == v for s in col_row) for v in dec_src.values()), 'every decode reads a column'
sigma_wire = rows[0]['sigma_wire']
outs = re.findall(r'\bw_\d+_\d+\b', atom['hout'])
print(f'{R} rows, {k} slots, {m} columns, {len(decodes)} decodes, {count} σ coefficients, Q = {Q}',
      file=sys.stderr)

vec = lambda xs: '![' + ', '.join(f'({x})' if str(x).startswith('-') else str(x) for x in xs) + ']'
P = prefix
L = [''.join(f'import {i}\n' for i in imports),
     f'/-! The decode stage of the refresh of every CRT slot, from the generic lemmas of `TopCrtGeneric`\n'
     f'(written by `generator/gen_decode_crt_generic.py`). -/\n',
     'set_option maxRecDepth 65536\n',
     'namespace RefreshTopProof\n',
     'open Mxx.Primitives MxxRuntime MxxExponentLut\n',
     f'/-! The stage\'s numbers: {R} rows, {k} slots, {m} columns, {len(decodes)} decoded coefficients, '
     f'{count} σ coefficients. -/\n',
     f'example : {a.Q} = {Q} := rfl',
     f'example : {a.gadget_q} = {G_q} := rfl',
     f'example : {a.N} = {n} := rfl\n',
     f'/-- The slots of the decoding modulus. -/\nabbrev {P}_q : Fin {k} → Nat := {vec(primes)}',
     f'/-- The decode stage\'s CRT basis. -/\nabbrev {P}_K : Fin {k} → Int := {vec(K)}\n',
     f'theorem {P}_basis : ∀ i j, {P}_K i ≡ (if i = j then 1 else 0) [ZMOD ({P}_q j : Int)] := by decide',
     f'theorem {P}_coprime : ∀ i j, i ≠ j → Nat.Coprime ({P}_q i) ({P}_q j) := by decide',
     f'theorem {P}_gt : ∀ i, 1 < {P}_q i := by decide',
     f'theorem {P}_prod : {a.Q} = ∏ i, {P}_q i := by\n'
     f'  simp only [Fin.prod_univ_succ, Fin.prod_univ_zero]\n  decide\n']
L.append(f"""/-- The decode stage outputs zero when, for each row `j`, every slot's residue is that of
`y_j + e'_j` for a small row `e'_j`. -/
theorem {thm} {{model : HashModel}} {{params : Stage_decode.Params}} {{inputs}} {{out : Int}}
    (h : Stage_decode.generatedRoot Backend.backend model params inputs out)
    {{G : ExactMatrix {a.gadget_q} {a.N} 1 {m}}} (hG : gadgetMatrixRuns Backend.backend {gbase} {m} G)
    (E : Fin {R} → ErrorMatrix {a.N} 1 {m}) (hE : ∀ j c i, 4 * |(E j 0 c).coeff i| + 4 < ({a.Q} : Int))""")
for j, rw in enumerate(rows):
    v = (f'(topValue model {defs[key]} (topTagI {rw["tag"]}) {defs[sigma]} {defs.get(rw["r"], rw["r"])} '
         f'{defs[rw["coef"]]} {defs[rw["expo"]]} G (E {j}))')
    for c in range(k):
        L.append(f'    (hres{j}_{c} : ResidueAt {defs[rw["residues"][c]]}\n      {v})')
L.append('    : out = 0 := by')
L.append(destruct(path, 'Stage_decode', 'generatedRoot', 'h').rstrip('\n'))
L.append(f'  have hQ1 : 1 < {a.Q} := by decide')
L.append(f'  have hQodd : {a.Q} % 2 = 1 := by decide')
L.append(f'  have hg : {w(gadget)} = G := gadgetMatrixRuns_unique {gadget} hG')
L.append(f'  have hσ : {sigma_wire} = sigmaFold (Q := {a.Q}) {count - 1} {sigma} := by')
for kk, f in fams:
    L.append(f'    obtain ⟨_, e{kk}⟩ := familyGet_eq (k := {kk}) {f}')
L.append(f'    simp only [{", ".join(deps(sigma_wire, set()))}]')
L.append(f'    rw [{", ".join(f"e{kk}" for kk, _ in fams)}]')
L.append('    rfl')
for j, rw in enumerate(rows):
    L.append(f'  have hA{j} := {rw["hash"]}.2.2')
    L.append(f'  have hr{j}0 := slice_row {rw["r0"]} rfl (by decide)')
    L.append(f'  have hr{j}1 := slice_row {rw["r1"]} rfl (by decide)')
    stop = {w(rw['r0']), w(rw['r1']), w(grebase), sigma_wire, w(rw['hash'])}
    L.append(f'  have he{j} : {rw["expected"]} = expectedRow (model.sample {a.Q} {a.N} 2 {m} {key}')
    L.append(f'      (completeHashTag (topTagI {rw["tag"]}) [])) (sigmaFold {count - 1} {sigma}) {rw["r"]} '
             f'{rw["coef"]} {rw["expo"]} G := by')
    L.append(f'    simp only [{", ".join(deps(rw["expected"], stop))}]')
    L.append(f'    rw [hσ, hr{j}0, hr{j}1, {grebase}.2, hg, hA{j}, topTagI, completeHashTag_append, List.append_nil]')
    L.append('    rfl')
    cases = ', '.join(f'liftOf_intro hres{j}_{c} {lifts[rw["lifts"][c]]}' for c in range(k))
    L.append(f'  have hc{j} : {rw["combined"]} = topValue model {key} (topTagI {rw["tag"]}) {sigma} {rw["r"]} '
             f'{rw["coef"]} {rw["expo"]} G (E {j}) := by')
    L.append(f'    simp only [{", ".join(deps(rw["combined"], set(rw["lifts"])))}]')
    L.append(f'    exact crt_matrix_gen (k := {k - 1}) (l := ![{", ".join(rw["lifts"])}]) hN')
    L.append(f'      {P}_q {P}_gt {P}_K {P}_basis {P}_coprime {P}_prod (fun i ↦ by')
    L.append('        fin_cases i')
    L.append(f'        exacts [{cases}])')
    L.append(f'  have hd{j} : {rw["diff"]} = reduceMatrix {a.Q} {a.N} 1 {m} (E {j}) := by')
    L.append(f'    simp only [{rw["diff"]}]')
    L.append(f'    rw [hc{j}, he{j}, topValue, matrixSub]')
    L.append('    exact add_sub_cancel_left _ _')
    for c in rw['cols']:
        L.append(f'  rw [hd{j}] at {c}')
zs = []
for dn in decodes:
    sl = next(s for s in col_row if w(s) == dec_src[dn])
    j = col_row[sl]
    L.append(f'  have z{dn[1:]} := column_zero_gen hN hQ1 hQodd (E {j}) (hE {j}) {sl} (by decide) {dn}')
    zs.append('z' + dn[1:])
L.append('  rw [hout]')
L.append(f'  simp only [{", ".join(deps(outs[0], set()))}]')
L.append(f'  simp only [{", ".join(zs)}]')
L.append('  norm_num\n')
L.append('end RefreshTopProof\n')
open(out_path, 'w').write('\n'.join(L))
