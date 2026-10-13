"""Write `TopDecodeCrt.lean` of parameter directory `d`: the decode stage of the refresh of every
CRT slot outputs zero when, for each row, the residue of every slot `c` modulo `q_c` is that of the
expected row plus one small error row (shared by the slots), modulo `Q₀`.

The stage lifts each slot's residue to `Q₀` (centered), recombines them with the CRT basis
(`crt_matrix`), subtracts the expected row and threshold-decodes every coefficient at `Q₀`.

Usage: `python3 gen_decode_crt.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from patterns import destruct, parse

d = sys.argv[1]
path = f'{d}/generated/Stage_decode.lean'
text = open(path).read()
lets, names, atoms = parse(path, 'generatedRoot')
atom = dict(zip(names, atoms))
body = text[text.index('abbrev generatedRoot.body'):]
defs = dict(re.findall(r'^  let (w_\S+) : [^\n]*?:= (.*)$', body, re.M))
defs.update(re.findall(r'^  let (w_\S+) := (witness\.\S+)$', body, re.M))
types = dict(re.findall(r'^  let (w_\S+) : ([^\n]*?) :=', body, re.M))
w = lambda h: 'w_' + h[1:]
named = [n for n in names if n != '-']
kind = lambda prefix: [n for n in named if atom[n].startswith('MxxRuntime.' + prefix)]


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


def arg(x, op, k):
    """Argument `k` of the let `x := MxxRuntime.op a b`."""
    return re.match(rf'MxxRuntime\.{op} (w_\S+) (w_\S+)', defs[x]).group(k + 1)


gadget = kind('gadgetMatrixRuns')[0]
rebases = kind('centeredRebaseRuns w_')
grebase = next(r for r in rebases if re.search(r'Runs (w_\d+_0) ', atom[r]).group(1) == w(gadget))
lifts = {re.search(r'Runs (w_\d+_0) (w_\d+_0)$', atom[r]).group(2): r for r in rebases if r != grebase}
hashes = kind('hashSample')
slices = kind('sliceMatrix')
decodes = kind('thresholdDecode')
assert len(lifts) == 8 and len(hashes) == 2 and len(decodes) == 96, (len(lifts), len(hashes), len(decodes))
src = lambda n: re.search(r'sliceMatrix (w_\d+_0)', atom[n]).group(1)
rows = [s for s in slices if src(s) in (w(hashes[0]), w(hashes[1]))]
cols = [s for s in slices if s not in rows]
assert len(rows) == 4 and len(cols) == 12
diff = [src(cols[0]), src(cols[6])]
inputs = [n for n in lets if 'inputs' in defs.get(n, '')]
sigma = next(n for n in inputs if types[n].strip() == 'Fin 8 → Int')
key = next(n for n in inputs if types[n].strip() == 'ByteArray')
coef, expo = [n for n in inputs if types[n].strip() == 'Int']
payload = next(n for n in inputs if types[n].strip().endswith(' 1 1'))
# Each row: `diff = combined - expected`, `combined` the CRT recombination of the four lifts.
combined = [arg(diff[j], 'matrixSub', 0) for j in range(2)]
expected = [arg(diff[j], 'matrixSub', 1) for j in range(2)]


def crt_lifts(x):
    """The four lifts the recombination `x` adds up, in slot order."""
    out = []
    for _ in range(3):
        out.append(arg(arg(x, 'matrixAdd', 1), 'matrixMulScalarRight', 0))
        x = arg(x, 'matrixAdd', 0)
    out.append(arg(x, 'matrixMulScalarRight', 0))
    return out[::-1]


slot_lifts = [crt_lifts(combined[j]) for j in range(2)]
residue = [[re.search(r'Runs (w_\d+_0) ', atom[lifts[l]]).group(1) for l in slot_lifts[j]] for j in range(2)]
sigma_wire = arg(arg(arg(expected[0], 'matrixSub', 0), 'matrixAdd', 0), 'matrixMulScalarLeft', 0)
row_wires = [[w(r) for r in rows[2 * j:2 * j + 2]] for j in range(2)]
outs = re.findall(r'\bw_\d+_\d+\b', atom['hout'])

L = ['import TopCrt\nimport TopAssembly\n',
     '/-! The decode stage of the refresh of every CRT slot (written by `generator/gen_decode_crt.py`). -/\n',
     'set_option maxRecDepth 65536\n',
     'namespace RefreshTopProof\n',
     'open Mxx.Primitives MxxRuntime MxxExponentLut\n']
L.append("""/-- The decode stage outputs zero when, for each row `j`, every slot's residue is that of
`y_j + e'_j` for a small row `e'_j`. -/
theorem decode_zero {model : HashModel} {params : Stage_decode.Params} {inputs} {out : Int}
    (h : Stage_decode.generatedRoot Backend.backend model params inputs out)
    {G : ExactMatrix Q1 N 1 6} (hG : gadgetMatrixRuns Backend.backend 32768 6 G)
    (E : Fin 2 → ErrorMatrix N 1 6) (hE : ∀ j c i, 4 * |(E j 0 c).coeff i| + 4 < (Q0 : Int))""")
for j in range(2):
    r = defs[payload] if j == 0 else '1'
    v = f'(topValue model {defs[key]} (topTagI {j}) {defs[sigma]} {r} {defs[coef]} {defs[expo]} G (E {j}))'
    for c in range(4):
        L.append(f'    (h{j}{c} : Residue {defs[residue[j][c]]}\n      {v})')
L.append('    : out = 0 := by')
L.append(destruct(path, 'Stage_decode', 'generatedRoot', 'h').rstrip('\n'))
L.append(f'  have hg : {w(gadget)} = G := gadgetMatrixRuns_unique {gadget} hG')
fams = [n for n in named if atom[n].startswith('MxxRuntime.familyGetStatic ' + sigma + ' ')]
L.append(f'  have hσ : {sigma_wire} = sigmaPoly {sigma} := by')
for k, f in enumerate(fams):
    L.append(f'    obtain ⟨_, e{k}⟩ := familyGet_eq (k := {k}) {f}')
L.append(f'    simp only [{", ".join(deps(sigma_wire, set()))}]')
L.append(f'    rw [{", ".join(f"e{k}" for k in range(len(fams)))}]')
L.append('    rfl')
for j in range(2):
    r = payload if j == 0 else '1'
    L.append(f'  have hA{j} := {hashes[j]}.2.2')
    for k in range(2):
        L.append(f'  have hr{j}{k} := slice_row {rows[2 * j + k]} rfl (by decide)')
    stop = set(row_wires[j]) | {w(grebase), sigma_wire, w(hashes[j])}
    L.append(f'  have he{j} : {expected[j]} = expectedRow (model.sample Q0 N 2 6 {key}')
    L.append(f'      (completeHashTag (topTagI {j}) [])) (sigmaPoly {sigma}) {r} {coef} {expo} G := by')
    L.append(f'    simp only [{", ".join(deps(expected[j], stop))}]')
    L.append(f'    rw [hσ, hr{j}0, hr{j}1, {grebase}.2, hg, hA{j}, topTagI, completeHashTag_append, List.append_nil]')
    L.append('    rfl')
    hs = ' '.join(f'h{j}{c}' for c in range(4))
    ls = ' '.join(lifts[l] for l in slot_lifts[j])
    L.append(f'  have hc{j} : {combined[j]} = topValue model {key} (topTagI {j}) {sigma} {r} {coef} {expo} G (E {j}) := by')
    L.append(f'    simp only [{", ".join(deps(combined[j], set(slot_lifts[j])))}]')
    L.append(f'    exact crt_matrix {hs} {ls}')
    L.append(f'  have hd{j} : {diff[j]} = reduceMatrix Q0 N 1 6 (E {j}) := by')
    L.append(f'    simp only [{diff[j]}]')
    L.append(f'    rw [hc{j}, he{j}, topValue, matrixSub_eq, add_sub_cancel_left]')
    for c in cols[6 * j:6 * j + 6]:
        L.append(f'  rw [hd{j}] at {c}')
zs = []
for dname in decodes:
    srcw = re.search(r'thresholdDecode \(2\) \(8\) \d+ (w_\d+_0)', atom[dname]).group(1)
    sl = next(c for c in cols if w(c) == srcw)
    j = 0 if cols.index(sl) < 6 else 1
    L.append(f'  have z{dname[1:]} := column_zeroQ0 (E {j}) (hE {j}) {sl} (by decide) {dname}')
    zs.append('z' + dname[1:])
L.append('  rw [hout]')
L.append(f'  simp only [{", ".join(deps(outs[0], set()))}]')
L.append(f'  simp only [{", ".join(zs)}]')
L.append('  norm_num\n')
L.append('end RefreshTopProof\n')
open(f'{d}/TopDecodeCrt.lean', 'w').write('\n'.join(L))
