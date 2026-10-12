"""Write `Part_<c_j>_OnlineKey.lean` and `Part_<c_j>_OnlinePart.lean`.

`onlineKeyHyp`: the preimage equations of the decoders over the keys the online root computes,
stated relationally (for every run of the atoms that compute the keys from the root inputs).

`onlinePart`: the online root's output is, residue by residue modulo `q₀`, the value
`y = β A'₀ + A'₁ - T + e'` of the refresh, given the semantic facts about the root inputs (the
chain theorems' hypotheses, renamed per chain), the hint and the one encoding, the preimage
equations and the noise bound.

Usage: `python3 gen_online_part.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from chain import WIRE, binder_groups, group_names, head, seg_args, subst
from gen_part import parts_of
from patterns import destruct_root, parse
from segment import abbreviations, qualify, segment, short

d = sys.argv[1]
ab = abbreviations(open(f'{d}/TopSetup.lean').read())
ATOM = re.compile(r"^\((h\d+_\d+(?:_\d+)?|e_\d+_(?:\d+|concat_\d+)) :")
SHARED = {'sm', 'hsm'}


def gadget_names(groups):
    """Global names of the gadgets and re-encoded gadgets a theorem binds."""
    types, out = {}, {}
    for g in groups:
        m = re.match(r"\{([^:]+) : ExactMatrix (\S+) N 1 (\d+)\}", ' '.join(g.split()))
        if m:
            for n in m.group(1).split():
                types[n] = (m.group(2), m.group(3))
    text = ' '.join(' '.join(g.split()) for g in groups)
    for m in re.finditer(r"\((\S+) : gadgetMatrixRuns Backend\.backend 32768 (\d+) (\S+?)\)", text):
        q, w = types[m.group(3)]
        out[m.group(3)] = f'G{q}_{w}'
        out[m.group(1)] = f'hG{q}_{w}'
    for m in re.finditer(r"\((\S+) : centeredRebaseRuns (\S+) (\S+?)\)", text):
        if m.group(3) in types:
            q, w = types[m.group(3)]
            out[m.group(3)] = f'R{q}_{w}'
            out[m.group(1)] = f'hR{q}_{w}'
    return out


class Top:
    """The top-level binders, deduplicated."""

    def __init__(self):
        self.groups, self.seen = [], set()

    def add(self, g):
        key = ' '.join(g.split())
        if key not in self.seen:
            self.seen.add(key)
            self.groups.append(key)


def chain_call(top, text, thm, suffix, fixed, internal):
    """Add the semantic binders of `thm` to `top` (renamed with `suffix`, except shared names,
    gadgets and `fixed`) and return its call over the destructured root. `internal` maps binder
    names supplied inside the proof to their arguments (those binders are not added)."""
    groups, concl = binder_groups(text, thm)
    gnames = gadget_names(groups)
    ren = {}
    atoms = []
    for g in groups:
        flat = ' '.join(g.split())
        if ATOM.match(flat):
            atoms.append(group_names(g)[0])
            continue
        if re.match(r"\{w_", flat) or re.match(r"\{(model|params)", flat):
            continue
        for n in group_names(g):
            if n in SHARED:
                ren[n] = n
            elif n in fixed:
                ren[n] = fixed[n]
            elif n in gnames:
                ren[n] = gnames[n]
            elif n in internal:
                ren[n] = n
            else:
                ren[n] = n + suffix
    explicit = []
    for g in groups:
        flat = ' '.join(g.split())
        if ATOM.match(flat) or re.match(r"\{(w_|model|params)", flat):
            continue
        names = group_names(g)
        if any(n in internal for n in names):
            continue
        ty = flat[flat.index(' : ') + 3:-1]
        if g.startswith('{'):
            for n in names:
                if re.match(r'[GR]Q|GM', ren[n]) or n in fixed:
                    continue
                top.add('{' + ren[n] + ' : ' + subst(ty, ren) + '}')
        else:
            top.add('(' + ren[names[0]] + ' : ' + subst(ty, ren) + ')')
            explicit += names
    args = ' '.join(f'({a} := {a})' if a.startswith('h') else f'({a} := rfl)' for a in atoms)
    sem = ' '.join(f'({n} := {ren[n]})' for n in explicit)
    inner = ' '.join(f'({n} := {v})' for n, v in internal.items())
    fixed_args = ' '.join(f'({n} := {v})' for n, v in fixed.items() if not n.startswith('h') and n in ren)
    return f'{thm} (model := model) {args} {sem} {inner} {fixed_args}', subst(concl, ren)



def closure(lets, names, atoms, body, targets):
    """The atoms the wires `targets` are computed from."""
    vals = dict(re.findall(r'let (w_\S+) : [^\n]*?:= (.*)', body))
    prod = {}
    for n, a in zip(names, atoms):
        if n == '-':
            continue
        if a.startswith('(∀'):
            i = a.rindex(') ((w_') if ') ((w_' in a else None
            outs = WIRE.findall(a[i:]) if i else WIRE.findall(a)[-1:]
        elif a.startswith('MxxIR.IterRuns'):
            outs = WIRE.findall(a[a.rindex(') (w_'):])
        else:
            outs = WIRE.findall(a)[-1:]
        for w in outs:
            prod.setdefault(w, n)
    A = dict(zip(names, atoms))
    seen, used, todo = set(), [], list(targets)
    while todo:
        w = todo.pop()
        if w in seen:
            continue
        seen.add(w)
        if w in vals and not vals[w].startswith(('inputs', 'witness')):
            todo += WIRE.findall(vals[w])
        elif w in prod and prod[w] not in used:
            used.append(prod[w])
            todo += WIRE.findall(A[prod[w]])
    return [n for n in names if n in used]


FRESH_SUMS = [413, 438, 460, 482, 504, 526]
SMUDGE_SUMS = [579, 634, 688, 742, 796, 850]
PRE = ['w_85_0', 'w_584_0', 'w_638_0', 'w_692_0', 'w_746_0', 'w_800_0']


def key_hyp(part, ns, path, lets, names, atoms, body, inputs):
    targets = [f'w_332_{k}' for k in range(4)] + ['w_52_0', 'w_70_0', 'w_342_0', 'w_355_0', 'w_417_0']
    targets += [f'w_{S}_{k}' for S in FRESH_SUMS + SMUDGE_SUMS for k in range(4)]
    used = closure(lets, names, atoms, body, targets)
    binders, _ = segment(path, ns, 'generatedRoot', used, ab, targets)
    inp = {w for w, _ in inputs}
    params = [b for b in binders if re.match(r'\{(w_\d+_0) :', b) and re.match(r'\{(w_\d+_0) :', b).group(1) in inp]
    rest = [b for b in binders if b not in params]
    for w in PRE:
        if not any(p.startswith('{' + w + ' ') for p in params):
            params.append(f'{{{w} : ExactMatrix Q1 N 16 1}}')
    params = [p.replace('{', '(', 1)[:-1] + ')' for p in params]
    kg = 'fun i ↦ blockModSwitch [q0, q1, q2, q3] 1 (![![w_332_0, w_332_1], ![w_332_2, w_332_3]] i 1)'
    kx = 'fun i ↦ ![![w_52_0, w_67_0], ![w_70_0, w_73_0]] i 0'
    cols = ', '.join(f'(blockModSwitch [q0, q1, q2, q3] 1 (![![w_{S}_0, w_{S}_1], ![w_{S}_2, w_{S}_3]] i 1) * w_417_0) 0 0'
                     for S in FRESH_SUMS)
    kw = f'fun i ↦ rowMat (fun k ↦ ![{cols}] k)'
    ksm = '![' + ', '.join(f'fun i ↦ ![![w_{S}_0, w_{S}_1], ![w_{S}_2, w_{S}_3]] i 1' for S in SMUDGE_SUMS) + ']'
    pre = '![' + ', '.join(PRE) + ']'
    return f'''/-- The preimage equations of the decoders over the keys the online root of part `{part}`
computes: for every run of the atoms that compute the keys, column `k` of the preimages, times the
hint keys `B`, is column `k` of the combination's keys plus the smudge keys. -/
def onlineKeyHyp (model : HashModel) (params : {ns}.Params)
{chr(10).join('    ' + p for p in params)}
    (B : Fin 2 → ExactMatrix Q1 N 1 16) (K₁ : Fin 2 → ExactMatrix Q1 N 1 6) : Prop :=
  ∀ {chr(10).join('    ' + r for r in rest)},
  ∀ (k : Fin 6) (i : Fin 2), B i * {pre} k =
    (({kg}) i * topRows w_342_0 + K₁ i * bottomRows w_342_0 - ({kx}) i * w_355_0 +
      ({kw}) i) * unitCol (q := Q1) k + {ksm} k i
''', [p[1:p.index(' :')] for p in params]


for part in parts_of(d):
    ns = f'Stage_online_eval_{part}'
    path = f'{d}/generated/{ns}.lean'
    text = open(path).read()
    lets, names, atoms = parse(path, 'generatedRoot')
    A = dict(zip(names, atoms))
    body = text[text.index('abbrev generatedRoot.body'):]
    inputs = re.findall(r'let (w_\d+_0) : ([^\n]*?) := inputs', body)
    files = {f: open(f'{d}/Part_{part}_{f}.lean').read()
             for f in ['OnlineChain', 'FreshChain', 'SmudgeChain', 'Online']}
    top = Top()
    for q, w in [('Q7', 14), ('Q6', 12), ('Q5', 10), ('Q0', 8), ('Q1', 6), ('M01', 4)]:
        top.add(f'{{G{q}_{w} : ExactMatrix {q} N 1 {w}}}')
    for q, w in [('Q7', 12), ('Q6', 10), ('Q5', 8), ('Q0', 6), ('Q1', 4)]:
        top.add(f'{{R{q}_{w} : ExactMatrix {q} N 1 {w}}}')
    calls = {}
    fixed = {'a': 'labelExp'}
    calls['mask'] = chain_call(top, files['OnlineChain'], 'maskChain', '_m',
                               {**fixed, 'gq': 'GQ1_6'}, {'hpay': 'hpay109'})
    calls['freshS'] = chain_call(top, files['FreshChain'], 'freshSamples', '_f', fixed,
                                 {'hpay': 'hpay380', 'o': 'o'})
    for k in range(6):
        calls[f'fresh{k}'] = chain_call(top, files['FreshChain'], f'freshCol{k}', '_fc', {'sp': 'sp_f'},
                                        {'hsamp': 'hfs', 'ζ': 'ζf', 'Bs': 'Bsf'})
    smudge_luts = []
    for k in range(6):
        t = files['SmudgeChain']
        hp = re.search(r'\(hpay : ∀ k : Fin 5, (w_\d+)_0 k', head(t, f'smudgeChain{k}')).group(1)
        calls[f'smudge{k}'] = chain_call(top, t, f'smudgeChain{k}', f'_s{k}', fixed,
                                         {'hpay': f'hpay{hp[2:]}'})
        smudge_luts.append(hp)
    calls['source'] = chain_call(top, files['Online'], 'rootLut18', '_x', {'gq': 'RQ1_4'}, {})
    kh, kparams = key_hyp(part, ns, path, lets, names, atoms, body, inputs)
    open(f'{d}/Part_{part}_OnlineKey.lean', 'w').write(f'''import Part_{part}_OnlineChain
import Part_{part}_FreshChain
import Part_{part}_SmudgeChain
import TopColumn

/-! The relational preimage hypothesis of part `{part}` (written by `generator/gen_online_part.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

set_option maxRecDepth 65536 in
{kh}
end RefreshTopProof.Part_{part}
''')
    break

