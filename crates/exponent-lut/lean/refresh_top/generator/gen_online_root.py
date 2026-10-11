"""Write `Part_<c_j>_Online.lean`: segment theorems of the online root, each over the atoms of one
part of the root (see `segment.py`), applied by the root theorem.

Usage: `python3 gen_online_root.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_luts import branch_call, round_args
from gen_part import parts_of
from chains import chain_of
from patterns import parse
from private_chain import root_offset
from segment import abbreviations, segment
from layouts import layouts

WIRE = re.compile(r'\bw_\d+_(?:\d+|concat_\d+)\b')


def tuple_wires(text):
    return WIRE.findall(text)


def root_lut(path, ns, names, atoms, lets, ab, loop, kind, q, qp, p, m, mp, layout, hlayout, qeq,
             width, scale=None, table=None):
    """The segment theorem `rootLut<k>` of the root LUT whose loop atom is `loop`: `kind` `'round'`
    (with the move down) or `'branches'`."""
    atom = dict(zip(names, atoms))
    k = loop[1:].split('_')[0]
    text = atom[loop]
    call = text[text.index('params i ('):text.index(') next)')]
    args = tuple_wires(call)
    h0, key, h1, c0, A00, A10, c1, A01, A11, f00, f01, f10, f11 = args
    init_text = text[text.index('(Int.toNat'):]
    count = int(re.search(r'\(Int\.toNat \((\d+)\)\)', init_text).group(1))
    init = tuple_wires(init_text)[:10]
    state = tuple_wires(init_text)[10:20]
    # The region: the init gets, and the atoms from the loop to the next loop or parallel.
    named = [x for x in names if x != '-']
    start = named.index(loop)
    stop = next(i for i in range(start + 1, len(named))
                if atom[named[i]].startswith('MxxIR.IterRuns') or atom[named[i]].startswith('(∀'))
    def defines(wire):
        return next(x for x in named if re.search(rf'\b{wire}$', atom[x]))
    init_gets = [defines(x) for x in init[6:10]]
    region = init_gets + named[start:stop]
    rn = [x for x in region]
    ra = [atom[x] for x in region]
    r = round_args(rn, ra, lets, state, width - 1)
    w = r['wire']
    help_gets = [x for x in region if re.match(rf'MxxRuntime\.familyGetStatic \S+ \({width - 1}\) ', atom[x])][:2]
    lut_a = '[114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]'
    hashes = [x for x in region if atom[x].startswith('MxxRuntime.hashSample') and lut_a in atom[x]]
    hash_of = {int(re.search(r'\.bytes \[99[^\]]*\], \.u64Le \((\d)\)', atom[x]).group(1)): x
               for x in hashes}
    level = int(re.search(r'\(\[\.u64Le \((\d+)\), \.integer', atom[hashes[0]]).group(1))
    sig, sig_odd, Fw, lr = ('sigma2', 'sigma2_odd', 'Fin 2', 'lut_round2') if width == 2 else \
        ('sigma', 'sigma_odd', 'Fin 4', 'lut_round')
    lb = 'lut_branches2' if width == 2 else 'lut_branches'
    last = width - 1
    used = list(region)
    fin = ''
    if kind == 'round':
        rev = {v: k2 for k2, v in lets.items()}
        def sum_let(x, y):
            return rev[f'MxxRuntime.matrixAdd {x} {y}']
        def modswitch_of(src):
            return next(x for x in named if re.match(
                rf'MxxRuntime\.blockModSwitchRuns \[[^\]]*\] \(1\) {src} ', atom[x]))
        kk = [[r['K0'][0], r['K0'][1]], [r['K1'][0], r['K1'][1]]]
        fin_k = [[modswitch_of(sum_let(state[2 + 2 * a + b], w(kk[a][b]))) for b in range(2)]
                 for a in range(2)]
        yx = [sum_let(w(r['Y'][jj]), w(r['X'][jj])) for jj in range(2)]
        fin_r = [modswitch_of(sum_let(state[jj], yx[jj])) for jj in range(2)]
        used += [x for x in [fin_k[0][0], fin_k[0][1], fin_k[1][0], fin_k[1][1], fin_r[0], fin_r[1]]
                 if x not in used]
    used = [x for x in named if x in used]
    extra = []
    if kind != 'round':
        rev = {v: k2 for k2, v in lets.items()}
        kk = [[r['K0'][0], r['K0'][1]], [r['K1'][0], r['K1'][1]]]
        extra = [rev[f'MxxRuntime.matrixAdd {state[2 + 2 * a + b]} {w(kk[a][b])}'] for a in range(2) for b in range(2)]
        yx = [rev[f'MxxRuntime.matrixAdd {w(r["Y"][jj])} {w(r["X"][jj])}'] for jj in range(2)]
        extra += [rev[f'MxxRuntime.matrixAdd {state[jj]} {yx[jj]}'] for jj in range(2)]
    binders, _ = segment(path, ns, 'generatedRoot', used, ab, extra)
    eqs = [b.split(' : ')[0][1:] for b in binders if b.startswith('(e_')]
    fam = f'![![{f00}, {f01}], ![{f10}, {f11}]]'
    src_target = (f"(fun j ↦ reducePoly {q} N ((({p} : Nat) : Int) * Xe (n := N) z * unitPair sp j) • g)"
                  if width == 2 else
                  f"(fun j ↦ reducePoly {q} N ((({p} : Nat) : Int) * Xe (n := N) (2 * z) * unitPair sp j) • g)")
    sem = f'''
    {{sm sp : ErrorPoly N}} (hsm : polyNorm sm ≤ 1) {{z : Nat}}
    {{g' : ExactMatrix {qp} N 1 {m}}} {{g : ExactMatrix {q} N 1 {m}}}
    (hg' : gadgetMatrixRuns Backend.backend 32768 {m} g') (hg : centeredRebaseRuns g' g)
    {{B : Nat}}
    (hinput : Rel sm ![![{A00}, {A01}], ![{A10}, {A11}]] {src_target} ![{c0}, {c1}] B)
    {{Bh : Nat}} (hhelp : ∀ i : {Fw}, Rel sm (lutA model {key} {q} {m} {level} ({sig} i))
      (fun j ↦ reducePoly {q} N ((({p} : Nat) : Int) *
        unitPair (substitution ({sig} i).toNat ({sig_odd} i) sm) j) • g) ![{h0} i, {h1} i] Bh)'''
    inv = '2' if width == 2 else '4'
    coef = 'lutCoefficient2' if width == 2 else 'lutCoefficient'
    bound = f'''(({count} * lutBranchBound{k} B Bh + (({m} + {m}) * N * Bh * 16384 + ({m} + {m}) * N * B * 16384 +
        (N * 1 * ({p} / 2) + {p} / 2)))'''
    if kind == 'round':
        value = (f"(Xe (z - z % 2) * {table} ⟨z % 2, Nat.mod_lt _ (by norm_num)⟩)" if width == 2 else
                 f"((((-1 : Int) ^ (z / 4) : Int) : ErrorPoly N) * {table} ⟨z % 4, Nat.mod_lt _ (by norm_num)⟩)")
        sem += f'''
    {{gq : ExactMatrix {qp} N 1 {mp}}} {{inverse : Nat}}
    (hinverse : (inverse : ExactPoly {qp} N) * (({inv} : Nat) : ExactPoly {qp} N) = 1) {{BB : Nat}}
    (hswitch : ∀ i : {Fw}, ∃ e : Fin 2 → ErrorMatrix N 1 {mp}, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly {qp} N (substitution ({sig} i).toNat ({sig_odd} i) sp) •
          {fam} 0 j i + {fam} 1 j i =
        ({coef} {qp} inverse (fun k ↦ reducePoly {qp} N ({table} k)) ({sig} i).toNat *
          reducePoly {qp} N ({scale} * unitPair sp j)) • gq + reduceMatrix {qp} N 1 {mp} (e j)) :
    Rel sm ![![{w(fin_k[0][0])}, {w(fin_k[0][1])}], ![{w(fin_k[1][0])}, {w(fin_k[1][1])}]]
      (fun j ↦ reducePoly {qp} N ({scale} * {value} * unitPair sp j) • gq)
      ![{w(fin_r[0])}, {w(fin_r[1])}]
      ({bound} + (N + 2) * ({p} / 2)) / {p} + {width} * (N * 1 * BB)) := by'''
    else:
        sumv = 'z' if width == 2 else '(2 * z)'
        rows = [sum_let_ for sum_let_ in []]
        rev = {v: k2 for k2, v in lets.items()}
        kk = [[r['K0'][0], r['K0'][1]], [r['K1'][0], r['K1'][1]]]
        keys = [[rev[f'MxxRuntime.matrixAdd {state[2 + 2 * a + b]} {w(kk[a][b])}'] for b in range(2)]
                for a in range(2)]
        yx = [rev[f'MxxRuntime.matrixAdd {w(r["Y"][jj])} {w(r["X"][jj])}'] for jj in range(2)]
        rws = [rev[f'MxxRuntime.matrixAdd {state[jj]} {yx[jj]}'] for jj in range(2)]
        sem += f''' :
    Rel sm ![![{keys[0][0]}, {keys[0][1]}], ![{keys[1][0]}, {keys[1][1]}]]
      (fun j ↦ ∑ i : {Fw}, scaleUpMatrix {q} {p}
        (reducePoly {qp} N (substitution ({sig} i).toNat ({sig_odd} i) (Xe {sumv})) •
          (reducePoly {qp} N (substitution ({sig} i).toNat ({sig_odd} i) sp) •
            {fam} 0 j i + {fam} 1 j i)))
      ![{rws[0]}, {rws[1]}] {bound}) := by'''
    st = f"({', '.join(state)}, ())"
    kj = {0: '0 0', 1: '0 1', 2: '1 0', 3: '1 1'}
    proof = f'''  subst {' '.join(eqs)}
''' if eqs else ''
    proof += f'''  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) {init_gets[0]}
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) {init_gets[1]}
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) {init_gets[2]}
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) {init_gets[3]}
  subst hwa hwb hwc hwd
  rw [show Int.toNat {count} = {count} from rfl] at {loop}
  obtain ⟨hloop, hloopSwitch⟩ := lutLoop{k} (fam := fun k j ↦ {fam} k j)
    (c := ![{c0}, {c1}]) (A := ![![{A00}, {A01}], ![{A10}, {A11}]])
    (final := {st}) hsm hg' hg hinput hhelp (count := {count}) (by decide) {loop}
  obtain ⟨_, hw19⟩ := familyGet_eq (k := {last}) {help_gets[0]}
  obtain ⟨_, hw24⟩ := familyGet_eq (k := {last}) {help_gets[1]}
  rw [hw19] at {r['help'][0]}
  rw [hw24] at {r['help'][2]}
'''
    for c in range(4):
        proof += (f"  have hA{c} : {w(hash_of[c])} = lutA model {key} {q} {m} {level} ({sig} ({last} : {Fw})) {kj[c]} :=\n"
                  f"    hashSample_eq {hash_of[c]}\n")
    proof += f"  rw [hA0] at {r['help'][0]}\n  rw [hA2] at {r['help'][1]}\n"
    proof += f"  rw [hA1] at {r['help'][2]}\n  rw [hA3] at {r['help'][3]}\n"
    proof += f'  have hhelp3 := hhelp {last}\n'
    args_ = branch_call(r, 'hAd')
    if kind == 'round':
        proof += f'''  have hfin := {lr} (q := {q}) (q' := {qp}) (p := {p}) (m := {m}) (m' := {mp}) {qeq} (by norm_num)
    (by norm_num) (by norm_num) {layout} {hlayout} rfl hsm hg' hg hinput hloop hloopSwitch hhelp3
    {args_}
    (T := {table}) (Δ' := {scale}) hinverse
    (fun i ↦ hswitch i) (K' := ![![{w(fin_k[0][0])}, {w(fin_k[0][1])}], ![{w(fin_k[1][0])}, {w(fin_k[1][1])}]])
    (R' := ![{w(fin_r[0])}, {w(fin_r[1])}])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [{fin_k[0][0]}, {fin_k[0][1]}, {fin_k[1][0]}, {fin_k[1][1]}])
    (fun j ↦ by fin_cases j; exacts [{fin_r[0]}, {fin_r[1]}])
  exact hfin
'''
    else:
        proof += f'''  have hsum := {lb} (q := {q}) (q' := {qp}) (p := {p}) (m := {m}) (m' := {mp}) {qeq} (by norm_num)
    (by norm_num) (by norm_num) {layout} {hlayout} rfl hsm hg' hg hinput hloop hloopSwitch hhelp3
    {args_}
  convert hsum using 1
  all_goals first
    | rfl
    | (funext a j; fin_cases a <;> fin_cases j <;> rfl)
    | (funext j; fin_cases j <;> rfl)
'''
    head = f'''
set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The root LUT `{k}`. -/
theorem rootLut{k} {{model : HashModel}} {{params : {ns}.Params}}
    {chr(10).join('    ' + b for b in binders).lstrip()}{sem}
'''
    return head + proof


d = sys.argv[1]
L = layouts(d)
ab = abbreviations(open(f'{d}/TopSetup.lean').read())
Q3 = '(((q3 : Nat) : Int) : ErrorPoly N)'
for part in parts_of(d):
    path = f'{d}/generated/Stage_online_eval_{part}.lean'
    ns = f'Stage_online_eval_{part}'
    text = open(path).read()
    lets_list, names, atoms = parse(path, 'generatedRoot')
    body = text[text.index('abbrev generatedRoot.body'):]
    body = body[:body.index('\n\n')]
    lets = dict(re.findall(r'^  let (w_\S+) : [^\n]*?:= (.*)$', body, re.M))
    lut_a = '[114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]'
    loops = [n for n, a in zip(names, atoms) if a.startswith('MxxIR.IterRuns') and lut_a in
             text[text.index('abbrev ' + re.search(r'(sequential_generatedRoot_\d+) ', a).group(1) + '.constraints_0 '):][:40000]]
    out = f'''import Part_{part}_RootLuts
import Part_{part}_Readout

/-! The segments of the online root (written by `generator/gen_online_root.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

/-- The table `[X, 1]` of the source LUT. -/
noncomputable def sourceTable (k : Fin 2) : ErrorPoly N := ![Xe 1, 1] k
'''
    out += root_lut(path, ns, names, atoms, lets, ab, loops[0], 'round', 'Q0', 'Q1', 'q3', 6, 4,
                    'Backend.' + L['Q1'], 'layoutQ1', 'q0_eq', 2, scale=Q3.replace('q3', chain_of(part)['p']),
                    table='sourceTable')
    out += root_lut(path, ns, names, atoms, lets, ab, loops[1], 'round', 'Q5', 'Q0', 'p4', 8, 6,
                    'Backend.' + L['Q0'], 'layoutQ0', 'q5_eq', 4, scale=Q3, table='readoutTable')
    out += root_lut(path, ns, names, atoms, lets, ab, loops[2], 'branches', 'Q0', 'Q1', 'q3', 6, 6,
                    'Backend.' + L['Q1'], 'layoutQ1', 'q0_eq', 4)
    rhs = '[114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115'
    tops = {WIRE.findall(a)[-1] for a in atoms if a.startswith('MxxRuntime.hashSample') and rhs in a}
    xbs = {WIRE.findall(a)[-1] for a in atoms
           if a.startswith('MxxRuntime.concatRows ') and WIRE.findall(a)[0] in tops}
    d1s = {WIRE.findall(a)[-1] for a in atoms
           if a.startswith('MxxRuntime.gadgetDecomposeRuns') and WIRE.findall(a)[0] in xbs}
    ds = {WIRE.findall(a)[-1] for a in atoms
          if a.startswith('MxxRuntime.compactCenteredRebaseRuns') and WIRE.findall(a)[0] in d1s}
    first_both = next(n for n, a in zip(names, atoms) if a.startswith('MxxRuntime.concatColumns ') and
                      any(re.match(rf'MxxRuntime\.matrixMul {WIRE.findall(a)[-1]} (\S+)$', v) and
                          v.split()[-1] in ds for v in lets.values()))
    out += root_offset(path, ns, names, atoms, lets, ab, first_both)
    out += f'\nend RefreshTopProof.Part_{part}\n'
    open(f'{d}/Part_{part}_Online.lean', 'w').write(out)
    print(part, loops)
    break
