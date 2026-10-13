"""Write `Part_<c_j>_KeysRoot.lean`: the keys of the online root's LUTs (segment theorems over the
root atoms), as `Σ_i lutKeys i` (moved down for a LUT round).

Usage: `python3 gen_keys_root.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_luts import round_args
from gen_part import parts_of
from gen_keys_luts import LAYOUT, on_info
from patterns import parse
from segment import abbreviations, segment, short

WIRE = re.compile(r'\bw_\d+_(?:\d+|concat_\d+)\b')
LUT_A = '[114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]'


def root_keys(path, ns, names, atoms, lets, ab, loop, kind):
    atom = dict(zip(names, atoms))
    k = loop[1:].split('_')[0]
    text = atom[loop]
    body_name = re.search(r'(sequential_generatedRoot_\d+)', text).group(1)
    info = on_info(path, body_name, ab)
    q, m, W, qp, mp = info['q'], info['m'], info['W'], info['qp'], info['mp']
    lay, hlay = LAYOUT[qp]
    sig = 'sigma2' if W == 2 else 'sigma'
    call = text[text.index('params i ('):text.index(') next)')]
    h0, key, h1, c0, A00, A10, c1, A01, A11, f00, f01, f10, f11 = WIRE.findall(call)
    init_text = text[text.index('(Int.toNat'):]
    count = int(re.search(r'\(Int\.toNat \((\d+)\)\)', init_text).group(1))
    init = WIRE.findall(init_text)[:10]
    state = WIRE.findall(init_text)[10:20]
    named = [x for x in names if x != '-']
    start = named.index(loop)
    stop = next(i for i in range(start + 1, len(named))
                if atom[named[i]].startswith('MxxIR.IterRuns') or atom[named[i]].startswith('(∀'))
    def defines(wire):
        return next(x for x in named if re.search(rf'\b{wire}$', atom[x]))
    init_gets = [defines(x) for x in init[6:10]]
    region = init_gets + named[start:stop]
    r = round_args(region, [atom[x] for x in region], lets, state, W - 1)
    w = r['wire']
    hashes = [x for x in region if atom[x].startswith('MxxRuntime.hashSample') and LUT_A in atom[x]]
    hash_of = {int(re.search(r'\.bytes \[99[^\]]*\], \.u64Le \((\d)\)', atom[x]).group(1)): x
               for x in hashes}
    level = info['level']
    rev = {v: k2 for k2, v in lets.items()}
    kk = [[r['K0'][0], r['K0'][1]], [r['K1'][0], r['K1'][1]]]
    sums = [[rev[f'MxxRuntime.matrixAdd {state[2 + 2 * a_ + b_]} {w(kk[a_][b_])}'] for b_ in range(2)]
            for a_ in range(2)]
    used = list(region)
    extra = [x for row in sums for x in row]
    if kind == 'round':
        def modswitch_of(src):
            return next(x for x in named if re.match(
                rf'MxxRuntime\.blockModSwitchRuns \[[^\]]*\] \(1\) {src} ', atom[x]))
        fin_k = [[modswitch_of(sums[a_][b_]) for b_ in range(2)] for a_ in range(2)]
        used += [fin_k[a_][b_] for a_ in range(2) for b_ in range(2)]
        basis2 = short(re.search(r'blockModSwitchRuns (\[[^\]]*\])', atom[fin_k[0][0]]).group(1), ab)
    used = [x for x in named if x in used]
    binders, _ = segment(path, ns, 'generatedRoot', used, ab, extra)
    eqs = [b.split(' : ')[0][1:] for b in binders if b.startswith('(e_')]
    LK = (f"lutKeys (q := {q}) (q' := {qp}) (m := {m}) (m' := {mp}) {m} {info['basis']} model {key} "
          f"{level} {sig} ![![{A00}, {A01}], ![{A10}, {A11}]] ![![{f00}, {f01}], ![{f10}, {f11}]]")
    S = f'(∑ i ∈ Finset.range {W}, {LK} i)'
    if kind == 'round':
        lhs = f'({w(fin_k[0][0])}, {w(fin_k[0][1])}, {w(fin_k[1][0])}, {w(fin_k[1][1])}, ())'
        concl = f'keyMap (blockModSwitch (p := {qp}) {basis2} 1) {S}'
    else:
        lhs = f'({sums[0][0]}, {sums[0][1]}, {sums[1][0]}, {sums[1][1]}, ())'
        concl = S
    st = f"({', '.join(state)}, ())"
    kj = {0: '0 0', 1: '0 1', 2: '1 0', 3: '1 1'}
    bl = '\n    '.join(binders)
    two = lambda key_: f"(fun j ↦ by fin_cases j; exacts [{r[key_][0]}, {r[key_][1]}])"
    sum_eqs = [f'e_{x[2:]}' for x in extra]
    keep = [e for e in eqs if e not in sum_eqs]
    proof = (f'  subst {" ".join(keep)}\n' if keep else '')
    proof += f"""  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) {init_gets[0]}
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) {init_gets[1]}
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) {init_gets[2]}
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) {init_gets[3]}
  subst hwa hwb hwc hwd
  rw [show Int.toNat {count} = {count} from rfl] at {loop}
  obtain ⟨hloop, hloopSwitch⟩ := keyLoop_{body_name} (count := {count}) (by decide) {loop}
"""
    for c in range(4):
        proof += (f"  have hA{c} : {w(hash_of[c])} = lutA model {key} {q} {m} {level} ({sig} ({W - 1} : Fin {W})) {kj[c]} :=\n"
                  f"    hashSample_eq {hash_of[c]}\n")
    proof += f"  rw [hA0] at {r['help'][0]}\n  rw [hA2] at {r['help'][1]}\n"
    proof += f"  rw [hA1] at {r['help'][2]}\n  rw [hA3] at {r['help'][3]}\n"
    proof += f"""  have hK4 := branchKey_of_stacks (q := {q}) (q' := {qp}) (m := {m}) (m' := {mp}) (by norm_num) Backend.{lay}
    {hlay} rfl rfl (digits := {m}) (c := ![{c0}, {c1}]) (A := ![![{A00}, {A01}], ![{A10}, {A11}]]) (index := {sig} {W - 1})
    {' '.join(r['hr'] + r['enc'][:5])}
    (sw := LoopState.switch {st}) (Xb := ![{w(r['Xb'][0])}, {w(r['Xb'][1])}]) {two('Xb')}
    (D' := ![{w(r['D1'][0])}, {w(r['D1'][1])}]) {two('D1')}
    (D := ![{w(r['D'][0])}, {w(r['D'][1])}]) {two('D')}
    (M := ![![{w(r['M0'][0])}, {w(r['M0'][1])}], ![{w(r['M1'][0])}, {w(r['M1'][1])}]]) {two('M0')} {two('M1')}
    (Asw := ![![{w(r['Asw'][0][0])}, {w(r['Asw'][0][1])}], ![{w(r['Asw'][1][0])}, {w(r['Asw'][1][1])}]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [{r['Asw'][0][0]}, {r['Asw'][0][1]}, {r['Asw'][1][0]}, {r['Asw'][1][1]}])
    (Z := ![{w(r['Z'][0])}, {w(r['Z'][1])}]) {two('Z')}
    (E' := ![{w(r['E1'][0])}, {w(r['E1'][1])}]) {two('E1')}
    (E := ![{w(r['E'][0])}, {w(r['E'][1])}]) {two('E')}
    (h := ![{r['help_rows'][0]}, {r['help_rows'][1]}]) (Ad := lutA model {key} {q} {m} {level} ({sig} ({W - 1} : Fin {W})))
    {' '.join(r['help'][:5])}
    (K := ![![{w(r['K0'][0])}, {w(r['K0'][1])}], ![{w(r['K1'][0])}, {w(r['K1'][1])}]]) {two('K0')} {two('K1')}
  rw [hloopSwitch] at hK4
  have hS : ({sums[0][0]}, {sums[0][1]}, {sums[1][0]}, {sums[1][1]}, ()) = {S} := by
    rw [{', '.join(sum_eqs)}]
    rw [Finset.sum_range_succ, ← hloop, lutKeys, dif_pos (by norm_num)]
    refine (show _ = keyTuple (LoopState.key {st}) +
      keyTuple ![![{w(r['K0'][0])}, {w(r['K0'][1])}], ![{w(r['K1'][0])}, {w(r['K1'][1])}]] from rfl).trans ?_
    rw [hK4]
    rfl
"""
    if kind == 'round':
        proof += f"""  rw [← hS]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · exact blockModSwitchRuns_unique {fin_k[0][0]} (blockModSwitchRuns_fn {fin_k[0][0]} _)
  · exact blockModSwitchRuns_unique {fin_k[0][1]} (blockModSwitchRuns_fn {fin_k[0][1]} _)
  · exact blockModSwitchRuns_unique {fin_k[1][0]} (blockModSwitchRuns_fn {fin_k[1][0]} _)
  · exact blockModSwitchRuns_unique {fin_k[1][1]} (blockModSwitchRuns_fn {fin_k[1][1]} _)
"""
    else:
        proof += "  exact hS\n"
    return f"""
set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The keys of the root LUT `{k}`. -/
theorem keyRootLut{k} {{model : HashModel}} {{params : {ns}.Params}}
    {bl} :
    {lhs} = {concl} := by
{proof}"""


def on_offset_keys(path, ns, names, atoms, lets, ab):
    """The keys of the online root's two private multiplications of the encoding of one."""
    from private_chain import private_mult
    atom = dict(zip(names, atoms))
    named = [x for x in names if x != '-']
    wire = lambda x: WIRE.findall(atom[x])[-1]
    rhs = '[114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115'
    tops = {WIRE.findall(a)[-1] for a in atoms if a.startswith('MxxRuntime.hashSample') and rhs in a}
    xbs = {WIRE.findall(a)[-1] for a in atoms if a.startswith('MxxRuntime.concatRows ') and WIRE.findall(a)[0] in tops}
    d1s = {WIRE.findall(a)[-1] for a in atoms if a.startswith('MxxRuntime.gadgetDecomposeRuns') and WIRE.findall(a)[0] in xbs}
    ds = {WIRE.findall(a)[-1] for a in atoms if a.startswith('MxxRuntime.compactCenteredRebaseRuns') and WIRE.findall(a)[0] in d1s}
    first = next(n for n, a in zip(names, atoms) if a.startswith('MxxRuntime.concatColumns ') and
                 any(re.match(rf'MxxRuntime\.matrixMul {WIRE.findall(a)[-1]} (\S+)$', v) and v.split()[-1] in ds
                     for v in lets.values()))
    pm1 = private_mult(named, atom, lets, first)
    hl01b = next(y for y in named if atom[y].startswith(f'MxxRuntime.concatRows {wire(pm1["c2"][0])} '))
    hlb = next(y for y in named if atom[y].startswith(f'MxxRuntime.concatRows {wire(hl01b)} '))
    second = next(y for y in named if atom[y].startswith(f'MxxRuntime.concatColumns {wire(hlb)} '))
    pm2 = private_mult(named, atom, lets, second)
    used = []
    for pm in (pm1, pm2):
        for key in ('enc', 'Xb', 'D1', 'D', 'M0', 'M1', 'A0', 'A1'):
            used += pm[key]
    used = [x for x in named if x in used]
    binders, _ = segment(path, ns, 'generatedRoot', used, ab, pm1['top'] + pm1['bot'] + pm2['top'] + pm2['bot'])
    eqs = [b.split(' : ')[0][1:] for b in binders if b.startswith('(e_')]
    bl = '\n    '.join(binders)
    two = lambda pm, key: f"(fun j ↦ by fin_cases j; exacts [{pm[key][0]}, {pm[key][1]}])"
    w2 = lambda pm, key: f"![{wire(pm[key][0])}, {wire(pm[key][1])}]"
    a2 = lambda pm: f"![![{wire(pm['A0'][0])}, {wire(pm['A0'][1])}], ![{wire(pm['A1'][0])}, {wire(pm['A1'][1])}]]"
    A = pm1['A']
    def call(pm, q, qp, m, mp, Ain):
        lay, hlay = LAYOUT[qp]
        return f"""privKey_of_stacks (q := {q}) (q' := {qp}) (m := {m}) (m' := {mp}) (by norm_num) Backend.{lay} {hlay} rfl rfl
    (digits := {m}) (c := ![{pm['c'][0]}, {pm['c'][1]}]) (A := {Ain}) {' '.join(pm['enc'])}
    (top := ![{pm['top'][0]}, {pm['top'][1]}]) (bot := ![{pm['bot'][0]}, {pm['bot'][1]}])
    (Xb := {w2(pm, 'Xb')}) {two(pm, 'Xb')} (D' := {w2(pm, 'D1')}) {two(pm, 'D1')} (D := {w2(pm, 'D')}) {two(pm, 'D')}
    (M := ![![{wire(pm['M0'][0])}, {wire(pm['M0'][1])}], ![{wire(pm['M1'][0])}, {wire(pm['M1'][1])}]])
    {two(pm, 'M0')} {two(pm, 'M1')} (A' := {a2(pm)})
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [{pm['A0'][0]}, {pm['A0'][1]}, {pm['A1'][0]}, {pm['A1'][1]}])"""
    Aone = f"![![{A[0][0]}, {A[0][1]}], ![{A[1][0]}, {A[1][1]}]]"
    return f"""
set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The keys of the encoding of one after the two private multiplications by the unit offset RHS. -/
theorem keyRootOffset {{model : HashModel}}
    {bl} :
    {a2(pm2)} = privKey (q := Q6) (q' := Q5) 10 [q0, q1, q2, q3, p4, p5]
      (privKey (q := Q7) (q' := Q6) 12 [q0, q1, q2, q3, p4, p5, p6] {Aone}
        ![{pm1['top'][0]}, {pm1['top'][1]}] ![{pm1['bot'][0]}, {pm1['bot'][1]}])
      ![{pm2['top'][0]}, {pm2['top'][1]}] ![{pm2['bot'][0]}, {pm2['bot'][1]}] := by
{('  subst ' + ' '.join(eqs) + chr(10)) if eqs else ''}  have h1 := {call(pm1, 'Q7', 'Q6', 12, 10, Aone)}
  have h2 := {call(pm2, 'Q6', 'Q5', 10, 8, a2(pm1))}
  rw [h2, h1]
"""


def pre_offset_keys(d, part, ab):
    """The keys the preprocessing computes for the encoding of one after the two private
    multiplications."""
    ns = f'Stage_preprocess_{part}'
    path = f'{d}/generated/{ns}.lean'
    used = ['h46_0', 'h48_0', 'h49_0', 'h51_0', 'h68_0', 'h69_0', 'h72_0', 'h73_0', 'h74_0', 'h75_0',
            'h76_0', 'h78_0', 'h79_0', 'h82_0', 'h83_0', 'h93_0', 'h94_0', 'h97_0', 'h98_0', 'h99_0',
            'h100_0', 'h101_0', 'h118_0', 'h119_0', 'h122_0', 'h123_0', 'h124_0', 'h125_0', 'h127_0',
            'h128_0', 'h138_0', 'h139_0', 'h142_0', 'h143_0', 'h144_0', 'h145_0']
    binders, _ = segment(path, ns, 'generatedRoot', used, ab)
    eqs = [b.split(' : ')[0][1:] for b in binders if b.startswith('(e_')]
    keep = [e for e in eqs if e not in ('e_67_0', 'e_92_0', 'e_117_0', 'e_137_0')]
    bl = '\n    '.join(binders)
    return f"""
set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The keys the preprocessing computes for the encoding of one after the two private
multiplications. -/
theorem preKeyOffset {{model : HashModel}}
    {bl} :
    ![![w_123_0, w_143_0], ![w_125_0, w_145_0]] = privKey (q := Q6) (q' := Q5) 10 [q0, q1, q2, q3, p4, p5]
      (privKey (q := Q7) (q' := Q6) 12 [q0, q1, q2, q3, p4, p5, p6] ![![w_27_0, w_29_0], ![w_28_0, w_30_0]]
        ![w_47_0, w_81_0] ![w_67_0, w_92_0])
      ![w_77_0, w_126_0] ![w_117_0, w_137_0] := by
{('  subst ' + ' '.join(keep) + chr(10)) if keep else ''}  have h1 := privKey_of_pre (q := Q7) (q' := Q6) (m := 12) (m' := 10) (digits := 12)
    (A := ![![w_27_0, w_29_0], ![w_28_0, w_30_0]]) h46_0 h51_0 (top := ![w_47_0, w_81_0]) (bot := ![w_67_0, w_92_0])
    (dT := ![w_48_0, w_82_0]) (rT := ![w_49_0, w_83_0]) (dB := ![w_68_0, w_93_0]) (rB := ![w_69_0, w_94_0])
    (fun j ↦ by fin_cases j; exacts [h48_0, h82_0]) (fun j ↦ by fin_cases j; exacts [h49_0, h83_0])
    (fun j ↦ by fin_cases j; exacts [h68_0, h93_0]) (fun j ↦ by fin_cases j; exacts [h69_0, h94_0])
    (M := ![![w_72_0, w_97_0], ![w_74_0, w_99_0]])
    (fun j ↦ by fin_cases j; exacts [h72_0, h97_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h99_0])
    (A' := ![![w_73_0, w_98_0], ![w_75_0, w_100_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h73_0, h98_0, h75_0, h100_0])
  have h2 := privKey_of_pre (q := Q6) (q' := Q5) (m := 10) (m' := 8) (digits := 10)
    (A := ![![w_73_0, w_98_0], ![w_75_0, w_100_0]]) h76_0 h101_0 (top := ![w_77_0, w_126_0]) (bot := ![w_117_0, w_137_0])
    (dT := ![w_78_0, w_127_0]) (rT := ![w_79_0, w_128_0]) (dB := ![w_118_0, w_138_0]) (rB := ![w_119_0, w_139_0])
    (fun j ↦ by fin_cases j; exacts [h78_0, h127_0]) (fun j ↦ by fin_cases j; exacts [h79_0, h128_0])
    (fun j ↦ by fin_cases j; exacts [h118_0, h138_0]) (fun j ↦ by fin_cases j; exacts [h119_0, h139_0])
    (M := ![![w_122_0, w_142_0], ![w_124_0, w_144_0]])
    (fun j ↦ by fin_cases j; exacts [h122_0, h142_0]) (fun j ↦ by fin_cases j; exacts [h124_0, h144_0])
    (A' := ![![w_123_0, w_143_0], ![w_125_0, w_145_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h123_0, h143_0, h125_0, h145_0])
  rw [h2, h1]
"""


d = sys.argv[1]
ab = abbreviations(open(f'{d}/TopSetup.lean').read())
for part in parts_of(d):
    path = f'{d}/generated/Stage_online_eval_{part}.lean'
    ns = f'Stage_online_eval_{part}'
    text = open(path).read()
    _, names, atoms = parse(path, 'generatedRoot')
    body = text[text.index('abbrev generatedRoot.body'):]
    body = body[:body.index('\n\n')]
    lets = dict(re.findall(r'^  let (w_\S+) : [^\n]*?:= (.*)$', body, re.M))
    loops = [nm for nm, a in zip(names, atoms) if a.startswith('MxxIR.IterRuns') and LUT_A in
             text[text.index('abbrev ' + re.search(r'(sequential_generatedRoot_\d+) ', a).group(1) + '.constraints_0 '):][:40000]]
    out = f'''import Part_{part}_KeysLuts

/-! The keys of the online root's LUTs (written by `generator/gen_keys_root.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof
'''
    out += root_keys(path, ns, names, atoms, lets, ab, loops[0], 'round')
    out += root_keys(path, ns, names, atoms, lets, ab, loops[1], 'round')
    out += root_keys(path, ns, names, atoms, lets, ab, loops[2], 'branches')
    out += on_offset_keys(path, ns, names, atoms, lets, ab)
    out += pre_offset_keys(d, part, ab)
    out += f'\nend RefreshTopProof.Part_{part}\n'
    open(f'{d}/Part_{part}_KeysRoot.lean', 'w').write(out)
    print(part, loops)
    break
