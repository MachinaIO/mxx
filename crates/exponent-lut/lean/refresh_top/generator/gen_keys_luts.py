"""Write `Part_<c_j>_KeysLuts.lean`: the keys of every LUT as `Σ_i branchKey i`, on the
preprocessing side (its LUT loop bodies, loops and parallels) and the online side.

Usage: `python3 gen_keys_luts.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_part import parts_of
from patterns import destruct_inline, parse_inline
from segment import abbreviations, short
from layouts import layouts
from ids import online, pre

LAYOUT = {ring: (lay, 'layout' + ring) for ring, lay in layouts(sys.argv[1]).items()}
# the decoding ring of instance `I` (laid out by `layoutR`, see `view.instance_names`)
LAYOUT['Ri0'] = LAYOUT['Q1']
LUT_A = '[114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]'


def body_info(path, name, ab):
    text = open(path).read()
    i = text.index(f'def {name} ')
    body = text[i:text.index('\n\n', i)]
    head = text[i:i + 2000]
    idx = re.findall(r'\((i_\d|_) : Nat\)', head[:head.index('(inputs')])
    types = dict((m.group(1), short(m.group(2), ab)) for m in
                 re.finditer(r'let (w_\S+) : (.*?) :=', body))
    q, m = re.match(r'ExactMatrix (\S+) 8 1 (\d+)', types['w_6_0']).groups()
    W, qp, mp = re.match(r'Fin (\d+) → ExactMatrix (\S+) 8 1 (\d+)', types['w_11_0']).groups()
    _, n, a = parse_inline(path, name)
    atom = dict(zip(n, a))
    basis = short(re.search(r'blockModSwitchRuns (\[[^\]]*\])', atom['h29_0']).group(1), ab)
    level = re.search(r'\[\.u64Le \((\d+)\), \.integer', atom['h3_0']).group(1)
    return dict(idx=idx, q=q, m=int(m), W=int(W), qp=qp, mp=int(mp), basis=basis, level=level)


def pre_step(part, d, name, ab):
    ns = f'Stage_preprocess_{part}'
    path = f'{d}/generated/{ns}.lean'
    info = body_info(path, name, ab)
    q, m, W, qp, mp = info['q'], info['m'], info['W'], info['qp'], info['mp']
    sig = 'sigma2' if W == 2 else 'sigma'
    lay, hlay = LAYOUT[qp]
    member = len(info['idx']) == 2
    obind = '{o : Nat} ' if member else ''
    oarg = 'o ' if member else ''
    dd = destruct_inline(path, ns, name, 'h')
    K = (f"branchKey (q := {q}) (q' := {qp}) (m := {m}) (m' := {mp}) {m} {info['basis']} ({sig} i).toNat "
         f"![![A00, A01], ![A10, A11]] (lutA model key {q} {m} {info['level']} ({sig} i)) "
         f"(fun k j ↦ ![![f00, f01], ![f10, f11]] k j ⟨i, hi⟩)")
    st = f'Keys {q} {mp}'
    return f'''
set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem preKeyStep_{name} {{model : HashModel}} {{params : {ns}.Params}} {obind}{{i : Nat}}
    (hi : i < {W}) {{state next : {st}}} {{key : ByteArray}}
    {{A00 A10 A01 A11 : ExactMatrix {q} N 1 {m}}} {{f00 f10 f01 f11 : Fin {W} → ExactMatrix {qp} N 1 {mp}}}
    (h : {ns}.{name} Backend.backend model params {oarg}i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, key, A00, A10, f00, A01, A11, f10, f01, f11,
        ()) next) :
    next = state + lutKeys (q := {q}) (q' := {qp}) (m := {m}) (m' := {mp}) {m} {info['basis']} model key {info['level']} {sig}
      ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
{dd}  have hidx : w_12_0 = Int.ofNat i := rfl
  rw [hidx] at h13_0 h23_0 h46_0 h51_0
  obtain ⟨_, e13⟩ := familyGet_eq (k := i) h13_0
  obtain ⟨_, e23⟩ := familyGet_eq (k := i) h23_0
  obtain ⟨_, e46⟩ := familyGet_eq (k := i) h46_0
  obtain ⟨_, e51⟩ := familyGet_eq (k := i) h51_0
  have hAd : lutA model key {q} {m} {info['level']} ({sig} i) = ![![w_3_0, w_33_0], ![w_4_0, w_34_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h33_0).symm
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h34_0).symm
  have hsw : (fun k j ↦ ![![f00, f01], ![f10, f11]] k j ⟨i, hi⟩) = ![![w_13_0, w_46_0], ![w_23_0, w_51_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact e13.symm
    · exact e46.symm
    · exact e23.symm
    · exact e51.symm
  have hK := branchKey_of_pre (q := {q}) (q' := {qp}) (m := {m}) (m' := {mp}) (by norm_num) Backend.{lay} {hlay}
    rfl rfl (digits := {m}) (basis := {info['basis']}) (index := {sig} i)
    (A := ![![A00, A01], ![A10, A11]]) (Ad := ![![w_3_0, w_33_0], ![w_4_0, w_34_0]])
    h7_0 h9_0 h18_0 h20_0 h10_0 h21_0 h5_0 h35_0
    (sw := ![![w_13_0, w_46_0], ![w_23_0, w_51_0]])
    (dA := ![w_14_0, w_47_0]) (rA := ![w_15_0, w_48_0]) (dB := ![w_24_0, w_52_0]) (rB := ![w_25_0, w_53_0])
    (dC := ![w_30_0, w_58_0]) (rC := ![w_31_0, w_59_0]) (dE := ![w_38_0, w_63_0]) (rE := ![w_39_0, w_64_0])
    (fun j ↦ by fin_cases j; exacts [h14_0, h47_0]) (fun j ↦ by fin_cases j; exacts [h15_0, h48_0])
    (fun j ↦ by fin_cases j; exacts [h24_0, h52_0]) (fun j ↦ by fin_cases j; exacts [h25_0, h53_0])
    (M := ![![w_28_0, w_56_0], ![w_36_0, w_61_0]])
    (fun j ↦ by fin_cases j; exacts [h28_0, h56_0]) (fun j ↦ by fin_cases j; exacts [h36_0, h61_0])
    (Asw := ![![w_29_0, w_57_0], ![w_37_0, w_62_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h29_0, h57_0, h37_0, h62_0])
    (fun j ↦ by fin_cases j; exacts [h30_0, h58_0]) (fun j ↦ by fin_cases j; exacts [h31_0, h59_0])
    (fun j ↦ by fin_cases j; exacts [h38_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h64_0])
    (K := ![![w_42_0, w_67_0], ![w_70_0, w_73_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h67_0]) (fun j ↦ by fin_cases j; exacts [h70_0, h73_0])
  rw [← hAd, ← hsw] at hK
  rw [lutKeys, dif_pos hi, ← hK, hout]
  rfl
'''


def pre_loop(part, d, name, ab):
    ns = f'Stage_preprocess_{part}'
    path = f'{d}/generated/{ns}.lean'
    info = body_info(path, name, ab)
    q, m, W, qp, mp = info['q'], info['m'], info['W'], info['qp'], info['mp']
    sig = 'sigma2' if W == 2 else 'sigma'
    member = len(info['idx']) == 2
    obind = '{o : Nat} ' if member else ''
    oarg = 'o ' if member else ''
    st = f'Keys {q} {mp}'
    LK = (f"lutKeys (q := {q}) (q' := {qp}) (m := {m}) (m' := {mp}) {m} {info['basis']} model key "
          f"{info['level']} {sig} ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]]")
    return f'''
/-- The loop `{name}` accumulates the keys of its first `count` branches. -/
theorem preKeyLoop_{name} {{model : HashModel}} {{params : {ns}.Params}} {obind}{{key : ByteArray}}
    {{A00 A10 A01 A11 : ExactMatrix {q} N 1 {m}}} {{f00 f10 f01 f11 : Fin {W} → ExactMatrix {qp} N 1 {mp}}}
    {{count : Nat}} (hcount : count ≤ {W}) {{init final : {st}}}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : {st}) ↦
      {ns}.{name} Backend.backend model params {oarg}i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, key, A00, A10, f00, A01, A11, f10,
          f01, f11, ()) next) count init final) :
    final = init + ∑ i ∈ Finset.range count, {LK} i := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : {st}) ↦ k ≤ {W} → s = init + ∑ i ∈ Finset.range k, {LK} i)
    (fun _ ↦ by simp) (fun i current next hinv hbody hk ↦ ?_) hruns hcount
  rw [preKeyStep_{name} (by omega) hbody, hinv (by omega), Finset.sum_range_succ, add_assoc]
'''



def on_info(path, name, ab):
    text = open(path).read()
    i = text.index(f'def {name} ')
    body = text[i:text.index('\n\n', i)]
    head = text[i:i + 3000]
    idx = re.findall(r'\((i_\d|_) : Nat\)', head[:head.index('(inputs')])
    types = dict((m.group(1), short(m.group(2), ab)) for m in
                 re.finditer(r'let (w_\S+) : (.*?) :=', body))
    q, m = re.match(r'ExactMatrix (\S+) 8 1 (\d+)', types['w_15_0']).groups()
    W, qp, mp = re.match(r'Fin (\d+) → ExactMatrix (\S+) 8 1 (\d+)', types['w_79_0']).groups()
    _, n, a = parse_inline(path, name)
    atom = dict(zip(n, a))
    basis = short(re.search(r'blockModSwitchRuns (\[[^\]]*\])', atom['h37_0']).group(1), ab)
    level = re.search(r'\[\.u64Le \((\d+)\), \.integer', atom['h6_0']).group(1)
    nxt = re.search(r'familyGetDynamic \S+ (w_\d+_0) ', atom['h82_0']).group(1)
    one = re.search(rf'let {nxt} : Int := \(w_2_0 \+ (w_\d+_0)\)', body).group(1)
    return dict(idx=idx, q=q, m=int(m), W=int(W), qp=qp, mp=int(mp), basis=basis, level=level,
                nxt=nxt, one=one)


def on_step(part, d, name, ab):
    ns = f'Stage_online_eval_{part}'
    path = f'{d}/generated/{ns}.lean'
    info = on_info(path, name, ab)
    q, m, W, qp, mp = info['q'], info['m'], info['W'], info['qp'], info['mp']
    sig = 'sigma2' if W == 2 else 'sigma'
    lay, hlay = LAYOUT[qp]
    member = len(info['idx']) == 2
    obind = '{o : Nat} ' if member else ''
    oarg = 'o ' if member else ''
    dd = destruct_inline(path, ns, name, 'hbody')
    st = f'LoopState {q} {qp} {mp}'
    LK = (f"lutKeys (q := {q}) (q' := {qp}) (m := {m}) (m' := {mp}) {m} {info['basis']} model key "
          f"{info['level']} {sig} ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]]")
    inputs = ('(state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,\n'
              '        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,\n'
              '        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,\n'
              '        fam10, fam11, ())')
    cur = inputs.replace('state', 'current')
    return f"""
set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem keyStep_{name} {{model : HashModel}} {{params : {ns}.Params}} {obind}{{i : Nat}}
    (hi : i < {W - 1}) {{state next : {st}}} {{key : ByteArray}} {{h0 h1 : Fin {W} → ExactMatrix {q} N 1 {m}}}
    {{c0 A00 A10 c1 A01 A11 : ExactMatrix {q} N 1 {m}}}
    {{fam00 fam01 fam10 fam11 : Fin {W} → ExactMatrix {qp} N 1 {mp}}}
    (hswitch : state.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i, by omega⟩)
    (hbody : {ns}.{name} Backend.backend model params {oarg}i
      {inputs} next) :
    keyTuple next.key = keyTuple state.key + {LK} i ∧
      next.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i + 1, by omega⟩ := by
{dd}  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : {info['nxt']} = Int.ofNat (i + 1) := by
    simp only [{info['nxt']}, {info['one']}, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hnext] at h82_0 h84_0 h86_0 h88_0
  obtain ⟨_, hs0⟩ := familyGet_eq h82_0
  obtain ⟨_, hs1⟩ := familyGet_eq h84_0
  obtain ⟨_, hs2⟩ := familyGet_eq h86_0
  obtain ⟨_, hs3⟩ := familyGet_eq h88_0
  have hAd : lutA model key {q} {m} {info['level']} ({sig} i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  have hK := branchKey_of_stacks (q := {q}) (q' := {qp}) (m := {m}) (m' := {mp}) (by norm_num) Backend.{lay}
    {hlay} rfl rfl (digits := {m}) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := {sig} i)
    h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (sw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![w_3_0, w_10_0]) (Ad := ![![w_6_0, w_11_0], ![w_7_0, w_12_0]])
    h6_0_1 h8_0 h11_0_1 h13_0 h14_0 (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  rw [← hAd, hswitch] at hK
  subst hout
  refine ⟨?_, ?_⟩
  · rw [lutKeys, dif_pos (by omega), ← hK]
    rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The loop `{name}` accumulates the keys of its first `count` branches. -/
theorem keyLoop_{name} {{model : HashModel}} {{params : {ns}.Params}} {obind}{{key : ByteArray}}
    {{h0 h1 : Fin {W} → ExactMatrix {q} N 1 {m}}} {{c0 A00 A10 c1 A01 A11 : ExactMatrix {q} N 1 {m}}}
    {{fam00 fam01 fam10 fam11 : Fin {W} → ExactMatrix {qp} N 1 {mp}}} {{count : Nat}} (hcount : count ≤ {W - 1})
    {{final : {st}}}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : {st}) ↦
      {ns}.{name} Backend.backend model params {oarg}i
        {cur} next) count
      (0, 0, 0, 0, 0, 0, fam00 0, fam01 0, fam10 0, fam11 0, ()) final) :
    keyTuple final.key = ∑ i ∈ Finset.range count, {LK} i ∧
      final.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : {st}) ↦ ∀ hk : k ≤ {W - 1},
      keyTuple s.key = ∑ i ∈ Finset.range k, {LK} i ∧
        s.switch = fun k' j ↦ ![![fam00, fam01], ![fam10, fam11]] k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨rfl, ?_⟩
    funext k j
    fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := keyStep_{name} (by omega) hsw hbody
    exact ⟨by rw [hrel', hrel, Finset.sum_range_succ], hsw'⟩
"""


def on_round(part, d, k, kind, ab):
    """The keys of the online member LUT `parallel_generatedRoot_<k>`: `kind` `'round'` (moved
    down), `'branches'`, `'offset'` (plus the offset's keys) or `'smudge'` (moved down, weighted and
    placed)."""
    from gen_luts import round_args
    from patterns import destruct, parse
    ns = f'Stage_online_eval_{part}'
    path = f'{d}/generated/{ns}.lean'
    text = open(path).read()
    name = f'parallel_generatedRoot_{k}'
    body = text[text.index(f'abbrev {name}.body'):]
    body = body[:body.index('\n\n')]
    lets = dict(re.findall(r'^  let (w_\S+) : [^\n]*?:= (.*)$', body, re.M))
    _, n, a = parse(path, name)
    atom = dict(zip(n, a))
    loop_atom = next(x for x in n if x != '-' and atom[x].startswith('MxxIR.IterRuns'))
    loop = re.search(r'(sequential_parallel_generatedRoot_\d+_\d+)', atom[loop_atom]).group(1)
    state = re.findall(r'\b(w_\d+_\d+)\b', atom[loop_atom].split(') (Int.toNat')[1])[-10:]
    init_gets = [x for x in n if x != '-' and re.match(r'MxxRuntime\.familyGetStatic \S+ \(0\) ', atom[x])][:4]
    help_gets = [x for x in n if x != '-' and re.match(r'MxxRuntime\.familyGetStatic \S+ \(3\) ', atom[x])][:2]
    hashes = [x for x in n if x != '-' and atom[x].startswith('MxxRuntime.hashSample')]
    hash_of = {int(re.search(r'\.bytes \[99[^\]]*\], \.u64Le \((\d)\)', atom[x]).group(1)): x for x in hashes}
    r = round_args(n, a, lets, state)
    w = r['wire']
    info = on_info(path, loop, ab)
    q, m, qp, mp = info['q'], info['m'], info['qp'], info['mp']
    lay, hlay = LAYOUT[qp]
    level = info['level']
    dd = destruct(path, ns, name, 'h')
    offset = kind == 'offset'
    off_bind = ' {off : Blocks ' + q + f' {mp}' + '}' if offset else ''
    off_in = ', off.1, off.2.1, off.2.2.1, off.2.2.2.1, off.2.2.2.2.1, off.2.2.2.2.2.1' if offset else ''
    st = f"({', '.join(state)}, ())"
    kj = {0: '0 0', 1: '0 1', 2: '1 0', 3: '1 1'}
    LK = (f"lutKeys (q := {q}) (q' := {qp}) (m := {m}) (m' := {mp}) {m} {info['basis']} model key "
          f"{level} sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]]")
    S = f'(∑ i ∈ Finset.range 4, {LK} i)'
    rev = {v: k2 for k2, v in lets.items()}
    kk = [[r['K0'][0], r['K0'][1]], [r['K1'][0], r['K1'][1]]]
    sums = [[rev[f'MxxRuntime.matrixAdd {state[2 + 2 * a_ + b_]} {w(kk[a_][b_])}'] for b_ in range(2)]
            for a_ in range(2)]
    out_q = qp if kind in ('round', 'smudge') else q
    out_m = 1 if kind == 'smudge' else mp
    if kind == 'round' or kind == 'smudge':
        def modswitch_of(src):
            return next(x for x in n if x != '-' and re.match(
                rf'MxxRuntime\.blockModSwitchRuns \[[^\]]*\] \(1\) {src} ', atom[x]))
        fin_k = [[modswitch_of(sums[a_][b_]) for b_ in range(2)] for a_ in range(2)]
        basis2 = short(re.search(r'blockModSwitchRuns (\[[^\]]*\])', atom[fin_k[0][0]]).group(1), ab)
    if kind == 'round':
        concl = f'keyMap (blockModSwitch (p := {qp}) {basis2} 1) {S}'
    elif kind == 'branches':
        concl = S
    elif kind == 'offset':
        concl = f'{S} + (off.1, off.2.1, off.2.2.1, off.2.2.2.1, ())'
    else:
        concl = (f'keyMap (fun x ↦ multiplyMonomial (blockModSwitch (p := {qp}) {basis2} 1 x * '
                 f'(dec (inner := 6) 6 (matrixPolynomial [smudgeWeight{k} ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] : '
                 f'ExactMatrix {qp} N 1 1))) (Int.fdiv (Int.ofNat o) 20)) {S}')
    head = f"""
set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem keyRound{k} {{model : HashModel}} {{key : ByteArray}} {{params : {ns}.Params}} {{o : Nat}}
    {{fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix {qp} N 1 {mp}}} {{h0 h1 : Fin 4 → ExactMatrix {q} N 1 {m}}}
    {{c0 A00 A10 c1 A01 A11 : ExactMatrix {q} N 1 {m}}}{off_bind} {{outputs : Blocks {out_q} {out_m}}}
    (h : {ns}.{name} Backend.backend model params o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11{off_in}, ()) outputs) :
    (outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1, ()) = {concl} := by
"""
    proof = f"""{dd}  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) {init_gets[0]}
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) {init_gets[1]}
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) {init_gets[2]}
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) {init_gets[3]}
  rw [hwa, hwb, hwc, hwd] at {loop_atom}
  rw [show Int.toNat 3 = 3 from rfl] at {loop_atom}
  obtain ⟨hloop, hloopSwitch⟩ := keyLoop_{loop} (count := 3) (by decide) {loop_atom}
"""
    for c in range(4):
        proof += (f"  have hA{c} : {w(hash_of[c])} = lutA model key {q} {m} {level} (sigma (3 : Fin 4)) {kj[c]} :=\n"
                  f"    hashSample_eq {hash_of[c]}\n")
    proof += f"  rw [hA0] at {r['help'][0]}\n  rw [hA2] at {r['help'][1]}\n"
    proof += f"  rw [hA1] at {r['help'][2]}\n  rw [hA3] at {r['help'][3]}\n"
    two = lambda key: f"(fun j ↦ by fin_cases j; exacts [{r[key][0]}, {r[key][1]}])"
    proof += f"""  have hK4 := branchKey_of_stacks (q := {q}) (q' := {qp}) (m := {m}) (m' := {mp}) (by norm_num) Backend.{lay}
    {hlay} rfl rfl (digits := {m}) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma 3)
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
    (h := ![{r['help_rows'][0]}, {r['help_rows'][1]}]) (Ad := lutA model key {q} {m} {level} (sigma (3 : Fin 4)))
    {' '.join(r['help'][:5])}
    (K := ![![{w(r['K0'][0])}, {w(r['K0'][1])}], ![{w(r['K1'][0])}, {w(r['K1'][1])}]]) {two('K0')} {two('K1')}
  rw [hloopSwitch] at hK4
  have hS : ({sums[0][0]}, {sums[0][1]}, {sums[1][0]}, {sums[1][1]}, ()) = {S} := by
    rw [Finset.sum_range_succ, ← hloop, lutKeys, dif_pos (by norm_num)]
    refine (show _ = keyTuple (LoopState.key {st}) +
      keyTuple ![![{w(r['K0'][0])}, {w(r['K0'][1])}], ![{w(r['K1'][0])}, {w(r['K1'][1])}]] from rfl).trans ?_
    rw [hK4]
    rfl
"""
    if kind == 'round':
        proof += f"""  rw [← hS, hout]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl))) <;>
    exact blockModSwitchRuns_unique ‹_› (blockModSwitchRuns_fn ‹_› _)
"""
    proof_round = proof
    if kind == 'round':
        proof = proof_round.replace("""  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl))) <;>
    exact blockModSwitchRuns_unique ‹_› (blockModSwitchRuns_fn ‹_› _)
""", f"""  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · exact blockModSwitchRuns_unique {fin_k[0][0]} (blockModSwitchRuns_fn {fin_k[0][0]} _)
  · exact blockModSwitchRuns_unique {fin_k[0][1]} (blockModSwitchRuns_fn {fin_k[0][1]} _)
  · exact blockModSwitchRuns_unique {fin_k[1][0]} (blockModSwitchRuns_fn {fin_k[1][0]} _)
  · exact blockModSwitchRuns_unique {fin_k[1][1]} (blockModSwitchRuns_fn {fin_k[1][1]} _)
""")
    elif kind == 'smudge':
        slice_atom = next(x for x in n if x != '-' and re.match(r'MxxRuntime\.sliceMatrix \S+ 0 1 \(Int\.fmod', atom[x]))
        row = re.match(r'MxxRuntime\.sliceMatrix (\S+) ', atom[slice_atom]).group(1)
        weight_wire = re.search(r'(w_\d+_0)$', atom[slice_atom]).group(1)
        dec_atom = next(x for x in n if x != '-' and re.match(rf'MxxRuntime\.gadgetDecomposeRuns backend 32768 \d+ {weight_wire} ', atom[x]))
        reb_atom = next(x for x in n if x != '-' and re.match(rf'MxxRuntime\.compactCenteredRebaseRuns 16384 {w(dec_atom)} ', atom[x]))
        concats = [x for x in n if x != '-' and re.match(r'MxxRuntime\.concatColumns ', atom[x])
                   and (re.search(rf'{row}$', atom[x]) or re.search(rf'{row[:-2]}_concat_\d+$', atom[x]))]
        weights = re.findall(r'concatColumns (w_\d+_0|w_\d+_concat_\d+) (w_\d+_0)', ' '.join(atom[x] for x in concats))
        firsts = [weights[0][0]] + [b_ for _, b_ in weights]
        fdiv = next(x for x in lets if lets[x].startswith('(Int.fdiv ((Int.ofNat'))
        monos = [x for x in lets if re.match(rf'MxxRuntime\.multiplyMonomial \S+ {fdiv}$', lets[x])]
        prods = [re.match(r'MxxRuntime\.multiplyMonomial (\S+) ', lets[x]).group(1) for x in monos]
        proof += f"""  simp only [concatColumns] at {" ".join(concats)}
  have hrow : ∀ c : Fin 20, {row} 0 c = (matrixPolynomial [smudgeWeight{k} c] : ExactMatrix {qp} N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [{", ".join(concats)}, {", ".join(firsts)}, smudgeWeight{k}]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; rfl
  rw [hd] at {slice_atom}
  have hK : {weight_wire} = matrixPolynomial [smudgeWeight{k} ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry {slice_atom} (by simp; omega)]
    exact hrow _
  have hW : {w(reb_atom)} = dec (inner := 6) 6 (matrixPolynomial [smudgeWeight{k} ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] :
      ExactMatrix {qp} N 1 1) := by
    rw [rebase_self (by norm_num) {reb_atom}, ← hK]
    exact dec_eq {dec_atom}
  rw [← hS, hout]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
"""
        for (a_, b_), mono, prod in zip([(0, 0), (0, 1), (1, 0), (1, 1)], monos[:4], prods[:4]):
            proof += f"""  · show multiplyMonomial (matrixMul {re.match(r'MxxRuntime.matrixMul (\S+) ', lets[prod]).group(1)} {w(reb_atom)}) {fdiv} = _
    rw [hW, blockModSwitchRuns_unique {fin_k[a_][b_]} (blockModSwitchRuns_fn {fin_k[a_][b_]} _)]
    rfl
"""
    elif kind == 'branches':
        proof += """  rw [← hS, hout]
  try rfl
"""
    elif kind == 'offset':
        proof += """  rw [← hS, hout]
  try rfl
"""
    return head + proof


def pre_par(part, d, k, kind, ab, weights=None):
    # every decoder's smudge digits have the weights of decoder `0`'s LUT
    weights = weights or f'smudgeWeight{online(560, part)}'
    """The keys the preprocessing computes in its LUT parallel `parallel_generatedRoot_<k>`."""
    from patterns import destruct, parse
    ns = f'Stage_preprocess_{part}'
    path = f'{d}/generated/{ns}.lean'
    text = open(path).read()
    name = f'parallel_generatedRoot_{k}'
    body = text[text.index(f'abbrev {name}.body'):]
    body = body[:body.index('\n\n')]
    lets = dict(re.findall(r'^  let (w_\S+) : [^\n]*?:= (.*)$', body, re.M))
    _, n, a = parse(path, name)
    atom = dict(zip(n, a))
    loop = re.search(r'(sequential_parallel_generatedRoot_\d+_\d+)', atom['h10_3']).group(1)
    info = body_info(path, loop, ab)
    q, m, qp, mp = info['q'], info['m'], info['qp'], info['mp']
    LK = (f"lutKeys (q := {q}) (q' := {qp}) (m := {m}) (m' := {mp}) {m} {info['basis']} model key "
          f"{info['level']} sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]]")
    S = f'(∑ i ∈ Finset.range 4, {LK} i)'
    dd = destruct(path, ns, name, 'h')
    offset = kind == 'offset'
    off_bind = f' {{off : Keys {q} {mp}}}' if offset else ''
    off_in = ', off.1, off.2.1, off.2.2.1, off.2.2.2.1' if offset else ''
    out_q = qp if kind in ('round', 'smudge') else q
    out_m = 1 if kind == 'smudge' else mp
    if kind in ('round', 'smudge'):
        basis2 = short(re.search(r'blockModSwitchRuns (\[[^\]]*\])', atom['h11_0']).group(1), ab)
    if kind == 'round':
        concl = f'keyMap (blockModSwitch (p := {qp}) {basis2} 1) {S}'
    elif kind == 'branches':
        concl = S
    elif kind == 'offset':
        concl = f'{S} + off'
    else:
        concl = (f'keyMap (fun x ↦ multiplyMonomial (blockModSwitch (p := {qp}) {basis2} 1 x * '
                 f'(dec (inner := 6) 6 (matrixPolynomial [{weights} ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] : '
                 f'ExactMatrix {qp} N 1 1))) (Int.fdiv (Int.ofNat o) 20)) {S}')
    head = f"""
set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem preKeyRound{k} {{model : HashModel}} {{key : ByteArray}} {{params : {ns}.Params}} {{o : Nat}}
    {{A00 A10 A01 A11 : ExactMatrix {q} N 1 {m}}} {{f00 f10 f01 f11 : Fin 4 → ExactMatrix {qp} N 1 {mp}}}{off_bind}
    {{outputs : Keys {out_q} {out_m}}}
    (h : {ns}.{name} Backend.backend model params o
      (key, A00, A10, f00, A01, A11, f10, f01, f11{off_in}, ()) outputs) :
    outputs = {concl} := by
{dd}  rw [show Int.toNat 4 = 4 from rfl] at h10_3
  have hloop := preKeyLoop_{loop} (count := 4) (by decide) h10_3
  rw [show ((0 : ExactMatrix {q} N 1 {mp}), (0 : ExactMatrix {q} N 1 {mp}), (0 : ExactMatrix {q} N 1 {mp}),
    (0 : ExactMatrix {q} N 1 {mp}), ()) = (0 : Keys {q} {mp}) from rfl, zero_add] at hloop
"""
    if kind == 'round':
        proof = f"""  rw [hout, ← hloop]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · exact blockModSwitchRuns_unique h11_0 (blockModSwitchRuns_fn h11_0 _)
  · exact blockModSwitchRuns_unique h12_0 (blockModSwitchRuns_fn h12_0 _)
  · exact blockModSwitchRuns_unique h13_0 (blockModSwitchRuns_fn h13_0 _)
  · exact blockModSwitchRuns_unique h14_0 (blockModSwitchRuns_fn h14_0 _)
"""
    elif kind == 'branches':
        proof = """  rw [hout, ← hloop]
"""
    elif kind == 'offset':
        proof = """  rw [hout, ← hloop]
  rfl
"""
    else:
        concats = [x for x in n if x != '-' and atom[x].startswith('MxxRuntime.concatColumns ')]
        consts = [x for x in lets if lets[x].startswith('(MxxRuntime.matrixPolynomial')]
        bms = [x for x in n if x != '-' and atom[x].startswith('MxxRuntime.blockModSwitchRuns')]
        prods = [x for x in lets if lets[x].startswith('MxxRuntime.matrixMul ')]
        proof = f"""  simp only [concatColumns] at {" ".join(concats)}
  have hrow : ∀ c : Fin 20, w_32_0 0 c = (matrixPolynomial [{weights} c] : ExactMatrix {qp} N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [{", ".join(concats)}, {", ".join(consts)}, {weights}]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; rfl
  rw [hd] at h33_0
  have hK : w_33_0 = matrixPolynomial [{weights} ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry h33_0 (by simp; omega)]
    exact hrow _
  have hW : w_35_0 = dec (inner := 6) 6 (matrixPolynomial [{weights} ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] :
      ExactMatrix {qp} N 1 1) := by
    rw [rebase_self (by norm_num) h35_0, ← hK]
    exact dec_eq h34_0
  rw [hout, ← hloop]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
"""
        for b in bms:
            proof += f"""  · rw [← blockModSwitchRuns_unique {b} (blockModSwitchRuns_fn {b} _), ← hW]
    rfl
"""
    return head + proof


d = sys.argv[1]
ab = abbreviations(open(f'{d}/TopSetup.lean').read())
for part in parts_of(d):
    ppath = f'{d}/generated/Stage_preprocess_{part}.lean'
    ptext = open(ppath).read()
    bodies = []
    for nm in re.findall(r'^def (sequential_\w+) ', ptext, re.M):
        i = ptext.index(f'abbrev {nm}.constraints_0 ')
        if LUT_A in ptext[i:i + ptext[i:].index('\n\n')]:
            bodies.append(nm)
    out = [f'''import TopKeyForms
import TopLut2
import Stage_preprocess_{part}
import Stage_online_eval_{part}
import Part_{part}_Smudge

/-! The keys of every LUT (written by `generator/gen_keys_luts.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof
''']
    for nm in bodies:
        out.append(pre_step(part, d, nm, ab))
        out.append(pre_loop(part, d, nm, ab))
    opath = f'{d}/generated/Stage_online_eval_{part}.lean'
    otext = open(opath).read()
    obodies = []
    for nm in re.findall(r'^def (sequential_\w+) ', otext, re.M):
        i = otext.index(f'abbrev {nm}.constraints_0 ')
        if LUT_A in otext[i:i + otext[i:].index('\n\n')]:
            obodies.append(nm)
    for nm in obodies:
        out.append(on_step(part, d, nm, ab))
    out.append(on_round(part, d, online(121, part), 'round', ab))
    out.append(on_round(part, d, online(392, part), 'round', ab))
    out.append(on_round(part, d, online(393, part), 'branches', ab))
    out.append(on_round(part, d, online(312, part), 'offset', ab))
    out.append(pre_par(part, d, pre(42, part), 'round', ab))
    out.append(pre_par(part, d, pre(217, part), 'round', ab))
    out.append(pre_par(part, d, pre(219, part), 'branches', ab))
    out.append(pre_par(part, d, pre(152, part), 'offset', ab))
    for nm in bodies:
        mm = re.match(r'sequential_parallel_generatedRoot_(\d+)_10$', nm)
        if mm and int(mm.group(1)) not in [pre(k, part) for k in (42, 217, 219, 152)]:
            out.append(pre_par(part, d, int(mm.group(1)), 'smudge', ab))
    smudge_text = open(f'{d}/Part_{part}_Smudge.lean').read()
    for k in re.findall(r'theorem lut(\d+) ', smudge_text):
        out.append(on_round(part, d, int(k), 'smudge', ab))
    out.append(f'\nend RefreshTopProof.Part_{part}\n')
    open(f'{d}/Part_{part}_KeysLuts.lean', 'w').write(''.join(out))
    print(obodies)
    print(part, bodies)
    break
