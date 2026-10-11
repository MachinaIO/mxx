"""LUT loop lemmas: one generated loop iteration adds a branch, and the loop accumulates the first
`W - 1` branches (see `TopLut.lean`).

`loop_lemmas(path, ns, name, ...)` returns the Lean text of `lutStep<tag>` and `lutLoop<tag>` for
the loop body definition `name` of stage file `path` (namespace `ns`).
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from patterns import destruct_inline, parse_inline


def loop_lemmas(path, ns, name, tag, q, qp, p, m, mp, layout, hlayout, qeq, level, bound,
                params='params', extra_args='', width=4):
    """`name`: the loop body (`sequential_…`), whose inputs are the state (10 blocks), the helper
    rows `h0`, the hash key, `h1`, the input encoding `c 0, A 0 0, A 1 0, c 1, A 0 1, A 1 1`, and
    the switching families. `extra_args`: the leading Nat arguments of the body before `i` (e.g.
    the member index of a body nested in a member parallel). `width`: `4` (`TopLut`) or `2`
    (`TopLut2`)."""
    d = destruct_inline(path, ns, name, 'hbody')
    _, n, a = parse_inline(path, name)
    gets = [x for x, y in zip(n, a) if x != '-' and y.startswith('MxxRuntime.familyGetDynamic')]
    help_gets, switch_gets = gets[:2], gets[2:]
    atom = dict(zip(n, a))
    nxt = re.search(r'familyGetDynamic \S+ (w_\d+_0) ', atom[switch_gets[0]]).group(1)
    text = open(path).read()
    body = text[text.index(f'def {name} '):]
    one = re.search(rf'let {nxt} : Int := \(w_2_0 \+ (w_\d+_0)\)', body).group(1)
    st = f'LoopState {q} {qp} {mp}'
    args = extra_args + ' ' if extra_args else ''
    text = f'''
/-- The error bound added by one branch of the LUT `{tag}`. -/
abbrev lutBranchBound{tag} (B Bh : Nat) : Nat :=
  ({m} + {m}) * N * Bh * {bound} + ({m} + {m}) * N * B * {bound} + (N * 1 * ({p} / 2) + {p} / 2)

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- One iteration of the LUT loop `{tag}` adds branch `i` to the accumulated relation and carries
the switching ciphertext of branch `i + 1`. -/
theorem lutStep{tag} {{model : HashModel}} {{key : ByteArray}} {{sm sp μ : ErrorPoly N}}
    (hsm : polyNorm sm ≤ 1)
    {{g' : ExactMatrix {qp} N 1 {m}}} {{g : ExactMatrix {q} N 1 {m}}}
    (hg' : gadgetMatrixRuns Backend.backend 32768 {m} g') (hg : centeredRebaseRuns g' g)
    {{c : Fin 2 → ExactMatrix {q} N 1 {m}}} {{A : Fin 2 → Fin 2 → ExactMatrix {q} N 1 {m}}} {{B : Nat}}
    (henc : Rel sm A (fun j ↦ reducePoly {q} N ((({p} : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {{h0 h1 : Fin 4 → ExactMatrix {q} N 1 {m}}} {{Bh : Nat}}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key {q} {m} {level} (sigma i))
      (fun j ↦ reducePoly {q} N ((({p} : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {{fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix {qp} N 1 {mp}}}
    {{i : Nat}} (hi : i < 3) {{state next : {st}}}
    (hswitch : state.switch = fun k j ↦ fam k j ⟨i, by omega⟩)
    {{T : Fin 2 → ExactMatrix {q} N 1 {mp}}} {{Bs : Nat}} (hstate : Rel sm state.key T state.row Bs)
    {{{params}}} {extra_binders(extra_args)}
    (hbody : {ns}.{name} Backend.backend model {params} {args}i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0, c 1, A 0 1, A 1 1, fam 0 0,
        fam 0 1, fam 1 0, fam 1 1, ()) next) :
    Rel sm next.key (T + branchTarget (q := {q}) (p := {p}) μ sp fam ⟨i, by omega⟩) next.row
        (Bs + lutBranchBound{tag} B Bh) ∧
      next.switch = fun k j ↦ fam k j ⟨i + 1, by omega⟩ := by
{d}  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : {nxt} = Int.ofNat (i + 1) := by
    simp only [{nxt}, {one}, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at {" ".join(help_gets)}
  rw [hnext] at {" ".join(switch_gets)}
  obtain ⟨_, hw3⟩ := familyGet_eq {help_gets[0]}
  obtain ⟨_, hw10⟩ := familyGet_eq {help_gets[1]}
{"".join(f"  obtain ⟨_, hs{k}⟩ := familyGet_eq {x}{chr(10)}" for k, x in enumerate(switch_gets))}  subst hw3 hw10
  have hhelp' := hhelp ⟨i, by omega⟩
  have hAd : lutA model key {q} {m} {level} (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  simp only [hAd] at hhelp'
  have hbranch := lut_branch (n := N) (q := {q}) (q' := {qp}) (p := {p}) (m := {m}) (m' := {mp})
    (log := 3) (bound := {bound}) (index := sigma i) hN hpow {qeq} (by norm_num) (by norm_num)
    {layout} {hlayout} rfl hg' hg hsm
    henc h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (bsw := state.switch) (Xb := ![w_32_0, w_51_0])
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
    (h := ![_, _]) (sigma_odd ⟨i, by omega⟩) hhelp' h6_0_1 h8_0 h11_0_1 h13_0 h14_0
    (Y := ![w_44_0, w_63_0]) (X := ![w_45_0, w_64_0])
    (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h44_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h45_0, h64_0])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  subst hout
  rw [hswitch] at hbranch
  refine ⟨?_, ?_⟩
  · convert hstate.add hbranch using 1
    · funext k j
      fin_cases k <;> fin_cases j <;> rfl
    · funext j
      fin_cases j <;> rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The generated loop `{tag}` accumulates the first `count` branches and carries the switching
ciphertext of branch `count`. -/
theorem lutLoop{tag} {{model : HashModel}} {{key : ByteArray}} {{sm sp μ : ErrorPoly N}}
    (hsm : polyNorm sm ≤ 1)
    {{g' : ExactMatrix {qp} N 1 {m}}} {{g : ExactMatrix {q} N 1 {m}}}
    (hg' : gadgetMatrixRuns Backend.backend 32768 {m} g') (hg : centeredRebaseRuns g' g)
    {{c : Fin 2 → ExactMatrix {q} N 1 {m}}} {{A : Fin 2 → Fin 2 → ExactMatrix {q} N 1 {m}}} {{B : Nat}}
    (henc : Rel sm A (fun j ↦ reducePoly {q} N ((({p} : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {{h0 h1 : Fin 4 → ExactMatrix {q} N 1 {m}}} {{Bh : Nat}}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key {q} {m} {level} (sigma i))
      (fun j ↦ reducePoly {q} N ((({p} : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {{fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix {qp} N 1 {mp}}} {{count : Nat}} (hcount : count ≤ 3)
    {{final : {st}}} {{{params}}} {extra_binders(extra_args)}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : {st}) ↦
      {ns}.{name} Backend.backend model {params} {args}i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
          current.2.2.2.2.2.1, current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1,
          current.2.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0,
          c 1, A 0 1, A 1 1, fam 0 0, fam 0 1, fam 1 0, fam 1 1, ()) next) count
      (0, 0, 0, 0, 0, 0, fam 0 0 0, fam 0 1 0, fam 1 0 0, fam 1 1 0, ()) final) :
    Rel sm final.key (∑ i ∈ Finset.range count, lutBranchTarget (q := {q}) (p := {p}) μ sp fam i)
        final.row (count * lutBranchBound{tag} B Bh) ∧
      final.switch = fun k j ↦ fam k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : {st}) ↦ ∀ hk : k ≤ 3,
      Rel sm s.key (∑ i ∈ Finset.range k, lutBranchTarget (q := {q}) (p := {p}) μ sp fam i) s.row
        (k * lutBranchBound{tag} B Bh) ∧
        s.switch = fun k' j ↦ fam k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨?_, ?_⟩
    · rw [Finset.range_zero, Finset.sum_empty, zero_mul]
      have hrow : (![0, 0] : Fin 2 → ExactMatrix {q} N 1 {mp}) = 0 := by
        funext j
        fin_cases j <;> rfl
      simp only [LoopState.key, LoopState.row, hrow]
      exact Rel.zero
    · funext k j
      fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := lutStep{tag} hsm hg' hg henc hhelp (by omega) hsw hrel hbody
    refine ⟨?_, hsw'⟩
    rw [Finset.sum_range_succ, lutBranchTarget, dif_pos (by omega), add_mul, one_mul]
    exact hrel'
'''

    if width == 2:
        text = text.replace('Fin 4', 'Fin 2').replace('sigma_odd', 'sigma2_odd')
        text = re.sub(r'\bsigma\b', 'sigma2', text)
        text = text.replace('lutBranchTarget', 'lutBranchTarget2')
        text = text.replace('branchTarget (', 'branchTarget2 (')
        text = text.replace('i < 3', 'i < 1').replace('≤ 3', '≤ 1')
    return text

def extra_binders(extra_args):
    return ' '.join(f'{{{x} : Nat}}' for x in extra_args.split()) if extra_args else ''


def round_args(names, atoms, lets, state, last=3):
    """The arguments of the fourth branch of a LUT round, matched in the atoms `names`/`atoms` of
    the definition holding it, whose `let`s are `lets` (name → value) and whose loop's final state
    is the tuple of wires `state` (10 names). Returns a dict of hypothesis and wire names."""
    atom = dict(zip(names, atoms))
    named = [(n, a) for n, a in zip(names, atoms) if n != '-']
    wire = lambda n: 'w_' + n[1:].split('_concat')[0] if '_concat' not in n else None
    out = {}
    autos = [(n, a) for n, a in named if a.startswith('MxxRuntime.ringAutomorphismRuns')]
    autos = autos[-6:]
    out['hr'] = [n for n, _ in autos]
    rot = {re.search(r'Runs \(.*\) (w_\d+_0) (w_\d+_0)$', a).group(2): n for n, a in autos}
    # The concatenations of the routed blocks and of the helper.
    concat = [(n, a) for n, a in named if a.startswith('MxxRuntime.concat')]
    def concat_of(first):
        return next(n for n, a in concat if re.match(rf'MxxRuntime\.concat\w+ {first} ', a))
    def out_of(n):
        return re.search(r'(w_\d+_(?:\d+|concat_\d+))$', atom[n]).group(1)
    routed = [re.search(r'(w_\d+_0)$', atom[n]).group(1) for n in out['hr']]
    hl01 = concat_of(routed[0]); hl = concat_of(out_of(hl01))
    hr01 = concat_of(routed[3]); hr = concat_of(out_of(hr01))
    hboth = concat_of(out_of(hl))
    out['enc'] = [hl01, hl, hr01, hr, hboth]
    both = out_of(hboth)
    # The helper: the concatenation that starts with the fetched helper row.
    helper_rows = [re.search(rf'familyGetStatic \S+ \({last}\) (w_\d+_0)', a).group(1) for n, a in named
                   if re.match(rf'MxxRuntime\.familyGetStatic \S+ \({last}\) ', a)][:2]
    hhl01 = concat_of(helper_rows[0]); hhl = concat_of(out_of(hhl01))
    hhr01 = concat_of(helper_rows[1]); hhr = concat_of(out_of(hhr01))
    hH = concat_of(out_of(hhl))
    out['help'] = [hhl01, hhl, hhr01, hhr, hH]
    H = out_of(hH)
    out['help_rows'] = helper_rows
    sw = state[6:10]
    def by_first(op, first):
        return next(n for n, a in named if re.match(rf'MxxRuntime\.{op} {first} ', a))
    xb = [by_first('concatRows', sw[0]), by_first('concatRows', sw[1])]
    out['Xb'] = xb
    out['D1'] = [by_first('gadgetDecomposeRuns backend 32768 \\d+', out_of(x)) for x in xb]
    out['D'] = [by_first('compactCenteredRebaseRuns 16384', out_of(x)) for x in out['D1']]
    mul = {v: k for k, v in lets.items() if v.startswith('MxxRuntime.matrixMul ')}
    bd = [mul[f'MxxRuntime.matrixMul {both} {out_of(x)}'] for x in out['D']]
    def slice_of(src, a, b):
        return next(n for n, at in named if re.match(rf'MxxRuntime\.sliceMatrix {src} {a} {b} 0 \d+ ', at))
    out['M0'] = [slice_of(x, 1, 2) for x in bd]
    out['M1'] = [slice_of(x, 2, 3) for x in bd]
    out['X'] = [slice_of(x, 0, 1) for x in bd]
    out['Asw'] = [[by_first('blockModSwitchRuns \\[[^\\]]*\\] \\(1\\)', out_of(x)) for x in out['M0']],
                  [by_first('blockModSwitchRuns \\[[^\\]]*\\] \\(1\\)', out_of(x)) for x in out['M1']]]
    out['Z'] = [by_first('concatRows', out_of(x)) for x in out['Asw'][0]]
    out['E1'] = [by_first('gadgetDecomposeRuns backend 32768 \\d+', out_of(x)) for x in out['Z']]
    out['E'] = [by_first('compactCenteredRebaseRuns 16384', out_of(x)) for x in out['E1']]
    he = [mul[f'MxxRuntime.matrixMul {H} {out_of(x)}'] for x in out['E']]
    out['K0'] = [slice_of(x, 1, 2) for x in he]
    out['K1'] = [slice_of(x, 2, 3) for x in he]
    out['Y'] = [slice_of(x, 0, 1) for x in he]
    out['wire'] = out_of
    return out


def branch_call(r, sigma3_help):
    """The hypotheses of `lut_branches`/`lut_round` after `hhelp` from a `round_args` match."""
    w = r['wire']
    two = lambda key: f"(fun j ↦ by fin_cases j; exacts [{r[key][0]}, {r[key][1]}])"
    lines = [
        ' '.join(r['hr'] + r['enc'][:5]),
        f"(Xb := ![{w(r['Xb'][0])}, {w(r['Xb'][1])}]) {two('Xb')}",
        f"(D' := ![{w(r['D1'][0])}, {w(r['D1'][1])}]) {two('D1')}",
        f"(D := ![{w(r['D'][0])}, {w(r['D'][1])}]) {two('D')}",
        f"(M := ![![{w(r['M0'][0])}, {w(r['M0'][1])}], ![{w(r['M1'][0])}, {w(r['M1'][1])}]])",
        f"{two('M0')}", f"{two('M1')}",
        f"(Asw := ![![{w(r['Asw'][0][0])}, {w(r['Asw'][0][1])}], ![{w(r['Asw'][1][0])}, {w(r['Asw'][1][1])}]])",
        f"(fun k j ↦ by fin_cases k <;> fin_cases j; exacts [{r['Asw'][0][0]}, {r['Asw'][0][1]}, {r['Asw'][1][0]}, {r['Asw'][1][1]}])",
        f"(Z := ![{w(r['Z'][0])}, {w(r['Z'][1])}]) {two('Z')}",
        f"(E' := ![{w(r['E1'][0])}, {w(r['E1'][1])}]) {two('E1')}",
        f"(E := ![{w(r['E'][0])}, {w(r['E'][1])}]) {two('E')}",
        ' '.join(r['help'][:5]),
        f"(Y := ![{w(r['Y'][0])}, {w(r['Y'][1])}]) (X := ![{w(r['X'][0])}, {w(r['X'][1])}])",
        f"(K := ![![{w(r['K0'][0])}, {w(r['K0'][1])}], ![{w(r['K1'][0])}, {w(r['K1'][1])}]])",
        f"{two('Y')}", f"{two('X')}", f"{two('K0')}", f"{two('K1')}",
    ]
    return '\n    '.join(lines)


def member_lut(path, ns, k, kind, q, qp, p, m, mp, layout, hlayout, qeq, level, scale=None,
               table=None, offset=False):
    """The LUT of member `o` in the parallel `parallel_generatedRoot_<k>` (loop
    `sequential_parallel_generatedRoot_<k>_<l>`): `kind = 'round'` (with the move down to `qp`,
    the output scale `scale` and the table `table`) or `'branches'` (before the move down, plus
    the offset encoding when `offset`)."""
    from patterns import destruct, parse
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
    tag = str(k)
    level = int(re.search(r'\(\[\.u64Le \((\d+)\), \.integer', atom[hashes[0]]).group(1))
    loops = loop_lemmas(path, ns, loop, tag, q, qp, p, m, mp, layout, hlayout, qeq, level, 16384,
                        extra_args='o')
    dd = destruct(path, ns, name, 'h')
    off_bind = ' {off : Blocks ' + q + f' {mp}' + '}' if offset else ''
    off_in = ', off.1, off.2.1, off.2.2.1, off.2.2.2.1, off.2.2.2.2.1, off.2.2.2.2.2.1' if offset else ''
    st = f"({', '.join(state)}, ())"
    kj = {0: '0 0', 1: '0 1', 2: '1 0', 3: '1 1'}
    hAd = ''.join(
        f"  have hA{c} : {w(hash_of[c])} = lutA model key {q} {m} {level} (sigma (3 : Fin 4)) {kj[c]} :=\n"
        f"    hashSample_eq {hash_of[c]}\n" for c in range(4))
    hAd += f"  rw [hA0] at {r['help'][0]}\n  rw [hA2] at {r['help'][1]}\n"
    hAd += f"  rw [hA1] at {r['help'][2]}\n  rw [hA3] at {r['help'][3]}\n"
    head = f'''
set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem lut{tag} {{model : HashModel}} {{key : ByteArray}} {{params}} {{o : Nat}}
    {{fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix {qp} N 1 {mp}}} {{h0 h1 : Fin 4 → ExactMatrix {q} N 1 {m}}}
    {{c0 A00 A10 c1 A01 A11 : ExactMatrix {q} N 1 {m}}}{off_bind} {{outputs : Blocks {qp if kind != 'branches' else q} {1 if kind == 'smudge' else mp}}}
    (h : {ns}.{name} Backend.backend model params o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11{off_in}, ()) outputs)
    {{sm sp : ErrorPoly N}} (hsm : polyNorm sm ≤ 1) {{z : Nat}}
    {{g' : ExactMatrix {qp} N 1 {m}}} {{g : ExactMatrix {q} N 1 {m}}}
    (hg' : gadgetMatrixRuns Backend.backend 32768 {m} g') (hg : centeredRebaseRuns g' g)
    {{B : Nat}}
    (hinput : Rel sm ![![A00, A01], ![A10, A11]]
      (fun j ↦ reducePoly {q} N ((({p} : Nat) : Int) * Xe (n := N) (2 * z) * unitPair sp j) • g)
      ![c0, c1] B)
    {{Bh : Nat}} (hhelp : ∀ i : Fin 4, Rel sm (lutA model key {q} {m} {level} (sigma i))
      (fun j ↦ reducePoly {q} N ((({p} : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)'''
    if kind == 'smudge':
        head += f'''
    (ho : o < 160) {{gq : ExactMatrix {qp} N 1 {mp}}} (hgq : gadgetMatrixRuns Backend.backend 32768 {mp} gq) {{inverse : Nat}}
    (hinverse : (inverse : ExactPoly {qp} N) * ((4 : Nat) : ExactPoly {qp} N) = 1) {{BB : Nat}}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 {mp}, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly {qp} N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient {qp} inverse (fun k ↦ reducePoly {qp} N ({table} k)) (sigma i).toNat *
          reducePoly {qp} N ({scale} * unitPair sp j)) • gq + reduceMatrix {qp} N 1 {mp} (e j)) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly {qp} N (Xe (n := N) (o / 20)) • reducePoly {qp} N
        (((smudgeWeight{tag} ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩ : Int) : ErrorPoly N) *
          ({scale} * ((((-1 : Int) ^ (z / 4) : Int) : ErrorPoly N) *
            {table} ⟨z % 4, Nat.mod_lt _ (by norm_num)⟩)) * unitPair sp j) •
        (1 : ExactMatrix {qp} N 1 1))
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ({mp} * ((3 * lutBranchBound{tag} B Bh + (({m} + {m}) * N * Bh * 16384 + ({m} + {m}) * N * B * 16384 +
        (N * 1 * ({p} / 2) + {p} / 2)) + (N + 2) * ({p} / 2)) / {p} + 4 * (N * 1 * BB)) * (32768 / 2)) := by'''
    elif kind == 'round':
        head += f'''
    {{gq : ExactMatrix {qp} N 1 {mp}}} {{inverse : Nat}}
    (hinverse : (inverse : ExactPoly {qp} N) * ((4 : Nat) : ExactPoly {qp} N) = 1) {{BB : Nat}}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 {mp}, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly {qp} N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient {qp} inverse (fun k ↦ reducePoly {qp} N ({table} k)) (sigma i).toNat *
          reducePoly {qp} N ({scale} * unitPair sp j)) • gq + reduceMatrix {qp} N 1 {mp} (e j)) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly {qp} N ({scale} * ((((-1 : Int) ^ (z / 4) : Int) : ErrorPoly N) *
        {table} ⟨z % 4, Nat.mod_lt _ (by norm_num)⟩) * unitPair sp j) • gq)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((3 * lutBranchBound{tag} B Bh + (({m} + {m}) * N * Bh * 16384 + ({m} + {m}) * N * B * 16384 +
        (N * 1 * ({p} / 2) + {p} / 2)) + (N + 2) * ({p} / 2)) / {p} + 4 * (N * 1 * BB)) := by'''
    else:
        off_h = f'''
    {{toff : Fin 2 → ExactMatrix {q} N 1 {mp}}} {{Boff : Nat}}
    (hoff : Rel sm ![![off.1, off.2.1], ![off.2.2.1, off.2.2.2.1]] toff
      ![off.2.2.2.2.1, off.2.2.2.2.2.1] Boff)''' if offset else ''
        head += off_h + f''' :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ (∑ i : Fin 4, scaleUpMatrix {q} {p}
        (reducePoly {qp} N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * z))) •
          (reducePoly {qp} N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i))){" + toff j" if offset else ""})
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((3 * lutBranchBound{tag} B Bh + (({m} + {m}) * N * Bh * 16384 + ({m} + {m}) * N * B * 16384 +
        (N * 1 * ({p} / 2) + {p} / 2))){" + Boff" if offset else ""}) := by'''
    proof = f'''
{dd}  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) {init_gets[0]}
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) {init_gets[1]}
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) {init_gets[2]}
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) {init_gets[3]}
  rw [hwa, hwb, hwc, hwd] at {loop_atom}
  rw [show Int.toNat 3 = 3 from rfl] at {loop_atom}
  obtain ⟨hloop, hloopSwitch⟩ := lutLoop{tag} (fam := fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j)
    (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]])
    (final := {st}) hsm hg' hg hinput hhelp (count := 3) (by decide) {loop_atom}
  obtain ⟨_, hw19⟩ := familyGet_eq (k := 3) {help_gets[0]}
  obtain ⟨_, hw24⟩ := familyGet_eq (k := 3) {help_gets[1]}
  rw [hw19] at {r['help'][0]}
  rw [hw24] at {r['help'][2]}
{hAd}  have hhelp3 := hhelp 3
'''
    args = branch_call(r, 'hAd')
    if kind == 'round':
        # The moved-down sums: `blockModSwitch (state + branch)`.
        rev = {v: k2 for k2, v in lets.items()}
        def sum_let(x, y):
            return rev[f'MxxRuntime.matrixAdd {x} {y}']
        def modswitch_of(src):
            return next(x for x in n if x != '-' and re.match(
                rf'MxxRuntime\.blockModSwitchRuns \[[^\]]*\] \(1\) {src} ', atom[x]))
        kk = [[r['K0'][0], r['K0'][1]], [r['K1'][0], r['K1'][1]]]
        fin_k = [[modswitch_of(sum_let(state[2 + 2 * a + b], w(kk[a][b]))) for b in range(2)]
                 for a in range(2)]
        yx = [sum_let(w(r['Y'][jj]), w(r['X'][jj])) for jj in range(2)]
        fin_r = [modswitch_of(sum_let(state[jj], yx[jj])) for jj in range(2)]
        proof += f'''  have hfin := lut_round (q := {q}) (q' := {qp}) (p := {p}) (m := {m}) (m' := {mp}) {qeq} (by norm_num)
    (by norm_num) (by norm_num) {layout} {hlayout} rfl hsm hg' hg hinput hloop hloopSwitch hhelp3
    {args}
    (T := {table}) (Δ' := {scale}) hinverse
    (fun i ↦ hswitch i) (K' := ![![{w(fin_k[0][0])}, {w(fin_k[0][1])}], ![{w(fin_k[1][0])}, {w(fin_k[1][1])}]])
    (R' := ![{w(fin_r[0])}, {w(fin_r[1])}])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [{fin_k[0][0]}, {fin_k[0][1]}, {fin_k[1][0]}, {fin_k[1][1]}])
    (fun j ↦ by fin_cases j; exacts [{fin_r[0]}, {fin_r[1]}])
  rw [hout]
  convert hfin using 1
  all_goals first
    | rfl
    | (funext a j; fin_cases a <;> fin_cases j <;> rfl)
    | (funext j; fin_cases j <;> rfl)
'''
    elif kind == 'smudge':
        # The weight row `w_d` of the digits and the moved-down LUT output at `qp`.
        rev = {v: k2 for k2, v in lets.items()}
        def sum_let(x, y):
            return rev[f'MxxRuntime.matrixAdd {x} {y}']
        def modswitch_of(src):
            return next(x for x in n if x != '-' and re.match(
                rf'MxxRuntime\.blockModSwitchRuns \[[^\]]*\] \(1\) {src} ', atom[x]))
        kk = [[r['K0'][0], r['K0'][1]], [r['K1'][0], r['K1'][1]]]
        fin_k = [[modswitch_of(sum_let(state[2 + 2 * a_ + b_], w(kk[a_][b_]))) for b_ in range(2)]
                 for a_ in range(2)]
        yx = [sum_let(w(r['Y'][jj]), w(r['X'][jj])) for jj in range(2)]
        fin_r = [modswitch_of(sum_let(state[jj], yx[jj])) for jj in range(2)]
        slice_atom = next(x for x in n if x != '-' and re.match(r'MxxRuntime\.sliceMatrix \S+ 0 1 \(Int\.fmod', atom[x]))
        row = re.match(r'MxxRuntime\.sliceMatrix (\S+) ', atom[slice_atom]).group(1)
        weight_wire = re.search(r'(w_\d+_0)$', atom[slice_atom]).group(1)
        dec_atom = next(x for x in n if x != '-' and re.match(rf'MxxRuntime\.gadgetDecomposeRuns backend 32768 \d+ {weight_wire} ', atom[x]))
        reb_atom = next(x for x in n if x != '-' and re.match(rf'MxxRuntime\.compactCenteredRebaseRuns 16384 {w(dec_atom)} ', atom[x]))
        concats = [x for x in n if x != '-' and re.match(r'MxxRuntime\.concatColumns ', atom[x])
                   and (re.search(rf'{row}$', atom[x]) or re.search(rf'{row[:-2]}_concat_\d+$', atom[x]))]
        weights = [w_ for w_ in re.findall(r'concatColumns (w_\d+_0|w_\d+_concat_\d+) (w_\d+_0)', ' '.join(atom[x] for x in concats))]
        firsts = [weights[0][0]] + [b_ for _, b_ in weights]
        consts = [re.search(r'\[\((\d+)\)\]', lets[x]).group(1) for x in firsts]
        fdiv = next(x for x in lets if lets[x].startswith('(Int.fdiv ((Int.ofNat'))
        monos = [x for x in lets if re.match(rf'MxxRuntime\.multiplyMonomial \S+ {fdiv}$', lets[x])]
        placed = monos + [re.match(r'MxxRuntime\.multiplyMonomial (\S+) ', lets[x]).group(1) for x in monos]
        proof += f'''  have hfin := lut_round (q := {q}) (q' := {qp}) (p := {p}) (m := {m}) (m' := {mp}) {qeq} (by norm_num)
    (by norm_num) (by norm_num) {layout} {hlayout} rfl hsm hg' hg hinput hloop hloopSwitch hhelp3
    {args}
    (T := {table}) (Δ' := {scale}) hinverse
    (fun i ↦ hswitch i) (K' := ![![{w(fin_k[0][0])}, {w(fin_k[0][1])}], ![{w(fin_k[1][0])}, {w(fin_k[1][1])}]])
    (R' := ![{w(fin_r[0])}, {w(fin_r[1])}])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [{fin_k[0][0]}, {fin_k[0][1]}, {fin_k[1][0]}, {fin_k[1][1]}])
    (fun j ↦ by fin_cases j; exacts [{fin_r[0]}, {fin_r[1]}])
  -- The weight `w_(o mod 20)` of the member's digit.
  simp only [concatColumns] at {" ".join(concats)}
  have hrow : ∀ c : Fin 20, {row} 0 c = (matrixPolynomial [smudgeWeight{tag} c] : ExactMatrix {qp} N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [{", ".join(concats)}, {", ".join(firsts)}, smudgeWeight{tag}]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; rfl
  rw [hd] at {slice_atom}
  have hK : {weight_wire} = matrixPolynomial [smudgeWeight{tag} ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry {slice_atom} (by simp; omega)]
    exact hrow _
  have hfin2 := hfin.finalizeCoefficient (by norm_num) hN {layout} {hlayout} rfl hgq _ hK {dec_atom} {reb_atom}
  have hpl := rel_monomial hfin2 (k := o / 20) (by simp only [N]; omega)
  have hq : {fdiv} = ((o / 20 : Nat) : Int) := by
    simp only [{fdiv}]
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by norm_num)]; rfl
  rw [hout]
  convert hpl using 1
  all_goals simp only [{", ".join(placed)}, hq]
  all_goals first
    | (funext a j; fin_cases a <;> fin_cases j <;> rfl)
    | (funext j; fin_cases j <;> rfl)
'''
    else:
        proof += f'''  have hsum := lut_branches (q := {q}) (q' := {qp}) (p := {p}) (m := {m}) (m' := {mp}) {qeq} (by norm_num)
    (by norm_num) (by norm_num) {layout} {hlayout} rfl hsm hg' hg hinput hloop hloopSwitch hhelp3
    {args}
'''
        if offset:
            proof += '  have htot := hsum.add hoff\n'
        else:
            proof += '  have htot := hsum\n'
        proof += '''  rw [hout]
  convert htot using 1
  all_goals first
    | rfl
    | (funext a j; fin_cases a <;> fin_cases j <;> rfl)
    | (funext j; fin_cases j <;> rfl)
'''
    if kind == 'smudge':
        loops = f"\n/-- The digit weights of the smudge LUT `{tag}`. -/\ndef smudgeWeight{tag} : Fin 20 → Int := ![{', '.join(consts)}]\n" + loops
    return loops + head + proof
