"""Step-zero closure for gen_s0o.py (module Dio2Step0).

From the actual source targets of online zero (s0a) this module assembles the
refreshed roots of `Online0RefreshedRoots`:

* `refreshed_pairs`: every refreshed target pair, from the per-part CRT rows
  (Dio2Step0Rows), the key-stage agreement (Dio2Step0Agree), the sampled
  preimages of the claim and the actual targets (Dio2Step0Targets);
* the key of every part read from the actual target-selection loop: packed
  root keys for non-label targets, label-0 hashes for forest members;
* one root lemma per exported non-label target class, identifying its packed
  root key with the next root key of import one;
* the refreshed hint from the zero-plaintext anchor parts;
* `Complete.step0_transition_of` and the closed `Complete.step0_transition`.

Label members stay quantified; helper branches are the exported helper target
classes.  No part, member, digit or coefficient is enumerated.
"""
import re

import dio2_meta as meta
import s0o_agree as agree
import s0o_targets as targets
from xmodel import strip_parens

ONLINE = 'W_Stage_online_0'
KEYS = 'W_Stage_keys_online_0_0'
WIRE = re.compile(r'w_\d+_\d+')
CTX = 'initial.context.online'
TAC = '(by simp only [Int.ofNat_eq_coe, decide_eq_true_eq, Step0Targets.pairIndex, Fin.val_mk]; (try split_ifs) <;> omega)'


def split_top(s):
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch in '([':
            depth += 1
        if ch in ')]':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


class Step0:
    def __init__(self, geo, T, PF):
        self.geo, self.T, self.PF = geo, T, PF
        self.n = geo.n
        self.m = geo.columns
        self.root = T.root
        view = geo.view
        self.R = view['labels']['root_members']
        self.C = view['labels']['target_columns']
        self.L = T.labels
        self.ls = T.label_start
        self.targets = geo.targets
        self.parts = geo.parts
        self.nlt = T.nlt
        self.cls = {tp: (t, cls) for tp, (t, cls) in enumerate(T.nonlabel)}

    def fact(self, atom):
        return f'F.c{atom.frag}.{atom.field}'

    def input_term(self, wire):
        node = self.root.node(wire)
        meta.require(node[0] == 'input', f'{wire} is not an online input')
        return targets.claim_input_term(self.geo, 'onlineInputs', node[1])

    def producer(self, wire, kind):
        node = self.root.node(wire)
        meta.require(node[0] == 'atom' and node[1].kind == kind, f'{wire}: expected {kind}')
        return node[1]


# --------------------------------------------------------------------------- claim plumbing

def runs_index(geo, stage_module):
    claim = (geo.package / 'worst' / 'W_Claim.lean').read_text()
    i = claim.index('def Runs ')
    body = claim[claim.index(':=\n', i) + 3:claim.index('\n\nend', i)]
    conj = [c.strip() for c in body.split(' ∧\n')]
    k = next(j for j, c in enumerate(conj) if c.startswith(stage_module + '.generatedRoot '))
    return 'hrun' + '.2' * k + ('.1' if k < len(conj) - 1 else '')


def anchors_source(S):
    geo = S.geo
    anchors = runs_index(geo, 'W_Stage_anchors')
    publics = runs_index(geo, 'W_Stage_anchor_publics')
    return f'''/-- The anchor publics are the public matrices of the sampled anchors. -/
theorem anchor_public (hrun : W_GeneratedClaim.Runs model external execution) :
    execution.«stage_2».1 = execution.«stage_1».1 ∧
      execution.«stage_2».2.1 = execution.«stage_1».2.2.1 := by
  obtain ⟨_, hb⟩ := {publics}
  have h := anchors_trapdoor {anchors}
  simp only [W_Stage_anchor_publics.generatedRoot.body,
    W_Stage_anchor_publics.generatedRoot.constraints_0] at hb
  rw [hb]
  exact ⟨trapdoorSample_public h.1, trapdoorSample_public h.2⟩

'''


def refreshed_source(S):
    geo, n, T = S.geo, S.n, S.T
    pin, pargs = geo.part_inputs, geo.part_args
    K = f'{KEYS}.generatedRoot'
    keyfacts = (geo.package / 'worst' / f'{KEYS}_Facts.lean')
    ksrc = '\n'.join(p.read_text() for p in sorted((geo.package / 'worst').glob(f'{KEYS}_Facts*.lean')))
    frag = None
    hout_frag = None
    for line in ksrc.splitlines():
        mm = re.match(r'structure generatedRoot\.Facts_(\d+) ', line)
        if mm:
            frag = mm.group(1)
        if line.strip() == 'hout : outputs = generatedRoot.outs ctx' and frag is not None:
            hout_frag = frag
    meta.require(hout_frag is not None, 'key stage output fact')
    count = len(geo.primes)
    eqs, solved = [], []
    for k, prime in enumerate(geo.primes):
        pre = pargs[pin.index(prime['wires']['preimage'])].split()[0]
        term = S.input_term(pre)
        eqs.append(f'''  have e{k} : generatedRoot.{pre} {CTX} = {term} := by
    rw [generatedRoot.{pre}, initial.context.online_inputs]; rfl''')
        proj = '.2' * k + ('.1' if k < count - 1 else '')
        solved.append(f'    (fun part ↦ by rw [hA, e{k}]; have := P{proj} part; rw [hk] at this; exact this)')
    return f'''/-- Every refreshed target pair of the actual run, under the next mask, with the
actual target-selection keys and the certified fresh error. -/
theorem refreshed_pairs (hrun : W_GeneratedClaim.Runs model external execution) {{state : Nat}}
    (initial : State0Witness model external execution (Xe (n := {n}) (2 * state)))
    {{next result : ErrorPoly {n}}} {{fresh : Fin {geo.fresh_cells} → ErrorPoly {n}}}
    {{smudge : Fin {geo.smudge_cells} → ErrorPoly {n}}}
    (R : Online0Sources.Relations {CTX} (execution.«stage_2».1) initial.mask
      initial.payload next result fresh smudge) :
    ∃ label : Fin {S.L} → ErrorPoly {n},
      (∀ (m : Fin {S.R}) (member : Fin {S.L}), m.val ≠ {S.R - 1} → member.val = m.val * {S.C} →
        reducePoly modulus_0 {n} (label member) = reducePoly modulus_0 {n} (online1Labels external m)) ∧
      ∀ t : Fin {S.targets}, MxxExponentLut.Rel next
        (fun r c ↦ MxxExponentLut.rowOf
          (generatedRoot.{T.family}_3 {CTX} (StepTransition.partIndex t c)) r)
        (fun c ↦ reducePoly modulus_0 {n} (Step0Targets.stepTarget initial.payload next result label
          (StepTransition.partIndex t c)) • generatedRoot.{T.gadget} {CTX})
        (StepTransition.nextRows {CTX} t) {S.fresh_bound} := by
  obtain ⟨label, hbridge, targets⟩ := Step0Targets.target_rows hrun initial R
  refine ⟨label, hbridge, ?_⟩
  obtain ⟨kw, kb⟩ := Online0Claim.key_runs hrun
  let ck : Step0Rows.KeyCtx model := ⟨Online0Claim.keyInputs external execution, kw⟩
  have Fk := {K}.facts ck kb
  have links := Step0Agree.key_links {CTX} ck initial.context.online_facts Fk
    initial.context.online_inputs rfl
  have P := (claim_preimages hrun).1
  have hA : execution.«stage_2».1 = execution.«stage_1».1 := (anchor_public hrun).1
  have hk := Fk.c{hout_frag}.hout
{chr(10).join(eqs)}
  have rows := Step0Rows.rows initial.context.online_facts Fk links R
{chr(10).join(solved)}
    targets
  refine StepTransition.paired_parts {CTX} (bound := fun _ ↦ {S.fresh_bound})
    (fresh := fun t c ↦ Step0Rows.freshError fresh (StepTransition.partIndex t c)) ?_ ?_
  · intro t c row column coefficient
    exact (coeff_natAbs_le_polyNorm _ _).trans (R.fresh_small _)
  · intro t c
    exact rows _

'''


# --------------------------------------------------------------------------- part keys

def selection_prefix(S):
    """The destructuring of the actual target-selection loop at a part."""
    T = S.T
    Sel = T.gen.info('A', T.select_scope)
    head = agree.Lemma.destruct_lines(Sel.exists + [f'c{k}' for k in Sel.frags], 'h', 'th', '  ')
    out_path = targets.and_proj(len(Sel.conj[Sel.out_atom.frag]) - 1, len(Sel.conj[Sel.out_atom.frag]))
    head.append(f'  have hout := c{Sel.out_atom.frag}{out_path}')
    head.append('  simp only [Prod.mk.injEq] at hout')
    return Sel, (f'  have h := {T.fact(T.select_atom)} part\n  dsimp only [{Sel.name}] at h\n' +
                 '\n'.join(head) + '\n')


def conj_path(Sel, wire):
    node = Sel.node(wire)
    meta.require(node[0] == 'atom', f'{Sel.name}.{wire} is not an atom')
    a = node[1]
    group = Sel.conj[a.frag]
    j = next(k for k, b in enumerate(group) if b is a)
    return a, f'c{a.frag}{targets.and_proj(j, len(group))}'


def keys_source(S):
    T, n, m = S.T, S.n, S.m
    Sel, prefix = selection_prefix(S)
    outs = [w for _, w in Sel.outputs()]
    sel, sel_p = conj_path(Sel, outs[3])
    nl, lab = [strip_parens(x) for x in sel.args[1].strip('[]').split(',')]
    nla, nla_p = conj_path(Sel, nl)
    meta.require(nla.kind == 'familyGetDynamic', 'non-label key read')
    keyfam = strip_parens(nla.args[0])
    meta.require(Sel.s.wires[keyfam].src == 'input', 'non-label key family')
    root_keys = T.forest_args  # placeholder to keep the root argument order visible
    red, red_p = conj_path(Sel, lab)
    meta.require(red.kind == 'modulusReduceRuns', 'label key reduction')
    comp_sel, comp_p = conj_path(Sel, strip_parens(red.args[0]))
    meta.require(comp_sel.kind == 'select', 'label key component selection')
    cw = [strip_parens(x) for x in comp_sel.args[1].strip('[]').split(',')]
    blocks = []
    for j, w in enumerate(cw):
        cat, cat_p = conj_path(Sel, w)
        meta.require(cat.kind == 'concatRows', 'label key rows')
        rows = []
        for r, x in enumerate(strip_parens(y) for y in cat.args[:2]):
            s, s_p = conj_path(Sel, x)
            meta.require(s.kind == 'select', 'label key prefix selection')
            pick = strip_parens(s.args[1].strip('[]').split(',')[1])
            hs, hs_p = conj_path(Sel, pick)
            meta.require(hs.kind == 'hashSample', 'label-0 key hash')
            rows.append((s_p, hs_p, hs))
        blocks.append((cat_p, rows))
    hs0 = blocks[0][1][0][2]
    pre = strip_parens(hs0.args[1])
    comps = split_top(strip_parens(hs0.args[2])[1:-1])
    idx = re.fullmatch(r'\.u64Le \((.*)\)', comps[1]).group(1)
    sq = re.sub(r'[()\s]', '', idx)
    meta.require(sq == f'Int.fdivInt.fmod-{S.ls}+Int.fdivInt.ofNati_02{S.L}{S.C}',
                 f'label-0 key member index {idx}')
    S.label0_prefix = pre
    hdr = (f'    {{model : HashModel}} {{params : Params}}\n'
           f'    (ctx : generatedRoot.Ctx Backend.backend model params) {{outputs : _}}\n'
           f'    (F : generatedRoot.Facts ctx outputs) (part : Fin {S.parts})')
    key_lines = []
    for j, (cat_p, rows) in enumerate(blocks):
        key_lines.append(f'''  · rw [hc.{j + 1} {TAC}]
    obtain ⟨e0, e1⟩ := concatRows_two {cat_p}
    fin_cases r <;> simp only [Fin.zero_eta, Fin.mk_one, Fin.isValue]
    · rw [e0, (Step0Targets.select2 {rows[0][0]}).2 {TAC}, MxxExponentLut.hashSample_eq {rows[0][1]}, hidx]; rfl
    · rw [e1, (Step0Targets.select2 {rows[1][0]}).2 {TAC}, MxxExponentLut.hashSample_eq {rows[1][1]}, hidx]; rfl''')
    return f'''/-- The key of a non-label part is its packed root key. -/
theorem target_nonlabel_key
{hdr} (t : Nat) (ht : t < {S.nlt}) (j : Fin 2)
    (hpart : (part.val / 2 < {S.ls} ∧ part.val / 2 = t) ∨
      ({S.ls + S.L} ≤ part.val / 2 ∧ part.val / 2 = t + {S.L}))
    (hj : part.val % 2 = j.val) :
    generatedRoot.{T.family}_3 ctx part = generatedRoot.{S.keyfam_root} ctx (Step0Targets.pairIndex t ht j) := by
{prefix}  have hjv := j.isLt
  have hi := Step0Targets.select2 {S.int_path}
  have vi : {S.int_wire} = (t : Int) := by
    rcases hpart with ⟨h1, h2⟩ | ⟨h1, h2⟩
    · have := hi.2 {TAC}
      rw [this]; simp only [Int.ofNat_eq_coe]; omega
    · have := hi.1 {TAC}
      rw [this]; simp only [Int.ofNat_eq_coe]; omega
  have s3 := (Step0Targets.select2 {sel_p}).1 {TAC}
  have n3 := Step0Targets.family_get {nla_p} (Step0Targets.pairIndex t ht j) {TAC}
  rw [hout.2.2.2.1, s3]; exact n3

/-- The key of the label-0 part of forest member {S.C} m is the next label-0 key of m. -/
theorem target_label0_key
{hdr} (member : Fin {S.L}) (j : Fin 2)
    (hmember : part.val / 2 = {S.ls} + member.val) (hj : part.val % 2 = j.val)
    (hzero : member.val % {S.C} = 0) (r : Fin 2) :
    MxxExponentLut.rowOf (generatedRoot.{T.family}_3 ctx part) r =
      labelKeysG model (generatedRoot.{S.seed_wire} ctx) modulus_0 {m}
        {pre}
        ((member.val / {S.C} : Nat) : Int) r j := by
{prefix}  have hm := member.isLt
  have hjv := j.isLt
  have s3 := (Step0Targets.select2 {sel_p}).2 {TAC}
  have hr := modulusReduce_same (by decide) {red_p}
  have hc := Step0Targets.select2 {comp_p}
  have hidx : Int.fdiv (Int.fmod (-{S.ls} + Int.fdiv (Int.ofNat part.val) 2) {S.L}) {S.C} =
      ((member.val / {S.C} : Nat) : Int) := by
    simp only [Int.fdiv_eq_ediv, Int.fmod_eq_emod, Int.ofNat_eq_natCast]
    simp only [show (0 : Int) ≤ {S.C} from by decide, show (0 : Int) ≤ {S.L} from by decide, true_or, if_true,
      sub_zero, Int.reduceAbs]
    omega
  rw [hout.2.2.2.1, s3, hr]
  fin_cases j <;> (try simp only [Fin.val_mk, Fin.isValue, Fin.zero_eta, Fin.mk_one] at hj)
{chr(10).join(key_lines)}

'''


def nonlabel_setup(S):
    """Integer selection of the non-label pair inside the selection loop (as in s0o_targets)."""
    T = S.T
    Sel = T.gen.info('A', T.select_scope)
    outs = [w for _, w in Sel.outputs()]
    int_sel = [a for a in Sel.s.atoms if a.kind == 'select' and not a.loop and
               Sel.s.wires.get(strip_parens(a.args[-1])) is not None and
               strip_parens(a.args[-1]) in Sel.exists and strip_parens(a.args[-1]) not in outs]
    int_sel = [a for a in int_sel if re.search(r'Int', str(Sel.s.wires[strip_parens(a.args[-1])].type or ''))]
    meta.require(len(int_sel) == 1, 'non-label index selection')
    S.int_wire = strip_parens(int_sel[0].args[-1])
    group = Sel.conj[int_sel[0].frag]
    ij = next(k for k, b in enumerate(group) if b is int_sel[0])
    S.int_path = f'c{int_sel[0].frag}{targets.and_proj(ij, len(group))}'
    # The packed root key family: the selection loop input feeding the non-label key read.
    sel, _ = conj_path(Sel, outs[3])
    nl = strip_parens(sel.args[1].strip('[]').split(',')[0])
    nla, _ = conj_path(Sel, nl)
    keyfam = strip_parens(nla.args[0])
    _, sargs, _ = agree.call_args(T.select_atom)
    sargs = [strip_parens(x) for _, x in agree.leaves(sargs)]
    S.keyfam_root = sargs[Sel.s.wires[keyfam].term.count('.2')].split()[0]
    S.seed_wire = sargs[-1].split()[0]
    pack = S.producer(S.keyfam_root, 'familyPack')
    S.pack_fact = S.fact(pack)
    S.entries = WIRE.findall(pack.args[1])
    meta.require(len(S.entries) == T.nl2, 'packed root keys')
    S.fresh_bound = None


# --------------------------------------------------------------------------- root lemmas

def entry_info(S, e):
    red = S.producer(e, 'modulusReduceRuns')
    z = strip_parens(red.args[0])
    node = S.root.node(z)
    meta.require(node[0] == 'atom', f'{z}: entry producer')
    return red, node[1]


def hash_info(S, x):
    a = S.producer(x, 'hashSample')
    pre = strip_parens(a.args[1])
    comps = split_top(strip_parens(a.args[2])[1:-1])
    meta.require(comps[-2] == '.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116]', f'{x}: component tag')
    k = int(re.fullmatch(r'\.u64Le \((\d+)\)', comps[-1]).group(1))
    pc = []
    for c in comps[:-2]:
        for w in WIRE.findall(c):
            wi = S.root.s.wires[w]
            meta.require(wi.src == 'let', f'{w}: hash component is not a constant')
            c = c.replace(w, '(' + wi.term.strip() + ')')
        pc.append(c)
    meta.require(strip_parens(a.args[3]) == S.seed_wire, f'{x}: hash seed')
    return a, pre, pc, k


def root_lemmas(S):
    out = []
    S.hash_classes = {}
    for tp in range(S.nlt):
        t, cls = S.cls[tp]
        if cls['kind'] == 'anchor':
            continue
        ents = []
        for j in range(2):
            red, cat = entry_info(S, S.entries[2 * tp + j])
            meta.require(cat.kind == 'concatRows', f'class {tp}: entry rows')
            hx = hash_info(S, strip_parens(cat.args[0]))
            hy = hash_info(S, strip_parens(cat.args[1]))
            ents.append((red, cat, hx, hy))
        pre, pc = ents[0][2][1], ents[0][2][2]
        for e in ents:
            for h in (e[2], e[3]):
                meta.require(h[1] == pre and h[2] == pc, f'class {tp}: key tags differ')
        meta.require([ents[0][2][3], ents[0][3][3], ents[1][2][3], ents[1][3][3]] == [0, 2, 1, 3],
                     f'class {tp}: key components')
        comps = '[' + ', '.join(pc) + ']'
        S.hash_classes[tp] = (pre, comps)
        cases = []
        for j in range(2):
            red, cat, hx, hy = ents[j]
            for proj, h in (('1', hx), ('2', hy)):
                cases.append(f'''  · rw [Step0Targets.familyPack_entry {S.pack_fact} _ {2 * tp + j} (by simp) rfl]
    simp only [List.get_eq_getElem, List.getElem_cons_zero, List.getElem_cons_succ]
    rw [modulusReduce_same (by decide) {S.fact(red)}, (concatRows_two {S.fact(cat)}).{proj},
      MxxExponentLut.hashSample_eq {S.fact(h[0])}, hseed]
    rfl''')
        out.append(f'''/-- The refreshed pair of exported target {t} under its actual root key. -/
theorem root_{tp} {{model : HashModel}} {{co : Step0Rows.OnlineCtx model}} {{outputs : _}}
    (F : generatedRoot.Facts co outputs) {{key : ByteArray}} (hseed : generatedRoot.{S.seed_wire} co = key)
    {{next : ErrorPoly {S.n}}} {{target : Fin {S.parts} → ErrorPoly {S.n}}} {{value : Fin 2 → ErrorPoly {S.n}}}
    (hv : ∀ c, target (StepTransition.partIndex ⟨{t}, by decide⟩ c) = value c)
    (hp : MxxExponentLut.Rel next
      (fun r c ↦ MxxExponentLut.rowOf (generatedRoot.{S.T.family}_3 co (StepTransition.partIndex ⟨{t}, by decide⟩ c)) r)
      (fun c ↦ reducePoly modulus_0 {S.n} (target (StepTransition.partIndex ⟨{t}, by decide⟩ c)) • generatedRoot.{S.T.gadget} co)
      (StepTransition.nextRows co ⟨{t}, by decide⟩) {S.fresh_bound}) :
    MxxExponentLut.Rel next (keyPairG model key modulus_0 {S.m} {pre} {comps})
      (fun c ↦ reducePoly modulus_0 {S.n} (value c) • generatedRoot.{S.T.gadget} co)
      (StepTransition.nextRows co ⟨{t}, by decide⟩) {S.fresh_bound} := by
  refine rel_congr hp (fun r j ↦ ?_) (fun j ↦ by rw [hv]) (fun _ ↦ rfl) le_rfl
  rw [target_nonlabel_key co F _ {tp} (by decide) j (by simp only [StepTransition.partIndex]; omega)
    (by simp only [StepTransition.partIndex]; omega)]
  fin_cases j <;> fin_cases r <;> simp only [Fin.zero_eta, Fin.mk_one, Fin.isValue, Step0Targets.pairIndex]
''' + '\n'.join(cases) + '\n')
    return '\n'.join(out)


# --------------------------------------------------------------------------- hint

def hint_setup(S, fields):
    """The anchor parts, their keys and the exported hint chain."""
    T, m = S.T, S.m
    S.anchor_tps = [tp for tp in range(S.nlt) if S.cls[tp][1]['kind'] == 'anchor']
    meta.require(S.anchor_tps == list(range(S.anchor_tps[0], S.anchor_tps[0] + len(S.anchor_tps))),
                 'anchor classes are consecutive')
    S.anchor_targets = [S.cls[tp][0] for tp in S.anchor_tps]
    meta.require(S.anchor_targets == list(range(S.anchor_targets[0], S.anchor_targets[0] + len(S.anchor_tps))),
                 'anchor targets are consecutive')
    S.first_anchor_part = 2 * S.anchor_targets[0]
    meta.require(S.first_anchor_part + 2 * len(S.anchor_tps) == S.parts, 'anchor parts are the last parts')
    Ac = S.geo.anchor
    S.used = -(-Ac // m)
    P = 2 * len(S.anchor_tps)
    meta.require(S.used <= P, 'anchor columns fit the anchor parts')
    # Keys: every used anchor part reads a slice of the padded next anchor.
    S.akeys = []
    concat_input = None
    for k in range(S.used):
        e = S.entries[2 * S.anchor_tps[0] + k]
        red, sl = entry_info(S, e)
        meta.require(sl.kind == 'sliceMatrix', f'anchor key {k}: slice')
        start = int(strip_parens(sl.args[3]))
        meta.require(start == m * k, f'anchor key {k}: slice start')
        src = strip_parens(sl.args[0])
        cat = S.producer(src, 'concatColumns')
        if concat_input is None:
            concat_input = cat
        meta.require(cat is concat_input, 'anchor keys slice one padded anchor')
        S.akeys.append((S.fact(red), S.fact(sl)))
    S.anchor_cat_fact = S.fact(concat_input)
    S.anchor_wire = strip_parens(concat_input.args[0])
    meta.require(S.input_term(S.anchor_wire) == 'execution.«stage_2».2.1', 'padded anchor is the next anchor public')
    # The exported hint: a slice of the column concatenation of the anchor parts.
    S.hint_wire = dict(S.root.outputs())[fields['hint'][0][0]]
    hs = S.producer(S.hint_wire, 'sliceMatrix')
    S.hint_slice_fact = S.fact(hs)
    top = strip_parens(hs.args[0])
    chain = []
    w = top
    while True:
        node = S.root.node(w)
        meta.require(node[0] == 'atom', f'{w}: hint chain')
        a = node[1]
        if a.kind != 'concatColumns':
            break
        left, right = strip_parens(a.args[0]), strip_parens(a.args[1])
        chain.append((S.fact(a), right))
        w = left
    chain.reverse()
    bottom = w
    parts = [bottom] + [r for _, r in chain]
    meta.require(len(parts) == P, 'hint concatenates every anchor part')
    S.hint_top = top
    S.hint_chain = chain  # chain[j-1] is the concatenation adding part j
    S.hint_parts = []
    for j, pw in enumerate(parts):
        g = S.producer(pw, 'familyGetStatic')
        meta.require(meta.integer(g.args[1]) == S.first_anchor_part + j, f'hint part {j}')
        S.hint_parts.append(S.fact(g))


def hint_source(S):
    n, m, Ac = S.n, S.m, S.geo.anchor
    T = S.T
    bound = S.root_bound
    out = []
    for k in range(S.used):
        tp = S.anchor_tps[k // 2]
        comp = k % 2
        red_f, sl_f = S.akeys[k]
        out.append(f'''/-- Anchor key slice {k}: the packed anchor key is the next anchor's columns. -/
theorem anchor_key{k} {{model : HashModel}} {{co : Step0Rows.OnlineCtx model}} {{outputs : _}}
    (F : generatedRoot.Facts co outputs) {{A : ExactMatrix modulus_0 {n} 2 {Ac}}}
    (hA : generatedRoot.{S.anchor_wire} co = A) (r : Fin 2) (c : Fin {m}) (d : Fin {Ac})
    (hd : d.val = {m} * {k} + c.val) :
    generatedRoot.{S.keyfam_root} co (Step0Targets.pairIndex {tp} (by decide) {comp}) r c = A r d := by
  have hc := c.isLt
  have hdl := d.isLt
  rw [Step0Targets.familyPack_entry {S.pack_fact} _ {2 * tp + comp} (by simp) rfl]
  simp only [List.get_eq_getElem, List.getElem_cons_zero, List.getElem_cons_succ]
  rw [modulusReduce_same (by decide) {red_f}]
  obtain ⟨-, -, -, -, -, -, -, -, hs⟩ := {sl_f}
  rw [hs r c (by simp) (by simp; omega), {S.anchor_cat_fact}, dif_pos (by simp; omega), hA]
  congr 1 <;> (apply Fin.ext; simp) <;> omega
''')
    t0 = S.anchor_targets[0]
    rels = '\n'.join(f'''    (h{t} : MxxExponentLut.Rel next
      (fun r c ↦ MxxExponentLut.rowOf (generatedRoot.{T.family}_3 co (StepTransition.partIndex ⟨{t}, by decide⟩ c)) r)
      (fun c ↦ reducePoly modulus_0 {n} (target (StepTransition.partIndex ⟨{t}, by decide⟩ c)) • generatedRoot.{T.gadget} co)
      (StepTransition.nextRows co ⟨{t}, by decide⟩) {S.fresh_bound})''' for t in S.anchor_targets)
    obtains = '\n'.join(f'  obtain ⟨e{t}, he{t}, r{t}⟩ := h{t}' for t in S.anchor_targets)
    keyeqs = '\n'.join(f'''  have k{t} : ∀ c r, MxxExponentLut.rowOf (generatedRoot.{T.family}_3 co (StepTransition.partIndex ⟨{t}, by decide⟩ c)) r =
      MxxExponentLut.rowOf (generatedRoot.{S.keyfam_root} co (Step0Targets.pairIndex {tp} (by decide) c)) r := by
    intro c r
    rw [target_nonlabel_key co F _ {tp} (by decide) c (by simp only [StepTransition.partIndex]; omega)
      (by simp only [StepTransition.partIndex]; omega)]''' for tp, t in zip(S.anchor_tps, S.anchor_targets))
    # the error matrix: part k contributes columns [m k, m (k + 1))
    def eterm(k):
        t = S.anchor_targets[k // 2]
        col = 'c.val' if k == 0 else f'c.val - {m * k}'
        return f'e{t} {k % 2} 0 ⟨{col}, by omega⟩'
    ebody = ''
    for k in range(S.used - 1):
        ebody += f'    if h{k + 1} : c.val < {m * (k + 1)} then {eterm(k)}\n    else '
    ebody += eterm(S.used - 1)
    eb_cases = '\n'.join(f'    · exact (he{S.anchor_targets[k // 2]} _ _ _ _).trans (by norm_num)' for k in range(S.used))
    parts_have = '\n'.join(f'  have p{j} := Step0Targets.family_get {S.hint_parts[j]} ⟨{S.first_anchor_part + j}, by decide⟩ rfl'
                           for j in range(S.used))
    P = len(S.hint_parts)
    # top levels beyond the used parts always go left
    top_rw = []
    for j in range(P - 1, S.used - 1, -1):
        fact, _ = S.hint_chain[j - 1]
        top_rw.append(f'{fact}, dif_pos (show c.val < {m * j} by omega)')
    cases = []
    for k in range(S.used):
        rw = []
        for j in range(S.used - 1, 0, -1):
            fact, _ = S.hint_chain[j - 1]
            if j > k:
                rw.append(f'{fact}, dif_pos (show c.val < {m * j} by omega)')
            elif j == k:
                rw.append(f'{fact}, dif_neg (show ¬ c.val < {m * j} by omega)')
                break
        rw.append(f'p{k}')
        for i in range(S.used - 1):
            if i < k:
                rw.append(f'dif_neg (show ¬ c.val < {m * (i + 1)} by omega)')
            elif i == k:
                rw.append(f'dif_pos (show c.val < {m * (i + 1)} by omega)')
                break
        t = S.anchor_targets[k // 2]
        tp = S.anchor_tps[k // 2]
        col = 'c.val' if k == 0 else f'c.val - {m * k}'
        cases.append([f'rw [{", ".join(rw)}]',
                      f'rw [show (⟨{S.first_anchor_part + k}, by decide⟩ : Fin {S.parts}) = StepTransition.partIndex ⟨{t}, by decide⟩ {k % 2} from rfl,',
                      f'  part ⟨{t}, by decide⟩ {k % 2} e{t} (by decide) {tp} (by decide) k{t} r{t}]',
                      'simp only [Fin.val_mk]',
                      f'rw [anchor_key{k} F hA 0 ⟨{col}, by omega⟩ c (by simp <;> omega), anchor_key{k} F hA 1 ⟨{col}, by omega⟩ c (by simp <;> omega)]'])

    def emit(k, indent):
        if k == S.used - 1:
            return [indent + x for x in cases[k]]
        lines = [indent + f'by_cases hc{k + 1} : c.val < {m * (k + 1)}']
        body = cases[k]
        lines.append(indent + '· ' + body[0])
        lines += [indent + '  ' + x for x in body[1:]]
        rest = emit(k + 1, indent + '  ')
        lines.append(indent + '· ' + rest[0].strip())
        lines += rest[1:]
        return lines
    case_text = '\n'.join(emit(0, '  ')) + '\n'
    out.append(f'''/-- **The refreshed hint**: the first anchor-width columns of the zero-plaintext anchor
parts are the next anchor masked by the next mask, with their certified fresh errors. -/
theorem hint {{model : HashModel}} {{co : Step0Rows.OnlineCtx model}} {{outputs : _}}
    (F : generatedRoot.Facts co outputs) {{A : ExactMatrix modulus_0 {n} 2 {Ac}}}
    (hA : generatedRoot.{S.anchor_wire} co = A) {{next : ErrorPoly {n}}} {{target : Fin {S.parts} → ErrorPoly {n}}}
    (hz : ∀ part : Fin {S.parts}, {S.first_anchor_part} ≤ part.val → target part = 0)
{rels} :
    ∃ e : ErrorMatrix {n} 1 {Ac}, CoeffBound e {bound} ∧
      generatedRoot.{S.hint_wire} co =
        maskPart (reducePoly modulus_0 {n} next) (MxxExponentLut.rowOf A) + reduceMatrix modulus_0 {n} 1 {Ac} e := by
{obtains}
{keyeqs}
  have part : ∀ (t : Fin {S.targets}) (c : Fin 2) (e : Fin 2 → ErrorMatrix {n} 1 {m})
      (ht : {t0} ≤ t.val) (ttp : Nat) (htp : ttp < {S.nlt}) (hk : ∀ c r,
        MxxExponentLut.rowOf (generatedRoot.{T.family}_3 co (StepTransition.partIndex t c)) r =
        MxxExponentLut.rowOf (generatedRoot.{S.keyfam_root} co (Step0Targets.pairIndex ttp htp c)) r)
      (hr : ∀ j, StepTransition.nextRows co t j =
        reducePoly modulus_0 {n} next •
          (fun r c ↦ MxxExponentLut.rowOf (generatedRoot.{T.family}_3 co (StepTransition.partIndex t c)) r) 0 j +
        (fun r c ↦ MxxExponentLut.rowOf (generatedRoot.{T.family}_3 co (StepTransition.partIndex t c)) r) 1 j -
        (fun c ↦ reducePoly modulus_0 {n} (target (StepTransition.partIndex t c)) • generatedRoot.{T.gadget} co) j +
        reduceMatrix modulus_0 {n} 1 {m} (e j)) (col : Fin {m}),
      generatedRoot.{S.part_family} co (StepTransition.partIndex t c) 0 col =
        reducePoly modulus_0 {n} next * generatedRoot.{S.keyfam_root} co (Step0Targets.pairIndex ttp htp c) 0 col +
          generatedRoot.{S.keyfam_root} co (Step0Targets.pairIndex ttp htp c) 1 col +
          reducePoly modulus_0 {n} (e c 0 col) := by
    intro t c e ht ttp htp hk hr col
    have h := congrFun (congrFun (hr c) 0) col
    simp only at h
    rw [hk, hk, hz _ (by simp only [StepTransition.partIndex]; omega)] at h
    change generatedRoot.{S.part_family} co (StepTransition.partIndex t c) 0 col = _ at h
    rw [h]
    simp [MxxExponentLut.rowOf, reduceMatrix]
  let e : ErrorMatrix {n} 1 {Ac} := fun _ c ↦
{ebody}
  refine ⟨e, fun r c k ↦ ?_, ?_⟩
  · simp only [e]
    split_ifs
{eb_cases}
  funext r c
  have hr : r = 0 := Subsingleton.elim _ _
  subst hr
  have hc := c.isLt
{parts_have}
  have hslice : generatedRoot.{S.hint_wire} co 0 c = generatedRoot.{S.hint_top} co 0 ⟨c.val, by omega⟩ := by
    obtain ⟨-, -, -, -, -, -, -, -, hs⟩ := {S.hint_slice_fact}
    rw [hs 0 c (by simp) (by simp; omega)]
    congr 1
    apply Fin.ext
    simp
  rw [hslice{', ' + ', '.join(top_rw) if top_rw else ''}]
  simp only [Matrix.add_apply, maskPart, Matrix.smul_apply, MxxExponentLut.rowOf, smul_eq_mul,
    reduceMatrix, e]
{case_text}''')
    return '\n'.join(out)


# --------------------------------------------------------------------------- assembly

def transport_fields(S):
    src = (S.geo.package / 'proof' / 'Dio2Step0Transport.lean').read_text()
    i = src.index('structure Online0RefreshedRoots')
    j = src.index('/-- Transport every', i)
    body = src[i:j]
    fields = {}
    for fld in re.split(r'\n  (?=\w+ :)', body)[1:]:
        name = fld.split(' :')[0].strip()
        paths = re.findall(r'execution\.«stage_11»((?:\.[12])+)', fld)
        bound = re.findall(r'\(([\d +]+)\)\s*$', fld.strip())
        fields[name] = (paths, fld)
    meta.require(list(fields) == ['labels', 'state', 'privateUnit', 'stateHelpers', 'smudgeHelpers', 'hint'],
                 f'Online0RefreshedRoots fields {list(fields)}')
    bm = re.search(r'CoeffBound error \(([\d +]+)\)', fields['hint'][1])
    meta.require(bm is not None, 'refreshed root bound')
    S.root_bound = f'({bm.group(1)})'
    for name in ['labels', 'state', 'privateUnit', 'stateHelpers', 'smudgeHelpers']:
        meta.require(fields[name][1].rstrip().endswith(S.root_bound), f'{name}: bound')
    return fields


def route_lemmas(S):
    src = (S.geo.package / 'proof' / 'Dio2StepTransition.lean').read_text()
    out = {}
    for m in re.finditer(r'theorem (route_\w+) \{backend[^\n]*\n[^\n]*\n[^\n]*\(facts : generatedRoot\.Facts ctx outputs\)( \(member : Fin (\d+)\))? :\n\s*(.*?) := by', src, re.S):
        name, members, concl = m.group(1), m.group(3), m.group(4)
        dyn = re.fullmatch(r'familyGetDynamic \(generatedRoot\.(w_\d+_\d+) ctx\) (.*)\n\s*\(generatedRoot\.(w_\d+_\d+) ctx member\)', concl.strip(), re.S)
        if dyn:
            out[dyn.group(3)] = (name, int(members), dyn.group(2))
            continue
        st = re.fullmatch(r'generatedRoot\.(w_\d+_\d+) ctx = generatedRoot\.(w_\d+_\d+) ctx ⟨(\d+), by omega⟩', concl.strip())
        if st:
            out[st.group(1)] = (name, None, int(st.group(3)))
    return out


def final_source(S, fields):
    n, T = S.n, S.T
    outs = dict(S.root.outputs())
    routes = route_lemmas(S)
    hout_atom = S.root.out_atom
    hout = f'F.c{hout_atom.frag}.{hout_atom.field}'
    result = 'Xe (n := {n}) (2 * (entry table0 ((state + bit : Nat) : Int)).toNat)'.replace('{n}', str(n))
    def wires(name):
        return [outs[p] for p in fields[name][0]]
    def static_pair(name, tp, value):
        w0, w1 = wires(name)
        r0, r1 = routes[w0], routes[w1]
        t = S.cls[tp][0]
        meta.require(r0[1] is None and r0[2] == 2 * t and r1[2] == 2 * t + 1, f'{name}: static route')
        head = f'''    have h := root_{tp} F hseed (value := fun c ↦ {value})
      (fun c ↦ by fin_cases c <;> rfl) (hp ⟨{t}, by decide⟩)'''
        if name == 'privateUnit':
            return head + f'''
    refine rel_congr h (fun r j ↦ ?_) (fun _ ↦ rfl) (fun j ↦ ?_) (by norm_num)
    · fin_cases r <;> fin_cases j <;> rfl
    · fin_cases j
      · exact (StepTransition.{r0[0]} {CTX} F).symm
      · exact (StepTransition.{r1[0]} {CTX} F).symm'''
        return head + f'''
    refine rel_congr h (fun r j ↦ rfl) (fun _ ↦ rfl) (fun j ↦ ?_) (by norm_num)
    fin_cases j
    · exact (StepTransition.{r0[0]} {CTX} F).symm
    · exact (StepTransition.{r1[0]} {CTX} F).symm'''
    # classes by kind
    kind = {S.cls[tp][1]['kind']: tp for tp in range(S.nlt) if S.cls[tp][1]['kind'] in ('private', 'public', 'state', 'sigma')}
    priv, pub, stat = kind['private'], kind['public'], kind['state']
    tp_of_target = {S.cls[tp][0]: tp for tp in range(S.nlt)}
    # labels
    l0, l1 = wires('labels')
    lr0, lr1 = routes[l0], routes[l1]
    meta.require(lr0[1] == S.R, 'label-0 route members')
    pub_t = S.cls[pub][0]
    labels = f'''  · intro m
    rw [hout]
    by_cases hm : m.val = {S.R - 1}
    · have hm' : m = {S.R - 1} := Fin.ext hm
      subst hm'
      have h := root_{pub} F hseed
        (value := fun c ↦ online1Labels external {S.R - 1} * unitPair next c)
        (fun c ↦ by
          simp only [show online1Labels external {S.R - 1} = 1 from Complete.label1_last_payload hrun, one_mul]
          fin_cases c <;> rfl)
        (hp ⟨{pub_t}, by decide⟩)
      refine rel_congr h (fun r j ↦ rfl) (fun _ ↦ rfl) (fun j ↦ ?_) (by norm_num)
      fin_cases j
      · exact (dynamic_route _ (by simp [MxxIR.selectInt, StepTransition.partIndex])
          (StepTransition.{lr0[0]} {CTX} F {S.R - 1})).symm
      · exact (dynamic_route _ (by simp [MxxIR.selectInt, StepTransition.partIndex])
          (StepTransition.{lr1[0]} {CTX} F {S.R - 1})).symm
    · have hlt : m.val * {S.C} < {S.L} := by omega
      let fm : Fin {S.L} := ⟨m.val * {S.C}, hlt⟩
      let t : Fin {S.targets} := ⟨{S.ls} + m.val * {S.C}, by omega⟩
      refine rel_congr (hp t) (fun r j ↦ ?_) (fun j ↦ ?_) (fun j ↦ ?_) (by norm_num)
      · rw [target_label0_key {CTX} F _ fm j
          (by simp only [StepTransition.partIndex, t, fm]; omega)
          (by simp only [StepTransition.partIndex, t]; omega)
          (by simp only [fm]; omega) r, hseed]
        congr 2
        simp only [fm]; omega
      · show reducePoly modulus_0 {n} (Step0Targets.stepTarget initial.payload next ({result}) label
          (StepTransition.partIndex t j)) • _ = _
        have hv : Step0Targets.stepTarget initial.payload next ({result}) label (StepTransition.partIndex t j) =
            label fm * unitPair next j := by
          unfold Step0Targets.stepTarget
          rw [dif_pos (by simp only [StepTransition.partIndex, t]; omega)]
          congr 2
          · apply Fin.ext; simp only [StepTransition.partIndex, t, fm]; omega
          · apply Fin.ext; simp only [StepTransition.partIndex, t]; omega
        rw [hv, map_mul, map_mul, hbridge m fm hm rfl]
      · fin_cases j
        · exact (dynamic_route _ (by simp [MxxIR.selectInt, StepTransition.partIndex, t, hm]; omega)
            (StepTransition.{lr0[0]} {CTX} F m)).symm
        · exact (dynamic_route _ (by simp [MxxIR.selectInt, StepTransition.partIndex, t, hm]; omega)
            (StepTransition.{lr1[0]} {CTX} F m)).symm'''
    def helpers(name, sigmas):
        w0, w1 = wires(name)
        r0, r1 = routes[w0], routes[w1]
        lst = re.search(r'MxxIR\.selectInt \(\(Int\.ofNat member\.val\)\) \[([^\]]*)\]', r0[2])
        meta.require(lst is not None, f'{name}: route targets')
        tlist = [int(strip_parens(x)) for x in lst.group(1).split(',')]
        meta.require(len(tlist) == r0[1], f'{name}: route members')
        src = f'  · intro branch\n    rw [hout]\n    fin_cases branch\n'
        for b, t in enumerate(tlist):
            tp = tp_of_target[t]
            k = S.cls[tp][1].get('automorphism')
            fix = (f'rw [substitution_one_of ({sigmas}_odd {b}) (show (sigmaOf {sigmas} {b}).toNat = 1 from rfl)]; '
                   if k == 1 else '')
            src += f'''    · have h := root_{tp} F hseed
        (value := fun c ↦ unitPair (substitution (sigmaOf {sigmas} {b}).toNat ({sigmas}_odd {b}) next) c)
        (fun c ↦ by {fix}fin_cases c <;> rfl) (hp ⟨{t}, by decide⟩)
      refine rel_congr h (fun r j ↦ rfl) (fun _ ↦ rfl) (fun j ↦ ?_) (by norm_num)
      fin_cases j
      · exact (dynamic_route _ (by simp [MxxIR.selectInt, StepTransition.partIndex])
          (StepTransition.{r0[0]} {CTX} F {b})).symm
      · exact (dynamic_route _ (by simp [MxxIR.selectInt, StepTransition.partIndex])
          (StepTransition.{r1[0]} {CTX} F {b})).symm
'''
        return src.rstrip('\n')
    sh = re.search(r'\[\.integer \(sigmaOf (\w+) branch\)\]', fields['stateHelpers'][1]).group(1)
    mh = re.search(r'\[\.integer \(sigmaOf (\w+) branch\)\]', fields['smudgeHelpers'][1]).group(1)
    hz_hi = S.first_anchor_part
    anchor_lo = S.anchor_targets[0] - S.L
    anchor_vals = ' ∨\n        '.join(f'(if part.val / 2 < {S.ls} then part.val / 2 else part.val / 2 - {S.L}) = {tp}'
                                       for tp in S.anchor_tps)
    rcases = ' | '.join('h' for _ in S.anchor_tps)
    hint_rels = ' '.join(f'(hp ⟨{t}, by decide⟩)' for t in S.anchor_targets)
    return f'''theorem substitution_one {{n : Nat}} (h : Odd 1) (x : ErrorPoly n) :
    substitution (n := n) (R := Int) 1 h x = x := by
  have e : substitution (n := n) (R := Int) 1 h = RingHom.id _ := by
    apply AdjoinRoot.ringHom_ext
    · ext r
      simp [substitution]
    · simp [substitution]
  rw [e]; rfl

theorem substitution_one_of {{n k : Nat}} (hk : Odd k) (h : k = 1) (x : ErrorPoly n) :
    substitution (n := n) (R := Int) k hk x = x := by
  subst h
  exact substitution_one hk x

theorem dynamic_route {{f : Fin {S.parts} → ExactMatrix modulus_0 {n} 1 {S.m}}} {{index : Int}}
    {{value : ExactMatrix modulus_0 {n} 1 {S.m}}} (part : Fin {S.parts}) (hindex : (part.val : Int) = index)
    (h : familyGetDynamic f index value) : value = f part :=
  Online0Labels.read_at part hindex h

end Step0

open Step0 in
/-- **Step zero, from the actual source targets.** Every refreshed export of online
zero is a certified encoding under the next mask with the next root keys; the label
plaintexts are the requirement's next external labels, and import one carries them
into the shared step-one invariant. -/
theorem Complete.step0_transition_of {{model : HashModel}} {{external : GeneratedClaim.ExternalInputs}}
    {{execution : GeneratedClaim.Execution}} (hrun : W_GeneratedClaim.Runs model external execution)
    (state bit : Nat) (hsrc : Online0SourceTargets model external execution state bit) :
    State1 model external execution
      ({result}) := by
  obtain ⟨initial, next, fresh, smudge, R⟩ := hsrc
  obtain ⟨label, hbridge, hp⟩ := refreshed_pairs hrun initial R
  have F := initial.context.online_facts
  have hseed : generatedRoot.{S.seed_wire} {CTX} = external.input_3 := by
    rw [generatedRoot.{S.seed_wire}, initial.context.online_inputs]; rfl
  have hA : generatedRoot.{S.anchor_wire} {CTX} = execution.«stage_1».2.2.1 := by
    rw [generatedRoot.{S.anchor_wire}, initial.context.online_inputs]; exact (anchor_public hrun).2
  have hout := {hout}
  refine state1_of_refreshed_roots hrun (sm := next) (sp := initial.payload)
    (gadget := generatedRoot.{T.gadget} {CTX}) R.next_small initial.payload_small
    R.gadget (Complete.payload_matrix_eq initial.payload_eq) ?_
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
{labels}
  · rw [hout]
{static_pair('state', stat, result.join(['(', ')']) + ' * unitPair initial.payload c')}
  · rw [hout]
{static_pair('privateUnit', priv, 'unitPair initial.payload c')}
{helpers('stateHelpers', sh)}
{helpers('smudgeHelpers', mh)}
  · rw [hout]
    refine hint F hA (fun part hpart ↦ ?_) {hint_rels}
    unfold Step0Targets.stepTarget
    rw [dif_neg (by omega)]
    have hanchor : {anchor_vals} := by
      split_ifs <;> omega
    rcases hanchor with {rcases} <;> rw [h] <;> rfl

/-- **Step zero, closed.** The actual source targets of online zero (s0a) feed the
refreshed-root assembly above; nothing beyond the run, the state and the label is assumed. -/
theorem Complete.step0_transition : ∀ {{model : HashModel}} {{external : GeneratedClaim.ExternalInputs}}
    {{execution : GeneratedClaim.Execution}}, W_GeneratedClaim.Runs model external execution →
    ∀ (state bit : Nat), state ≤ 1 → bit ≤ 1 →
    Label model external execution 0 (Xe (n := {n}) (2 * bit)) →
    State0 model external execution (Xe (n := {n}) (2 * state)) →
    State1 model external execution
      ({result}) :=
  fun hrun state bit hstate hbit hlabel hinput ↦
    Complete.step0_transition_of hrun state bit
      (online0_source_targets hrun state bit hstate hbit hinput hlabel)

end Dio2Proof
'''


HEADER = '''import Dio2Step0Targets
import {links_module}
import Dio2Step0Transport
import Dio2Online0SourcesPacking
import Dio2Preimage

/-!
# Step zero

Generated by `gen_s0o.py` (module `s0o_step0.py`) from checked export
metadata. The refreshed roots of online zero are assembled from the actual
source targets (`online0_source_targets`), the per-part CRT rows, the
key-stage agreement, the claim's sampled preimages and the actual target
selection; import one then constructs `State1`. Label members stay
quantified; helper branches are the exported helper target classes.
-/

set_option maxRecDepth 16384
set_option maxHeartbeats 16000000
set_option Elab.async false

namespace Dio2Proof
open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh
  RefreshTopProof RefreshTopProof.Gen RefreshTopProof.GenIface RefreshTopProof.GenInst
open W_Stage_online_0 Dio2NoiseCertificate

namespace Step0

variable {model : HashModel} {external : GeneratedClaim.ExternalInputs}
  {execution : GeneratedClaim.Execution}

'''


RELCONGR = '''theorem rel_congr {{next : ErrorPoly {n}}} {{K K' : Fin 2 → Fin 2 → ExactMatrix modulus_0 {n} 1 {m}}}
    {{t t' c c' : Fin 2 → ExactMatrix modulus_0 {n} 1 {m}}} {{B B' : Nat}}
    (h : MxxExponentLut.Rel next K t c B) (hK : ∀ r j, K r j = K' r j) (ht : ∀ j, t j = t' j)
    (hc : ∀ j, c j = c' j) (hB : B ≤ B') : MxxExponentLut.Rel next K' t' c' B' := by
  have e1 : K = K' := funext fun r ↦ funext fun j ↦ hK r j
  have e2 : t = t' := funext ht
  have e3 : c = c' := funext hc
  subst e1 e2 e3
  exact h.mono hB

'''


def step0_source(geo, T, PF, links_module):
    S = Step0(geo, T, PF)
    nonlabel_setup(S)
    sources = (geo.package / 'proof' / 'Dio2Online0Sources.lean').read_text()
    fb = re.search(r'fresh_small : ∀ cell, polyNorm \(fresh cell\) ≤ (\d+)', sources)
    meta.require(fb is not None, 'fresh plaintext bound')
    S.fresh_bound = int(fb.group(1))
    S.part_family = geo.view['refresh']['family_output'][0].split()[0]
    fields = transport_fields(S)
    hint_setup(S, fields)
    text = (HEADER.replace('{links_module}', links_module) + anchors_source(S) + refreshed_source(S) + keys_source(S) +
            RELCONGR.format(n=S.n, m=S.m) + root_lemmas(S) + '\n' + hint_source(S) + '\n' +
            final_source(S, fields))
    return text
