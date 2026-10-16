"""Step-zero refresh target relations for gen_s0o.py.

Every refresh part reads its target ciphertext and key rows from the actual
target-selection loop: non-label targets from three packed families, label
targets from the symbolic label forest.  This module emits:

* exact selection lemmas for both branches of that loop (integer index
  arithmetic is discharged by omega; no part is enumerated);
* one relation lemma per exported non-label target circuit (private, public,
  state, sigma, each automorphism helper and the zero anchors), obtained from
  the existing class theorems and the actual source relations;
* the label forest relation for an arbitrary label member;
* `target_rows`, the `Step0Rows.TargetRows` premise for every part.

Key rows are identified with the class theorems' key expressions by unfolding
the actual definitional lets, slices, zero-count loops and hash samples of the
exported graph.  No plaintext, noise or key equality is assumed.
"""
import re

import dio2_meta as meta
import s0o_agree as agree
from xmodel import app_split, strip_parens

ONLINE = 'W_Stage_online_0'


def and_proj(index, count):
    if count == 1:
        return ''
    if index == count - 1:
        return '.2' * index
    return '.2' * index + '.1'


class Targets:
    def __init__(self, geo, gen):
        self.geo = geo
        self.gen = gen
        self.root = gen.info('A', 'generatedRoot')
        pin, pargs = geo.part_inputs, geo.part_args
        w0 = geo.primes[0]['wires']
        self.family = pargs[pin.index(w0['target'])].split()[0][:-2]
        node = self.root.node(self.family + '_0')
        meta.require(node[0] == 'loop', 'target family is not a loop output')
        self.select_atom = node[1]
        self.select_scope, args, outs = agree.call_args(self.select_atom)
        leaves = [strip_parens(x) for _, x in agree.leaves(args)]
        meta.require(len(leaves) == 11, 'unexpected target selection arity')
        (self.fam_c, self.lab_c0, self.lab_c1, self.fam_a0, self.lab_a00, self.lab_a01,
         self.fam_a1, self.lab_a10, self.lab_a11, self.fam_keys, self.seed) = leaves
        self.lists = {}
        for fam in (self.fam_c, self.fam_a0, self.fam_a1):
            atom = self.root.prod.get(fam)
            meta.require(atom is not None and atom[0] == 'atom' and atom[1].kind == 'familyPack',
                         f'{fam} is not a packed family')
            items = re.findall(r'w_\d+_\d+', atom[1].args[1])
            self.lists[fam] = (items, atom[1])
        self.nl2 = len(self.lists[self.fam_c][0])
        meta.require(self.nl2 % 2 == 0 and all(len(self.lists[f][0]) == self.nl2
                                               for f in (self.fam_a0, self.fam_a1)), 'family sizes')
        self.nlt = self.nl2 // 2
        classes = geo.view['target_classes']
        labels = [c for c in classes if c['kind'] == 'labels']
        meta.require(len(labels) == 1, 'label class')
        self.label_start = labels[0]['start']
        self.labels = labels[0]['stop'] - labels[0]['start']
        self.nonlabel = []
        for c in classes:
            if c['kind'] == 'labels':
                continue
            for t in range(c['start'], c['stop']):
                self.nonlabel.append((t, c))
        meta.require(len(self.nonlabel) == self.nlt, 'non-label target count mismatch')
        meta.require(self.label_start == 3 and [t for t, _ in self.nonlabel[:3]] == [0, 1, 2],
                     'unexpected target class order')
        lab = self.root.node(self.lab_c0)
        meta.require(lab[0] == 'loop', 'label family is not a loop output')
        self.forest_atom = lab[1]
        self.forest_scope, fargs, fouts = agree.call_args(self.forest_atom)
        self.forest_args = [strip_parens(x) for _, x in agree.leaves(fargs)]
        self.forest_outs = [strip_parens(x).split()[0] for _, x in agree.leaves(fouts)]
        meta.require(len(self.forest_outs) == 6, 'forest loop outputs')

    def fact(self, atom):
        return f'F.c{atom.frag}.{atom.field}'

    def entry(self, fam, idx):
        return self.lists[fam][0][idx]

    def pack_fact(self, fam):
        return self.fact(self.lists[fam][1])


PRELUDE = r'''theorem familyPack_entry {α : Type} {size : Nat} {count : Int} {values : List α}
    {f : Fin size → α} (h : familyPack count values f) (i : Fin size) (k : Nat)
    (hk : k < values.length) (hi : i.val = k) : f i = values.get ⟨k, hk⟩ := by
  obtain ⟨-, -, h⟩ := h
  obtain ⟨p, hp, ep⟩ := h i
  rw [ep]
  congr 1
  exact Fin.ext (by simp only; omega)

theorem select2 {α : Type} {i : Int} {a b out : α} (h : select i [a, b] out) :
    (i = 0 → out = a) ∧ (i = 1 → out = b) := by
  obtain ⟨p, hp, rfl⟩ := h
  obtain ⟨v, hv⟩ := p
  simp only [List.length_cons, List.length_nil] at hv
  constructor
  · intro hi
    have : v = 0 := by simp only at hp; omega
    subst this; rfl
  · intro hi
    have : v = 1 := by simp only at hp; omega
    subst this; rfl

theorem family_get {α : Type} {count : Nat} {f : Fin count → α} {i : Int} {out : α}
    (h : familyGetDynamic f i out) (k : Fin count) (hk : i = k.val) : out = f k := by
  obtain ⟨p, hp, rfl⟩ := h
  congr 1
  exact Fin.ext (by omega)

/-- One refreshed entry encodes `x` under keys `(a0, a1)` with error at most `B`. -/
def EntryRow {q n m : Nat} (s : ExactPoly q n) (c a0 a1 G : ExactMatrix q n 1 m)
    (x : ErrorPoly n) (B : Nat) : Prop :=
  ∃ e : ErrorMatrix n 1 m, CoeffBound e B ∧
    c = maskPart s ![a0, a1] - reducePoly q n x • G + reduceMatrix q n 1 m e

theorem entry_of {q n m : Nat} {s : ExactPoly q n} {c a0 a1 G G' : ExactMatrix q n 1 m}
    {K : Fin 2 → ExactMatrix q n 1 m} {X : ExactPoly q n} {x : ErrorPoly n}
    {e : ErrorMatrix n 1 m} {B B' : Nat}
    (h : c = maskPart s K - X • G' + reduceMatrix q n 1 m e) (he : CoeffBound e B)
    (hK : K = ![a0, a1]) (hX : X • G' = reducePoly q n x • G) (hB : B ≤ B') :
    EntryRow s c a0 a1 G x B' :=
  ⟨e, coeffBound_mono he hB, by rw [h, hK, hX]⟩

theorem entry_mono {q n m : Nat} {s : ExactPoly q n} {c a0 a1 G : ExactMatrix q n 1 m}
    {x : ErrorPoly n} {B B' : Nat} (h : EntryRow s c a0 a1 G x B) (hB : B ≤ B') :
    EntryRow s c a0 a1 G x B' := by
  obtain ⟨e, he, hc⟩ := h
  exact ⟨e, coeffBound_mono he hB, hc⟩

theorem rel_comp {q n m : Nat} {sm : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {t c : Fin 2 → ExactMatrix q n 1 m} {B : Nat} (h : MxxExponentLut.Rel sm A t c B) (j : Fin 2) :
    ∃ e : ErrorMatrix n 1 m, CoeffBound e B ∧
      c j = maskPart (reducePoly q n sm) (fun k ↦ A k j) - t j + reduceMatrix q n 1 m e := by
  obtain ⟨e, he, hc⟩ := h
  exact ⟨e j, he j, by rw [hc j, maskPart]⟩

theorem rel_of_entries {q n m : Nat} {sm : ErrorPoly n}
    {c0 c1 a00 a01 a10 a11 G : ExactMatrix q n 1 m} {x : Fin 2 → ErrorPoly n} {B : Nat}
    (h0 : EntryRow (reducePoly q n sm) c0 a00 a10 G (x 0) B)
    (h1 : EntryRow (reducePoly q n sm) c1 a01 a11 G (x 1) B) :
    MxxExponentLut.Rel sm ![![a00, a01], ![a10, a11]] (fun j ↦ reducePoly q n (x j) • G)
      ![c0, c1] B := by
  obtain ⟨e0, he0, hc0⟩ := h0
  obtain ⟨e1, he1, hc1⟩ := h1
  refine ⟨![e0, e1], fun j ↦ ?_, fun j ↦ ?_⟩
  · fin_cases j
    · exact he0
    · exact he1
  · fin_cases j
    · simpa [maskPart] using hc0
    · simpa [maskPart] using hc1

theorem rel_target_congr {q n m : Nat} {sm : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {t t' c : Fin 2 → ExactMatrix q n 1 m} {B : Nat} (h : MxxExponentLut.Rel sm A t c B)
    (ht : ∀ j, t j = t' j) : MxxExponentLut.Rel sm A t' c B := by
  obtain ⟨e, he, hc⟩ := h
  exact ⟨e, he, fun j ↦ by rw [hc j, ht j]⟩

'''


def int_tac():
    return ('(by simp only [Int.ofNat_eq_coe] at *; split_ifs <;> simp_all <;> omega)')


TAC = '(by simp only [Int.ofNat_eq_coe, decide_eq_true_eq, pairIndex, Fin.val_mk]; (try split_ifs) <;> omega)'


def selection_source(T):
    """Exact branch lemmas of the actual target-selection loop."""
    geo = T.geo
    S = T.gen.info('A', T.select_scope)
    parts = geo.parts
    fam = T.family
    O = f'{ONLINE}.generatedRoot'

    def conj(wire):
        node = S.node(wire)
        meta.require(node[0] == 'atom', f'{S.name}.{wire} is not an atom')
        a = node[1]
        group = S.conj[a.frag]
        j = next(k for k, b in enumerate(group) if b is a)
        return a, f'c{a.frag}{and_proj(j, len(group))}'
    outs = [w for _, w in S.outputs()]
    meta.require(len(outs) == 4, 'target selection outputs')
    head = []
    head += agree.Lemma.destruct_lines(S.exists + [f'c{k}' for k in S.frags], 'h', 'th', '  ')
    out_path = and_proj(len(S.conj[S.out_atom.frag]) - 1, len(S.conj[S.out_atom.frag]))
    head.append(f'  have hout := c{S.out_atom.frag}{out_path}')
    head.append('  simp only [Prod.mk.injEq] at hout')
    # per output: flag select, label select, families
    info = []
    for k in range(3):
        sel, sel_p = conj(outs[k])
        flag = strip_parens(sel.args[0])
        nl, lab = [strip_parens(x) for x in sel.args[1].strip('[]').split(',')]
        lsel, lsel_p = conj(lab)
        comp = strip_parens(lsel.args[0])
        g0, g1 = [strip_parens(x) for x in lsel.args[1].strip('[]').split(',')]
        ga0, ga0_p = conj(g0)
        ga1, ga1_p = conj(g1)
        nla, nla_p = conj(nl)
        info.append(dict(sel=sel_p, lsel=lsel_p, g0=ga0_p, g1=ga1_p, nl=nla_p))
    hdr = (f'    {{model : HashModel}} {{params : Params}}\n'
           f'    (ctx : generatedRoot.Ctx Backend.backend model params) {{outputs : _}}\n'
           f'    (F : generatedRoot.Facts ctx outputs) (part : Fin {parts})')
    fact = T.fact(T.select_atom)
    labs = [(T.lab_c0, T.lab_c1), (T.lab_a00, T.lab_a01), (T.lab_a10, T.lab_a11)]
    src = f'''/-- The label branch of the actual target-selection loop. -/
theorem target_label_entries
{hdr} (member : Fin {T.labels}) (j : Fin 2)
    (hmember : part.val / 2 = {T.label_start} + member.val) (hj : part.val % 2 = j.val) :
'''
    src += ' ∧\n'.join(f'    generatedRoot.{fam}_{k} ctx part = ![generatedRoot.{a} ctx, generatedRoot.{b} ctx] j member'
                       for k, (a, b) in enumerate(labs)) + ' := by\n'
    src += f'  have h := {fact} part\n  dsimp only [{S.name}] at h\n'
    src += '\n'.join(head) + '\n  have hm := member.isLt\n'
    for k, d in enumerate(info):
        src += (f'  have s{k} := (select2 {d["sel"]}).2 {TAC}\n'
                f'  have t{k} := select2 {d["lsel"]}\n'
                f'  have g{k}0 := family_get {d["g0"]} member {TAC}\n'
                f'  have g{k}1 := family_get {d["g1"]} member {TAC}\n')
    src += '  refine ⟨?_, ?_, ?_⟩\n'
    for k in range(3):
        proj = ['hout.1', 'hout.2.1', 'hout.2.2.1'][k]
        src += (f'  · rw [{proj}, s{k}]\n'
                f'    fin_cases j <;> (try simp only [Fin.val_mk, Fin.isValue, Fin.zero_eta, Fin.mk_one] at hj)\n'
                f'    · exact (t{k}.1 {TAC}).trans g{k}0\n'
                f'    · exact (t{k}.2 {TAC}).trans g{k}1\n')
    # non-label branch
    sel0 = S.prod[strip_parens(conj(outs[0])[0].args[1].strip('[]').split(',')[0])][1]
    idx_wire = strip_parens(sel0.args[1])
    # the integer existential selecting the non-label target
    int_sel = [a for a in S.s.atoms if a.kind == 'select' and not a.loop and
               S.s.wires.get(strip_parens(a.args[-1])) is not None and
               strip_parens(a.args[-1]) in S.exists and strip_parens(a.args[-1]) not in outs]
    int_sel = [a for a in int_sel if re.search(r'Int', str(S.s.wires[strip_parens(a.args[-1])].type or ''))]
    meta.require(len(int_sel) == 1, 'non-label index selection')
    iw = strip_parens(int_sel[0].args[-1])
    group = S.conj[int_sel[0].frag]
    ij = next(k for k, b in enumerate(group) if b is int_sel[0])
    ipath = f'c{int_sel[0].frag}{and_proj(ij, len(group))}'
    src += f'''
/-- The non-label branch of the actual target-selection loop. -/
theorem target_nonlabel_entries
{hdr} (t : Nat) (ht : t < {T.nlt}) (j : Fin 2)
    (hpart : (part.val / 2 < {T.label_start} ∧ part.val / 2 = t) ∨
      ({T.label_start + T.labels} ≤ part.val / 2 ∧ part.val / 2 = t + {T.labels}))
    (hj : part.val % 2 = j.val) :
    generatedRoot.{fam}_0 ctx part = generatedRoot.{T.fam_c} ctx (pairIndex t ht j) ∧
    generatedRoot.{fam}_1 ctx part = generatedRoot.{T.fam_a0} ctx (pairIndex t ht j) ∧
    generatedRoot.{fam}_2 ctx part = generatedRoot.{T.fam_a1} ctx (pairIndex t ht j) := by
  have h := {fact} part
  dsimp only [{S.name}] at h
'''
    src += '\n'.join(head) + '\n'
    src += f'''  have hjv := j.isLt
  have hi := select2 {ipath}
  have vi : {iw} = (t : Int) := by
    rcases hpart with ⟨h1, h2⟩ | ⟨h1, h2⟩
    · have := hi.2 {TAC}
      rw [this]; simp only [Int.ofNat_eq_coe]; omega
    · have := hi.1 {TAC}
      rw [this]; simp only [Int.ofNat_eq_coe]; omega
'''
    for k, d in enumerate(info):
        src += (f'  have s{k} := (select2 {d["sel"]}).1 {TAC}\n'
                f'  have n{k} := family_get {d["nl"]} (pairIndex t ht j) {TAC}\n')
    src += '  refine ⟨?_, ?_, ?_⟩\n'
    for k in range(3):
        proj = ['hout.1', 'hout.2.1', 'hout.2.2.1'][k]
        src += f'  · rw [{proj}, s{k}]; exact n{k}\n'
    return src


def pair_index_source(T):
    return f'''/-- The packed non-label entry of pair `t`, component `j`. -/
def pairIndex (t : Nat) (ht : t < {T.nlt}) (j : Fin 2) : Fin {T.nl2} :=
  ⟨2 * t + j.val, by omega⟩

'''


PRELUDE2 = r'''theorem entry_of_one {q n m : Nat} {s : ExactPoly q n} {c a0 a1 G G' : ExactMatrix q n 1 m}
    {K : Fin 2 → ExactMatrix q n 1 m} {x : ErrorPoly n}
    {e : ErrorMatrix n 1 m} {B B' : Nat}
    (h : c = maskPart s K - G' + reduceMatrix q n 1 m e) (he : CoeffBound e B)
    (hK : K = ![a0, a1]) (hX : G' = reducePoly q n x • G) (hB : B ≤ B') :
    EntryRow s c a0 a1 G x B' :=
  ⟨e, coeffBound_mono he hB, by rw [h, hK, hX]⟩

theorem zero_loop {S : Type} {body : Nat → S → S → Prop} {c : Nat} {init final : S}
    (h : MxxIR.IterRuns body c init final) (hc : c = 0) : final = init := by
  subst hc
  cases h
  rfl

theorem autoPlaintext_nat {n : Nat} (k : Nat) (hodd : Odd k) (sm : ErrorPoly n) :
    autoPlaintext (k : Int) sm = substitution k hodd sm := by
  simp [autoPlaintext, hodd]

theorem sum_single {q n m : Nat} (hn : 0 < n) (next p : ErrorPoly n) (G : ExactMatrix q n 1 m)
    (j : Fin 2) :
    (∑ o : Fin n, fun j ↦ reducePoly q n (Xe (n := n) (o : Nat)) •
      (reducePoly q n (1 * (if (o : Nat) = 0 then next else 0) * unitPair p j) • G)) j =
      reducePoly q n (next * unitPair p j) • G := by
  rw [Finset.sum_apply, Finset.sum_eq_single ⟨0, hn⟩]
  · simp [Xe]
  · intro o _ ho
    have : (o : Nat) ≠ 0 := fun h ↦ ho (Fin.ext h)
    simp [this]
  · intro h
    exact absurd (Finset.mem_univ _) h

theorem sum_single_value {n : Nat} (hn : 0 < n) (next : ErrorPoly n) :
    (1 : ErrorPoly n) * ∑ o : Fin n, Xe (n := n) (o : Nat) *
      (if (o : Nat) = 0 then next else 0) = next := by
  rw [one_mul, Finset.sum_eq_single ⟨0, hn⟩]
  · simp [Xe]
  · intro o _ ho
    have : (o : Nat) ≠ 0 := fun h ↦ ho (Fin.ext h)
    simp [this]
  · intro h
    exact absurd (Finset.mem_univ _) h

'''


def norm_items(T, wire, haves, items, seen):
    """Simp facts that unfold `wire` down to class-theorem atoms."""
    if wire in seen:
        return
    seen.add(wire)
    root = T.root
    node = root.node(wire)
    if node[0] == 'let':
        items.append(f'generatedRoot.{wire}')
        for x in agree.WIRE.findall(node[1]):
            norm_items(T, x, haves, items, seen)
    elif node[0] == 'atom':
        a = node[1]
        if a.kind == 'sliceMatrix':
            r = strip_parens(a.args[1])
            if r.isdigit() and strip_parens(a.args[2]) == str(int(r) + 1) and strip_parens(a.args[3]) == '0':
                xw = strip_parens(a.args[0])
                ty = root.s.wires[xw].type or ''
                mm = re.search(r'ExactMatrix \S+ \S+ (\d+) (\d+)', ty)
                meta.require(mm is not None, f'slice source type {xw}: {ty}')
                items.append(f'(sliceMatrix_row (⟨{r}, by decide⟩ : Fin {mm.group(1)}) {T.fact(a)})')
                norm_items(T, strip_parens(a.args[0]), haves, items, seen)
        elif a.kind == 'hashSample':
            items.append(f'MxxExponentLut.hashSample_eq {T.fact(a)}')
        elif a.kind in ('familyGetStatic', 'familyGetDynamic'):
            k = strip_parens(a.args[1])
            if k.isdigit():
                name = f'z{len(haves)}'
                haves.append(f'  have {name} := family_get {T.fact(a)} ⟨{k}, by omega⟩ rfl')
                items.append(name)
                norm_items(T, strip_parens(a.args[0]), haves, items, seen)
    elif node[0] == 'loop':
        a, path = node[1], node[2]
        if a.loop.kind == 'seq' and str(a.loop.count) == '0':
            init = agree.navigate(agree.parse_tuple(a.loop.init), path)
            init = strip_parens(init)
            name = f'z{len(haves)}'
            rhs = f'generatedRoot.{init} initial.context.online' if agree.WIRE.fullmatch(init) else init
            haves.append(f'  have {name} : generatedRoot.{wire} initial.context.online = {rhs} :=\n'
                         f'    congrArg (fun s ↦ s{path}) (zero_loop {T.fact(a)} rfl)')
            items.append(name)
            if agree.WIRE.fullmatch(init):
                norm_items(T, init, haves, items, seen)


SIMP_BASE = ['List.get_eq_getElem', 'List.getElem_cons_zero', 'List.getElem_cons_succ', 'Matrix.cons_val_zero', 'Matrix.cons_val_one', 'Matrix.head_cons', 'Fin.zero_eta',
             'Fin.mk_one', 'Fin.isValue', 'MxxRuntime.matrixMul', 'MxxRuntime.matrixAdd', 'zero_add']


def theorem_text(T, module, name):
    text = (T.geo.package / 'proof' / (module + '.lean')).read_text()
    m = re.search(r'^theorem ' + re.escape(name) + r' (.*?) := by', text, re.S | re.M)
    meta.require(m is not None, f'missing theorem {name}')
    return m.group(1)


def class_lemma(T, tp, cls, budget_defs):
    """The relation lemma of non-label target `tp` (pair index) and its plaintext/budget."""
    geo = T.geo
    n = geo.n
    c0, c1 = T.entry(T.fam_c, 2 * tp), T.entry(T.fam_c, 2 * tp + 1)
    a0 = [T.entry(T.fam_a0, 2 * tp), T.entry(T.fam_a0, 2 * tp + 1)]
    a1 = [T.entry(T.fam_a1, 2 * tp), T.entry(T.fam_a1, 2 * tp + 1)]
    kind = cls['kind']
    ctx = 'initial.context.online'
    G = T.gadget
    lines = [f'  have F := {ctx}_facts', f'  have hGG : generatedRoot.{T.class_gadget} {ctx} = '
             f'generatedRoot.{G} {ctx} :=', f'    gadgetMatrixRuns_unique {T.class_gadget_fact} {T.gadget_fact}']
    layout = 'Backend.layout0 Online0Claim.layout_some Online0Claim.layout_exact'
    entry = 'entry_of'
    comps = []
    closers = []
    if kind == 'anchor':
        meta.require(len({c0, c1, *a0, *a1}) == 1, 'anchor entries are not one zero wire')
        z = c0
        node = T.root.node(z)
        meta.require(node[0] == 'let' and node[1].startswith('(0 :'), 'anchor entry is not zero')
        x = 'fun _ ↦ 0'
        budget = 'b1_anchor'
        body = [f'  have F := {ctx}_facts', '  intro j', '  fin_cases j']
        for j in range(2):
            idx = 2 * tp + j
            body += [f'  · refine ⟨0, fun _ _ _ ↦ by simp, ?_⟩',
                     f'    rw [familyPack_entry {T.pack_fact(T.fam_c)} _ {idx} (by simp) rfl,',
                     f'      familyPack_entry {T.pack_fact(T.fam_a0)} _ {idx} (by simp) rfl,',
                     f'      familyPack_entry {T.pack_fact(T.fam_a1)} _ {idx} (by simp) rfl]',
                     f'    funext a b',
                     f'    simp [generatedRoot.{z}, maskPart, reduceMatrix]']
        return x, budget, body
    if kind == 'private':
        text = theorem_text(T, 'Dio2Online0Refresh', 'online0_target_priv')
        cw = re.findall(r'!\[generatedRoot\.(w_\d+_\d+) ctx, generatedRoot\.(w_\d+_\d+) ctx\] j =', text)
        meta.require(cw == [(c0, c1)], 'private target wires')
        lines += [f'  have hpriv := Online0Claim.{T.private_rel} hrun initial F.c{T.gadget72_frag}.h{T.class_gadget[2:]}',
                  f'  have row := online0_target_priv {ctx} F (by decide) {layout} hpriv']
        comps = ['row 0', 'row 1']
        closers = ['rw [hGG, one_mul]; rfl'] * 2
        x, budget = 'unitPair initial.payload', 'b1_private'
        seed = True
    elif kind == 'public':
        lines += ['  obtain ⟨eg, heg, hg⟩ := R.g', '  obtain ⟨e1, he1, hc1⟩ := R.one',
                  '  rw [← hGG] at hg hc1',
                  f'  have pub := online0_target_pub {ctx} F (by decide) {layout} heg he1 hg hc1']
        comps = ['pub.1', 'pub.2']
        closers = ['rw [hGG]; rfl', 'rw [hGG]; simp [unitPair]']
        x, budget, seed = 'unitPair next', 'b1_public', False
    elif kind == 'state':
        lines += ['  obtain ⟨es, hes, hcs⟩ := R.state',
                  f'  have st := online0_target_state {ctx} F (by decide) {layout}',
                  f'    (x := fun j ↦ reducePoly modulus_0 {n} (result * unitPair initial.payload j))',
                  '    (fun j ↦ hes j) (fun j ↦ by rw [hcs j]; rfl)']
        comps = ['st 0', 'st 1']
        closers = ['rw [hGG]; rfl'] * 2
        x, budget, seed = 'fun j ↦ result * unitPair initial.payload j', 'b1_stateTarget', False
    elif kind == 'sigma':
        lines += ['  obtain ⟨e1, he1, hc1⟩ := R.one', '  rw [← hGG] at hc1',
                  f'  have hσ := rel_target_congr R.sigma (t\' := ∑ o : Fin {n}, fun j ↦',
                  f'    reducePoly modulus_0 {n} (Xe (n := {n}) (o : Nat)) •',
                  f'      (reducePoly modulus_0 {n} (1 * (if (o : Nat) = 0 then next else 0) *',
                  f'        unitPair initial.payload j) • generatedRoot.{G} {ctx}))',
                  f'    (fun j ↦ (sum_single (by decide) next initial.payload _ j).symm)',
                  f'  have sg := online0_target_sigma {ctx} F (by decide) {layout} hσ he1 hc1',
                  f'  have hsum := sum_single_value (by decide : 0 < {n}) next']
        comps = ['rel_comp sg 0', 'rel_comp sg 1']
        closers = ['rw [hGG, hsum]; rfl'] * 2
        x, budget, seed = 'unitPair next', 'b1_sigma', False
    elif kind == 'Aut':
        k = cls['automorphism']
        odd = f'(⟨{(k - 1) // 2}, rfl⟩ : Odd {k})'
        name = f'online0_aut_{k}'
        rel = f'Online0Claim.imported_rel_helper_Aut_{k}_0'
        sw = f'Online0Sources.Preprocess.claim_aut{k}_switching'
        coeff = f'Online0Sources.Preprocess.aut{k}Coefficients'
        lines += ['  obtain ⟨e1, he1, hc1⟩ := R.one', '  rw [← hGG] at hc1',
                  f'  have hhelp0 := {rel} hrun initial F.c{T.gadget_frag}.h{G[2:]} 0',
                  f'  have hhelp : MxxExponentLut.Rel initial.mask (keyPairG model (generatedRoot.w_7_0 {ctx})',
                  f'      modulus_0 {geo.columns} Online0Helpers.helperPrefix [.integer {k}])',
                  f'      (fun j ↦ reducePoly modulus_0 {n} (((1 : Nat) : Int) *',
                  f'        unitPair (substitution {k} {odd} initial.mask) j) • generatedRoot.{G} {ctx})',
                  f'      ![{T.aut_rows[k][0]} {ctx} 0, {T.aut_rows[k][1]} {ctx} 0] 1 := by',
                  f'    rw [seed]',
                  f'    simpa [MxxIR.selectInt, autoPlaintext_nat {k} {odd}, keyPairG, rootKey] using hhelp0',
                  f'  obtain ⟨err, herr, hseq⟩ := {sw} initial F.c{T.gadget_frag}.h{G[2:]} 1 0',
                  f'  have hC : {coeff} 0 = 1 := by',
                  f'    have hidx : (sigmaOf Online0Sources.Preprocess.aut{k}Sigmas (0 : Fin 1)).toNat = {k} := by decide',
                  f'    simp only [{coeff}, PreprocessSwitch.coefficientCoefficient,',
                  f'      LutKey.keyTableG, LutKey.keyTermC, Finset.sum_range_one, hidx]',
                  f'    rw [ringAutomorphism_eq (by decide) (by decide) {k} {odd}]',
                  f'    simp [matrixMulScalarLeft, liftInteger, multiplyMonomial, matrixPolynomial, MxxIR.selectInt]',
                  f'  have res := {name} {ctx} F (by decide) {layout}',
                  f'    (sm := initial.mask) (sp := initial.payload) (μ := next) initial.mask_small',
                  f'    (by simpa using R.sigma) hhelp (Δ\' := 1) (C := 1)',
                  f'    (ν := substitution {k} {odd} next)',
                  f'    ⟨err, herr, by simpa [hC] using hseq⟩',
                  f'    ((polyNorm_substitution_le (by decide) {odd} (by decide) next).trans R.next_small)',
                  f'    (by simp) he1 hc1']
        comps = ['rel_comp res 0', 'rel_comp res 1']
        closers = ['rw [hGG, one_mul]; rfl'] * 2
        x, budget, seed = f'unitPair (substitution {k} {odd} next)', 'b1_automorphism', True
    else:
        raise meta.MetadataError(f'unsupported target class {kind}')
    if seed:
        lines.insert(3, f'  have seed : generatedRoot.w_7_0 {ctx} = {T.seed_rhs} := by')
        lines.insert(4, f'    change {ctx}.inputs{T.seed_path} = _')
        lines.insert(5, f'    rw [initial.context.online_inputs]; rfl')
    body = list(lines)
    body += ['  intro j', '  fin_cases j']
    for j in range(2):
        idx = 2 * tp + j
        haves, items, seen = [], [], set()
        for w in (a0[j], a1[j]):
            norm_items(T, w, haves, items, seen)
        if seed and kind == 'private':
            items.append('seed')
        ent = 'entry_of_one' if (kind == 'public' and j == 1) else 'entry_of'
        body += [f'  · obtain ⟨e, he, h⟩ := {comps[j]}']
        body += ['  ' + hv for hv in haves]
        body += [f'    rw [familyPack_entry {T.pack_fact(T.fam_c)} _ {idx} (by simp) rfl]',
                 f'    refine {ent} h he ?_ ?_ (by norm_num [{", ".join(budget_defs)}])',
                 f'    · rw [familyPack_entry {T.pack_fact(T.fam_a0)} _ {idx} (by simp) rfl,',
                 f'        familyPack_entry {T.pack_fact(T.fam_a1)} _ {idx} (by simp) rfl]',
                 '      funext k',
                 f'      fin_cases k <;> (try simp only [{", ".join(SIMP_BASE + items)}]) <;> (try rfl)',
                 f'    · {closers[j]}']
    return x, budget, body


def setup_classes(T):
    geo = T.geo
    T.gadget = geo_gadget = None
    pin, pargs = geo.part_inputs, geo.part_args
    fivers = [a for a in pargs if a.endswith('_5')]
    T.gadget = pargs[pargs.index(fivers[0]) + 1]
    ga = T.root.node(T.gadget)
    meta.require(ga[0] == 'atom' and ga[1].kind == 'gadgetMatrixRuns', 'part gadget')
    T.gadget_fact = T.fact(ga[1])
    T.gadget_frag = ga[1].frag
    priv = theorem_text(T, 'Dio2Online0Refresh', 'online0_target_priv')
    m = re.search(r'unitPair tp j\) • generatedRoot\.(w_\d+_\d+) ctx\)', priv)
    meta.require(m is not None, 'class gadget')
    T.class_gadget = m.group(1)
    cg = T.root.node(T.class_gadget)
    meta.require(cg[0] == 'atom' and cg[1].kind == 'gadgetMatrixRuns', 'class gadget fact')
    T.class_gadget_fact = T.fact(cg[1])
    T.gadget72_frag = cg[1].frag
    inputs = (geo.package / 'proof' / 'Dio2Online0Inputs.lean').read_text()
    meta.require('theorem imported_rel_row_dio2_one_private_0 ' in inputs, 'private import relation')
    T.private_rel = 'imported_rel_row_dio2_one_private_0'
    m = re.search(r'theorem imported_rel_row_dio2_one_private_0 .*?rootKey model (\(Claim0\.initialInputs external execution\)[.\d]*)',
                  inputs, re.S)
    meta.require(m is not None, 'seed projection')
    T.seed_rhs = m.group(1)
    helper = (geo.package / 'proof' / 'Dio2Online0Helper.lean').read_text()
    T.aut_rows = {}
    seed_wire = None
    for c in geo.view['target_classes']:
        if c['kind'] != 'Aut':
            continue
        k = c['automorphism']
        text = theorem_text(T, 'Dio2Online0Helper', f'online0_aut_{k}')
        m = re.search(r'keyPairG hashModel \(generatedRoot\.(w_\d+_\d+) ctx\).*?!\[generatedRoot\.(w_\d+_\d+) ctx 0, '
                      r'generatedRoot\.(w_\d+_\d+) ctx 0\] Bh', text, re.S)
        meta.require(m is not None, f'aut {k} helper rows')
        seed_wire = m.group(1)
        T.aut_rows[k] = (f'generatedRoot.{m.group(2)}', f'generatedRoot.{m.group(3)}')
    meta.require(seed_wire == 'w_7_0', 'label seed wire is not w_7_0')
    node = T.root.node(seed_wire)
    meta.require(node[0] == 'input', 'seed is not an input')
    T.seed_path = node[1]


def nonlabel_source(T, only=None):
    """NonLabelRow, one lemma per exported non-label target, and their table."""
    n = T.geo.n
    budget_defs = ['b1_private', 'b1_public', 'b1_stateTarget', 'b1_sigma', 'b1_automorphism',
                   'b1_anchor', 'b1_g', 'b1_one', 'b1_state', 'b1_packed', 'b1_helper']
    src = f'''variable {{model : HashModel}} {{external : GeneratedClaim.ExternalInputs}}
  {{execution : GeneratedClaim.Execution}} {{value : ErrorPoly {n}}}

/-- The packed non-label pair `t` encodes `x` with error at most `B` in both components. -/
def NonLabelRow (initial : State0Witness model external execution value) (t : Nat)
    (ht : t < {T.nlt}) (x : Fin 2 → ErrorPoly {n}) (B : Nat) : Prop :=
  ∀ j : Fin 2, EntryRow (reducePoly modulus_0 {n} initial.mask)
    (generatedRoot.{T.fam_c} initial.context.online (pairIndex t ht j))
    (generatedRoot.{T.fam_a0} initial.context.online (pairIndex t ht j))
    (generatedRoot.{T.fam_a1} initial.context.online (pairIndex t ht j))
    (generatedRoot.{T.gadget} initial.context.online) (x j) B

'''
    T.nl_info = []
    for tp, (t, cls) in enumerate(T.nonlabel):
        if only is not None and str(tp) not in only.split(','):
            continue
        x, budget, body = class_lemma(T, tp, cls, budget_defs)
        T.nl_info.append((tp, t, cls, x, budget))
        src += f'''/-- Exported target {t} ({cls['kind']}{' ' + str(cls.get('automorphism')) if cls['kind'] == 'Aut' else ''}). -/
theorem target_{tp} (hrun : W_GeneratedClaim.Runs model external execution) {{state : Nat}}
    (initial : State0Witness model external execution (Xe (n := {n}) (2 * state)))
    {{anchor : ExactMatrix modulus_0 {n} 2 {T.geo.anchor}}} {{next result : ErrorPoly {n}}}
    {{fresh : Fin {T.geo.fresh_cells} → ErrorPoly {n}}} {{smudge : Fin {T.geo.smudge_cells} → ErrorPoly {n}}}
    (R : Online0Sources.Relations initial.context.online anchor initial.mask initial.payload
      next result fresh smudge) :
    NonLabelRow initial {tp} (by decide) ({x}) {budget} := by
''' + '\n'.join(body) + '\n\n'
    return src


# --------------------------------------------------------------------------- label forest member

def claim_tuple(geo, name):
    """The term tree of a Claim0 input tuple definition (right-nested or balanced)."""
    s = (geo.package / 'proof' / 'Dio2State0.lean').read_text()
    i = s.index(f'def {name}')
    body = s[s.index(':=', i) + 2:s.index('\n\n', i)].strip()
    return agree.parse_tuple(body)


def claim_input_term(geo, name, path):
    """The external/execution term at a projection path of a Claim0 input tuple."""
    tree = claim_tuple(geo, name)
    leaf = agree.navigate(tree, path)
    meta.require(isinstance(leaf, str), f'{name}{path}: not a leaf')
    return strip_parens(leaf)


def requirement_equal(geo, a, b):
    """The requirement loop proving two external families equal, as (field, swapped)."""
    run = (geo.package / 'proof' / 'Dio2RunFacts.lean').read_text()
    m = re.search(r'theorem requirement \(hrun[^\n]*\n\s*Dio2Requirement\.Facts ([^\n]*?) :=', run)
    meta.require(m is not None, 'requirement facts statement')
    args = m.group(1).split()
    req = (geo.package / 'proof' / 'Dio2Requirement.lean').read_text()
    for f, x, y in re.findall(r'\n  (loop_\d+) : ∀ n\d+ < \d+, \(famAt (x\d+) n\d+\) = \(famAt (x\d+) n\d+\)\n', req):
        pa, pb = args[int(x[1:])], args[int(y[1:])]
        if (pa, pb) == (a, b):
            return f, False
        if (pa, pb) == (b, a):
            return f, True
    raise meta.MetadataError(f'no requirement loop equates {a} and {b}')


def forest_premises(T):
    src = (T.geo.package / 'proof' / 'Dio2Step0Forest.lean').read_text()
    i = src.index('theorem online0_label_target_from_roots')
    j = src.index(':= by', i)
    stmt = src[i:j]
    f = re.escape(T.forest_scope)
    unit = re.search(r'\(unit : Online0Labels\.Node sm \(' + f + r'\.(w_\d+_\d+) ctx\) 1 (.*?)\)\n', stmt)
    meta.require(unit is not None, 'forest unit premise')
    gadget = unit.group(1)
    unit_wires = re.findall(f + r'\.(w_\d+_\d+) ctx', unit.group(2))
    meta.require(len(unit_wires) == 7, 'forest unit node wires')
    lab = re.search(r'\(labels : ∀ member : Fin (\d+),\n(.*?)\)\n    \(hs : ', stmt, re.S)
    meta.require(lab is not None, 'forest label premise')
    hs = re.search(r'\(hs : Rel sm (.*?) b1_public\)', stmt, re.S)
    meta.require(hs is not None, 'forest public premise')
    return gadget, unit_wires, int(lab.group(1)), lab.group(2), hs.group(1)


def label_member_source(T, PF):
    geo = T.geo
    n = geo.n
    fname = T.forest_scope
    R = geo.view['labels']['root_members']
    C = geo.view['labels']['target_columns']
    members = T.labels
    ctx = 'initial.context.online'
    gadget, unit_wires, label_count, label_text, hs_text = forest_premises(T)
    fscope = T.gen.scA[fname]
    def root_of(w):
        wi = fscope.wires[w]
        meta.require(wi.src == 'input', f'{fname}.{w}: not a forest input')
        return T.forest_args[wi.term.count('.2')].split()[0]
    # unit node: rows read the constant label-0 root, keys are its hashes, payload the constant.
    unit_root = [root_of(w) for w in unit_wires]
    lines = []
    for k, rw in enumerate(unit_root):
        node = T.root.node(rw)
        if k < 2:
            meta.require(node[0] == 'atom' and node[1].kind == 'familyGetStatic', f'unit row {rw}')
            meta.require(meta.integer(node[1].args[1]) == R - 1, 'unit row is the constant member')
            lines.append(f'(family_get {T.fact(node[1])} ⟨{R - 1}, by decide⟩ rfl).symm')
        elif k < 6:
            meta.require(node[0] == 'atom' and node[1].kind == 'hashSample', f'unit key {rw}')
            lines.append(f'''    · show _ = generatedRoot.{rw} {ctx}
      rw [MxxExponentLut.hashSample_eq {T.fact(node[1])}, seed]; rfl''')
        else:
            lines.append(f'''    · show _ = generatedRoot.{rw} {ctx}
      funext a b
      fin_cases a; fin_cases b
      simp [initial_label0_value, message, MxxIR.selectInt, generatedRoot.{rw}]''')
    unit_payload_root = unit_root[6]
    # label nodes: seed and exponent family of the forest.
    lab_wires = re.findall(re.escape(fname) + r'\.(w_\d+_\d+) ctx', label_text)
    lab_seed = re.search(r'rootKey hashModel \(' + re.escape(fname) + r'\.(w_\d+_\d+) ctx\)', label_text).group(1)
    lab_exps = re.search(r'intFamilyValue \(\(' + re.escape(fname) + r'\.(w_\d+_\d+) ctx\)\)', label_text).group(1)
    seed_root, exps_root = root_of(lab_seed), root_of(lab_exps)
    meta.require(seed_root == 'w_7_0', 'forest label seed is the online seed')
    labels_src = (geo.package / 'proof' / 'Dio2Step0Labels.lean').read_text()
    em = re.search(r'noncomputable def initial_label_value[^\n]*\n[^\n]*\n  \(message 1 \(\(\(intFamilyValue \(\(\(\((Claim0\.initialInputs external execution\)[.\d]*)\)\)\)', labels_src)
    meta.require(em is not None, 'initial label exponent family')
    exps_rhs = '(' + em.group(1)
    enode = T.root.node(exps_root)
    meta.require(enode[0] == 'input', 'forest exponent family is a root input')
    online_ext = claim_input_term(geo, 'onlineInputs', enode[1])
    initial_ext = claim_input_term(geo, 'initialInputs', exps_rhs[len('(Claim0.initialInputs external execution)'):])
    field, swapped = requirement_equal(geo, online_ext, initial_ext)
    hrel = 'h.symm' if swapped else 'h'
    text = label_text.replace(' ctx', ' fctx').replace('hashModel', 'model').replace('Online0Labels.Node sm', 'Online0Labels.Node initial.mask')
    text = text.replace('∀ member', '∀ m').replace('member.val', 'm.val').replace(') member)', ') m)')
    text = re.sub(r'\(\(' + re.escape(fname) + r'\.(w_\d+_\d+) fctx\) member\)', r'((' + fname + r'.\1 fctx) m)', text)
    hs_ = hs_text.replace(' ctx', ' fctx').replace('\n      ts ', '\n      (fun j ↦ reducePoly modulus_0 ' + str(n) + ' (unitPair next j) • generatedRoot.' + T.gadget + ' ' + ctx + ')\n      ')
    pub_tp = 1
    e0, e1 = 2 * pub_tp, 2 * pub_tp + 1
    fam_facts = [T.pack_fact(f) for f in (T.fam_c, T.fam_a0, T.fam_a1)]
    rw0 = ', '.join(f'familyPack_entry {x} _ {e0} (by simp) rfl' for x in fam_facts)
    rw1 = ', '.join(f'familyPack_entry {x} _ {e1} (by simp) rfl' for x in fam_facts)
    fargs = ', '.join(f'(generatedRoot.{a.split()[0]} {ctx})' for a in T.forest_args)
    ftuple = '(' + ', ('.join(f'(generatedRoot.{a.split()[0]} {ctx})' for a in T.forest_args) + ', ()' + ')' * len(T.forest_args)
    outs = T.forest_outs
    col, sha_fact, croot, broot = PF.columns
    gnode = T.root.node(root_of(gadget))
    meta.require(gnode[0] == 'atom' and gnode[1].kind == 'gadgetMatrixRuns', 'forest gadget fact')
    gfact = T.fact(gnode[1])
    return f'''theorem node_eq {{q n m : Nat}} {{sm : ErrorPoly n}} {{G : ExactMatrix q n 1 m}} {{B : Nat}}
    {{c0 c1 k00 k01 k10 k11 c0' c1' k00' k01' k10' k11' : ExactMatrix q n 1 m}}
    {{payload payload' : ExactMatrix q n 1 1}}
    (h : Online0Labels.Node sm G B c0 c1 k00 k01 k10 k11 payload)
    (e0 : c0 = c0') (e1 : c1 = c1') (e2 : k00 = k00') (e3 : k01 = k01') (e4 : k10 = k10')
    (e5 : k11 = k11') (e6 : payload = payload') :
    Online0Labels.Node sm G B c0' c1' k00' k01' k10' k11' payload' := by
  subst e0 e1 e2 e3 e4 e5 e6; exact h

/-- The label target of an arbitrary forest member, through the actual forest;
non-constant label-0 members carry exactly the next external label. -/
theorem target_label_member (hrun : W_GeneratedClaim.Runs model external execution) {{state : Nat}}
    (initial : State0Witness model external execution (Xe (n := {n}) (2 * state)))
    {{next : ErrorPoly {n}}}
    (hpub : NonLabelRow initial {pub_tp} (by decide) (unitPair next) b1_public) (member : Fin {members}) :
    ∃ α : ErrorPoly {n},
      (∀ m : Fin {R}, m.val ≠ {R - 1} → member.val = m.val * {C} →
        reducePoly modulus_0 {n} α = reducePoly modulus_0 {n} (online1Labels external m)) ∧
      SmulGain α 1 ∧ ∀ j : Fin 2,
      EntryRow (reducePoly modulus_0 {n} initial.mask)
        (![generatedRoot.{outs[0]} {ctx}, generatedRoot.{outs[1]} {ctx}] j member)
        (![generatedRoot.{outs[2]} {ctx}, generatedRoot.{outs[3]} {ctx}] j member)
        (![generatedRoot.{outs[4]} {ctx}, generatedRoot.{outs[5]} {ctx}] j member)
        (generatedRoot.{T.gadget} {ctx}) (α * unitPair next j) b1_label := by
  have F := {ctx}_facts
  have seed : generatedRoot.w_7_0 {ctx} = {T.seed_rhs} := by
    change {ctx}.inputs{T.seed_path} = _
    rw [initial.context.online_inputs]; rfl
  obtain ⟨wit, body⟩ := {T.fact(T.forest_atom)} member
  let fctx : {fname}.Ctx Backend.backend model W_GeneratedClaim.stage_11_params member.val :=
    ⟨{ftuple}, wit⟩
  have FF := {fname}.facts fctx body
  have hG : gadgetMatrixRuns Backend.backend {geo.primes[0]['base']} {geo.primes[0]['digits']} ({fname}.{gadget} fctx) := {gfact}
  have unit0 := Dio2Proof.online0_initial_label0_nodes hrun initial {gfact} {R - 1}
  have unit : Online0Labels.Node initial.mask ({fname}.{gadget} fctx) 1
      {' '.join(f'({fname}.{w} fctx)' for w in unit_wires)} := by
    refine node_eq unit0 {lines[0]}
      {lines[1]} ?_ ?_ ?_ ?_ ?_
{chr(10).join(lines[2:])}
  have exps : generatedRoot.{exps_root} {ctx} = {exps_rhs} := by
    rw [generatedRoot.{exps_root}, initial.context.online_inputs]
    funext i
    have h := (Dio2Proof.Complete.requirement hrun).{field} i.val i.isLt
    rw [RefreshTopProof.GenIface.famAt_of_lt _ i.isLt, RefreshTopProof.GenIface.famAt_of_lt _ i.isLt] at h
    exact {hrel}
  have labels : ∀ m : Fin {label_count},
{text} := by
    intro m
    have h := Dio2Proof.online0_initial_label_nodes hrun initial {gfact} m
    have k25 : {fname}.{lab_seed} fctx = {T.seed_rhs} := seed
    have k30 : {fname}.{lab_exps} fctx = {exps_rhs} := exps
    rw [k25, k30]
    simpa [initial_label_value, initial_label_keys] using h
  have hs : MxxExponentLut.Rel initial.mask {hs_} b1_public := by
    have h0 := hpub 0
    have h1 := hpub 1
    rw [{rw0}] at h0
    rw [{rw1}] at h1
    exact rel_of_entries h0 h1
  obtain ⟨α, hpα, hgain, hrel⟩ := online0_label_target_from_roots member fctx FF (by decide) (by decide)
    Backend.layout0 Online0Claim.layout_some Online0Claim.layout_exact hG unit labels hs
  refine ⟨α, ?_, hgain, fun j ↦ ?_⟩
  · intro m hm hmember
    have unitP : {fname}.{unit_wires[6]} fctx = Step0Payload.P 0 := by
      show generatedRoot.{unit_payload_root} {ctx} = _
      funext a b
      fin_cases a; fin_cases b
      simp [Step0Payload.P, message, generatedRoot.{unit_payload_root}]
    have hcount := fun i ↦ ({sha_fact}.2.2.2.2.2 i).2
    have hf := Step0Payload.forest_payload _ fctx FF unitP (Step0Payload.count_le fctx FF hcount)
    have hl := Step0Payload.member_payload hrun m hm member hmember fctx FF hcount
      (fun i ↦ ({sha_fact}.2.2.2.2.2 i).1) exps
    exact congrFun (congrFun (hpα.symm.trans (hf.trans hl)) 0) 0
  obtain ⟨e, he, h⟩ := rel_comp hrel j
  refine ⟨e, he, ?_⟩
  fin_cases j
  · refine h.trans ?_
    simp only [smul_smul, ← map_mul]; rfl
  · refine h.trans ?_
    simp only [smul_smul, ← map_mul]; rfl

'''


# --------------------------------------------------------------------------- assembly

def assembly_source(T):
    geo = T.geo
    n, parts, nlt = geo.n, geo.parts, T.nlt
    ls, L = T.label_start, T.labels
    le = ls + L
    R = geo.view['labels']['root_members']
    C = geo.view['labels']['target_columns']
    meta.require(parts == 2 * geo.targets and geo.targets == nlt + L, 'target counts')
    arms = []
    for tp, t, cls, x, budget in T.nl_info:
        v = x.replace('initial.payload', 'payload')
        arms.append(f'  | {tp} => {v}')
    rows_arms = '\n'.join(f'  | {tp}, _ => target_{tp} hrun initial R' for tp, *_ in T.nl_info)
    sel = f'(if part.val / 2 < {ls} then part.val / 2 else part.val / 2 - {L})'
    hdr = f'''(hrun : W_GeneratedClaim.Runs model external execution) {{state : Nat}}
    (initial : State0Witness model external execution (Xe (n := {n}) (2 * state)))
    {{anchor : ExactMatrix modulus_0 {n} 2 {geo.anchor}}} {{next result : ErrorPoly {n}}}
    {{fresh : Fin {geo.fresh_cells} → ErrorPoly {n}}} {{smudge : Fin {geo.smudge_cells} → ErrorPoly {n}}}
    (R : Online0Sources.Relations initial.context.online anchor initial.mask initial.payload
      next result fresh smudge)'''
    return f'''/-- Plaintexts of the exported non-label target pairs. -/
noncomputable def nonlabelValue (payload next result : ErrorPoly {n}) : Nat → Fin 2 → ErrorPoly {n}
{chr(10).join(arms)}
  | _ => fun _ ↦ 0

/-- The exported target of non-label pair `t`. -/
def nonlabelTarget (t : Nat) : Nat := if t < {ls} then t else t + {L}

/-- Plaintext of every refresh part: label parts carry the forest plaintext. -/
noncomputable def stepTarget (payload next result : ErrorPoly {n}) (label : Fin {L} → ErrorPoly {n})
    (part : Fin {parts}) : ErrorPoly {n} :=
  if h : {ls} ≤ part.val / 2 ∧ part.val / 2 < {le} then
    label ⟨part.val / 2 - {ls}, by omega⟩ * unitPair next ⟨part.val % 2, Nat.mod_lt _ (by decide)⟩
  else
    nonlabelValue payload next result {sel}
      ⟨part.val % 2, Nat.mod_lt _ (by decide)⟩

theorem targetBudget_pair (part : Fin {parts}) :
    Online0Assembly.targetBudget part =
      Online0Assembly.targetBudget ⟨2 * (part.val / 2), by omega⟩ := by
  unfold Online0Assembly.targetBudget
  simp only
  split_ifs <;> omega

/-- Every exported non-label target, at the certified budget of its parts. -/
theorem nonlabel_rows {hdr} :
    ∀ (t : Nat) (ht : t < {nlt}), NonLabelRow initial t ht (nonlabelValue initial.payload next result t)
      (Online0Assembly.targetBudget ⟨2 * nonlabelTarget t, by unfold nonlabelTarget; split_ifs <;> omega⟩)
{rows_arms}
  | _ + {nlt}, ht => absurd ht (by omega)

/-- The refresh target premise of `Step0Rows.rows`, for every part, with the
actual forest plaintexts; label members bridge to the next external labels. -/
theorem target_rows {hdr} :
    ∃ label : Fin {L} → ErrorPoly {n},
      (∀ (m : Fin {R}) (member : Fin {L}), m.val ≠ {R - 1} → member.val = m.val * {C} →
        reducePoly modulus_0 {n} (label member) = reducePoly modulus_0 {n} (online1Labels external m)) ∧
      Step0Rows.TargetRows initial.context.online initial.mask
        (stepTarget initial.payload next result label) := by
  have F := initial.context.online_facts
  choose label hbridge hgain hrow using
    fun member : Fin {L} ↦ target_label_member hrun initial (target_1 hrun initial R) member
  refine ⟨label, fun m member hm hmember ↦ hbridge member m hm hmember, fun part ↦ ?_⟩
  have hj : part.val % 2 < 2 := Nat.mod_lt _ (by decide)
  by_cases hl : {ls} ≤ part.val / 2 ∧ part.val / 2 < {le}
  · obtain ⟨h0, h1, h2⟩ := target_label_entries initial.context.online F part
      ⟨part.val / 2 - {ls}, by omega⟩ ⟨part.val % 2, hj⟩ (by simp only; omega) rfl
    have row := hrow ⟨part.val / 2 - {ls}, by omega⟩ ⟨part.val % 2, hj⟩
    rw [← h0, ← h1, ← h2] at row
    have hb : Online0Assembly.targetBudget part = b1_label := by
      unfold Online0Assembly.targetBudget
      split_ifs <;> first | rfl | omega
    rw [hb]
    unfold stepTarget
    rw [dif_pos hl]
    exact row
  · have ht : {sel} < {nlt} := by
      split_ifs <;> omega
    obtain ⟨h0, h1, h2⟩ := target_nonlabel_entries initial.context.online F part _ ht
      ⟨part.val % 2, hj⟩ (by split_ifs <;> omega) rfl
    have row := nonlabel_rows hrun initial R _ ht ⟨part.val % 2, hj⟩
    rw [← h0, ← h1, ← h2] at row
    have hb : Online0Assembly.targetBudget part =
        Online0Assembly.targetBudget ⟨2 * nonlabelTarget
          {sel}, by
            unfold nonlabelTarget; split_ifs <;> omega⟩ := by
      rw [targetBudget_pair]
      congr 2
      unfold nonlabelTarget
      split_ifs <;> omega
    have hv : stepTarget initial.payload next result label part =
        nonlabelValue initial.payload next result
          {sel} ⟨part.val % 2, hj⟩ :=
      dif_neg hl
    show EntryRow _ _ _ _ _ _ _
    rw [hb, hv]
    exact row

'''


TARGETS_HEADER = '''import Dio2RunFacts
import Dio2Step0Payload
import Dio2Step0Rows
import Dio2Online0Refresh
import Dio2Online0Helper
import Dio2Online0SourcesTables
import Dio2Step0Forest
import Dio2Step0Labels

/-!
# Step-zero refresh targets

Generated by `gen_s0o.py` (module `s0o_targets.py`) from checked export
metadata. Every refresh part reads its target ciphertext and keys from the
actual target-selection loop; each exported non-label target circuit is
related to its actual source relation, and the label targets come from the
symbolic label forest for an arbitrary member. No part, member or
coefficient is enumerated, and no noise inequality is assumed.
-/

set_option maxRecDepth 16384
set_option maxHeartbeats 16000000
set_option Elab.async false

namespace Dio2Proof
open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh
  RefreshTopProof RefreshTopProof.Gen RefreshTopProof.GenIface RefreshTopProof.GenInst
open W_Stage_online_0 Dio2NoiseCertificate

namespace Step0Targets

'''


def targets_source(T, PF):
    setup_classes(T)
    nonlabel = nonlabel_source(T)
    i = nonlabel.index('variable {model : HashModel}')
    text = (TARGETS_HEADER + PRELUDE + PRELUDE2 + pair_index_source(T) + selection_source(T) +
            nonlabel + label_member_source(T, PF) + assembly_source(T) +
            'end Step0Targets\n\nend Dio2Proof\n')
    return text
