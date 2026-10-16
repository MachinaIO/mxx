"""Step-zero label-forest payloads for gen_s0o.py (module Dio2Step0Payload).

The exported label forest multiplies constant `1 x 1` payloads: every leaf
carries `X^(s e)` for its selected exponent `e` (or the unit `X^0` when it is
inactive), and every merge multiplies a left node by its right node unless the
right node lies past the active count.  This module emits

* the generic payload algebra (`P`, products, the zero tail and the periodic
  reduction modulo the ring dimension);
* one payload lemma for the leaf scope and one per merge scope, read from the
  exported constraints (dynamic reads, selections and index arithmetic) with
  every member quantified;
* `forest_payload`: the root payload of an arbitrary forest member is the
  monomial of the sum of its active leaf exponents, with the level counts
  bounded from the exported count arithmetic;
* `member_payload`: for a non-constant next label member, that sum is the
  requirement's own next label exponent, so the forest payload equals the
  next external label (`online1Labels`).

No leaf, member, level count or coefficient is enumerated.
"""
import re

import dio2_meta as meta
import s0o_agree as agree
from xmodel import strip_parens

WIRE = re.compile(r'w_\d+_\d+')


def lets(scope):
    return {name: wire.term for name, wire in scope.wires.items() if wire.src == 'let'}


class Forest:
    """The exported label forest of online zero and its levels."""

    def __init__(self, geo, gen, module, T):
        import gen_adapters as adapters
        self.adapters = adapters
        self.geo, self.gen, self.T = geo, gen, T
        view = geo.view
        entry = meta.one((e for e in view['label_forest_scopes']['online_0'] if e['children']),
                         'label forest')
        meta.require(entry['scope'] == T.forest_scope, 'forest scope differs from the target loop')
        self.name = entry['scope']
        self.members = entry['count']
        self.scope = module.scopes[self.name]
        self.info = gen.info('A', self.name)
        counts = {e['scope']: e['count'] for e in view['label_forest_scopes']['online_0']}
        children = entry['children']
        self.leaf = module.scopes[children[0]]
        self.merges = [module.scopes[c] for c in children[1:]]
        self.sizes = [counts[c] for c in children]
        for a, b in zip(self.sizes, self.sizes[1:]):
            meta.require(b == (a + 1) // 2, 'forest levels must halve (ceiling)')
        meta.require(self.sizes[-1] == 1, 'forest must end in one root')
        loops = {a.loop.child: a for a in self.scope.loops}
        self.calls = []
        for c in children:
            atom = loops[c]
            meta.require(atom.loop.kind == 'par', f'{c}: forest level is not a parallel loop')
            outs = WIRE.findall(' '.join(atom.loop.outs))
            meta.require(len(outs) == 7, f'{c}: forest level must output seven families')
            self.calls.append((atom, outs))
        self.count_wires = [strip_parens(a.loop.args[0]) for a, _ in self.calls]
        # The leaf count is the active count of the whole forest.
        self.cnt = self.count_wires[0]
        n = geo.n
        self.n = n
        self.levels = len(self.merges)
        self.leaves = self.sizes[0]
        self.width = 2 ** self.levels
        meta.require(self.width >= self.leaves, 'forest width')

    def fact(self, wire):
        node = self.info.node(wire)
        meta.require(node[0] in ('atom', 'loop'), f'{self.name}.{wire}: no fact')
        atom = node[1]
        return f'F.c{atom.frag}.{atom.field}'


def payload_prelude(n):
    return rf'''/-- The constant payload `X^e`. -/
noncomputable def P (e : Int) : ExactMatrix modulus_0 {n} 1 1 :=
  fun _ _ ↦ reducePoly modulus_0 {n} (message 1 e)

theorem message_one_add {{n : Nat}} (hn : 0 < n) (a b : Int) :
    message (n := n) 1 a * message 1 b = message 1 (a + b) := by
  have hroot : AdjoinRoot.root (negacyclicModulus n Int) ^ (2 * n) = 1 := by
    rw [pow_mul', Negacyclic.root_pow_n]; simp
  simp only [message, Int.cast_one, one_mul, ← pow_add]
  have hM : (0 : Int) < ((2 * n : Nat) : Int) := by omega
  rw [pow_eq_pow_mod _ hroot, pow_eq_pow_mod (((a + b) % ((2 * n : Nat) : Int)).toNat) hroot]
  congr 1
  have hM2 : ((2 * n : Nat) : Int) = 2 * (n : Int) := by push_cast; ring
  rw [hM2]
  have hM' : (0 : Int) < 2 * (n : Int) := by omega
  have ha := Int.emod_nonneg a (ne_of_gt hM')
  have hb := Int.emod_nonneg b (ne_of_gt hM')
  have hab := Int.emod_nonneg (a + b) (ne_of_gt hM')
  apply Int.ofNat.inj
  simp only [Int.ofNat_eq_natCast]
  push_cast
  rw [Int.toNat_of_nonneg ha, Int.toNat_of_nonneg hb, Int.toNat_of_nonneg hab,
    Int.emod_emod_of_dvd _ (dvd_refl _), ← Int.add_emod]

theorem P_mul (a b : Int) : matrixMulScalarLeft (P a) (P b) = P (a + b) := by
  unfold P
  rw [payload_mul rfl rfl, message_one_add (by decide)]

/-- A binary forest merge in payloads. The right node is read only when it is
active; a dropped right subtree lies past the active leaf count. -/
theorem merge_core {{t : Nat → Int}} {{cnt count : Int}} {{w i : Nat}}
    (ht : ∀ k : Nat, cnt ≤ k → t k = 0) (hc : cnt ≤ count * w)
    {{left right out : ExactMatrix modulus_0 {n} 1 1}}
    (hl : left = P (∑ r ∈ Finset.range w, t (2 * i * w + r)))
    (hr : (i : Int) * 2 + 1 < count → right = P (∑ r ∈ Finset.range w, t ((2 * i + 1) * w + r)))
    (hout : out = if ((i : Int) * 2 + 1 < count) then matrixMulScalarLeft left right else left) :
    out = P (∑ r ∈ Finset.range (2 * w), t (i * (2 * w) + r)) := by
  have hsplit : ∑ r ∈ Finset.range (2 * w), t (i * (2 * w) + r) =
      ∑ r ∈ Finset.range w, t (2 * i * w + r) + ∑ r ∈ Finset.range w, t ((2 * i + 1) * w + r) := by
    rw [two_mul w, Finset.sum_range_add]
    congr 1
    · exact Finset.sum_congr rfl fun r _ ↦ by congr 1; ring
    · exact Finset.sum_congr rfl fun r _ ↦ by congr 1; ring
  rw [hsplit, hout]
  split_ifs with hact
  · rw [hl, hr hact, P_mul]
  · have hz : ∑ r ∈ Finset.range w, t ((2 * i + 1) * w + r) = 0 := by
      refine Finset.sum_eq_zero fun r _ ↦ ht _ ?_
      have h1 : count * (w : Int) ≤ ((i : Int) * 2 + 1) * w :=
        Int.mul_le_mul_of_nonneg_right (by omega) (by omega)
      push_cast
      nlinarith
    rw [hz, add_zero, hl]

/-- Leaves past the active count contribute nothing. -/
theorem sum_tail {{t : Nat → Int}} {{a b : Nat}} (ht : ∀ k : Nat, a ≤ k → t k = 0) (hab : a ≤ b) :
    ∑ r ∈ Finset.range b, t r = ∑ r ∈ Finset.range a, t r := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hab
  have hz : ∑ x ∈ Finset.range d, t (a + x) = 0 :=
    Finset.sum_eq_zero fun r _ ↦ ht (a + r) (by omega)
  rw [Finset.sum_range_add, hz, add_zero]

/-- Every selector count is at most the number of hashed rows. -/
theorem column_count_le (seed : Blob) (rows columns indexBits selected : Nat) :
    (Sha256Columns.column seed rows columns indexBits selected).2 ≤ rows := by
  unfold Sha256Columns.column
  have key : ∀ (l : List Nat) (acc : Nat × Nat),
      (l.foldl (fun result row ↦
        if Sha256Columns.bit seed (row * columns + selected) = true then
          (result.1 + row * 2 ^ (result.2 * indexBits), result.2 + 1)
        else result) acc).2 ≤ acc.2 + l.length := by
    intro l
    induction l with
    | nil => intro acc; simp
    | cons x xs ih =>
      intro acc
      simp only [List.foldl_cons, List.length_cons]
      split_ifs
      · exact (ih _).trans (by simp only; omega)
      · exact (ih _).trans (by omega)
  simpa using key (List.range rows) (0, 0)

'''


def leaf_source(F):
    ad = F.adapters
    leaf = F.leaf
    binders, call, _ = ad.header(leaf)
    per, flat = ad.atom_names(leaf)
    per = ad.use_rfl_out(per, flat)
    text = dict(flat)
    outs = ad.first_out(flat)[1]
    out_items = [strip_parens(x) for x in meta.xmodel.split_top(
        strip_parens(outs[len('outputs = '):]), ',')]
    flat_outs = WIRE.findall(outs)
    meta.require(len(flat_outs) == 7, 'leaf must output seven values')
    pay = flat_outs[6]
    sel = meta.one((n for n, t in flat if t.startswith('MxxRuntime.select ') and t.endswith(' ' + pay)),
                   'leaf payload selection')
    m = re.match(r'MxxRuntime\.select (w_\d+_\d+) \[(w_\d+_\d+), (w_\d+_\d+)\] ' + pay + '$', text[sel])
    meta.require(m is not None, 'leaf payload selection shape')
    flag, unit, mono = m.groups()
    meta.require(leaf.wires[unit].src == 'input', 'leaf unit payload is not an input')
    unit_path = leaf.wires[unit].term
    mterm = leaf.wires[mono].term
    mm = re.fullmatch(r'MxxRuntime\.multiplyMonomial (w_\d+_\d+) (w_\d+_\d+)', mterm)
    meta.require(mm is not None, 'leaf monomial')
    one, exp = mm.groups()
    meta.require(re.fullmatch(r'\(1 : .*\)', leaf.wires[one].term.strip()) is not None, 'leaf monomial base')
    em = re.fullmatch(r'\((w_\d+_\d+) \* (w_\d+_\d+)\)', leaf.wires[exp].term.strip())
    meta.require(em is not None, 'leaf exponent scale')
    value, scalew = em.groups()
    scale = meta.integer(leaf.wires[scalew].term)
    get = meta.one((n for n, t in flat if t.startswith('MxxRuntime.familyGetDynamic ') and t.endswith(' ' + value)),
                   'leaf exponent read')
    gm = re.match(r'MxxRuntime\.familyGetDynamic (w_\d+_\d+) (w_\d+_\d+) ' + value + '$', text[get])
    fam, idx = gm.groups()
    meta.require(leaf.wires[fam].src == 'input', 'leaf exponent family is not an input')
    fam_path = leaf.wires[fam].term
    fam_type = leaf.wires[fam].type
    flag_term = ad.wire_term(leaf, flag)
    cond = re.fullmatch(r'\(if \(decide \(\(\(Int\.ofNat i_1\)\) < (inputs[.\d]*)\)\) then 1 else 0\)', flag_term)
    meta.require(cond is not None, f'leaf activity flag shape: {flag_term}')
    count_path = cond.group(1)
    index = ad.wire_term(leaf, idx)
    # leafTerm parameters: every input projection the index reads, plus the count and the family.
    paths = sorted(set(re.findall(r'inputs(?:\.2)*\.1', index)) | {count_path, fam_path},
                   key=lambda p: p.count('.2'))
    names = {p: f'x{p.count(".2")}' for p in paths}
    def types(p):
        w = meta.one((w for w in leaf.wires.values() if w.src == 'input' and w.term == p), f'leaf input {p}')
        return w.type
    params = ' '.join(f'({names[p]} : {types(p)})' for p in paths)
    def subst(term):
        for p in sorted(paths, key=lambda p: -p.count('.2')):
            term = term.replace(p, names[p])
        return term.replace('i_1', 'k')
    F.leaf_paths = paths
    F.leaf_scale = scale
    idx_paths = [p for p in paths if p not in (count_path, fam_path)]
    F.idx_paths = idx_paths
    term_index = subst(index)
    call_args = ' '.join(names[p] for p in paths)
    in_args = ' '.join(p for p in paths)
    idx_params = ' '.join(f'({names[p]} : {types(p)})' for p in idx_paths)
    idx_args = ' '.join(names[p] for p in idx_paths)
    src = f'''/-- The exponent index read by one forest leaf. -/
noncomputable def leafIndex {idx_params} (k : Nat) : Int :=
  {term_index}

/-- The exponent contributed by one forest leaf (zero when inactive). -/
noncomputable def leafTerm {params} (k : Nat) : Int :=
  if (Int.ofNat k) < {names[count_path]} then intFamilyValue {names[fam_path]} (leafIndex {idx_args} k) * {scale} else 0

theorem leafTerm_zero {''.join('{' + names[p] + ' : ' + types(p) + '} ' for p in paths)}(k : Nat)
    (hk : {names[count_path]} ≤ k) : leafTerm {call_args} k = 0 := by
  unfold leafTerm
  rw [if_neg (by simp only [Int.ofNat_eq_natCast]; omega)]

/-- One exported forest leaf: its payload is the monomial of its exponent, or the unit. -/
theorem leaf_payload {binders}
    (unit : {unit_path} = P 0)
    (h : {call}) :
    outputs{'.2' * 6}.1 = P (leafTerm {in_args} i_1) := by
  obtain {ad.pattern(leaf, per)} := h
  have payload_value := int_family_value {get}
  have sel := Online0Labels.select_pair {sel}
  simp only at sel ⊢
  rw [sel]
  unfold leafTerm leafIndex
  by_cases hk : (Int.ofNat i_1) < {count_path}
  · simp only [hk, decide_true, ↓reduceIte]
    funext r c
    have row_zero : r = 0 := Subsingleton.elim _ _
    have column_zero : c = 0 := Subsingleton.elim _ _
    subst r c
    simpa only [payload_value, one_mul] using
      (mono_entry (one_entry (q := modulus_0) (n := {F.n})) _)
  · simp only [hk, decide_false, Bool.false_eq_true, ↓reduceIte]
    exact unit

'''
    F.leaf_names = names
    F.leaf_count_path = count_path
    F.leaf_fam_path = fam_path
    F.unit_path = unit_path
    return src


def merge_source(F, k, scope, size):
    ad = F.adapters
    binders, call, _ = ad.header(scope)
    per, flat = ad.atom_names(scope)
    per = ad.use_rfl_out(per, flat)
    text = dict(flat)
    outs = WIRE.findall(ad.first_out(flat)[1])
    meta.require(len(outs) == 7, f'{scope.name}: merge must output seven values')
    pay = outs[6]
    sel = meta.one((n for n, t in flat if t.startswith('MxxRuntime.select ') and t.endswith(' ' + pay)),
                   'merge payload selection')
    m = re.match(r'MxxRuntime\.select (w_\d+_\d+) \[(w_\d+_\d+), (w_\d+_\d+)\] ' + pay + '$', text[sel])
    flag, left, prod = m.groups()
    pm = re.fullmatch(r'MxxRuntime\.matrixMulScalarLeft (w_\d+_\d+) (w_\d+_\d+)', scope.wires[prod].term.strip())
    meta.require(pm is not None and pm.group(1) == left, f'{scope.name}: merge product')
    right = pm.group(2)
    reads = {}
    for n, t in flat:
        g = re.match(r'MxxRuntime\.familyGetDynamic (w_\d+_\d+) (w_\d+_\d+) (w_\d+_\d+)$', t)
        if g:
            reads[g.group(3)] = (n, g.group(1), g.group(2))
    meta.require(left in reads and right in reads, f'{scope.name}: payload reads')
    lname, lfam, lidx = reads[left]
    rname, rfam, ridx = reads[right]
    meta.require(lfam == rfam and scope.wires[lfam].src == 'input', f'{scope.name}: payload family')
    fam_path = scope.wires[lfam].term
    lterm = ad.wire_term(scope, lidx)
    rterm = ad.wire_term(scope, ridx)
    squash = lambda term: re.sub(r'[()\s]', '', term)
    meta.require(squash(lterm) == 'Int.ofNati_1*2', f'{scope.name}: left index {lterm}')
    meta.require(squash(rterm) == f'Int.ofNati_1*2+1%{size}', f'{scope.name}: right index {rterm}')
    flag_term = ad.wire_term(scope, flag)
    cond = re.fullmatch(r'\(if \(decide \((.*) < (inputs[.\d]*)\)\) then 1 else 0\)', flag_term)
    meta.require(cond is not None, f'{scope.name}: activity flag {flag_term}')
    count_path = cond.group(2)
    meta.require(count_path == 'inputs.1', f'{scope.name}: count is not the first input')
    # The first reads of each index carry the member witnesses.
    def first_read(idx):
        return next(n for n, t in flat if re.match(r'MxxRuntime\.familyGetDynamic \S+ ' + idx + ' ', t))
    lfirst, rfirst = first_read(lidx), first_read(ridx)
    lget = f'Online0Labels.read_at leftMember leftIndex {lname}' if lname != lfirst else 'leftRead'
    rget = f'Online0Labels.read_at rightMember rightIndex {rname}' if rname != rfirst else 'rightRead'
    return f'''/-- Forest level {k}: the merged payload is the product of its active children. -/
theorem merge{k}_payload {binders}
    {{t : Nat → Int}} {{cnt : Int}} {{w : Nat}}
    (ht : ∀ k : Nat, cnt ≤ k → t k = 0) (hc : cnt ≤ inputs.1 * w) (hcs : inputs.1 ≤ {size})
    (hin : ∀ m : Fin {size}, {fam_path} m = P (∑ r ∈ Finset.range w, t (m.val * w + r)))
    (h : {call}) :
    outputs{'.2' * 6}.1 = P (∑ r ∈ Finset.range (2 * w), t (i_1 * (2 * w) + r)) := by
  obtain {ad.pattern(scope, per)} := h
  obtain ⟨leftMember, leftIndex, leftRead⟩ := {lfirst}
  have get_left := {lget}
  obtain ⟨rightMember, rightIndex, rightRead⟩ := {rfirst}
  have get_right := {rget}
  have select_payload := Online0Labels.select_pair {sel}
  have hl : leftMember.val = 2 * i_1 := by simp only [Int.ofNat_eq_natCast] at leftIndex; omega
  refine merge_core ht hc (left := {left}) (right := {right}) ?_ (fun hact ↦ ?_) ?_
  · rw [get_left, hin, hl]
  · have hr : rightMember.val = 2 * i_1 + 1 := by
      simp only [Int.ofNat_eq_natCast] at rightIndex hact; omega
    rw [get_right, hin, hr]
  · show {pay} = _
    rw [select_payload]
    by_cases hactive : {cond.group(1)} < inputs.1
    · simp only [hactive, decide_true, ↓reduceIte]
      rw [if_pos (by simpa only [Int.ofNat_eq_natCast] using hactive)]
    · simp only [hactive, decide_false, Bool.false_eq_true, ↓reduceIte]
      rw [if_neg (by simpa only [Int.ofNat_eq_natCast] using hactive)]

'''


def count_source(F):
    """Counts of the forest levels: active leaves fit the level, and the level fits its family."""
    s, fname = F.scope, F.name
    out = []
    unfold = set()
    def deps(w, acc):
        wi = s.wires[w]
        if wi.src == 'let':
            acc.add(w)
            for x in WIRE.findall(wi.term):
                deps(x, acc)
        return acc
    prev = None
    for lvl, cw in enumerate(F.count_wires[1:], start=1):
        w = 2 ** (lvl - 1)
        acc = deps(cw, set())
        sels = []
        for x in sorted(acc | {cw}):
            pass
        wi = s.wires[cw]
        lines = []
        if wi.src == 'witness':
            node = F.info.node(cw)
            atom = node[1]
            meta.require(node[0] == 'atom' and atom.kind == 'select', f'{fname}.{cw}: count producer')
            lines.append(f'    rw [Online0Labels.select_pair F.c{atom.frag}.{atom.field}]')
            acc = set()
            for x in WIRE.findall(' '.join(atom.args)):
                deps(x, acc)
        names = ', '.join(f'{fname}.{x}' for x in sorted(acc))
        simp = f'    simp only [{names}, decide_eq_true_eq]' if acc else '    skip'
        hyp = '' if prev is None else ' at *'
        out.append(f'''  have c{lvl} : {fname}.{F.cnt} ctx ≤ {fname}.{cw} ctx * (({w} : Nat) : Int) ∧
      {fname}.{cw} ctx ≤ {F.sizes[lvl - 1]} := by
{chr(10).join(lines)}
{simp}
    all_goals (try push_cast)
    all_goals (first | omega | (split <;> omega) | skip)''')
        prev = lvl
    return '\n'.join(out)


def forest_source(F):
    fname = F.name
    call31 = F.calls[0][0]
    leaf_args = [strip_parens(a) for a in call31.loop.args]
    # Map leaf input positions to forest wires.
    def arg_of(path):
        idx = path.count('.2')
        return f'({fname}.{leaf_args[idx].split()[0]} ctx)'
    lt_args = ' '.join(arg_of(p) for p in F.leaf_paths)
    unit_wire = leaf_args[F.unit_path.count('.2')].split()[0]
    t = f'leafTerm {lt_args}'
    levels = []
    leaf_out = F.calls[0][1][6]
    levels.append(f'''  have l0 : ∀ m : Fin {F.sizes[0]}, {fname}.{leaf_out} ctx m = P (∑ r ∈ Finset.range 1,
      {t} (m.val * 1 + r)) := by
    intro m
    rw [Finset.sum_range_one, mul_one, add_zero]
    exact leaf_payload unit ({F.fact(leaf_out)} m)''')
    for lvl in range(1, len(F.calls)):
        out = F.calls[lvl][1][6]
        levels.append(f'  have l{lvl} := fun m : Fin {F.sizes[lvl]} ↦ merge{lvl}_payload ht c{lvl}.1 c{lvl}.2 l{lvl - 1} ({F.fact(out)} m)')
    root_out = F.calls[-1][1][6]
    # The forest root payload wire reads the single root member.
    root_pay = None
    for w, node in F.info.prod.items():
        if node[0] == 'atom' and node[1].kind == 'familyGetStatic' and strip_parens(node[1].args[0]) == root_out:
            root_pay = (w, node[1])
    meta.require(root_pay is not None, 'forest root payload read')
    rw, ratom = root_pay
    F.root_payload = rw
    F.unit_wire = unit_wire
    F.lt_args = lt_args
    return f'''/-- The exported root payload of one label-forest member is the monomial of
the sum of its active leaf exponents, for an arbitrary member. -/
theorem forest_payload {{backend : BackendContext}} {{hashModel : HashModel}} {{params : Params}}
    (member : Fin {F.members}) (ctx : {fname}.Ctx backend hashModel params member.val)
    {{outputs : _}} (F : {fname}.Facts ctx outputs)
    (unit : {fname}.{unit_wire} ctx = P 0) (hcnt : {fname}.{F.cnt} ctx ≤ {F.leaves}) :
    {fname}.{rw} ctx = P (∑ r ∈ Finset.range {F.leaves}, {t} r) := by
  have ht : ∀ k : Nat, {fname}.{F.cnt} ctx ≤ k → {t} k = 0 :=
    fun k hk ↦ leafTerm_zero k hk
{count_source(F)}
{chr(10).join(levels)}
  obtain ⟨p, hp, hv⟩ := F.c{ratom.frag}.{ratom.field}
  have hp0 : p = 0 := Fin.ext (by omega)
  subst hp0
  rw [hv]
  refine (l{len(F.calls) - 1} 0).trans (congrArg P ?_)
  simp only [Fin.val_zero, zero_mul, zero_add]
  all_goals exact sum_tail (fun k hk ↦ ht k (by omega)) (by norm_num)

'''


def member_source(F):
    """The non-constant next label members are forest members on the label-0 column."""
    geo, T = F.geo, F.T
    fname = F.name
    labels = geo.view['labels']
    R = labels['root_members']
    C = labels['target_columns']
    meta.require(F.members == labels['rows'] * C, 'forest members are rows times target columns')
    meta.require(R == labels['rows'] + 1, 'label-0 roots are the label rows plus the constant one')
    run = (geo.package / 'proof' / 'Dio2RunFacts.lean').read_text()
    m = re.search(r'noncomputable def label1Exponent \(external : GeneratedClaim\.ExternalInputs\)\n'
                  r'    \(member : Fin (\d+)\) : Int :=\n  ([^\n]*)\n', run)
    meta.require(m is not None and int(m.group(1)) == R, 'label1Exponent shape')
    body = m.group(2)
    tail = re.search(r'\)\) % (\d+)\)\) \(\(\(Int\.ofNat member\.val\) % (\d+)\) \* (\d+)\)\.toNat\), 0\]\) \* (\d+)\)$', body)
    meta.require(tail is not None, 'label1Exponent member index and scale')
    M, rows, C2, S = (int(x) for x in tail.groups())
    meta.require(rows == R - 1 and C2 == C, 'label1Exponent member index')
    meta.require(S == F.leaf_scale and M * S == 2 * F.n, 'label exponent scale')
    meta.require(f'(if (decide ((Int.ofNat member.val) = {R - 1})) then (1 : Int) else 0)' in body,
                 'label1Exponent constant member')
    # Leaf argument wires of the forest: count, block and bits are family reads / lets.
    call31 = F.calls[0][0]
    leaf_args = [strip_parens(a).split()[0] for a in call31.loop.args]
    names = F.leaf_names
    pos = {names[p]: leaf_args[p.count('.2')] for p in F.leaf_paths}
    count_w = pos[names[F.leaf_count_path]]
    fam_w = pos[names[F.leaf_fam_path]]
    others = [w for p, w in pos.items() if w not in (count_w, fam_w)]
    meta.require(len(others) == 2, 'leaf block and bits arguments')
    reads = {}
    for w in [count_w] + others:
        node = F.info.node(w)
        if node[0] == 'atom' and node[1].kind == 'familyGetDynamic':
            a = node[1]
            reads[w] = (strip_parens(a.args[0]), strip_parens(a.args[1]), f'F.c{a.frag}.{a.field}')
    meta.require(count_w in reads, 'forest count read')
    bits_w = meta.one((w for w in others if w in reads), 'forest bits read')
    block_w = meta.one((w for w in others if w not in reads), 'forest block')
    cfam, cidx, cfact = reads[count_w]
    bfam, bidx, bfact = reads[bits_w]
    meta.require(cidx == bidx, 'count and bits share their column index')
    sizes = {}
    for fam in (cfam, bfam):
        sizes[fam] = meta.family_count(F.scope.wires[fam].type)
    meta.require(sizes[cfam] == sizes[bfam] == C, 'selector families have one column per target column')
    # Column index and block are member arithmetic.
    def deps(w, acc):
        wi = F.scope.wires[w]
        if wi.src == 'let':
            acc.add(w)
            for x in WIRE.findall(wi.term):
                deps(x, acc)
        return acc
    col_defs = ', '.join(f'{fname}.{x}' for x in sorted(deps(cidx, set())))
    blk_defs = ', '.join(f'{fname}.{x}' for x in sorted(deps(block_w, set())))
    # The selector family facts of the online root.
    fargs = T.forest_args
    cpos = [strip_parens(a).split()[0] for a in fargs]
    croot = cpos[int(F.scope.wires[cfam].term.count('.2'))]
    broot = cpos[int(F.scope.wires[bfam].term.count('.2'))]
    shan = T.root.node(croot)
    meta.require(shan[0] == 'atom' and shan[1].kind == 'sha256BitColumns', 'selector columns producer')
    sa = shan[1]
    meta.require(strip_parens(sa.args[-1]) == croot and strip_parens(sa.args[-2]) == broot, 'selector columns outputs')
    seed, crows, ccols, cbits = (strip_parens(x) for x in sa.args[:4])
    meta.require(int(ccols) == C, 'selector columns dimensions')
    # The forest is sized to the largest selector count; with odd level sizes its right
    # reads wrap, so the payload equals the requirement exponent only if every count is at
    # most the leaf count.  That bound is proved here only when it is structural (rows ==
    # leaves); a computed SHA-256 maximum is not kernel-checkable.
    meta.require(int(crows) == F.leaves,
                 f'selector count bound: hashed rows {crows} exceed forest leaves {F.leaves}; '
                 'the forest/requirement agreement needs every column count <= leaves, which is '
                 'a computed SHA-256 property here (no structural proof)')
    col = f'Sha256Columns.column {seed} {crows} {ccols} {cbits}'
    F.columns = (col, f'F.c{sa.frag}.{sa.field}', croot, broot)
    t = f'leafTerm {F.lt_args}'
    idx_call = ' '.join(f'({fname}.{pos[names[p]]} ctx)' for p in F.idx_paths)
    count_le = f'''/-- The active leaf count is a selector count, hence at most the number of leaves. -/
theorem count_le {{backend : BackendContext}} {{hashModel : HashModel}} {{params : Params}}
    {{member : Fin {F.members}}} (ctx : {fname}.Ctx backend hashModel params member.val)
    {{outputs : _}} (F : {fname}.Facts ctx outputs)
    (hcount : ∀ i : Fin {C}, {fname}.{cfam} ctx i = (({col} i.val).2 : Int)) :
    {fname}.{count_w} ctx ≤ {F.leaves} := by
  obtain ⟨p, hp, hv⟩ := {cfact}
  rw [hv, hcount]
  exact_mod_cast column_count_le _ _ _ _ _

'''
    return count_le + f'''theorem P_mod (e : Int) : P (e % {M} * {S}) = P (e * {S}) := by
  unfold P
  funext r c
  simp only [message]
  congr 3
  push_cast
  omega

/-- A non-constant label member `m` is forest member `{C} m`; its root payload is
exactly the next external label of `m`, by the requirement's own exponent sum. -/
theorem member_payload {{model : HashModel}} {{external : GeneratedClaim.ExternalInputs}}
    {{execution : GeneratedClaim.Execution}} (hrun : W_GeneratedClaim.Runs model external execution)
    {{backend : BackendContext}} {{hashModel : HashModel}} {{params : Params}}
    (m : Fin {R}) (hm : m.val ≠ {R - 1}) (member : Fin {F.members}) (hmember : member.val = m.val * {C})
    (ctx : {fname}.Ctx backend hashModel params member.val)
    {{outputs : _}} (F : {fname}.Facts ctx outputs)
    (hcount : ∀ i : Fin {C}, {fname}.{cfam} ctx i = (({col} i.val).2 : Int))
    (hbits : ∀ i : Fin {C}, {fname}.{bfam} ctx i = (({col} i.val).1 : Int))
    (hexps : {fname}.{fam_w} ctx = external.input_0) :
    P (∑ r ∈ Finset.range {F.leaves}, {t} r) =
      fun _ _ ↦ reducePoly modulus_0 {F.n} (online1Labels external m) := by
  have hlt : m.val < {R - 1} := by omega
  have hcol : {fname}.{cidx} ctx = ((⟨0, by decide⟩ : Fin {C}).val : Int) := by
    simp only [{col_defs}, hmember, Int.ofNat_eq_natCast]
    push_cast; omega
  have h4 := Online0Labels.read_at (⟨0, by decide⟩ : Fin {C}) hcol.symm {cfact}
  have h21 := Online0Labels.read_at (⟨0, by decide⟩ : Fin {C}) hcol.symm {bfact}
  have h19 : {fname}.{block_w} ctx = (m.val : Int) := by
    simp only [{blk_defs}, hmember, Int.ofNat_eq_natCast]
    push_cast; omega
  have hsel : (if (decide ((Int.ofNat m.val) = {R - 1})) then (1 : Int) else 0) = 0 := by
    simp only [Int.ofNat_eq_natCast, decide_eq_true_eq]; split_ifs <;> omega
  have hidx : (((Int.ofNat m.val) % {R - 1}) * {C}).toNat = m.val * {C} := by
    simp only [Int.ofNat_eq_natCast]; omega
  have hmod : ((Int.ofNat (m.val * {C})) % {C}).toNat = 0 := by
    simp only [Int.ofNat_eq_natCast]; omega
  have hdiv : (Int.ofNat (m.val * {C})) / {C} = (m.val : Int) := by
    simp only [Int.ofNat_eq_natCast]; omega
  have hsum : ∀ r, {t} r = (if (Int.ofNat r) < {fname}.{count_w} ctx then
      intFamilyValue ({fname}.{fam_w} ctx) (leafIndex {idx_call} r) else 0) * {S} := by
    intro r
    unfold leafTerm
    split_ifs <;> simp
  rw [Finset.sum_congr rfl fun r _ ↦ hsum r, ← Finset.sum_mul, ← P_mod, h4, h21, h19, hcount, hbits,
    hexps, online1Labels, Complete.label1_payload hrun m]
  unfold P
  simp only [Complete.label1Exponent, hsel, hidx, MxxIR.selectInt, Int.toNat_zero,
    List.getElem?_cons_zero, Option.getD_some, famAt_of_lt _ (show m.val * {C} < {F.members} by omega),
    hmod, hdiv]
  simp only [leafIndex, intFamilyValue, famAt]
  funext r c
  congr 3
  clear * -
  simp only [Int.ofNat_eq_natCast, Nat.cast_zero, zero_add, ↓reduceDIte, decide_eq_true_eq]
  apply congrArg (· % ({M} : Int))
  refine Finset.sum_congr rfl fun x _ ↦ ?_
  split_ifs <;> simp_all <;> omega

'''


HEADER = """import Dio2Step0Forest
import Dio2RunFacts
import Dio2Online1Inputs

/-!
# Step-zero label-forest payloads

Generated by `gen_s0o.py` (module `s0o_payload.py`) from checked export
metadata. Every forest member and level is quantified; the leaf exponent,
the level counts and the next-label exponent are read from the exported
constraints and the requirement circuit. No numerical assumption is made.
-/

set_option maxRecDepth 16384
set_option maxHeartbeats 16000000
set_option Elab.async false

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh
  RefreshTopProof RefreshTopProof.Gen RefreshTopProof.GenIface
  RefreshTopProof.GenInst RefreshTopProof.GenFinal GenDecode
open W_Stage_online_0
open scoped BigOperators

namespace Step0Payload

"""


def payload_source(geo, gen, module, T):
    F = Forest(geo, gen, module, T)
    text = HEADER + payload_prelude(geo.n) + leaf_source(F)
    for k, (scope, size) in enumerate(zip(F.merges, F.sizes), start=1):
        text += merge_source(F, k, scope, size)
    text += forest_source(F)
    text += member_source(F)
    text += 'end Step0Payload\n\nend Dio2Proof\n'
    return text, F
