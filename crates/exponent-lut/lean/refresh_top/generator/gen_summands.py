#!/usr/bin/env python3
"""Generate `Gen_Summands.lean` (WP F): the closed summand facts of `GenFinalInst.final_inst` at a
symbolic instance, from the instance Facts of the preprocess, encrypt and online stages, the Claim's
`Runs` and the requirement (`GenPart.Req`).

Method. D's chain test theorems (`Gen_Inst_Stage_online_eval_Chains.lean`, `chain_h*`, conclusion
`True`) compose the online segment theorems `seg_*` along the PRF chains; every chain hypothesis is a
binder `{x_hK}` / `(hx_hK : …)`. This generator re-emits a chain's body (its `node_have` steps, verbatim)
inside a theorem over the three stage contexts and replaces every binder by
* a value (data binders: the PRF start `μ = 1`, the label payloads `α k = X^(s₀ a_k)`, gadgets
  `gadgetOf`/`rebasedGadget`, the right-hand-side values `ν`, chosen one-hot selectors and keys,
  the LUT coefficients `keyTableG … 0 0`, the LUT values `centeredIntLift (Σ …)`), or
* a proof term: encrypt bridge facts (label entries, the PRF top encoding, LUT helper keys; D's
  `Gen_Inst_Stage_encrypt.roles` steps), preprocess facts (`rhs_*`, `switch_*` of
  `Gen_Inst_Stage_preprocess_Keys` / `Gen_Wiring`), the stage links (`Gen_Stage_Links`), the
  one-hot rows (`Gen_Wiring.onehot_*`), `ValidExternals` bounds and the requirement's label relation.
Binders are recognized by their names' kinds and the stage links of the online wires they mention
(no wire ids).

Summands (each over the three stage contexts; open hypotheses are listed in each docstring):
* `mask_summand` — `hg`/`heg`, `hβ` (mask chain + `mask_inst_form` + `mask_inst`; open: `hin`).
* `fresh_summand` — `hf`/`hef` with `E j = GenPart.errE external j` (sampler chain + column sums +
  `fresh_inst` + `sampler_value_gen`).
* `smudge_summand` — `hs`/`hes` and `hTH`/`hM` (smudge chain + `smudge_*` / `smudge_hTH_*` of
  `Gen_Online_Smudge` + `Gen_Wiring.smudge_digits_keys_*`).
* `hint_summand` (`hb`/`heb`), `pre_summand` (`hpre`; open: `hD`/`hY`), `keys_summand` (shared `Ks`),
  `preimage_summand` (`hP`/`hPd`, bound = the trapdoor cutoff).

Closed bounds. The bounds of `heg`, `hef`, `hes`, `hM` are stated as `B ≤ bound_<summand>_<B> i` with
`bound_… i` a per-instance numeral, checked in the proof by `interval_cases` + `decide`. The numerals
come from a probe build: `--probe` emits a `probe_bound` tactic that logs every bound's value per
instance (`BOUND <summand>.<B> <i> <value>`); `--bounds LOG[,LOG…]` reads such build logs.

Output: `<out>_Pre.lean` (lemmas, certificates, bound tables), one module `<out>_<Summand>.lean` per
summand, and `<out>` importing them all (one module of all summands exceeds 20 GB at n = 2^17).

Large-n rules the emitted proofs follow (each one cost minutes or a timeout at n = 2^17): no `rfl`,
`congr` or `convert` closing step at default transparency on polynomial terms (`with_reducible rfl`,
`convert (config := { closePre := false, closePost := false })`, `Xe_congr_nat` + `ring`); every
`bucket_of_cells*` step gets `M`/`W` explicitly (`pin_bucket_dims`).

With `--sigma` the stages are those of the caller's encoding of `σ_0 G` (`Stage_sigma_*`, one
instance per chain) and the output is `Gen_Summands_Sigma.lean` with `mask_summand` only: the row
`g` of the chain's instance, its keys and the PRF outputs `zN` as the requirement's sums (`hz`, the
`mask_inst` hypothesis of the parts reading the row).

Usage: gen_summands.py <param dir> [--out FILE] [--probe | --bounds LOG[,LOG…]] [--test-chains] [--sigma]
  <param dir>: the pipeline output dir (generated/, roles.json, Gen_Inst_Stage_online_eval_Chains.lean,
  Gen_Inst_Stage_encrypt.lean, Gen_Inst_Stage_preprocess_Keys.lean, Gen_Online_Smudge.lean, Gen_Requirement.lean).
  Two passes: build the `--probe` output (`_Pre`, then the summand modules), then regenerate with
  `--bounds <the summand modules' build logs>` and build again.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import xmodel  # noqa: E402
import gen_stage_links as SL  # noqa: E402
import gen_wiring as GW  # noqa: E402

#: The stage triple: the parts', or with `--sigma` the stages of the caller's encoding of `σ_0 G`.
SIGMA = '--sigma' in sys.argv
PRE, ENC, ON = (('Stage_sigma_preprocess', 'Stage_sigma_encrypt', 'Stage_sigma_online_eval') if SIGMA
                else ('Stage_preprocess', 'Stage_encrypt', 'Stage_online_eval'))


def stage_ns(M):
    """The namespace stem of a stage's generated instance modules (`gen_instance`, `gen_instance_pre`)."""
    return M.replace('Stage_', '').title().replace('_', '')


PRE_KEYS = f'RefreshTopProof.GenInst.{stage_ns(PRE)}Keys'
WIRING = GW.NS
#: The online stage's params (set by `Ctx`).
ON_PARAMS = None


class Fail(Exception):
    pass


WIRE_RE = re.compile(r'(?:\b(\w+)\.)?(parallel_generatedRoot_\d+)\.(w_\d+_\d+) (\w+)')


# --------------------------------------------------------------------------- parsing

def theorem_text(text, name):
    m = re.search(r'^theorem ' + re.escape(name) + r'\b', text, re.M)
    if not m:
        raise Fail(f'no theorem {name}')
    end = text.find('\n\n', m.start())
    return text[m.start(): end if end >= 0 else len(text)]


def split_header(th):
    """(binders text, conclusion, body) of `theorem name binders : concl := by body`."""
    i = th.index(':= by\n') if ':= by\n' in th else th.index(':=\n')
    head, body = th[:i], th[i:].split('\n', 1)[1]
    # last top-level ' : ' in head separates the conclusion
    depth, k = 0, None
    for j, ch in enumerate(head):
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        elif depth == 0 and k is None and (head.startswith(' : ', j) or head.startswith(' :\n', j)):
            k = j
    return head[:k], head[k + 3:].strip(), body


def binders(head):
    """Top-level bracketed binders: list of (bracket, names, type)."""
    out, i = [], 0
    while i < len(head):
        ch = head[i]
        if ch in '({[':
            j = xmodel.matching(head, i)
            inner = head[i + 1:j]
            parts = xmodel.split_top(inner, ' : ')
            if len(parts) >= 2:
                names = parts[0].split()
                typ = ' : '.join(parts[1:])
            else:
                names, typ = [inner.strip()], ''
            out.append((ch, names, re.sub(r'\s+', ' ', typ).strip()))
            i = j + 1
        else:
            i += 1
    return out


def node_haves(body):
    """[(name, expr)] of the `node_have` steps (continuation lines joined)."""
    out = []
    for ln in body.split('\n'):
        m = re.match(r'^  node_have (\S+) := (.*)$', ln)
        if m:
            out.append([m.group(1), m.group(2)])
        elif ln.startswith('    ') and out:
            out[-1][1] += '\n' + ln
    return [tuple(x) for x in out]


def kind_of(name):
    m = re.match(r"^(.*?)_(h\d+|pm\d+)$", name)
    return (m.group(1), m.group(2)) if m else (name, None)


# --------------------------------------------------------------------------- context

class Ctx:
    def __init__(self, pdir):
        self.pdir = pdir
        self.gdir = f'{pdir}/generated'
        self.roles = json.load(open(f'{pdir}/roles.json'))
        calls = [c for c in xmodel.parse_claim(self.gdir) if c.get('module')]
        self.stages = {c['module']: SL.Stage(self.gdir, c) for c in calls}
        self.O = self.stages[ON]
        self.P = self.stages[PRE]
        self.E = self.stages[ENC]
        global ON_PARAMS
        ON_PARAMS = self.O.params
        self.links_text, _ = SL.gen(self.gdir)
        self.S = self.O.S
        self.links = self.parse_links()
        self.chains = open(f'{pdir}/Gen_Inst_{ON}_Chains.lean').read()
        self.n = int(re.search(r'\{sm sp : ErrorPoly (\d+)\}', self.chains).group(1))
        self.enc = open(f'{pdir}/Gen_Inst_{ENC}.lean').read()
        self.keys = open(f'{pdir}/Gen_Inst_{PRE}_Keys.lean').read()
        self.smudge = '' if SIGMA else open(f'{pdir}/Gen_Online_Smudge.lean').read()
        self.efacts = open(f'{self.gdir}/{ENC}_Facts.lean').read()
        self.wiring = GW.gen(self.gdir, pdir)
        self.claim = open(f'{self.gdir}/Claim.lean').read()
        self.types, self.vproj = GW.claim_info(self.gdir)
        self.ninputs = len(self.types)

    def parse_links(self):
        """online wire -> list of (src stage, src wire, link term builder)."""
        out = {}
        for m in re.finditer(r'^theorem (link_(Stage_\w+?)_(w_\d+_\d+)_(Stage_\w+?)_(w_\d+_\d+)) ([^\n]*\n(?:    [^\n]*\n)*?)    (\S+) (\S+) = (\S+) (\S+) :=', self.links_text, re.M):
            name, A, a, B, b = m.group(1, 2, 3, 4, 5)
            hyps = m.group(6)
            lhs_w = m.group(7).split('.')[-1]
            outlink = 'houtA' in hyps
            if B == ON:
                src, sw, ow = A, a, b
            elif A == ON:
                src, sw, ow = B, b, a
            else:
                continue
            # orientation: we want `online = source`
            online_first = (lhs_w == ow)
            out.setdefault(ow, []).append(dict(name=name, src=src, sw=sw, outlink=outlink,
                                               A=A, B=B, online_first=online_first))
        return out

    def hyp_args(self, M):
        t = {PRE: 'P', ENC: 'E', ON: 'O'}[M]
        return t

    def link_term(self, ow, src):
        """A proof of `online.ow ctx = src.sw cS`."""
        for L in self.links.get(ow, []):
            if L['src'] != src:
                continue
            a, b = self.hyp_args(L['A']), self.hyp_args(L['B'])
            if L['outlink']:
                args = f'hout{a} hinst{a} hin{b} hroot{b}'
            else:
                args = f'hin{a} hroot{a} hin{b} hroot{b}'
            t = f'(StageLinks.{L["name"]} {args})'
            return (t if L['online_first'] else f'{t}.symm'), L['sw']
        return None, None


def online_wires(typ):
    return sorted(set(re.findall(r'parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx', typ)), key=lambda w: [int(x) for x in w[2:].split('_')])


def link_sources(C, wires):
    """stage -> [(online wire, src wire)] for wires with links."""
    out = {}
    for w in wires:
        for L in C.links.get(w, []):
            out.setdefault(L['src'], []).append((w, L['sw']))
    return out


def qualify(text, M, inst):
    """Qualify an instance-scope text of stage M (written with `open M`)."""
    text = re.sub(r'(?<![\w.])(parallel_generatedRoot_\d+|parallel_parallel_generatedRoot_\d+_\d+|sequential_\w+|modulus_\d+|mod_\d+|sel_\d+)\b',
                  M + r'.\1', text)
    return text


# --------------------------------------------------------------------------- the encrypt facts

class EncFacts:
    """D's encrypt `roles` steps (verbatim, re-scoped to the summand theorem)."""

    def __init__(self, C):
        th = theorem_text(C.enc, 'roles')
        _, _, body = split_header(th)
        self.lines = []
        cur = None
        for ln in body.split('\n'):
            if re.match(r'^  (node_have|have) ', ln):
                cur = [ln]
                self.lines.append(cur)
            elif ln.startswith('    ') and cur is not None:
                cur.append(ln)
        self.steps = {}
        self.masks = []
        for L in self.lines:
            txt = '\n'.join(L)
            m = re.match(r'^  node_have (\S+) := (.*)$', txt, re.S)
            if m:
                self.steps[m.group(1)] = m.group(2)
            elif txt.startswith('  have m_w_') or re.match(r'^  rw \[h_w_\d+_\d+\] at m_w_', txt):
                self.masks.append(txt)
        self.mask_lines = [ln for ln in body.split('\n') if re.match(r'^  (have m_w_|rw \[h_w_\d+_\d+\] at m_w_)', ln)]

    def rescope(self, text):
        t = qualify(text, ENC, None)
        t = re.sub(r'\bF\.', 'FE.', t)
        t = re.sub(r'\bctx\b', 'cE', t)
        t = re.sub(r'\bhsmsmall\b', 'hsm', t)
        t = re.sub(r'\bhspsmall\b', 'hsp', t)
        t = re.sub(r'\bi_0\b', '(iS : Nat)', t)
        return t

    def step_of_loop(self, loop):
        """The step `s_h<loop>_<k>` of a parallel loop (label entries, branch keys)."""
        for k, v in self.steps.items():
            if re.match(r'^s_h' + str(loop) + r'_\d+$', k):
                return k, self.rescope(v)
        return None, None

    def step_of_output(self, wire):
        """The root step whose `(c0 := …)` is `wire` (source / input encodings)."""
        for k, v in self.steps.items():
            if f'(c0 := (parallel_generatedRoot_' in v and re.search(r'\(c0 := \(parallel_generatedRoot_\d+\.' + wire + r' ctx\)\)', v):
                return k, self.rescope(v)
        return None, None


# --------------------------------------------------------------------------- generic lemmas

LEMMAS = r'''
/-! ## Generic lemmas -/

theorem ternary_secret {q n : Nat} {x : ExactMatrix q n 1 1} (h : uniformIntervalSample (-1) 1 x) :
    ∃ s : ErrorPoly n, polyNorm s ≤ 1 ∧ x = fun _ _ ↦ reducePoly q n s := by
  obtain ⟨w, hw, hb⟩ := uniformIntervalSample_ternary h
  refine ⟨w 0 0, hb 0 0, ?_⟩
  rw [hw]; funext r c
  rw [Subsingleton.elim r 0, Subsingleton.elim c 0]; rfl

theorem message_zero {n : Nat} (c : Int) : message (n := n) c 0 = (c : ErrorPoly n) := by
  simp [message]

/-- A selector cell: `message s 0` of a one-hot entry. -/
theorem message_onehot {n W : Nat} {x : Int} {k sel : Fin W} (h : x = if k = sel then 1 else 0) :
    message (n := n) x 0 = if k = sel then (1 : ErrorPoly n) else 0 := by
  rw [message_zero, h]; split_ifs <;> simp

theorem famAt_cell {M W : Nat} (f : Fin (M * W) → Int) (o : Fin M) (j : Fin W) :
    famAt f (o * W + j) = f (cellAt o j) := by
  have hlt : (o : Nat) * W + j < M * W := by
    have := o.isLt; have := j.isLt
    calc (o : Nat) * W + j < o * W + W := by omega
      _ = (o + 1) * W := by ring
      _ ≤ M * W := Nat.mul_le_mul_right _ (by omega)
  rw [famAt_of_lt _ hlt]
  congr 1
  ext
  simp only [cellAt]
  ring

/-- A key index read from a table entry in `[0, K)`. -/
theorem key_of_entry {K : Nat} {x : Int} (h0 : 0 ≤ x) (h1 : x < K) :
    ∃ k : Fin K, ((k : Nat) : Int) = x :=
  ⟨⟨x.toNat, by omega⟩, by simp [Int.toNat_of_nonneg h0]⟩

/-- A label payload `x1 X^x2` with the requirement's label relation is the monomial `X^(s0 x0)`. -/
theorem label_msg {n : Nat} (s0 : Nat) {x0 x1 x2 : Int} (h1 : x1 * x1 = 1)
    (h2 : x2 * 2 + (1 - x1) * (n : Int) = x0 * (2 * (s0 : Int))) (hx0 : 0 ≤ x0) (he : 0 ≤ x2) (he2 : x2 < 2 * n) :
    message (n := n) x1 x2 = Xe (n := n) (s0 * x0.toNat) := by
  apply message_Xe he he2
  have hx1 : x1 = 1 ∨ x1 = -1 := by
    have : (x1 - 1) * (x1 + 1) = 0 := by ring_nf; linarith
    rcases mul_eq_zero.1 this with h | h
    · left; linarith
    · right; linarith
  have hk : ((s0 * x0.toNat : Nat) : Int) = s0 * x0 := by push_cast; rw [Int.toNat_of_nonneg hx0]
  rcases hx1 with rfl | rfl
  · left; refine ⟨rfl, ?_⟩; rw [hk]; linarith
  · right; refine ⟨rfl, ?_⟩; rw [hk]; linarith

/-- The value of a literal list selection lies in the list's range (`List.all` certificate). -/
theorem selectInt_range {lo hi : Int} {l : List Int} (h : l.all (fun x ↦ decide (lo ≤ x ∧ x < hi)) = true)
    (h0 : lo ≤ 0 ∧ 0 < hi) (s : Int) : lo ≤ MxxIR.selectInt s l ∧ MxxIR.selectInt s l < hi := by
  unfold MxxIR.selectInt
  cases hx : l[s.toNat]? with
  | none => simpa using h0
  | some x =>
    have hm : x ∈ l := List.mem_of_getElem? hx
    have := List.all_eq_true.1 h x hm
    simpa using this

/-- A key table's scale `a` moved into the unit pair (`hswitch` with a scale `c = a`). -/
theorem keyTableG_scale {q n : Nat} (a b : Int) (S : ExactMatrix q n 1 1) (u : ErrorPoly n) (c : Int) (hc : c = a) :
    keyTableG [a] [b] S 0 0 * reducePoly q n (1 * u) = keyTableG [1] [b] S 0 0 * reducePoly q n ((c : ErrorPoly n) * u) := by
  subst hc
  simp only [keyTableG, matrixMulScalarLeft, matrixPolynomial_single, one_mul, map_mul, map_intCast, Int.cast_one]
  ring

theorem polyNorm_one_le {n : Nat} (hn : 1 < n) : polyNorm (1 : ErrorPoly n) ≤ 1 := by
  simpa [Xe] using polyNorm_Xe (n := n) hn 0

theorem polyNorm_subst_one {n : Nat} (hn : 1 < n) (index : Nat) (hodd : Odd index) :
    polyNorm (substitution index hodd (1 : ErrorPoly n)) ≤ 1 := by
  simpa [Xe] using polyNorm_substitution_Xe_gen hn hodd 0

theorem binTable_const {q n P : Nat} (tv : Fin P → Int) (h : Int) (h0 : ∀ k : Fin P, tv k = if k.val % 2 = 1 then h else 0) :
    (fun k : Fin P ↦ ((tv k : Int) : ExactPoly q n)) = fun k ↦ reducePoly q n (Gen.binTable (n := n) P (fun _ ↦ h) k) := by
  funext k
  unfold Gen.binTable
  rw [h0 k]
  split_ifs <;> simp

theorem famAt_nonneg {N : Nat} (f : Fin N → Int) (h : ∀ c, 0 ≤ f c) (k : Nat) : 0 ≤ famAt f k := by
  unfold famAt; split_ifs <;> simp [h]

theorem prf_bucket_val {W : Nat} (s t a : Nat → Int) (o : Nat) (sel : Fin W)
    (hs : ∀ k : Fin W, s (o * W + k) = if k = sel then 1 else 0) :
    (∑ k ∈ Finset.range W, s ((Int.ofNat o * (W : Int) + Int.ofNat k).toNat) * a (t ((Int.ofNat o * (W : Int) + Int.ofNat k).toNat)).toNat) =
      a (t (o * W + sel)).toNat := by
  have e : ∀ k, ((Int.ofNat o * (W : Int) + Int.ofNat k).toNat) = o * W + k := fun k ↦ by
    simp only [Int.ofNat_eq_natCast]; omega
  simp only [e]
  exact RefreshTopProof.GenFinal.onehot_pick a s t o sel hs

theorem binTable_of {q n P : Nat} (tv : Fin P → Int) (g : Nat → Int) (h0 : ∀ k : Fin P, tv k = if k.val % 2 = 1 then g (k.val / 2) else 0) :
    (fun k : Fin P ↦ ((tv k : Int) : ExactPoly q n)) = fun k ↦ reducePoly q n (Gen.binTable (n := n) P g k) := by
  funext k
  unfold Gen.binTable
  rw [h0 k]
  split_ifs <;> simp

theorem Xe_mul_Xe {n : Nat} (a b : Nat) : Xe (n := n) a * Xe (n := n) b = Xe (n := n) (a + b) := by
  simp only [Xe, pow_add]

/-- Equal exponents give equal monomials (used instead of `congr`/`rfl`, which unfold `X ^ k` at large `n`). -/
theorem Xe_congr_nat {n a b : Nat} (h : a = b) : Xe (n := n) a = Xe (n := n) b := h ▸ rfl
'''

#: The column placements of the fresh and smudge summands (`GenOnlineSmudge.colPlace_eq_place`).
LEMMAS_COLUMNS = r'''
theorem blk_colPlace {n q C m i : Nat} (y : ExactMatrix q n 1 m) (cc : Fin C) :
    blk (colPlace q n C m i y) cc = if (cc : Nat) = i then y else 0 := by
  rw [colPlace_eq_place]
  exact blk_tensor_unitRow (tensor_place y) cc

theorem column_rel_of_colsum {n q m C F W : Nat} (hF : W * C ≤ F) {sm : ErrorPoly n}
    {fams : Tup (Fin F → ExactMatrix q n 1 m) 6} {final : Tup (ExactMatrix q n (1 * 1) (C * m)) 6}
    (hcol : ∀ c, final.get c = ∑ i ∈ Finset.range C, colPlace q n C m i (∑ j ∈ Finset.range W, famAt (fams.get c) (W * i + j)))
    {t : Fin F → Fin 2 → ExactMatrix q n 1 m} {B : Nat} (hmem : ∀ f, EncRel sm (Tup.ev fams f) (t f) B) (cc : Fin C) :
    EncRel sm (blkT final cc)
      (∑ j : Fin W, t ⟨W * cc + j, by have := cc.isLt; have := j.isLt; nlinarith⟩) (W * B) := by
  have hg : ∀ c, (blkT final cc).get c =
      ∑ j : Fin W, (Tup.ev fams ⟨W * cc + j, by have := cc.isLt; have := j.isLt; nlinarith⟩).get c := by
    intro c
    rw [blkT_get, hcol c, blk_sum, Finset.sum_eq_single (cc : Nat)]
    · rw [blk_colPlace, if_pos rfl, Finset.sum_range]
      refine Finset.sum_congr rfl fun j _ ↦ ?_
      rw [Tup.get_ev, famAt_of_lt _ (by have := cc.isLt; have := j.isLt; nlinarith)]
    · intro b _ hb
      rw [blk_colPlace, if_neg (Ne.symm hb)]
    · intro h
      exact absurd (Finset.mem_range.2 cc.isLt) h
  let g : Fin W → Fin F := fun j ↦ ⟨W * cc + j, by have := cc.isLt; have := j.isLt; nlinarith⟩
  have hsum := Rel.sum (Finset.univ : Finset (Fin W)) (A := fun j ↦ Enc.A (Tup.ev fams (g j)))
    (t := fun j ↦ t (g j)) (c := fun j ↦ Enc.c (Tup.ev fams (g j))) (B := fun _ ↦ B)
    (fun j _ ↦ hmem (g j))
  unfold EncRel
  rw [enc_sum_A _ _ hg, enc_sum_c _ _ hg]
  exact hsum.mono (le_of_eq (by simp))

'''


# --------------------------------------------------------------------------- the smudge summand

def chain_info(C, name):
    th = theorem_text(C.chains, name)
    head, concl, body = split_header(th)
    bs = binders(head)
    nh = node_haves(body)
    return bs, nh


def online_role(C, h):
    return C.roles[ON].get(f'parallel_parallel_generatedRoot_{C.O.inst.split("_")[-1]}_{h}', {}).get('role')



# --------------------------------------------------------------------------- requirement: labels

class Req:
    """The requirement facts (`GenRequirement.Facts`, argument externals) used here."""

    def __init__(self, C):
        rt = open(f'{C.pdir}/Gen_Requirement.lean').read()
        hdr = re.search(r'^structure Facts (.*) : Prop where', rt, re.M).group(1)
        xs = re.findall(r'\((x\d+) : ', hdr)
        calls = xmodel.parse_claim(C.gdir)
        req = next(c for c in calls if c.get('module') == 'Requirement_0')
        self.ext = {x: e.strip() for x, e in zip(xs, req['args'])}
        self.facts_term = 'GenRequirement.Facts ' + ' '.join(f'({e.strip()})' for e in req['args'][:len(xs)])
        self.fields = {}
        for m in re.finditer(r'^  (\w+) : (.*)$', rt[rt.index('structure Facts'):], re.M):
            self.fields[m.group(1)] = m.group(2)
        # label relation: famAt x0 = list ∧ x1 x1 = 1 ∧ x2 * 2 + (1 - x1) * N = x0 * M
        self.label = None
        for f, ty in self.fields.items():
            m = re.match(r'^∀ \w+ < (\d+), \(\(famAt (x\d+) \w+\) = \(MxxIR\.selectInt \(Int\.ofNat \w+\) \[([^\]]*)\]\)\) ∧ '
                         r'\(\(\(famAt (x\d+) \w+\) \* \(famAt \4 \w+\)\) = 1\) ∧ '
                         r'\(\(\(\(famAt (x\d+) \w+\) \* 2\) \+ \(\(1 - \(famAt \4 \w+\)\) \* (\d+)\)\) = \(\(famAt \2 \w+\) \* (\d+)\)\)$', ty)
            if m:
                self.label = dict(field=f, K=int(m.group(1)), a=self.ext[m.group(2)], alist=m.group(3),
                                  c=self.ext[m.group(4)], e=self.ext[m.group(5)], N=int(m.group(6)), M=int(m.group(7)))
        # broadcast and pinning loops: famAt (famAt X j) k = famAt (famAt X (sel j).toNat) k ; famAt (famAt X 0) k = famAt Y k
        self.bcast, self.pin = {}, {}
        for f, ty in self.fields.items():
            m = re.match(r'^∀ \w+ < (\d+), ∀ \w+ < (\d+), \(famAt \(famAt (x\d+) \w+\) \w+\) = \(famAt \(famAt \3 \(MxxIR\.selectInt \(Int\.ofNat \w+\) \[([^\]]*)\]\)\.toNat\) \w+\)$', ty)
            if m:
                self.bcast[self.ext[m.group(3)]] = (f, m.group(4))
            m = re.match(r'^∀ \w+ < (\d+), \(famAt \(famAt (x\d+) 0\) \w+\) = \(famAt (x\d+) \w+\)$', ty)
            if m:
                self.pin[self.ext[m.group(2)]] = (f, self.ext[m.group(3)])

    def label_source(self, ext):
        """The requirement x the per-instance label family `ext` is pinned to: (bcast field, zero list, pin field, x)."""
        b = self.bcast.get(ext)
        p = self.pin.get(ext)
        if not b or not p:
            raise Fail(f'label family {ext} not pinned by the requirement')
        if any(x.strip() != '0' for x in b[1].split(',')):
            raise Fail(f'label family {ext}: broadcast not to member 0')
        return b[0], b[1], p[0], p[1]


def hv_proj(C, ext):
    """`hv.2.….1` projection of the ValidExternals conjunct of `external.input_k` and its bounds."""
    k = ext.split('.')[-1]
    path, c = C.vproj[k]
    m = re.search(r'\(\((-?\d+) : Int\) ≤ .*?\) ≤ \((-?\d+) : Int\)\)', c)
    return path, (int(m.group(1)), int(m.group(2))) if m else (None, None)


# --------------------------------------------------------------------------- the chain emitter

STAGE_LETTER = {PRE: 'P', ENC: 'E', ON: 'O'}


class Chain:
    def __init__(self, G, name):
        self.G, self.C = G, G.C
        self.name = name
        self.bs, self.nh = chain_info(self.C, name)
        self.head = kind_of(name[len('chain_'):] if False else name.replace('chain_', 'x_'))[1]
        self.sub = {}
        self.types = {}
        for br, names, ty in self.bs:
            for nm in names:
                self.types[nm] = ty
        self.order = [nm for _, names, _ in self.bs for nm in names]
        self.norms = []

    # helpers
    def links(self, ty, stage=None):
        """rewrite list `[l1, l2, …]` turning the online wires of `ty` into `stage`'s wires."""
        C = self.C
        ws = online_wires(ty)
        srcs = link_sources(C, ws)
        if stage is None:
            cands = [s for s in srcs if s != ON]
            if not cands:
                return [], None
            stage = max(cands, key=lambda s: len(srcs[s]))
        terms = []
        for w in ws:
            t, sw = C.link_term(w, stage)
            if t:
                terms.append(t)
        return terms, stage

    def value_of(self, nm):
        return self.sub.get(nm, nm)

    def subst(self, text):
        names = sorted(self.sub, key=len, reverse=True)
        if not names:
            return text
        pat = re.compile(r"(?<![\w'.])(" + '|'.join(re.escape(x) for x in names) + r")(?![\w'])")
        return pat.sub(lambda m: self.sub[m.group(1)], text)


def online_hash_keys(G, ws):
    """Rewrites turning online key wires `ws` that the online instance hash-samples itself (the four
    components of one `hashPair`) into the encrypt stage's `hashPair` of the linked key."""
    if not ws:
        return []
    C = G.C
    ofacts = open(f'{C.gdir}/{ON}_Facts.lean').read()
    comp, key = {}, None
    for w in ws:
        f = find_field(ofacts, r'hashSample .* \(parallel_generatedRoot_\d+\.' + w + r' ctx\)$')
        if not f:
            raise Fail(f'online key {w} is neither linked nor hash-sampled')
        m = re.search(r'\.u64Le \((\d+)\)\]\) \(\(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\)\)', f[2])
        comp[int(m.group(1))] = f'F.{f[0]}.{f[1]}'
        key = m.group(2)
    if sorted(comp) != [0, 1, 2, 3]:
        raise Fail(f'online keys {ws}: components {sorted(comp)}')
    kt, _ = C.link_term(key, ENC)
    if not kt:
        raise Fail(f'online hash key {key} not linked to {ENC}')
    return [f'← hashPair_of_samples (pre := []) {comp[0]} {comp[1]} {comp[2]} {comp[3]}', kt]


def scope_sub(text, n):
    """D's chain text -> the summand theorem's names."""
    t = text.replace('(backend := backend)', '(backend := Backend.backend)')
    t = t.replace('(params := params)', f'(params := {ON_PARAMS})')
    t = t.replace('(i_0 := i_0)', '(i_0 := (iS : Nat))')
    t = t.replace('(hb := hb)', '(hb := rfl)')
    t = re.sub(r'(?<![\w.])backend\b(?! :=)', 'Backend.backend', t)
    t = re.sub(r'(?<![\w.])i_0\b(?! :=)', '(iS : Nat)', t)
    return t

def pre_ref(C, w):
    return f'({PRE}.{C.P.inst}.{w} cP)'


def enc_ref(C, w):
    return f'({ENC}.{C.E.inst}.{w} cE)'


def find_field(facts, pat):
    """(cK, field) of the Facts field whose type matches `pat`."""
    cur = None
    for ln in facts.split('\n'):
        m = re.match(r'^structure \S+\.Facts_(\d+) ', ln)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r'^  (h\w+) : (.*)$', ln)
        if m and cur is not None and re.search(pat, m.group(2)):
            return f'c{cur}', m.group(1), m.group(2)
    return None



def perm_of(idx, L):
    """The `perm` (and its `hindex : index i % (2·2^L) = 1 + 2·perm i` proof) of an odd index table `idx`
    (`fun i : Fin (2 ^ L) ↦ (selectInt i [..]).toNat`); `Equiv.refl` when the table is `1 + 2 i` mod 2^(L+1)."""
    tab = [int(x) for x in re.findall(r'-?\d+', re.search(r'\[(.*?)\]', idx).group(1))]
    W = 2 ** int(L)
    res = [x % (2 * W) for x in tab]
    if len(tab) != W or any(r % 2 != 1 for r in res) or sorted((r - 1) // 2 for r in res) != list(range(W)):
        raise Fail(f'index table is not an odd permutation mod {2 * W}')
    if all(r == 1 + 2 * i for i, r in enumerate(res)):
        return '(Equiv.refl _)'
    f = f'(fun i : Fin (2 ^ {L}) ↦ (⟨({idx} i % (2 * 2 ^ {L}) / 2) % 2 ^ {L}, Nat.mod_lt _ (by positivity)⟩ : Fin (2 ^ {L})))'
    return f'(Equiv.ofBijective {f} (by decide))'

NORM_TAC = ('(by intros; (try simp only [Xe_mul_Xe, mul_one, one_mul]); first | exact polyNorm_one_le (by norm_num) | '
            'exact polyNorm_subst_one (by norm_num) _ _ | exact polyNorm_Xe (by norm_num) _ | '
            'exact polyNorm_substitution_Xe_gen (by norm_num) _ _)')


def emit_chain(G, ch):
    """Values / proofs for every binder of chain `ch`; returns the body tactic lines."""
    C = G.C
    n = C.n
    sub = ch.sub
    ctx_names = {'backend', 'hashModel', 'params', 'i_0', 'ctx', 'outputs', 'F', 'hb', 'sm', 'sp', 'hsm'}
    req = G.req
    tys = ch.types
    names = [nm for nm in ch.order if nm not in ctx_names and not nm.startswith('Fact')]
    # pass 1: kinds independent of the bucket order
    for nm in names:
        k, tag = kind_of(nm)
        ty = scope_sub(tys[nm], n)
        if k == 'μ':
            sub[nm] = f'(1 : ErrorPoly {n})'
        elif k == 'α':
            L = req.label
            K = re.match(r'Fin (\d+) →', ty).group(1)
            s0 = L['M'] // 2
            sub[nm] = f'(fun k : Fin {K} ↦ Xe (n := {n}) ({s0} * (famAt {L["a"]} k).toNat))'
            G.alpha, G.s0, G.K = sub[nm], s0, int(K)
        elif k == 'hα':
            sub[nm] = ('(fun k ↦ by first | exact polyNorm_Xe (by norm_num) _ | '
                       '(simp only [Xe]; exact smulGain_root_pow (by norm_num) _))')
        elif k in ('hαμ', 'hμ'):
            sub[nm] = NORM_TAC
            ch.norms.append(nm)
        elif k == 'ha':
            terms, _ = ch.links(ty, ENC)
            enc_w = [C.link_term(w, ENC)[1] for w in online_wires(ty)]
            # the encrypt label entry loop producing the encodings (a Program in its own stages also
            # links its hash key, an input of both stages)
            loops = [int(w.split('_')[1]) for w in enc_w if w and G.enc.step_of_loop(int(w.split('_')[1]))[1]]
            if not loops:
                raise Fail(f'no encrypt step for the label encodings {enc_w}')
            loop = loops[0]
            sname, step = G.enc.step_of_loop(loop)
            G.enc_masks()
            key = f'lab{loop}'
            fld = find_field(C.efacts, r'parallel_parallel_generatedRoot_\d+_' + str(loop) + r' backend')
            body_ = fld[2][fld[2].index(', ') + 2:].rstrip(')')
            _, fargs = xmodel.app_split(body_)
            ins = [re.search(r'(w_\d+_\d+)', x).group(1) for x in xmodel.tuple_items(fargs[-2])]
            # variant 1 (`rel_1`, keys hash-sampled): inputs (s, key, 1, c, e), conclusions (G, rel, μ)
            v1 = 'EncLabelEntry.rel_1' in step
            cw, ew = (ins[3], ins[4]) if v1 else (ins[6], ins[7])
            G.label_need(cw, ew)
            G.pre(f'  have h{key} := {step}')
            if v1:
                G.pre(f'  choose G{key} hG{key} hrel{key} hmu{key} using h{key}')
                G.labmu = key
            else:
                G.pre(f'  choose G{key} hG{key} _ _ _ _ hrel{key} _ using h{key}')
            G.pre(f'  have hG{key}0 : ∀ k, G{key} k = G{key} 0 := fun k ↦ gadgetMatrixRuns_unique (hG{key} k) (hG{key} 0)')
            G.pre(f'  have he{key} : ∀ k, message (n := {n}) ({enc_ref(C, cw)} k) ({enc_ref(C, ew)} k) = {G.alpha} k := fun k ↦ by\n'
                  f'    rw [StageLinks.src_{ENC}_{cw} hinE hrootE, StageLinks.src_{ENC}_{ew} hinE hrootE]\n'
                  f'    exact label_payload hv hreq iS k')
            sub['G_' + tag] = f'(G{key} 0)'
            sub['hG_' + tag] = f'(hG{key} 0)'
            sub['Ba_' + tag] = '_'
            sub[nm] = f'(by intro k; rw [{", ".join(terms)}, ← hG{key}0 k, ← he{key} k]; exact hrel{key} k)'
        elif k == 'hpay':
            w = online_wires(ty)[0]
            t, ew = C.link_term(w, ENC)
            if t and G.labmu:
                # the payloads are the encrypt label entries' `μ` (a Program in its own stages)
                key = G.labmu
                sub[nm] = (f'(fun k ↦ by rw [{t}]; refine (hmu{key} k).trans ?_; dsimp only; '
                           f'rw [he{key} k])')
            else:
                G.hpay_need(w)
                sub[nm] = f'hpay_{w}'
    # pass 2: per-bucket and LUT binders in order
    for idx, nm in enumerate(names):
        k, tag = kind_of(nm)
        if nm in sub:
            continue
        ty = scope_sub(tys[nm], n)
        nxt = names[idx + 1] if idx + 1 < len(names) else ''
        nxt2 = names[idx + 2] if idx + 2 < len(names) else ''
        if k == "g'" and kind_of(nxt)[0] == "hg'" and kind_of(nxt2)[0] == 'g':
            # PRF top: g', g from the encrypt source encoding of the hs wires
            hs_nm = next(x for x in names[idx:] if kind_of(x)[0] == 'hs')
            hsty = scope_sub(tys[hs_nm], n)
            terms, _ = ch.links(hsty, ENC)
            wires = [C.link_term(w, ENC)[1] for w in online_wires(hsty)]
            c0 = [w for w in wires if w and G.enc.step_of_output(w)[0]]
            if not c0:
                raise Fail(f'{ch.name}: no encrypt source for {hs_nm}')
            sname, step = G.enc.step_of_output(c0[0])
            gw = re.search(rf'\({ENC}\.parallel_generatedRoot_\d+\.(w_\d+_\d+) cE\) FE\.c\d+\.h\w+ FE\.c\d+\.h\w+ \(c0 :=', step).group(1)
            f1 = find_field(C.efacts, r'centeredRebaseRuns \(parallel_generatedRoot_\d+\.w_\d+_\d+ ctx\) \(parallel_generatedRoot_\d+\.' + gw + r' ctx\)')
            gty = re.search(r'centeredRebaseRuns \(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\)', f1[2]).group(1)
            f2 = find_field(C.efacts, r'gadgetMatrixRuns backend \S+ \S+ \(parallel_generatedRoot_\d+\.' + gty + r' ctx\)')
            G.enc_masks()
            G.pre(f'  have htop{c0[0]} := {step}')
            G.seed_need(step, c0[0])
            # a Program in its own stages hash-samples the top's keys in the online instance as well
            terms += online_hash_keys(G, [w for w in online_wires(hsty) if not C.link_term(w, ENC)[0]])
            sub[nm] = enc_ref(C, gty)
            sub[nxt] = f'FE.{f2[0]}.{f2[1]}'
            sub[nxt2] = enc_ref(C, gw)
            sub[f'hg_{tag}'] = f'FE.{f1[0]}.{f1[1]}'
            sub[f'Bs_{tag}'] = '_'
            sub[hs_nm] = (f'(by rw [{", ".join(terms)}]; convert (config := {{ closePre := false, closePost := false }}) htop{c0[0]} using 4 <;> '
                          f'first | with_reducible rfl | exact (hseed{c0[0]}).symm | (rw [hseed{c0[0]}, mul_one, mul_one]; congr 1) | (congr 1))')
            continue
        if k == 'ν' and re.match(r'pm\d+$', tag or ''):
            # a root private key cell (preprocess `rootcell_*`): ν = its payload (= 1)
            h_nm = f'hrhs_{tag}'
            hty = scope_sub(tys[h_nm], n)
            src = link_sources(C, online_wires(hty))
            pre_w = [sw for (w, sw) in src.get(PRE, [])]
            rc = None
            for mrc in re.finditer(r'^theorem (rootcell_(w_\d+_\d+)) ', C.keys, re.M):
                th = theorem_text(C.keys, mrc.group(1))
                if all(re.search(r'\.' + w + r' ctx\)', th) for w in pre_w if w):
                    rc = mrc.group(1)
                    break
            if rc is None:
                raise Fail(f'{ch.name}: no root cell for {h_nm} ({pre_w})')
            rt = theorem_text(C.keys, rc)
            mm = re.search(r'\* message \(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\) \(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\) \* unitPair sp j\) • \(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\)', rt)
            cw, ew, gw = mm.groups()
            mg = re.search(r'unitPair sp j\) • (\S+)\) j \+ reduceMatrix', hty)
            gname = mg.group(1)
            G.pre_secret()
            G.pre_seed(cw, ew)
            sub[nm] = f'(1 : ErrorPoly {n})'
            sub[gname] = pre_ref(C, gw)
            G.pregad[gname] = gw
            sub[f'eB_{tag}'] = f'eR{tag}'
            sub[f'BB_{tag}'] = '_'
            sub[f'heB_{tag}'] = f'hbR{tag}'
            terms, _ = ch.links(hty, PRE)
            msc = re.search(r'reducePoly \S+ \d+ \((\(\(\(.*?\) : Nat\) : Int\)) \* ', hty)
            pcv = re.search(r'\(\(\((.*?) : Int\) : ErrorPoly \d+\) \* message', rt).group(1)
            sc = G.scale_cert(msc.group(1), f'({pcv})')
            G.pre(f'  obtain ⟨eR{tag}, hbR{tag}, heR{tag}⟩ := {PRE_KEYS}.{rc} FP hsp hP9')
            sub[h_nm] = (f"(by intro j; rw [{', '.join(terms)}]; have h := heR{tag} j; rw [hseedP{cw}] at h; "
                         f"convert (config := {{ closePre := false, closePost := false }}) h using 4 <;> first | with_reducible rfl | exact congrArg (fun x : Int ↦ (x : ErrorPoly {n})) {sc} | (congr 1))")
            continue
        if k == 'ν' and kind_of(nxt)[0] == 'hT':
            h_nm = next(x for x in names[idx:] if kind_of(x)[0] == 'hrhs')
            hty = scope_sub(tys[h_nm], n)
            src = link_sources(C, online_wires(hty))
            pre_w = [sw for (w, sw) in src.get(PRE, [])]
            loops = sorted({int(w.split('_')[1]) for w in pre_w}, key=lambda x: -x)
            rhs = None
            for L in loops:
                mm = re.search(r'^theorem (rhs_h' + str(L) + r'_\d+) ', C.keys, re.M)
                if mm:
                    rhs = mm.group(1)
                    break
            if rhs is None:
                raise Fail(f'{ch.name}: no rhs for {h_nm} ({pre_w})')
            rt = theorem_text(C.keys, rhs)
            mm = re.search(r'\* message \(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx i\) \(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx i\) \* unitPair sp j\) • \((rebasedGadget backend .*?)\) \+ reduceMatrix', rt)
            cw, ew, gad = mm.group(1), mm.group(2), mm.group(3)
            gad = qualify(gad, PRE, None).replace('backend', 'Backend.backend').replace('i_0', '(iS : Nat)')
            seg = next((e for (nmh, e) in ch.nh if re.search(r'\(ν := ' + re.escape(nm) + r'\)', e)), None)
            pp = re.search(r"\(p' := (.*?)\) \(ν :=", seg).group(1)
            G.pre_secret()
            bn = f'B{tag}'
            hpv = re.search(r"\(hp' : \(\(p' : Nat\) : Int\) = (.*?)\) :\n", rt).group(1)
            hpv = qualify(hpv, PRE, None).replace('i_0', '(iS : Nat)')
            ppc = G.scale_cert(f'((({scope_sub(pp, n)} : Nat)) : Int)', hpv)
            G.pre(f"  obtain ⟨eB{bn}, heB{bn}, hrhs{bn}⟩ := {PRE_KEYS}.{rhs} FP rfl hsp hP9 (p' := {scope_sub(pp, n)}) {ppc}")
            terms, _ = ch.links(hty, PRE)
            sub[nm] = f"(fun i' ↦ message (n := {n}) ({pre_ref(C, cw)} i') ({pre_ref(C, ew)} i'))"
            sub[f'hT_{tag}'] = f'({gad})'
            sub[f'eB_{tag}'] = f'eB{bn}'
            sub[f'BB_{tag}'] = '_'
            sub[h_nm] = f"(by intro i' j; rw [{', '.join(terms)}]; exact hrhs{bn} i' j)"
            sub[f'heB_{tag}'] = f'heB{bn}'
            G.buckets[tag] = dict(cw=cw, ew=ew, rhs=rhs)
            continue
        if k == 'sel':
            kty = scope_sub(tys[f'hkey_{tag}'], n)
            tw = online_wires(kty)[0]
            M, W = re.match(r'Fin (\d+) → Fin (\d+)', ty).groups()
            Kk = re.match(r'Fin \d+ → Fin (\d+)', scope_sub(tys[f'key_{tag}'], n)).group(1)
            info = G.buckets[tag]
            grp, b = G.onehot_of(info['rhs'])
            G.pre_secret()
            if grp not in G.oh_done:
                G.oh_done.add(grp)
                hs = ' '.join(f'hreq.{f}' for f in G.oh[grp]['fields'])
                G.pre(f'  have hoh{grp} := fun o : Fin {M} ↦ {WIRING}.onehot_{grp} hv {hs} iS.isLt o.isLt')
                G.pre(f'  choose sel{grp} hsel{grp} using hoh{grp}')
            ext_c = G.ext_of_pre(info['cw'])
            ext_e = G.ext_of_pre(info['ew'])
            pe, be = hv_proj(C, ext_e)
            if be != (0, 0):
                raise Fail(f'{ch.name}: exponent family {ext_e} not zero ({be})')
            G.pre(f"  have hsel{tag} : ∀ (o : Fin {M}) (j : Fin {W}), message (n := {n}) ({pre_ref(C, info['cw'])} (cellAt o j))\n"
                  f"      ({pre_ref(C, info['ew'])} (cellAt o j)) = if j = sel{grp} o {b} then 1 else 0 := by\n"
                  f"    intro o j\n"
                  f"    have h := hsel{grp} o {b} j\n"
                  f"    rw [StageLinks.src_{PRE}_{info['cw']} hinP hrootP, StageLinks.src_{PRE}_{info['ew']} hinP hrootP]\n"
                  f"    have he := {pe} iS (cellAt o j)\n"
                  f"    rw [show {ext_e} iS (cellAt o j) = 0 by omega]\n"
                  f"    apply message_onehot\n"
                  f"    rw [← h]\n"
                  f"    show _ = famAt (famAt {ext_c} (iS : Nat)) _\n"
                  f"    rw [famAt_of_lt _ iS.isLt]\n"
                  f"    exact (famAt_cell (M := {M}) (W := {W}) ({ext_c} iS) o j).symm")
            et, ext_t = G.ext_of_online(tw)
            pt, bt = hv_proj(C, ext_t)
            G.pre(f"  have hkey{tag} : ∀ o : Fin {M}, ∃ k : Fin {Kk}, ((k : Nat) : Int) = {C.O.inst}.{tw} ctx (cellAt o (sel{grp} o {b})) := by\n"
                  f"    intro o\n"
                  f"    rw [{et}]\n"
                  f"    have h := {pt} iS (cellAt o (sel{grp} o {b}))\n"
                  f"    exact key_of_entry h.1 (by have := h.2; push_cast; omega)\n"
                  f"  choose key{tag} hkey{tag} using hkey{tag}")
            sub[nm] = f'(fun o ↦ sel{grp} o {b})'
            sub[f'hsel_{tag}'] = f'(fun o ↦ hsel{tag} o)'
            sub[f'key_{tag}'] = f'key{tag}'
            sub[f'hkey_{tag}'] = f'hkey{tag}'
            continue
        if k == 'G' and kind_of(nxt)[0] == 'hG':
            q, nn, m = re.match(r'ExactMatrix (.+) (\d+) 1 (\d+)$', ty).groups()
            hty = scope_sub(tys[nxt], n)
            base, digits = re.match(r'gadgetMatrixRuns Backend\.backend (\S+) (\S+) ', hty).groups()
            cert = G.gadget_cert(q, nn, m, base, digits)
            sub[nm] = f'(gadgetOf Backend.backend {base} {digits} {q} {nn} {m})'
            sub[nxt] = f'(gadgetOf_runs {cert})'
            if kind_of(nxt2)[0] == 'GR':
                q2 = re.match(r'ExactMatrix (.+) \d+ 1 \d+$', scope_sub(tys[nxt2], n)).group(1)
                sub[nxt2] = f'(rebasedGadget Backend.backend {base} {digits} {q} {q2} {nn} {m})'
                sub[names[idx + 3]] = f'(rebasedGadget_runs {G.gt_cert(q2)})'
            continue
        if k == "g'" and kind_of(nxt)[0] == "hg'":
            q, nn, m = re.match(r'ExactMatrix (.+) (\d+) 1 (\d+)$', ty).groups()
            hty = scope_sub(tys[nxt], n)
            base, digits = re.match(r'gadgetMatrixRuns Backend\.backend (\S+) (\S+) ', hty).groups()
            cert = G.gadget_cert(q, nn, m, base, digits)
            sub[nm] = f'(gadgetOf Backend.backend {base} {digits} {q} {nn} {m})'
            sub[nxt] = f'(gadgetOf_runs {cert})'
            tgt0 = xmodel.strip_parens(re.match(r"centeredRebaseRuns \S+ (.*)$", scope_sub(tys[nxt2], n)).group(1))
            if tgt0 in tys and tgt0 not in sub:
                q2, nn2, m2 = re.match(r'ExactMatrix (.+) (\d+) 1 (\d+)$', scope_sub(tys[tgt0], n)).groups()
                sub[tgt0] = f'(rebasedGadget Backend.backend {base} {digits} {q} {q2} {nn} {m})'
            tgt = ch.subst(re.match(r"centeredRebaseRuns \S+ (.*)$", scope_sub(tys[nxt2], n)).group(1))
            mm = re.match(r'^\(*rebasedGadget Backend\.backend \S+ \S+ \S+ (\S+|\(.*?\)) ', tgt)
            tq = mm.group(1) if mm else None
            if tq is None:
                mp_ = re.match(rf'^\(*{PRE}\.parallel_generatedRoot_\d+\.(w_\d+_\d+) cP\)*$', tgt)
                if mp_:
                    pf = open(f'{C.gdir}/{PRE}_Facts.lean').read()
                    f1 = find_field(pf, r'centeredRebaseRuns \(parallel_generatedRoot_\d+\.w_\d+_\d+ ctx\) \(parallel_generatedRoot_\d+\.' + mp_.group(1) + r' ctx\)')
                    gx = re.search(r'centeredRebaseRuns \(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\)', f1[2]).group(1)
                    f2 = find_field(pf, r'gadgetMatrixRuns backend \S+ \S+ \(parallel_generatedRoot_\d+\.' + gx + r' ctx\)')
                    sub[nm] = pre_ref(C, gx)
                    sub[nxt] = f'FP.{f2[0]}.{f2[1]}'
                    sub[nxt2] = f'FP.{f1[0]}.{f1[1]}'
                    G.gadfacts[tgt0] = f'(rebasedGadget_eq FP.{f2[0]}.{f2[1]} FP.{f1[0]}.{f1[1]}).symm'
                    continue
                me = re.match(rf'^\(*{ENC}\.parallel_generatedRoot_\d+\.(w_\d+_\d+) cE\)*$', tgt)
                if not me:
                    raise Fail(f'{ch.name}: rebase target of {nxt2}: {tgt[:80]}')
                f1 = find_field(C.efacts, r'centeredRebaseRuns \(parallel_generatedRoot_\d+\.w_\d+_\d+ ctx\) \(parallel_generatedRoot_\d+\.' + me.group(1) + r' ctx\)')
                gx = re.search(r'centeredRebaseRuns \(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\)', f1[2]).group(1)
                f2 = find_field(C.efacts, r'gadgetMatrixRuns backend \S+ \S+ \(parallel_generatedRoot_\d+\.' + gx + r' ctx\)')
                sub[nm] = enc_ref(C, gx)
                sub[nxt] = f'FE.{f2[0]}.{f2[1]}'
                sub[nxt2] = f'FE.{f1[0]}.{f1[1]}'
                continue
            sub[nxt2] = f'(rebasedGadget_runs {G.gt_cert(tq)})'
            continue
        if k == 'hodd':
            G.pre(f'  have {nm}_{ch.name} : {ty} := by decide')
            sub[nm] = f'{nm}_{ch.name}'
            continue
        if k == 'Bh':
            sub[nm] = '_'
            continue
        if k == 'hhelp':
            terms, _ = ch.links(ty, ENC)
            ews = [C.link_term(w, ENC)[1] for w in online_wires(ty)]
            loops = sorted({int(w.split('_')[1]) for w in ews if w})
            step = None
            for L in loops:
                sname, step = G.enc.step_of_loop(L)
                if step and 'EncLutBranchKey' in step:
                    break
                step = None
            if not step:
                raise Fail(f'{ch.name}: no branch key for {nm}')
            G.enc_masks()
            G.pre(f'  have hbk{L} := {step}')
            pc = G.enc_pc(L)
            msc = re.search(r'reducePoly \S+ \d+ \((\(\(\(.*?\) : Nat\) : Int\)) \* unitPair', ty)
            extra = ''
            if msc and pc:
                sc_ = G.scale_cert(msc.group(1), pc)
                extra = f' | exact congrArg (fun x : Int ↦ (x : ErrorPoly {n})) {sc_} | exact {sc_}'
            gb = re.search(r'unitPair \(substitution .*\) j\) • \(?(\S+?)\)?\) !\[', ty)
            gfx = G.gadfacts.get(gb.group(1)) if gb else None
            if gfx:
                extra += f' | exact {gfx}'
            sub[nm] = (f'(by intro k; obtain ⟨Gq, Gb, hGq, hGb, hrel⟩ := hbk{L} k; rw [{", ".join(terms)}]; '
                       f'rw [← rebasedGadget_eq hGq hGb] at hrel; convert (config := {{ closePre := false, closePost := false }}) hrel using 5 <;> first | with_reducible rfl{extra} | (congr 1))')
            continue
        if k == "Δ'":
            sub[nm] = f'(1 : ErrorPoly {n})'
            continue
        if k == 'hswitch':
            terms, _ = ch.links(ty, PRE)
            pws = [C.link_term(w, PRE)[1] for w in online_wires(ty)]
            loops = sorted({int(w.split('_')[1]) for w in pws if w})
            sw = None
            for L in loops:
                mm = re.search(r'^theorem (switch_h' + str(L) + r'_\d+) ', C.wiring, re.M)
                if mm:
                    sw = mm.group(1)
                    break
            if sw is None:
                raise Fail(f'{ch.name}: no switch for {nm}')
            st = theorem_text(C.wiring, sw)
            mc = re.search(r'\(\(fun \(i : Fin (\d+)\) ↦ (.*?)\) k \*\n', st)
            ktxt = qualify(mc.group(2), PRE, None).replace('i_0', '(iS : Nat)')
            W_ = mc.group(1)
            hb_ = binders(split_header(st)[0])
            bty = {nms[0]: ty for br, nms, ty in hb_ if len(nms) == 1}
            G.pre_secret()
            # the chain's scale of the unit pair
            e_ = ty.index(' * unitPair sp j)) •')
            d_, s_ = 0, e_
            while True:
                s_ -= 1
                if ty[s_] == ')':
                    d_ += 1
                elif ty[s_] == '(':
                    if d_ == 0:
                        break
                    d_ -= 1
            scale = ch.subst(ty[s_ + 1:e_]).strip()
            unit = xmodel.strip_parens(scale) in (f'1 : ErrorPoly {n}', '1')
            if 'hg0' in bty:
                t0, tg = toks(bty['g0']), toks(bty["g'"])
                th0 = toks(bty['hg0'])
                q0 = qualify(t0[1], PRE, None).replace('i_0', '(iS : Nat)')
                q = qualify(tg[1], PRE, None).replace('i_0', '(iS : Nat)')
                nn, m = xmodel.strip_parens(tg[2]), xmodel.strip_parens(tg[4])
                base, digits = th0[2], th0[3]
                cert = G.gadget_cert(q0, nn, m, base, digits)
                gq = f'(rebasedGadget Backend.backend {base} {digits} {q0} {q} {nn} {m})'
                gargs = f'(gadgetOf_runs {cert}) (rebasedGadget_runs {G.gt_cert(q)})'
            else:
                tq, th_ = toks(bty["g'"]), toks(bty["hg'"])
                q = qualify(tq[1], PRE, None).replace('i_0', '(iS : Nat)')
                nn, m = xmodel.strip_parens(tq[2]), xmodel.strip_parens(tq[4])
                base, digits = th_[2], th_[3]
                cert = G.gadget_cert(q, nn, m, base, digits)
                gq = f'(gadgetOf Backend.backend {base} {digits} {q} {nn} {m})'
                gargs = f'(gadgetOf_runs {cert})'
            sub[f'gq_{tag}'] = gq
            sub[f'BB_{tag}'] = '_'
            G.pre(f'  obtain ⟨eS{tag}, hbS{tag}, heS{tag}⟩ := {WIRING}.{sw} FP hsp hP9 {gargs}')
            if unit:
                sub[f'C_{tag}'] = f'(fun (i : Fin {W_}) ↦ {ktxt})'
                sub[nm] = f'(by intro k; refine ⟨eS{tag} k, hbS{tag} k, fun j ↦ ?_⟩; rw [{", ".join(terms)}]; exact heS{tag} k j)'
            else:
                ma = re.match(r'^\(\(keyTableG \(\(\[(.*?)\]\) : List Int\) ', ktxt)
                a_txt = ma.group(1)
                k1 = ktxt[:ma.start(1)] + '(1)' + ktxt[ma.end(1):]
                c_txt = '(' + re.match(r'^(.*) : ErrorPoly \d+$', xmodel.strip_parens(scale)).group(1) + ')'
                sc = G.scale_cert(c_txt, a_txt)
                sub[f'C_{tag}'] = f'(fun (i : Fin {W_}) ↦ {k1})'
                sub[nm] = (f'(by intro k; refine ⟨eS{tag} k, hbS{tag} k, fun j ↦ ?_⟩; rw [{", ".join(terms)}]; '
                           f'have h := heS{tag} k j; beta_reduce at h; rw [keyTableG_scale _ _ _ _ _ {sc}] at h; exact h)')
            G.switch[tag] = dict(sw=sw)
            continue
    # LUT values and output gadgets
    for nm in names:
        k, tag = kind_of(nm)
        if k == 'hwalsh':
            ty = ch.subst(scope_sub(tys[nm], n))
            mm = re.match(r'^∀ i, (∑ k, .*) = reducePoly (.+?) (\d+) \(\S+ i\)$', ty)
            m1 = re.match(r'^(∑ k, .*) = reducePoly (.+?) (\d+) \S+$', ty)
            rv = G.readout_value(ch, tag, ty, mm or m1, bool(mm))
            if rv:
                sub[f'ν_{tag}'], sub[nm] = rv
                continue
            if mm:
                sub[f'ν_{tag}'] = f'(fun i ↦ centeredIntLift ({mm.group(1)}))'
                sub[nm] = f'(fun _ ↦ (reducePoly_centeredIntLift {G.gt_cert(mm.group(2))} (by norm_num) _).symm)'
            elif m1:
                sub[f'ν_{tag}'] = f'(centeredIntLift ({m1.group(1)}))'
                sub[nm] = f'(reducePoly_centeredIntLift {G.gt_cert(m1.group(2))} (by norm_num) _).symm'
            else:
                raise Fail(f'{ch.name}: hwalsh shape {ty[-80:]}')
        if k == 'hG' and nm not in sub and f'gq_{tag}' in sub:
            tk = toks(sub[f'gq_{tag}'])
            sub[nm] = f'(gadgetOf_runs {G.gadget_cert(tk[4], tk[5], tk[6], tk[2], tk[3])})'
    for nm in ch.norms:
        ty = ch.subst(scope_sub(tys[nm], n))
        hs, hx = 'substitution' in ty, 'Xe' in ty
        lem = {(True, True): 'polyNorm_substitution_Xe_gen (by norm_num) _ _', (True, False): 'polyNorm_subst_one (by norm_num) _ _',
               (False, True): 'polyNorm_Xe (by norm_num) _', (False, False): 'polyNorm_one_le (by norm_num)'}[(hs, hx)]
        sub[nm] = f'(by intros; (try simp only [Xe_mul_Xe, mul_one, one_mul]); exact {lem})'
    missing = [nm for nm in names if nm not in sub]
    if missing:
        G.notes.append(f'{ch.name}: unresolved binders {missing[:12]}{" …" if len(missing) > 12 else ""}')
    body = []
    for nmh, e in ch.nh:
        body.append(f'  node_have {nmh} := {ch.subst(scope_sub(e, n))}')
    return body, missing


def toks(s):
    return [x for x in xmodel.split_top(xmodel.strip_parens(s.strip()), ' ') if x]


class Gen:
    def __init__(self, C):
        self.C = C
        self.enc = EncFacts(C)
        self.req = Req(C)
        self.pre_lines = []
        self.done = set()
        self.cert = []
        self.notes = []
        self.buckets = {}
        self.switch = {}
        self.oh_done = set()
        self.pregad = {}
        self.readouts = {}
        self.gadfacts = {}
        self.labmu = None      # the label entries' `μ` facts of a Program in its own stages
        self.top = []          # top-level theorems (label payload)
        self._gc = {}
        self.oh = self.parse_onehot()
        self.secret_terms()

    def pre(self, s):
        if s not in self.done:
            self.done.add(s)
            self.pre_lines.append(s)

    # ---- secrets
    def secret_terms(self):
        C = self.C
        hdr = split_header(theorem_text(C.enc, 'roles'))[0]
        self.enc_secret = dict((v, w) for w, v in re.findall(
            r'\(h_(w_\d+_\d+) : parallel_generatedRoot_\d+\.w_\d+_\d+ ctx = fun _ _ ↦ reducePoly \S+ \d+ (sm|sp)\)', hdr))
        self.stage_of = {}
        for v, w in self.enc_secret.items():
            m = re.search(rf'theorem src_{ENC}_' + w + r' .*?\n.*?\n.*?\n    \S+ cM = \((.*?)\) :=', C.links_text)
            self.stage_of[v] = m.group(1)
        m = re.search(r'\(h_(w_\d+_\d+) : parallel_generatedRoot_\d+\.w_\d+_\d+ ctx = fun _ _ ↦ reducePoly \S+ \d+ sp\)', C.keys)
        self.pre_sp = m.group(1)
        mm = re.search(rf'theorem src_{PRE}_' + self.pre_sp + r' .*?\n.*?\n.*?\n    \S+ cM = \((.*?)\) :=', C.links_text)
        if mm.group(1) != self.stage_of['sp']:
            raise Fail('preprocess and encrypt payload secrets differ')

    def enc_masks(self):
        C = self.C
        n = C.n
        for v in ('sm', 'sp'):
            w = self.enc_secret[v]
            self.pre(f'  have h_{w} : {ENC}.{C.E.inst}.{w} cE = fun _ _ ↦ reducePoly _ {n} {v} :=\n'
                     f'    (StageLinks.src_{ENC}_{w} hinE hrootE).trans h{v}E')
        for ln in self.enc.mask_lines:
            self.pre(self.enc.rescope(ln))

    def pre_secret(self):
        C = self.C
        w = self.pre_sp
        self.pre(f'  have hP9 : {PRE}.{C.P.inst}.{w} cP = fun _ _ ↦ reducePoly _ {C.n} sp :=\n'
                 f'    (StageLinks.src_{PRE}_{w} hinP hrootP).trans hspE')

    # ---- certificates
    def gadget_cert(self, q, n, m, base, digits):
        q = q.strip()
        key = (q, str(n), str(m), base, digits)
        if key in self._gc:
            return self._gc[key]
        name = f'gex_{len(self._gc)}'
        idep = '(iS : Nat)' in q
        qq = q.replace('(iS : Nat)', 'i_0')
        if idep:
            self.cert.append(f'''theorem {name} (i_0 : Nat) [h : Fact (i_0 < S)] :
    ∃ G : ExactMatrix {qq} {n} 1 {m}, gadgetMatrixRuns Backend.backend {base} {digits} G := by
  have := h.out
  interval_cases i_0 <;> exact gadget_exists ⟨_, by simp [Backend.backend]; rfl, rfl, rfl, rfl⟩
''')
            ref = f'({name} (iS : Nat))'
        else:
            self.cert.append(f'''theorem {name} :
    ∃ G : ExactMatrix {qq} {n} 1 {m}, gadgetMatrixRuns Backend.backend {base} {digits} G :=
  gadget_exists ⟨_, by simp [Backend.backend]; rfl, rfl, rfl, rfl⟩
''')
            ref = name
        self._gc[key] = ref
        return ref

    def enc_pc(self, loop):
        """The scale `pc` of an encrypt LUT branch key loop (its Comp parameter `p21 = [pc]`)."""
        sc = open(f'{self.C.pdir}/Gen_{ENC}_Scopes.lean').read()
        m = re.search(r'^theorem (parallel_parallel_generatedRoot_\d+_' + str(loop) + r')\.sem .*\n.*\n    (EncLutBranchKey\.Comp\w*) (.*) backend hashModel', sc, re.M)
        if not m:
            return None
        tk = [x for x in xmodel.split_top(m.group(3), ' ') if x]
        lst = xmodel.strip_parens(tk[-1])
        mm = re.match(r'^\[(.*)\]$', lst)
        if not mm:
            return None
        pc = xmodel.strip_parens(mm.group(1))
        return '(' + qualify(pc, ENC, None).replace('i_0', '(iS : Nat)') + ')'

    def readout_value(self, ch, tag, ty, m, per):
        """For an exponent-table (readout) LUT on a monomial PRF output: the value `ν = X^(s1 readoutExpG P W z)`
        and its `hwalsh` (`readout_value_of_walsh`), else None."""
        C, n = self.C, self.C.n
        ctext = ch.sub.get(f'C_{tag}', '')
        if 'keyTermE' not in ctext:
            return None
        mm = re.match(r'^\(fun \(i : Fin (\d+)\) ↦ (.*)\)$', ctext, re.S)
        W, kt = int(mm.group(1)), mm.group(2)
        L = W.bit_length() - 1
        if (1 << L) != W:
            return None
        inner = xmodel.strip_parens(kt)
        inner = inner[:inner.rindex(' 0 0')]
        it = toks(inner)
        # (keyTableG A B S : ExactMatrix Q N 1 1)
        b = re.match(r'^\(\(\[(.*)\]\) : List Int\)$', it[2]).group(1)
        b = xmodel.strip_parens(b)
        q = it[6]
        ssum = xmodel.strip_parens(it[3])
        tt = toks(re.match(r'^∑ k ∈ Finset\.range \(\(\d+\) : Int\)\.toNat, (.*)$', ssum, re.S).group(1))
        sig, o, tv, s = tt[1], int(xmodel.strip_parens(tt[2]).split(':')[0].strip(' ()')), tt[3], int(re.search(r'(\d+)', tt[5]).group(1))
        if n % o:
            return None
        s1, P = o, n // o
        hw = ch.types[f'hwalsh_{tag}']
        mu = re.search(r'\(hodd_\w+ k\) \(', hw)
        if not mu:
            return None
        j0 = mu.end() - 1
        mu_txt = ch.subst(scope_sub(hw[j0:xmodel.matching(hw, j0) + 1], n))
        keys = re.findall(r'\bkey(h\d+)\b', mu_txt)
        a = self.req.label['a']
        z = ' + '.join(f'(famAt {a} (key{k} i)).toNat' for k in keys) or '0'
        if per:
            mu_txt = re.sub(r' i\)$', ' i)', mu_txt)
        if ('rd', ctext) in self._gc:
            name = self._gc[('rd', ctext)]
            fresh = False
        else:
            name = f'rd_{len(self._gc)}'
            self._gc[('rd', ctext)] = name
            fresh = True
        sigf = sig.replace('((i : Nat))', '(i : Nat)')
        idx = f'(fun i : Fin (2 ^ {L}) ↦ ({sigf.replace("(i : Nat)", "((i : Fin (2 ^ " + str(L) + ")) : Nat)")}).toNat)'
        tab = [int(x) for x in re.findall(r'-?\d+', re.search(r'\[(.*?)\]', sig).group(1))]
        res = [x % (2 * W) for x in tab]
        if any(r % 2 != 1 for r in res) or sorted((r - 1) // 2 for r in res) != list(range(W)):
            return None
        if all(r == 1 + 2 * i for i, r in enumerate(res)):
            perm = '(Equiv.refl _)'
        else:
            f = f'(fun i : Fin (2 ^ {L}) ↦ (⟨({idx} i % (2 * 2 ^ {L}) / 2) % 2 ^ {L}, Nat.mod_lt _ (by positivity)⟩ : Fin (2 ^ {L})))'
            perm = f'(Equiv.ofBijective {f} (by decide))'
        bb = b.replace('(iS : Nat)', 'i_0')
        qq = q.replace('(iS : Nat)', 'i_0')
        kti = kt.replace('(iS : Nat)', 'i_0')
        idep = 'i_0' in bb + qq + kti
        hdr = f'(i_0 : Nat) [h : Fact (i_0 < S)] ' if idep else ''
        cases = '  have := h.out\n  interval_cases i_0 <;> decide\n' if idep else '  decide\n'
        app = ' (iS : Nat)' if idep else ''
        tvf = tv
        (self.cert if fresh else []).append(f'''theorem {name}_nonneg {hdr}: 0 ≤ (({bb} : Int)) := by
{cases}
set_option maxRecDepth 65536 in
theorem {name}_inv {hdr}: ((({bb} : Int)).toNat * 2 ^ {L}) % {qq} = 1 := by
{cases}
theorem {name}_hinv {hdr}:
    (((({bb} : Int)).toNat : Nat) : ExactPoly {qq} {n}) * ((2 ^ {L} : Nat) : ExactPoly {qq} {n}) = 1 :=
  cast_inverse {self.gt_cert(q).replace('(iS : Nat)', 'i_0')} ({name}_inv{' i_0' if idep else ''})

theorem {name}_hC {hdr}(i : Fin (2 ^ {L})) :
    (fun (i : Fin {W}) ↦ {kti}) i =
      lutFourierCoefficient (2 ^ {L}) (X {qq} ^ {s}) (((({bb} : Int)).toNat : Nat) : ExactPoly {qq} {n})
        (fun k : Fin (2 ^ {L}) ↦ reducePoly {qq} {n} (Xe (n := {n}) ({s1} * readoutVal ({P} / 2) (2 ^ {L}) k))) ({idx} i) := by
  have hb : (((({bb} : Int)).toNat : Nat) : ExactPoly {qq} {n}) = (({bb} : Int) : ExactPoly {qq} {n}) := by
    rw [← Int.cast_natCast, Int.toNat_of_nonneg ({name}_nonneg{' i_0' if idep else ''})]
  refine ({WIRING}.keyTableE_walsh_int (q := {qq}) (n := {n}) (s := {s}) (W := {W}) (o := {o})
    {self.gt_cert(q).replace('(iS : Nat)', 'i_0')} (by norm_num) (σ := {sig.replace('((i : Nat))', '(i : Nat)')}) (by revert i; decide) (by norm_num) 1 ({bb}) {tv.replace('(iS : Nat)', 'i_0')} (by decide)).trans ?_
  rw [Int.cast_one, one_mul, hb]
  congr 1
  funext k
  congr 2
  revert k
  decide
''')
        sp_ = f'reducePoly {q} {n}'
        hmu = ('(by (try simp only [Xe_mul_Xe, mul_one, one_mul]); exact congrArg (reducePoly _ _) (Xe_congr_nat (by ring)))'
               if 'Xe' in mu_txt else '(by simp [Xe])')
        nu = f'Xe (n := {n}) ({s1} * readoutExpG {P} (2 ^ {L}) ({z}))'
        if per:
            nuv = f'(fun i ↦ {nu})'
            pf = (f'(fun i ↦ by\n'
                  f'      have hv := readout_value_of_walsh (s0 := {s}) (s1 := {s1}) (log := {L}) (P := {P}) (by norm_num) (by norm_num) (by norm_num)\n'
                  f'        ({name}_hinv{app}) {idx} (by intro i; revert i; decide) {perm} (by intro i; revert i; decide) ({name}_hC{app})\n'
                  f'        (μ := {mu_txt}) (z := {z}) {hmu} (ν := centeredIntLift _) (reducePoly_centeredIntLift {self.gt_cert(q)} (by norm_num) _).symm\n'
                  f'      rw [reducePoly_centeredIntLift] at hv\n      · exact hv\n      · exact {self.gt_cert(q)}\n      · norm_num)')
        else:
            nuv = f'({nu})'
            pf = (f'(by\n'
                  f'      have hv := readout_value_of_walsh (s0 := {s}) (s1 := {s1}) (log := {L}) (P := {P}) (by norm_num) (by norm_num) (by norm_num)\n'
                  f'        ({name}_hinv{app}) {idx} (by intro i; revert i; decide) {perm} (by intro i; revert i; decide) ({name}_hC{app})\n'
                  f'        (μ := {mu_txt}) (z := {z}) {hmu} (ν := centeredIntLift _) (reducePoly_centeredIntLift {self.gt_cert(q)} (by norm_num) _).symm\n'
                  f'      rw [reducePoly_centeredIntLift] at hv\n      · exact hv\n      · exact {self.gt_cert(q)}\n      · norm_num)')
        self.readouts[tag] = dict(s0=s, s1=s1, P=P, L=L, z=z, q=q, name=name, idx=idx, perm=perm, app=app)
        return nuv, pf

    def scale_cert(self, c, a):
        """A proof of `c = a` (Int; per-instance values by `interval_cases`)."""
        key = ('scale', c, a)
        if key in self._gc:
            return self._gc[key]
        name = f'scale_{len(self._gc)}'
        cc, aa = c.replace('(iS : Nat)', 'i_0'), a.replace('(iS : Nat)', 'i_0')
        if 'i_0' in cc + aa:
            self.cert.append(f'''theorem {name} (i_0 : Nat) [h : Fact (i_0 < S)] :
    ({cc} : Int) = {aa} := by
  have := h.out
  interval_cases i_0 <;> decide
''')
            ref = f'({name} (iS : Nat))'
        else:
            self.cert.append(f'''theorem {name} : ({cc} : Int) = {aa} := by decide
''')
            ref = name
        self._gc[key] = ref
        return ref

    def gt_cert(self, q):
        q = xmodel.strip_parens(q.strip())
        m = re.match(r'^(?:(Stage_\w+)\.)?(mod_\d+) \(\(?Int\.ofNat \(iS : Nat\)\)?\)$', q)
        if m:
            M = m.group(1) or ON
            return f'(RefreshTopProof.GenInst.Certs.{M}.{m.group(2)}_gt (iS : Nat))'
        return '(by decide)'

    # ---- externals
    def src_rhs(self, M, w):
        m = re.search(r'theorem src_' + M + '_' + w + r' .*?\n.*?\n.*?\n    \S+ cM = (.*?) :=', self.C.links_text)
        return m.group(1) if m else None

    def ext_of_pre(self, w):
        r = self.src_rhs(PRE, w)
        m = re.fullmatch(r'\((external\.input_\d+)\) i', r or '')
        if not m:
            raise Fail(f'preprocess {w} is not a member family ({r})')
        return m.group(1)

    def ext_of_online(self, w):
        C = self.C
        m = re.search(rf'theorem ext_{ON}_' + w + r' .*?\n    \S+ cB = \((external\.input_\d+)\) i :=', C.links_text, re.S)
        if m and re.search(rf'^theorem ext_{ON}_' + w + ' ', C.links_text, re.M):
            th = theorem_text(C.links_text, f'ext_{ON}_{w}')
            e = re.search(r'cB = \((external\.input_\d+)\) i :=', th).group(1)
            hy = 'hinA hrootA houtA hinstA hinB hrootB'
            src = re.search(r'\{rA : (Stage_\w+)\.generatedRoot', th).group(1)
            a = STAGE_LETTER[src]
            return f'StageLinks.ext_{ON}_{w} hin{a} hroot{a} hout{a} hinst{a} hinO hrootO', e
        r = self.src_rhs(ON, w)
        mm = re.fullmatch(r'\((external\.input_\d+)\) i', r or '')
        if mm:
            return f'StageLinks.src_{ON}_{w} hinO hrootO', mm.group(1)
        raise Fail(f'online table {w} has no external source')

    # ---- one-hot groups (Gen_Wiring)
    def parse_onehot(self):
        out = {}
        for m in re.finditer(r'/-- The one-hot selector rows of the PRF with key cells (.*?) \(buckets in that\norder; `(\d+)` cells per member, `(\d+)` members\): (.*?) \(`GenRequirement\.Facts`\)\. -/\ntheorem onehot_(\S+) ',
                             self.C.wiring, re.S):
            keys = re.findall(r'`keys_(h\d+_\d+)`', m.group(1))
            fields = re.findall(r'`hreq\.(\w+)`', m.group(4))
            out[m.group(5)] = dict(keys=keys, fields=fields)
        return out

    def onehot_of(self, rhs):
        tag = rhs[len('rhs_'):]
        for g, d in self.oh.items():
            if tag in d['keys']:
                return g, d['keys'].index(tag)
        raise Fail(f'no one-hot group for {rhs}')

    # ---- labels
    def label_need(self, cw, ew):
        if any('theorem label_payload' in t for t in self.top):
            return
        C, R = self.C, self.req
        L = R.label
        ec = self.src_rhs(ENC, cw)
        ee = self.src_rhs(ENC, ew)
        ec = re.fullmatch(r'\((external\.input_\d+)\) i', ec).group(1)
        ee = re.fullmatch(r'\((external\.input_\d+)\) i', ee).group(1)
        bc, zc, pc, xc = R.label_source(ec)
        be, ze, pe, xe = R.label_source(ee)
        if xc != L['c'] or xe != L['e']:
            raise Fail('label families are not the requirement label relation')
        pa, _ = hv_proj(C, L['a'])
        pee, (lo, hi) = hv_proj(C, ee)
        s0 = L['M'] // 2
        n = C.n
        if L['N'] != n:
            raise Fail('label relation period')
        self.top.append(f'''/-- The encrypt label payloads `c X^e` (members `j`) are `X^(s₀ a_k)` (the requirement's label relation
`{L['field']}`, broadcast `{bc}`/`{be}`, pinned `{pc}`/`{pe}`). -/
theorem label_payload {{external : ExternalInputs}} (hv : ValidExternals external) (hreq : {R.facts_term})
    (j : Fin S) (k : Fin {L['K']}) :
    message (n := {n}) ({ec} j k) ({ee} j k) = Xe (n := {n}) ({s0} * (famAt {L['a']} k).toNat) := by
  have ec := hreq.{bc} j j.2 k k.2
  have ee := hreq.{be} j j.2 k k.2
  have z : ∀ j : Fin S, (MxxIR.selectInt (Int.ofNat (j : Nat)) [{zc}]).toNat = 0 := by decide
  have z' : ∀ j : Fin S, (MxxIR.selectInt (Int.ofNat (j : Nat)) [{ze}]).toNat = 0 := by decide
  rw [z j, hreq.{pc} k k.2, famAt_of_lt _ j.2, famAt_of_lt _ k.2] at ec
  rw [z' j, hreq.{pe} k k.2, famAt_of_lt _ j.2, famAt_of_lt _ k.2] at ee
  obtain ⟨-, h1, h2⟩ := hreq.{L['field']} k k.2
  have ec' : {ec} j k = famAt {xc} k := ec
  have ee' : {ee} j k = famAt {xe} k := ee
  have ba : 0 ≤ famAt {L['a']} (k : Nat) := by rw [famAt_of_lt _ k.2]; exact ({pa} k).1
  have be := {pee} j k
  rw [ee'] at be
  rw [ec', ee']
  exact label_msg {s0} h1 (by push_cast; linarith) ba be.1 (by have := be.2; push_cast; omega)
''')

    def hpay_need(self, w):
        C, R = self.C, self.req
        n = C.n
        fld = find_field(open(f'{C.gdir}/{ON}_Facts.lean').read(),
                         r'\(\(parallel_generatedRoot_\d+\.' + w + r' ctx\) i\)\)$')
        loop = re.search(r'(parallel_parallel_generatedRoot_\d+_\d+) backend', fld[2]).group(1)
        sc = open(f'{C.gdir}/{ON}.lean').read()
        d = sc[sc.index(f'def {loop} '):]
        d = d[:d.index('\n\n')]
        lists = re.findall(r':= \(MxxIR\.selectInt \(\(Int\.ofNat i_1\)\) \[(.*?)\]\)', d)
        cl, el = lists[0], lists[1]
        cname = re.search(r'(parallel_parallel_generatedRoot_\d+_\d+\.constraints_\d+) i_1', d).group(1)
        ctext = sc[sc.index(f'abbrev {cname} '):]
        ctext = ctext[:ctext.index('\n\n')]
        nconj = ctext.count(' ∧\n')
        proj = '.2' * nconj
        L = R.label
        s0 = L['M'] // 2
        self.pre(f'''  have hpay_{w} : ∀ k : Fin {L['K']}, {C.O.inst}.{w} ctx k = fun _ _ ↦ reducePoly _ {n} ({self.alpha} k) := by
    intro k
    have h := (F.{fld[0]}.{fld[1]} k){proj}
    rw [h, multiplyMonomial_liftInteger]
    have hl := RefreshTopProof.GenOnlineValues.label_rel k
    have h7 := (hreq.{L['field']} k k.2).1
    have hb := selectInt_range (lo := 0) (hi := 2 * {n}) (l := [{el}]) (by decide) (by norm_num) (Int.ofNat (k : Nat))
    have hb7 := selectInt_range (lo := 0) (hi := {2 ** 62}) (l := [{L['alist']}]) (by decide) (by norm_num) (Int.ofNat (k : Nat))
    show _ = fun _ _ ↦ reducePoly _ {n} (Xe (n := {n}) ({s0} * (famAt {L['a']} k).toNat))
    rw [h7]
    congr 2
    funext _ _
    congr 1
    apply message_Xe hb.1 (by exact_mod_cast hb.2)
    have hk : (({s0} * (MxxIR.selectInt (Int.ofNat (k : Nat)) [{L['alist']}]).toNat : Nat) : Int) =
        {s0} * MxxIR.selectInt (Int.ofNat (k : Nat)) [{L['alist']}] := by
      push_cast; rw [Int.toNat_of_nonneg hb7.1]
    rw [hk]; exact hl''')

    def pre_seed(self, cw, ew):
        """`hseedP<cw>`: a preprocess root cell's payload `message c e` is `1` (ValidExternals: c = 1, e = 0)."""
        C = self.C
        ec = re.fullmatch(r'\((external\.input_\d+)\) i', self.src_rhs(PRE, cw)).group(1)
        ee = re.fullmatch(r'\((external\.input_\d+)\) i', self.src_rhs(PRE, ew)).group(1)
        pc, bc = hv_proj(C, ec)
        pe, be = hv_proj(C, ee)
        if bc != (1, 1) or be != (0, 0):
            raise Fail(f'root cell payload bounds {bc} {be}')
        self.pre(f'''  have hseedP{cw} : message (n := {C.n}) {pre_ref(C, cw)} {pre_ref(C, ew)} = 1 := by
    rw [StageLinks.src_{PRE}_{cw} hinP hrootP, StageLinks.src_{PRE}_{ew} hinP hrootP]
    have b1 := {pc} iS
    have b2 := {pe} iS
    rw [show {ec} iS = 1 by omega, show {ee} iS = 0 by omega, message_zero]; simp''')

    def seed_need(self, step, c0):
        """`hseed<c0>`: the PRF top's payload `message c e` is `1` (ValidExternals: c = 1, e = 0)."""
        C = self.C
        iw = C.E.iw
        seen, todo, mm = set(), [c0], None
        while todo:
            w = todo.pop()
            if w in seen or w not in iw:
                continue
            seen.add(w)
            t = iw[w].get('term', '')
            m = re.match(r'^MxxRuntime\.multiplyMonomial (w_\d+_\d+) (w_\d+_\d+)$', t)
            if m:
                a = iw.get(m.group(1), {}).get('term', '')
                ma = re.match(r'^MxxRuntime\.liftInteger (w_\d+_\d+)$', a)
                if ma:
                    mm = (ma.group(1), m.group(2))
                    break
            if iw[w].get('src') == 'let':
                todo += re.findall(r'\bw_\d+_\d+\b', t)
        if not mm:
            raise Fail(f'no payload of the top {c0}')
        cw, ew = mm
        ec = re.fullmatch(r'\((external\.input_\d+)\) i', self.src_rhs(ENC, cw)).group(1)
        ee = re.fullmatch(r'\((external\.input_\d+)\) i', self.src_rhs(ENC, ew)).group(1)
        pc, bc = hv_proj(C, ec)
        pe, be = hv_proj(C, ee)
        if bc != (1, 1) or be != (0, 0):
            raise Fail(f'top payload bounds {bc} {be}')
        self.pre(f'''  have hseed{c0} : message (n := {C.n}) {enc_ref(C, cw)} {enc_ref(C, ew)} = 1 := by
    rw [StageLinks.src_{ENC}_{cw} hinE hrootE, StageLinks.src_{ENC}_{ew} hinE hrootE]
    have b1 := {pc} iS
    have b2 := {pe} iS
    rw [show {ec} iS = 1 by omega, show {ee} iS = 0 by omega, message_zero]; simp''')


# --------------------------------------------------------------------------- theorem assembly

def ctx_binders(C):
    out = []
    for M, a, cn, on, fn in ((PRE, 'P', 'cP', 'oP', 'FP'), (ENC, 'E', 'cE', 'oE', 'FE'),
                             (ON, 'O', 'ctx', 'outputs', 'F')):
        st = C.stages[M]
        out.append(f'''    {{r{a} : {M}.generatedRoot.Ctx Backend.backend hashModel {st.params}}}
    {{{cn} : {M}.{st.inst}.Ctx Backend.backend hashModel {st.params} (iS : Nat)}} {{{on}}} ({fn} : {M}.{st.inst}.Facts {cn} {on})
    (hin{a} : {cn}.inputs = StageLinks.{M}.rootArgs r{a} iS) (hroot{a} : r{a}.inputs = StageLinks.{M}.claimArgs external execution)
    (hout{a} : {st.stage_out} = {M}.generatedRoot.outs r{a}) (hinst{a} : StageLinks.{M}.rootOuts r{a} iS = {M}.{st.inst}.outs {cn})''')
    return '\n'.join(out)


def theorem_head(G, name, doc):
    C = G.C
    n = C.n
    return f'''/-- {doc} -/
theorem {name} {{hashModel : HashModel}} {{external : ExternalInputs}} {{execution : Execution}} {{iS : Fin S}}
    (hv : ValidExternals external) (hreq : {G.req.facts_term})
    {{sm sp : ErrorPoly {n}}} (hsm : polyNorm sm ≤ 1) (hsp : polyNorm sp ≤ 1)
    (hsmE : {G.stage_of['sm']} = fun _ _ ↦ reducePoly _ {n} sm) (hspE : {G.stage_of['sp']} = fun _ _ ↦ reducePoly _ {n} sp)
{ctx_binders(C)} :'''


def smudge_summand(G, ch, body):
    C = G.C
    n = C.n
    sm = C.smudge
    sname = re.search(r'^theorem (smudge_h\d+_\d+) ', sm, re.M).group(1)
    stag = sname[len('smudge_'):]
    st = theorem_text(sm, sname)
    m = re.search(r'∃ es : ErrorMatrix (\d+) \(1 \* 1\) \((\d+) \* 1\), CoeffBound es \((\d+) \* B\) ∧\n\s*\((parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx)\) = maskPart \(reducePoly (.*?) (\d+) sm\) !\[\((.*?)\), \((.*?)\)\] -', st)
    if not m:
        raise Fail('smudge statement shape')
    _, cols, _, outw, outname, q, _, k0, k1 = m.groups()
    th = theorem_text(sm, f'smudge_hTH_{stag}')
    w509 = re.search(r'\) - \((parallel_generatedRoot_\d+\.w_\d+_\d+ ctx)\)', th).group(1)
    last, lexpr = ch.nh[-1]
    mu = re.search(r'\(μ := ', lexpr)
    j = xmodel.matching(lexpr, mu.start())
    mu_txt = lexpr[mu.start() + len('(μ := '):j]
    mu_sub = ch.subst(scope_sub(mu_txt, n))
    keys = re.findall(r'\bkey(h\d+)\b', mu_sub)
    a = G.req.label['a']
    z = ' + '.join(f'(famAt {a} (key{t} o)).toNat' for t in keys) or '0'
    lut_tag = kind_of(last)[1]
    nu = ch.sub[f'ν_{lut_tag}']
    hw = ch.sub[f'hwalsh_{lut_tag}']
    dk = re.search(r'^theorem (smudge_digits_keys_\S+) ', C.wiring, re.M).group(1)
    Mm = re.search(r'\{μ ν : Fin (\d+) →', theorem_text(C.wiring, dk)).group(1)
    zx = z.replace('(key', '(key').replace(' o)', ' ⟨o, h⟩)')
    zf = f'(fun o : Nat ↦ if h : o < {Mm} then {zx} else 0)'
    q_ = scope_sub(q, n)
    k0, k1, outw, w509 = (scope_sub(x, n) for x in (k0, k1, outw, w509))
    concl = f'''
    ∃ (Bs BM : Nat) (Tsm : ExactMatrix {q_} {n} 1 {cols}) (es M : ErrorMatrix {n} 1 {cols}),
      CoeffBound es Bs ∧
      ({outw}) = maskPart (reducePoly {q_} {n} sm) ![({k0}), ({k1})] - Tsm + reduceMatrix {q_} {n} 1 {cols} es ∧
      Tsm - ({w509}) = reduceMatrix {q_} {n} 1 {cols} M ∧ CoeffBound M BM := by'''
    tail = f'''  obtain ⟨es, hes, hs⟩ := RefreshTopProof.GenOnlineSmudge.{sname} F {last}
  have hμz : ∀ o : Fin {Mm}, reducePoly {q_} {n} ({mu_sub} o) = reducePoly {q_} {n} (Xe (n := {n}) ({G.s0} * ({zf} o))) := by
    intro o
    simp only [o.isLt, dite_true, Xe_mul_Xe, mul_one, Fin.eta]
    exact congrArg (reducePoly _ _) (Xe_congr_nat (by ring))
  have hdig := {WIRING}.{dk} (iS : Nat) (μ := {mu_sub}) (ν := {nu}) (Δ' := 1)
    (z := {zf}) hμz {hw} (map_one _)
  obtain ⟨M, hM, hTH⟩ := RefreshTopProof.GenOnlineSmudge.smudge_hTH_{stag} F (sp := sp) (Δ' := 1) (ν := {nu}) (z := {zf}) hdig
  exact ⟨_, _, _, es, M, hes, hs, hTH, hM⟩'''
    head = theorem_head(G, f'smudge_summand',
                        f'**The smudge summand** (`final_inst`\'s `hs`/`hes`, `hTH`/`hM`): chain `{ch.name}` closed from the stage facts.')
    return head + concl + '\n  haveI : Fact ((iS : Nat) < S) := ⟨iS.isLt⟩\n' + '\n'.join(G.pre_lines + body) + '\n' + tail + '\n'



CTXN = {PRE: 'cP', ENC: 'cE', ON: 'ctx'}


def link_proof(C, M1, w1, M2, w2):
    """A proof of `M1.inst.w1 c1 = M2.inst.w2 c2` from the stage links (either orientation), or None."""
    for (A, a, B, b) in ((M1, w1, M2, w2), (M2, w2, M1, w1)):
        name = f'link_{A}_{a}_{B}_{b}'
        m = re.search(r'^theorem ' + name + r' ', C.links_text, re.M)
        if not m:
            continue
        th = theorem_text(C.links_text, name)
        x, y = STAGE_LETTER[A], STAGE_LETTER[B]
        if 'houtA' in th:
            args = f'hout{x} hinst{x} hin{y} hroot{y}'
        else:
            args = f'hin{x} hroot{x} hin{y} hroot{y}'
        lhs = re.search(r'\n    (\S+) \S+ = \S+ \S+ :=\n', th).group(1)
        pf = f'(StageLinks.{name} {args})'
        return pf if lhs == f'{M1}.{C.stages[M1].inst}.{w1}' else f'{pf}.symm'
    return None


def stage_refs(C, text, ren):
    """Rename the stage contexts of a statement: `ren` maps context names to ours."""
    for a, b in ren.items():
        text = re.sub(r'\b' + a + r'\b', b, text)
    return text


def section_module(text, name):
    pos = text.index(f'theorem {name} ')
    op = re.findall(r'^open (Stage_\w+)$', text[:pos], re.M)
    return op[-1] if op else None


def qual_section(text, M):
    if not M:
        return text
    return re.sub(r'(?<![\w.])(parallel_generatedRoot_\d+|modulus_\d+|mod_\d+|sel_\d+)\b', M + r'.\1', text)


def emit_hint(G):
    C = G.C
    sm = C.smudge
    th = theorem_text(sm, 'hint_online')
    head, concl, _ = split_header(th)
    hl = next(ty for br, nms, ty in binders(head) if nms == ['hlink'])
    mm = re.match(r'^(Stage_\w+)\.\S+\.(w_\d+_\d+) octx = (Stage_\w+)\.\S+\.(w_\d+_\d+) ectx$', hl)
    lk = link_proof(C, mm.group(1), mm.group(2), mm.group(3), mm.group(4))
    if lk is None:
        raise Fail('hint link')
    M = section_module(sm, 'hint_online')
    concl = qual_section(stage_refs(C, concl, {'octx': 'ctx', 'ectx': 'cE', 'i_0': '(iS : Nat)'}), M)
    G.pre_lines, G.done = [], set()
    G.enc_masks()
    w4 = G.enc_secret['sm']
    body = '\n'.join(G.pre_lines)
    return (theorem_head(G, 'hint_summand', '**The hint summand** (`final_inst`\'s `hb`/`heb`, `Bk k = rowOf (encrypt hint key) k`).')
            + f'\n    {concl} := by\n  haveI : Fact ((iS : Nat) < S) := ⟨iS.isLt⟩\n{body}\n'
            f'  exact RefreshTopProof.GenOnlineSmudge.hint_online FE hsm h_{w4} {lk}\n')


def emit_pre(G):
    C = G.C
    sm = C.smudge
    name = re.search(r'^theorem (hpre_h\d+_\d+) ', sm, re.M).group(1)
    th = theorem_text(sm, name)
    head, concl, _ = split_header(th)
    bs = {nms[0]: ty for br, nms, ty in binders(head) if len(nms) == 1}
    fi = open(f'{C.pdir}/Gen_Final_Inst.lean').read()
    pl = re.search(r'combination digits `(w_\d+_\d+)` =\s+`(w_\d+_\d+)`, action digits `(w_\d+_\d+)` = `(w_\d+_\d+)`', fi)
    online_of = {pl.group(2): pl.group(1), pl.group(4): pl.group(3)}
    sub, hyps, extra = {}, [], []
    for v in ('D', 'Y', 'Pm', 'A'):
        hn = {'D': 'hD', 'Y': 'hY', 'Pm': 'hP', 'A': 'hA'}[v]
        pw = re.search(r'\.(w_\d+_\d+) ctx\)$', bs[hn]).group(1)
        if pw in online_of:
            ow = online_of[pw]
            sub[v] = f'({ON}.{C.O.inst}.{ow} ctx)'
            hyps.append(f'    ({hn} : {ON}.{C.O.inst}.{ow} ctx = {PRE}.{C.P.inst}.{pw} cP)')
            extra.append(hn)
            continue
        got = None
        for M in (ON, ENC):
            for L in re.finditer(rf'^theorem link_(?:{PRE}_' + pw + r'_' + M + r'_(w_\d+_\d+)|' + M + rf'_(w_\d+_\d+)_{PRE}_' + pw + r') ', C.links_text, re.M):
                w = L.group(1) or L.group(2)
                pf = link_proof(C, M, w, PRE, pw)
                if pf:
                    got = (M, w, pf)
                    break
            if got:
                break
        if not got:
            raise Fail(f'pre summand: no link for {pw}')
        M, w, pf = got
        sub[v] = f'({M}.{C.stages[M].inst}.{w} {CTXN[M]})'
        extra.append(pf)
    concl = concl
    for v, s in sub.items():
        concl = re.sub(r'(?<![\w.])' + v + r'(?![\w.])', s, concl)
    concl = stage_refs(C, concl, {'ctx': 'cP'})
    concl = concl.replace(f'({ON}.' + C.O.inst, f'({ON}.' + C.O.inst)
    concl = re.sub(rf'({ON}\.\S+ )cP\)', r'\1ctx)', concl)
    concl = re.sub(rf'({ENC}\.\S+ )cP\)', r'\1cE)', concl)
    concl = concl.replace('i_0', '(iS : Nat)')
    head = theorem_head(G, 'pre_summand', '**The preimage summand** (`final_inst`\'s `hpre`): `hD`/`hY` are `GenFinalInst.pre_link` (the online decompositions are the preprocess ones).')
    head = head[:-2] + '\n' + '\n'.join(hyps) + ' :'
    return head + f'\n    {concl} := by\n  haveI : Fact ((iS : Nat) < S) := ⟨iS.isLt⟩\n  exact RefreshTopProof.GenOnlineSmudge.{name} FP ' + ' '.join(extra) + '\n'


def emit_keys(G):
    C = G.C
    sm = C.smudge
    sk = re.search(r'^theorem (smudge_keys_h\d+_\d+) ', sm, re.M).group(1)
    th = theorem_text(sm, sk)
    head, concl, _ = split_header(th)
    path = f'{C.pdir}/Gen_Inst_Keys.lean'
    if not os.path.exists(path):
        raise Fail('no Gen_Inst_Keys.lean')
    kt = open(path).read()
    mk = re.search(r'^theorem (member_keys_h\d+_\d+) ', kt, re.M).group(1)
    kth = theorem_text(kt, mk)
    kb = binders(split_header(kth)[0])
    args = []
    for br, nms, ty in kb:
        if br != '(':
            continue
        nm = nms[0]
        if nm in ('FP', 'FE', 'FO'):
            args.append({'FP': 'FP', 'FE': 'FE', 'FO': 'F'}[nm])
        elif nm == 'hspsmall':
            args.append('hsp')
        elif re.match(r'h_w_\d+_\d+$', nm):
            args.append('hP9')
        elif re.match(r'l[OE]_w_\d+_\d+$', nm):
            m = re.match(r'^\((Stage_\w+)\.\S+\.(w_\d+_\d+) \w+\) = \((Stage_\w+)\.\S+\.(w_\d+_\d+) \w+\)$', ty)
            pf = link_proof(C, m.group(1), m.group(2), m.group(3), m.group(4))
            if pf is None:
                raise Fail(f'keys link {ty}')
            args.append(pf)
        else:
            raise Fail(f'member_keys binder {nm}')
    concl = stage_refs(C, concl, {'octx': 'ctx', 'pctx': 'cP', 'i_0': '(iS : Nat)'})
    G.pre_lines, G.done = [], set()
    G.pre_secret()
    return (theorem_head(G, 'keys_summand', '**The shared smudge keys** (`Ks` of `final_inst`\'s `hs` and `hpre`): the online smudge column keys are the preprocess ones (`smudge_keys_*` + D\'s `member_keys_*`).')
            + f'\n    {concl} := by\n  haveI : Fact ((iS : Nat) < S) := ⟨iS.isLt⟩\n' + '\n'.join(G.pre_lines)
            + f'\n  exact RefreshTopProof.GenOnlineSmudge.{sk} F FP (RefreshTopProof.GenInst.Keys.{mk} {" ".join(args)})\n')



# --------------------------------------------------------------------------- the mask summand

def node_arg(expr, name):
    """Text of the named argument `(name := …)` of a node expression."""
    m = re.search(r'\(' + re.escape(name) + r' := ', expr)
    if not m:
        return None
    j = xmodel.matching(expr, m.start())
    return expr[m.end():j]


def sampler_certs(G, kt, W, n, q, h):
    """Certificates of a coefficient key table `kt` (`keyTableG [1] [b] (Σ keyTermC σ tv …)`) whose odd
    entries are the constant `h`: `<name>_hC` in `sampled_of_walsh`'s `hC` shape (binary table)."""
    key = ('sm', kt)
    if key in G._gc:
        return G._gc[key]
    name = f'sm_{len(G._gc)}'
    G._gc[key] = name
    L = W.bit_length() - 1
    inner = xmodel.strip_parens(kt)
    inner = inner[:inner.rindex(' 0 0')]
    it = toks(inner)
    b = xmodel.strip_parens(re.match(r'^\(\(\[(.*)\]\) : List Int\)$', it[2]).group(1))
    ssum = xmodel.strip_parens(it[3])
    tt = toks(re.match(r'^∑ k ∈ Finset\.range \(\(\d+\) : Int\)\.toNat, (.*)$', ssum, re.S).group(1))
    sig, tv, s = tt[1], tt[2], int(re.search(r'(\d+)', tt[4]).group(1))
    I = lambda x: x.replace('(iS : Nat)', 'i_0')
    bb, qq, tvv, hh, ktt = I(b), I(q), I(tv), I(h), I(kt)
    sigv = I(sig).replace('((i : Nat))', '(i : Nat)')
    gt = G.gt_cert(q).replace('(iS : Nat)', 'i_0')
    idx = f'(fun i : Fin (2 ^ {L}) ↦ ({sig.replace("((i : Nat))", "((i : Fin (2 ^ " + str(L) + ")) : Nat)")}).toNat)'
    G.cert.append(f'''theorem {name}_tab (i_0 : Nat) [h : Fact (i_0 < S)] :
    ∀ k : Fin {W}, {tvv} k = if k.val % 2 = 1 then {hh} else 0 := by
  have := h.out
  interval_cases i_0 <;> decide

theorem {name}_nonneg (i_0 : Nat) [h : Fact (i_0 < S)] : 0 ≤ (({bb} : Int)) := by
  have := h.out
  interval_cases i_0 <;> decide

set_option maxRecDepth 65536 in
theorem {name}_inv (i_0 : Nat) [h : Fact (i_0 < S)] : ((({bb} : Int)).toNat * 2 ^ {L}) % {qq} = 1 := by
  have := h.out
  interval_cases i_0 <;> decide

theorem {name}_hinv (i_0 : Nat) [Fact (i_0 < S)] :
    (((({bb} : Int)).toNat : Nat) : ExactPoly {qq} {n}) * ((2 ^ {L} : Nat) : ExactPoly {qq} {n}) = 1 :=
  cast_inverse {gt} ({name}_inv i_0)

theorem {name}_hC (i_0 : Nat) [Fact (i_0 < S)] (i : Fin (2 ^ {L})) :
    (fun (i : Fin {W}) ↦ {ktt}) i =
      lutFourierCoefficient (2 ^ {L}) (X {qq} ^ {s}) (((({bb} : Int)).toNat : Nat) : ExactPoly {qq} {n})
        (fun k ↦ reducePoly {qq} {n} (Gen.binTable (n := {n}) (2 ^ {L}) (fun _ ↦ {hh}) k)) ({idx.replace("(iS : Nat)", "i_0")} i) := by
  have hb : (((({bb} : Int)).toNat : Nat) : ExactPoly {qq} {n}) = (({bb} : Int) : ExactPoly {qq} {n}) := by
    rw [← Int.cast_natCast, Int.toNat_of_nonneg ({name}_nonneg i_0)]
  refine ({WIRING}.keyTableC_walsh' (q := {qq}) (n := {n}) (s := {s}) (W := {W}) {gt} (by norm_num)
    (σ := {sigv}) (by revert i; decide) (by norm_num) 1 _ {tvv}).trans ?_
  rw [Int.cast_one, one_mul, hb, binTable_const {tvv} _ ({name}_tab i_0)]
  rfl
''')
    res = dict(name=name, idx=idx, L=L, s=s, sig=sig)
    G._gc[key] = res
    return res


def mask_summand(G, ch, body):
    C, n = G.C, G.C.n
    nh = dict(ch.nh)
    O = f'{ON}.{C.O.inst}'
    # the sampler-offset LUT member segment and the offset source
    so = None
    for nmh, e in ch.nh:
        m = re.match(r'seg_(h\d+)_\d+ ', e)
        if m and online_role(C, m.group(1)[1:]) == 'lut.member.sampler-offset':
            so = (nmh, e, m.group(1))
    if so is None:
        raise Fail('no sampler-offset member')
    nmh, e, sotag = so
    mu = ch.subst(scope_sub(node_arg(e, 'μ'), n))
    hodd = ch.subst(scope_sub(node_arg(e, 'hodd'), n))
    hoff = node_arg(e, 'hoff')
    src = re.match(r'(\S+)', hoff).group(1)          # hoff_hX
    srcnode = nh[src]
    r_off = re.match(r'encRel_of_rel (\S+)', srcnode).group(1)
    muoff = ch.subst(scope_sub(node_arg(nh[r_off], 'μ'), n))
    segth = theorem_text(open(f'{C.pdir}/Gen_Inst_{ON}.lean').read(), re.match(r'(seg_h\d+_\d+)', e).group(1))
    famw = re.search(r'fam4 \(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\) \(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\) \(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\) \(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\)', segth).groups()
    sigma = re.search(r'branchTargetG \(q := \S+\) \(p := .*?\) (\(sig \[[^\]]*\] \d+\)) ', segth).group(1)
    fam = 'fam4 ' + ' '.join(f'({O}.{w} ctx)' for w in famw)
    # the sampler switch (preprocess table of the fam wires)
    pws = [C.link_term(w, PRE) for w in famw]
    loops = sorted({int(sw.split('_')[1]) for _, sw in pws if sw})
    sw = next(re.search(r'^theorem (switch_h' + str(L) + r'_\d+) ', C.wiring, re.M).group(1) for L in loops
              if re.search(r'^theorem switch_h' + str(L) + r'_\d+ ', C.wiring, re.M))
    st = theorem_text(C.wiring, sw)
    mc = re.search(r'\(\(fun \(i : Fin (\d+)\) ↦ (.*?)\) k \*\n', st)
    Wsm = int(mc.group(1))
    kt = qualify(mc.group(2), PRE, None).replace('i_0', '(iS : Nat)')
    Cw = f'(fun (i : Fin {Wsm}) ↦ {kt})'
    hb_ = binders(split_header(st)[0])
    bty = {nms[0]: ty for br, nms, ty in hb_ if len(nms) == 1}
    tq, th_ = toks(bty["g'"]), toks(bty["hg'"])
    q = qualify(tq[1], PRE, None).replace('i_0', '(iS : Nat)')
    nn, m = xmodel.strip_parens(tq[2]), xmodel.strip_parens(tq[4])
    base, digits = th_[2], th_[3]
    cert = G.gadget_cert(q, nn, m, base, digits)
    gq = f'(gadgetOf Backend.backend {base} {digits} {q} {nn} {m})'
    links = ', '.join(t for t, _ in pws)
    # mask_inst_form / mask_inst statements (G)
    fresh = open(f'{C.pdir}/Gen_Online_Fresh.lean').read()
    mif = theorem_text(fresh, 'mask_inst_form')
    hv0 = mif.index('(hval : ')
    hv1 = mif.index('fun _ ↦ ', hv0) + len('fun _ ↦ ')
    hconst = mif[hv1:xmodel.matching(mif, hv1) + 1]
    hconst_i = hconst.replace('i0', '(iS : Nat)')
    R = f'({ON}.mod_0 ((Int.ofNat (iS : Nat))))'
    smc = sampler_certs(G, kt, Wsm, n, q, hconst_i)
    # the readout feeding the sampler: its value form
    rtag = None
    for tg, rv in G.readouts.items():
        if f'readoutExpG' in mu:
            rtag = tg
    rr = [rv for tg, rv in G.readouts.items()]
    zsum = re.search(r'readoutExpG \d+ \(2 \^ \d+\) \((.*)\)\)\)$', mu)
    zexpr = zsum.group(1) if zsum else None
    if zexpr is None:
        raise Fail(f'mask: sampler input is not a readout value: {mu[:120]}')
    zN = f'(fun o : Fin {C.n} ↦ {zexpr.replace(" i)", " o)")})'
    M = int(re.search(r'\(μ : Fin (\d+) → ErrorPoly', mif).group(1))
    zN = f'(fun o : Fin {M} ↦ {zexpr.replace(" i)", " o)")})'
    gtR = G.gt_cert(R)
    Ls = smc['L']
    ssm = smc['s']
    sampled = lambda muo, z: (f'(by\n      have hv := sampled_of_walsh (s := {ssm}) (log := {Ls}) (by norm_num) (by norm_num) (Gen.binTable (n := {n}) (2 ^ {Ls}) (fun _ ↦ {hconst_i}))\n'
                              f'        ({smc["name"]}_hinv (iS : Nat)) {smc["idx"]}\n'
                              f'        {hodd} {perm_of(smc["idx"], Ls)} (by intro i; revert i; decide) ({smc["name"]}_hC (iS : Nat)) (μ := {muo}) (z := {z}) rfl\n'
                              f'        (ν := centeredIntLift _) (reducePoly_centeredIntLift {gtR} (by norm_num) _).symm\n')
    zr_m = re.search(r'readoutExpG (\d+) \(2 \^ (\d+)\)', mu)
    Pm, Lr = zr_m.group(1), zr_m.group(2)
    hval_pf = 'fun o ↦ ' + sampled(f'{mu} o', f'readoutExpG {Pm} (2 ^ {Lr}) _') + '      exact hv)'
    hoff_pf = sampled(muoff, f'readoutExpG {Pm} (2 ^ {Lr}) 0') + f'      rw [(show readoutExpG {Pm} (2 ^ {Lr}) 0 = 1 by decide)] at hv\n      exact hv)'
    nu = f'(fun o ↦ centeredIntLift (∑ k, reducePoly {R} {n} (substitution ({sigma} k).toNat ({hodd} k) ({mu} o)) * {Cw} k))'
    nuoff = f'(centeredIntLift (∑ k, reducePoly {R} {n} (substitution ({sigma} k).toNat ({hodd} k) {muoff}) * {Cw} k))'
    atom = re.search(r'\(h(\d+_\d+) : blockModSwitchRuns', mif).group(1)
    fs = find_field(open(f'{C.gdir}/{ON}_Facts.lean').read(), r'blockModSwitchRuns .* \(parallel_generatedRoot_\d+\.w_' + atom + r' ctx\)$')
    out_w = 'w_' + atom
    lines = [
        f'  obtain ⟨eM, hbM, heM⟩ := {WIRING}.{sw} FP hsp hP9 (gadgetOf_runs {cert})',
        f'  have hsw : ∀ (k : Fin {Wsm}) (j : Fin 2), reducePoly {R} {n} (substitution ({sigma} k).toNat ({hodd} k) sp) •',
        f'      {fam} 0 j k + {fam} 1 j k =',
        f'      ({Cw} k * reducePoly {R} {n} (1 * unitPair sp j)) • {gq} + reduceMatrix {R} {n} 1 {m} (eM k j) := by',
        f'    intro k j',
        f'    rw [{links}]',
        f'    exact heM k j',
        f'  obtain ⟨K, eg, heg, hg, hK⟩ := RefreshTopProof.GenOnlineFresh.mask_inst_form (iS : Nat) hsm {sigma} {hodd}',
        f'    ({fam}) {mu} {muoff}',
        f'    (fun o k ↦ polyNorm_substitution_Xe_gen (by norm_num) _ _) (fun k ↦ polyNorm_substitution_Xe_gen (by norm_num) _ _)',
        f'    {ch.nh[-1][0]} hsw hbM {nu} {nuoff}',
        f'    (fun o ↦ (reducePoly_centeredIntLift {gtR} (by norm_num) _).symm)',
        f'    (reducePoly_centeredIntLift {gtR} (by norm_num) _).symm',
        f'    {zN}',
        f'    ({hval_pf})',
        f'    {hoff_pf} F.{fs[0]}.{fs[1]}',
    ]
    # hz: the PRF outputs are the requirement's sums
    req = G.req
    L_ = req.label
    a = L_['a']
    outext = None
    zr_field = None
    for f_, ty in req.fields.items():
        mm = re.match(r'^∀ (\w+) < (\d+), \(famAt (x\d+) \1\) = \(1 - \(\((.*) % (\d+)\) / (\d+)\)\)$', ty)
        if mm:
            zr_field = (f_, mm.group(1), mm.group(4))
            outext = req.ext[mm.group(3)]
    if not zr_field:
        raise Fail('mask: no requirement mask formula')
    fz, v, zbody = zr_field
    for x, e_ in sorted(req.ext.items(), key=lambda kv: -int(kv[0][1:])):
        zbody = re.sub(r'\b' + x + r'\b', e_, zbody)
    zR = f'(fun {v} : Nat ↦ {zbody})'
    keys = re.findall(r'\bkey(h\d+)\b', zexpr)
    hz = [f'  have z0 : ∀ j : Fin S, (MxxIR.selectInt (Int.ofNat (j : Nat)) [{",".join(["0"] * C.S)}]).toNat = 0 := by decide',
          f'  have ha : ∀ k, 0 ≤ famAt {a} k := famAt_nonneg _ (fun c ↦ ({hv_proj(C, a)[0]} c).1)',
          f'  have hz : ∀ o : Fin {M}, (({zN} o : Nat) : Int) = {zR} (o : Nat) := by',
          f'    intro o']
    rws, es = [], []
    for kk, kt_ in enumerate(keys):
        info = G.buckets[kt_]
        grp, b = G.onehot_of(info['rhs'])
        sext = G.ext_of_pre(info['cw'])
        tw = online_wires(scope_sub(ch.types[f'hkey_{kt_}'], n))[0]
        et, text = G.ext_of_online(tw)
        bs, bt = req.bcast.get(sext), req.bcast.get(text)
        if not bs or not bt:
            raise Fail(f'mask: no broadcast loops for {sext}/{text}')
        Wb = int(re.match(r'Fin \d+ → Fin (\d+)', scope_sub(ch.types[f'sel_{kt_}'], n)).group(1))
        Mb = int(re.match(r'Fin (\d+) →', scope_sub(ch.types[f'sel_{kt_}'], n)).group(1))
        hz += [f'    have hs{kk} : ∀ k : Fin {Wb}, famAt (famAt {sext} 0) (o * {Wb} + k) = if k = sel{grp} o {b} then 1 else 0 := fun k ↦ by',
               f'      have h := hsel{grp} o {b} k',
               f'      have hl := hreq.{bs[0]} iS iS.2 (o * {Wb} + k) (by omega)',
               f'      rw [z0 iS] at hl',
               f'      rw [← hl]; exact h',
               f'    have hk{kk} : ((key{kt_} o : Nat) : Int) = famAt (famAt {text} 0) (o * {Wb} + sel{grp} o {b}) := by',
               f'      rw [hkey{kt_} o, {et}]',
               f'      have hl := hreq.{bt[0]} iS iS.2 (o * {Wb} + sel{grp} o {b}) (by omega)',
               f'      rw [z0 iS, famAt_of_lt _ iS.isLt] at hl',
               f'      rw [← hl, famAt_cell (M := {Mb}) (W := {Wb}) ({text} iS)]',
               f'    have e{kk} := prf_bucket_val (W := {Wb}) (famAt (famAt {sext} 0)) (famAt (famAt {text} 0)) (famAt {a}) o (sel{grp} o {b}) hs{kk}']
        es.append(f'e{kk}')
        rws.append(f'← hk{kk}')
    hz += [f'    push_cast at {" ".join(es)}',
           f'    simp only [{", ".join(es)}, zero_add]',
           f'    rw [{", ".join(rws)}, {", ".join(["Int.toNat_natCast"] * len(keys))}]',
           f'    push_cast [Int.toNat_of_nonneg (ha _)]',
           f'    ring']
    lines += hz
    # the requirement's mask formula (the parts' `mask_inst` hypothesis on `σ_0`)
    fz_ty = req.fields[fz]
    for x, e_ in sorted(req.ext.items(), key=lambda kv: -int(kv[0][1:])):
        fz_ty = re.sub(r'\b' + x + r'\b', e_, fz_ty)
    lines += [f'  exact ⟨{zN}, K, eg, _, heg, hg, hz, hreq.{fz}, hK⟩']
    # conclusion: from mask_inst_form / mask_inst
    cf = split_header(mif)[1]
    cf = cf.replace('i0', '(iS : Nat)')
    # the named key witness (`K = fun k ↦ …` of mask_inst_form), at the instance's wires
    kq = re.search(r' ∧\n\s*K = (fun k ↦ .*?)\s*$', cf, re.S)
    if not kq:
        raise Fail('mask_inst_form: no key witness')
    cf = cf[:kq.start()]
    kterm = re.sub(r'(?<![\w.])(w_\d+_\d+)\b', lambda m_: f'({O}.{m_.group(1)} ctx)', ' '.join(kq.group(1).split()))
    mb = re.search(r'∃ \(K : (.*?)\) \(eg : (.*?)\),\n\s*CoeffBound eg .*? ∧\n\s*(w_\d+_\d+) = (.*)$', cf, re.S)
    beta = re.search(r' - (reducePoly \S+ \S+ \d+ \(∑ o : Fin \d+, .*\)) • g\' \+', mb.group(4), re.S)
    if not beta:
        beta = re.search(r' - (reducePoly .* \(∑ o : Fin \d+, .*\)) • g\' \+', mb.group(4), re.S)
    betat = beta.group(1).replace('zN o', f'zN o')
    eq = mb.group(4).replace("g'", gq)
    concl = (f'∃ (zN : Fin {M} → Nat) (K : {mb.group(1)}) (eg : {mb.group(2)}) (Beg : Nat), CoeffBound eg Beg ∧\n'
             f'      {O}.{mb.group(3)} ctx = {eq} ∧\n'
             f'      (∀ o : Fin {M}, ((zN o : Nat) : Int) = {zR} (o : Nat)) ∧\n      ({fz_ty}) ∧\n      K = {kterm}')
    head = theorem_head(G, 'mask_summand', '**The mask summand** of chain `iS` (the parts\' `hg`/`heg`, and `mask_inst`\'s `hz` and `hσ`): chain `' + ch.name + '` closed from the stage facts and the mask row (`mask_inst_form`); the PRF outputs `zN` are the requirement\'s sums `zR`, whose mask formula gives `σ_0`.')
    return head + '\n    ' + concl + ' := by\n  haveI : Fact ((iS : Nat) < S) := ⟨iS.isLt⟩\n' + '\n'.join(G.pre_lines + body + lines) + '\n'



# --------------------------------------------------------------------------- the fresh summand

def sampler_certs_g(G, kt, W, n, q, glist):
    """Certificates of a coefficient key table with a binary table `g = listTable glist` (`<name>_hC` in
    `sampler_value_gen`'s `hC` shape)."""
    key = ('fs', kt, glist)
    if key in G._gc:
        return G._gc[key]
    name = f'fs_{len(G._gc)}'
    L = W.bit_length() - 1
    inner = xmodel.strip_parens(kt)
    inner = inner[:inner.rindex(' 0 0')]
    it = toks(inner)
    b = xmodel.strip_parens(re.match(r'^\(\(\[(.*)\]\) : List Int\)$', it[2]).group(1))
    ssum = xmodel.strip_parens(it[3])
    tt = toks(re.match(r'^∑ k ∈ Finset\.range \(\(\d+\) : Int\)\.toNat, (.*)$', ssum, re.S).group(1))
    sig, tv, s = tt[1], tt[2], int(re.search(r'(\d+)', tt[4]).group(1))
    I = lambda x: x.replace('(iS : Nat)', 'i_0')
    bb, qq, tvv, ktt = I(b), I(q), I(tv), I(kt)
    sigv = I(sig).replace('((i : Nat))', '(i : Nat)')
    gt = G.gt_cert(q).replace('(iS : Nat)', 'i_0')
    idx = f'(fun i : Fin (2 ^ {L}) ↦ ({sig.replace("((i : Nat))", "((i : Fin (2 ^ " + str(L) + ")) : Nat)")}).toNat)'
    g = f'(GenFinal.listTable {glist})'
    tdep = 'i_0' in tvv
    tab_hdr = '(i_0 : Nat) [h : Fact (i_0 < S)] ' if tdep else ''
    tab_pf = '  have := h.out\n  interval_cases i_0 <;> decide\n' if tdep else '  decide\n'
    G.cert.append(f'''theorem {name}_tab {tab_hdr}:
    ∀ k : Fin {W}, {tvv} k = if k.val % 2 = 1 then {g} (k.val / 2) else 0 := by
{tab_pf}
theorem {name}_nonneg (i_0 : Nat) [h : Fact (i_0 < S)] : 0 ≤ (({bb} : Int)) := by
  have := h.out
  interval_cases i_0 <;> decide

set_option maxRecDepth 65536 in
theorem {name}_inv (i_0 : Nat) [h : Fact (i_0 < S)] : ((({bb} : Int)).toNat * 2 ^ {L}) % {qq} = 1 := by
  have := h.out
  interval_cases i_0 <;> decide

theorem {name}_hinv (i_0 : Nat) [Fact (i_0 < S)] :
    (((({bb} : Int)).toNat : Nat) : ExactPoly {qq} {n}) * ((2 ^ {L} : Nat) : ExactPoly {qq} {n}) = 1 :=
  cast_inverse {gt} ({name}_inv i_0)

theorem {name}_hC (i_0 : Nat) [Fact (i_0 < S)] (i : Fin (2 ^ {L})) :
    (fun (i : Fin {W}) ↦ {ktt}) i =
      lutFourierCoefficient (2 ^ {L}) (X {qq} ^ {s}) (((({bb} : Int)).toNat : Nat) : ExactPoly {qq} {n})
        (fun k ↦ reducePoly {qq} {n} (Gen.binTable (n := {n}) (2 ^ {L}) {g} k)) ({idx} i) := by
  have hb : (((({bb} : Int)).toNat : Nat) : ExactPoly {qq} {n}) = (({bb} : Int) : ExactPoly {qq} {n}) := by
    rw [← Int.cast_natCast, Int.toNat_of_nonneg ({name}_nonneg i_0)]
  refine ({WIRING}.keyTableC_walsh' (q := {qq}) (n := {n}) (s := {s}) (W := {W}) {gt} (by norm_num)
    (σ := {sigv}) (by revert i; decide) (by norm_num) 1 _ {tvv}).trans ?_
  rw [Int.cast_one, one_mul, hb, binTable_of {tvv} _ ({name}_tab{" i_0" if tdep else ""})]
  rfl
''')
    res = dict(name=name, idx=idx, L=L, s=s)
    G._gc[key] = res
    return res


def prf_link(G, ch, keys, idx_var, jterm, req_vec):
    """Proof lines of `((zN f : Nat) : Int) = prfZB W a s t f` (the PRF output of member `f`) for the chain's
    bucket keys `keys`, at the requirement's member index `jterm`."""
    C, n = G.C, G.C.n
    req = G.req
    a = req.label['a']
    lines = []
    sels, tabs, hk = [], [], []
    for kk, kt_ in enumerate(keys):
        info = G.buckets[kt_]
        grp, b = G.onehot_of(info['rhs'])
        sext = G.ext_of_pre(info['cw'])
        tw = online_wires(scope_sub(ch.types[f'hkey_{kt_}'], n))[0]
        et, text = G.ext_of_online(tw)
        bs, bt = req.bcast.get(sext), req.bcast.get(text)
        if not bs or not bt:
            raise Fail(f'no broadcast loops for {sext}/{text}')
        Wb = int(re.match(r'Fin \d+ → Fin (\d+)', scope_sub(ch.types[f'sel_{kt_}'], n)).group(1))
        Mb = int(re.match(r'Fin (\d+) →', scope_sub(ch.types[f'sel_{kt_}'], n)).group(1))
        sels.append((sext, grp, b, bs, Wb))
        tabs.append(text)
        lines += [f'    have hk{kk} : ((key{kt_} {idx_var} : Nat) : Int) = famAt (famAt {text} {jterm}) ({idx_var} * {Wb} + sel{grp} {idx_var} {b}) := by',
                  f'      rw [hkey{kt_} {idx_var}, {et}]',
                  f'      have hl := hreq.{bt[0]} iS iS.2 ({idx_var} * {Wb} + sel{grp} {idx_var} {b}) (by omega)',
                  f'      rw [zj iS, famAt_of_lt _ iS.isLt] at hl',
                  f'      rw [← hl, famAt_cell (M := {Mb}) (W := {Wb}) ({text} iS)]']
    return lines, sels, tabs


def fresh_summand(G, ch, body):
    C, n = G.C, G.C.n
    nh = dict(ch.nh)
    O = f'{ON}.{C.O.inst}'
    last, lexpr = ch.nh[-1]
    # the placed members' family and the sampler member segment
    mpl = re.search(r'\(parallel_parallel_generatedRoot_\d+_(\d+)\.sem \(F\.c\d+\.h(\d+)_5 i\)\) \((r_h\d+) i\)', lexpr)
    if not mpl:
        raise Fail('fresh: no placement')
    seg_name = mpl.group(3)
    e = nh[seg_name]
    sm_tag = re.match(r'seg_(h\d+)_\d+', e).group(1)
    if online_role(C, sm_tag[1:]) != 'lut.member.sampler':
        raise Fail('fresh: placement source is not a sampler member')
    mu = ch.subst(scope_sub(node_arg(e, 'μ'), n))
    hodd = ch.subst(scope_sub(node_arg(e, 'hodd'), n))
    segth = theorem_text(open(f'{C.pdir}/Gen_Inst_{ON}.lean').read(), re.match(r'(seg_h\d+_\d+)', e).group(1))
    famw = re.search(r'fam4 \(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\) \(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\) \(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\) \(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\)', segth).groups()
    sigma = re.search(r'branchTargetG \(q := \S+\) \(p := .*?\) (\(sig \[[^\]]*\] \d+\)) ', segth).group(1)
    fam = 'fam4 ' + ' '.join(f'({O}.{w} ctx)' for w in famw)
    # the column sums of the placed members
    pw = 'w_' + mpl.group(2) + '_0'
    cs = None
    for mcs in re.finditer(r'^theorem (colsum_h(\d+)_\d+) ', C.smudge, re.M):
        th = theorem_text(C.smudge, mcs.group(1))
        if f'.{pw} ctx)' in th:
            cs = (mcs.group(1), th)
    if cs is None:
        raise Fail(f'fresh: no column sums of {pw}')
    csname, csth = cs
    mm = re.search(r'colPlace \S+ \d+ (\d+) (\d+) i \(∑ j ∈ Finset\.range (\d+), famAt', csth)
    Ccol, mcol, Wcol = int(mm.group(1)), int(mm.group(2)), int(mm.group(3))
    outs = re.findall(r'\(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\) = ∑ i', csth)
    fams = [re.sub(r'_\d+$', f'_{c}', pw) for c in range(6)]
    Fm = Ccol * Wcol
    famt = '(' + ', '.join(f'({O}.{w} ctx)' for w in fams) + ', ())'
    fint = '(' + ', '.join(f'({O}.{w} ctx)' for w in outs) + ', ())'
    projs = ' | '.join(f'exact hcs{".2" * c}{".1" if c < 5 else ""}' for c in range(6))
    # the sampler switch
    pws = [C.link_term(w, PRE) for w in famw]
    loops = sorted({int(sw.split('_')[1]) for _, sw in pws if sw})
    sw = next(re.search(r'^theorem (switch_h' + str(L) + r'_\d+) ', C.wiring, re.M).group(1) for L in loops
              if re.search(r'^theorem switch_h' + str(L) + r'_\d+ ', C.wiring, re.M))
    st = theorem_text(C.wiring, sw)
    mc = re.search(r'\(\(fun \(i : Fin (\d+)\) ↦ (.*?)\) k \*\n', st)
    Wsm = int(mc.group(1))
    kt = qualify(mc.group(2), PRE, None).replace('i_0', '(iS : Nat)')
    Cw = f'(fun (i : Fin {Wsm}) ↦ {kt})'
    hb_ = binders(split_header(st)[0])
    bty = {nms[0]: ty for br, nms, ty in hb_ if len(nms) == 1}
    tq, th_ = toks(bty["g'"]), toks(bty["hg'"])
    q = qualify(tq[1], PRE, None).replace('i_0', '(iS : Nat)')
    nn, m = xmodel.strip_parens(tq[2]), xmodel.strip_parens(tq[4])
    base, digits = th_[2], th_[3]
    cert = G.gadget_cert(q, nn, m, base, digits)
    gq = f'(gadgetOf Backend.backend {base} {digits} {q} {nn} {m})'
    links = ', '.join(t for t, _ in pws)
    # the error row (D's Gen_Part.errE): val, prfZB, member index
    part = open(f'{C.pdir}/Gen_Part.lean').read()
    me = re.search(r'GenFinal\.freshErr \d+ \d+ \(GenFinal\.samplerVal (\d+) (\d+) \(GenFinal\.listTable (\[[^\]]*\])\)\)\n\s*\(GenFinal\.prfZB (\d+) \(GenIface\.famAt (\S+)\)\n\s*!\[(.*)\]\n\s*!\[(.*)\]\)', part)
    if not me:
        raise Fail('fresh: no errE in Gen_Part')
    Psv, Wsv, glist, Wz, aext = me.group(1), me.group(2), me.group(3), me.group(4), me.group(5)
    Rp = int(re.search(r'^abbrev R : Nat := (\d+)', part, re.M).group(1))
    jterm = f'((iS : Nat) % {Rp})'
    srow = me.group(6).replace('GenIface.', '').replace(' j)', f' {jterm})')
    trow = me.group(7).replace('GenIface.', '').replace(' j)', f' {jterm})')
    val = f'(GenFinal.samplerVal {Psv} {Wsv} (GenFinal.listTable {glist}))'
    z = f'(fun f : Nat ↦ GenFinal.prfZB {Wz} (famAt {aext}) ![{srow}] ![{trow}] f)'
    fcs = sampler_certs_g(G, kt, Wsm, n, q, glist)
    R = f'({ON}.mod_0 ((Int.ofNat (iS : Nat))))'
    gtR = G.gt_cert(R)
    zs = re.search(r'readoutExpG \d+ \(2 \^ \d+\) \((.*)\)\)\)$', mu)
    if not zs:
        raise Fail(f'fresh: sampler input is not a readout value: {mu[:100]}')
    zexpr = zs.group(1)
    keys = re.findall(r'\bkey(h\d+)\b', zexpr)
    zN = f'(fun f : Fin {Fm} ↦ {zexpr.replace(" i)", " f)")})'
    nu = f'(fun f ↦ centeredIntLift (∑ k, reducePoly {R} {n} (substitution ({sigma} k).toNat ({hodd} k) ({mu} f)) * {Cw} k))'
    # instance -> member index of the requirement (broadcast list = i % R)
    blist = G.req.bcast[G.ext_of_pre(G.buckets[keys[0]]['cw'])][1]
    plk, sels, tabs = prf_link(G, ch, keys, 'f', jterm, None)
    srow_items = [x.strip() for x in xmodel.split_top(srow, ',')]
    try:
        sels = sorted(sels, key=lambda s_: srow_items.index(f'famAt (famAt {s_[0]} {jterm})'))
    except ValueError:
        raise Fail(f'fresh: selector rows {[s_[0] for s_ in sels]} not in errE {srow_items}')
    if [str(s_[2]) for s_ in sels] != [str(b) for b in range(len(sels))] or len({s_[1] for s_ in sels}) != 1:
        raise Fail(f'fresh: selector buckets {[(s_[1], s_[2]) for s_ in sels]} are not one one-hot group in errE order')
    hs_cases = []
    for b, (sext, grp, bb_, bs, Wb) in enumerate(sels):
        hs_cases += [f'      · show famAt (famAt {sext} {jterm}) (f * {Wb} + k) = if k = sel{grp} f {bb_} then 1 else 0',
                     f'        have h := hsel{grp} f {bb_} k',
                     f'        have hl := hreq.{bs[0]} iS iS.2 (f * {Wb} + k) (by omega)',
                     f'        rw [zj iS] at hl',
                     f'        rw [← hl]; exact h']
    Wb = sels[0][4]
    grp = sels[0][1]
    B = len(sels)
    atoms = re.findall(r'\((h\d+_0) : ', split_header(theorem_text(open(f'{C.pdir}/Gen_Online_Fresh.lean').read(), 'fresh_inst'))[0])
    ofacts = open(f'{C.gdir}/{ON}_Facts.lean').read()
    arefs = []
    for a_ in atoms:
        cur = None
        for ln in ofacts.split('\n'):
            mm2 = re.match(r'^structure \S+\.Facts_(\d+) ', ln)
            if mm2:
                cur = mm2.group(1)
            if ln.startswith(f'  {a_} :'):
                arefs.append(f'F.c{cur}.{a_}')
                break
    hz_rows = '![' + ', '.join(f'famAt (famAt {sx} {jterm})' for sx, *_ in sels) + ']'
    b_of_kk = [G.onehot_of(G.buckets[kt_]['rhs'])[1] for kt_ in keys]
    kk_of_b = [b_of_kk.index(b) if b in b_of_kk else b_of_kk.index(str(b)) for b in range(len(keys))]
    kv = '![' + ', '.join(f'(key{keys[kk]} f : Nat)' for kk in kk_of_b) + ']'
    trow_v = '![' + trow + ']'
    lines = [
        f'  have hcs := RefreshTopProof.GenOnlineSmudge.{csname} F',
        f'  have r_col := fun cc : Fin {Ccol} ↦ column_rel_of_colsum (C := {Ccol}) (W := {Wcol}) (F := {Fm}) (by decide) (sm := sm)',
        f'    (fams := {famt}) (final := {fint})',
        f'    (by intro c; fin_cases c <;> first | {projs})',
        f'    (fun f ↦ {last} f) cc',
        f'  obtain ⟨eF, hbF, heF⟩ := {WIRING}.{sw} FP hsp hP9 (gadgetOf_runs {cert})',
        f'  have hsw : ∀ (k : Fin {Wsm}) (j : Fin 2), reducePoly {R} {n} (substitution ({sigma} k).toNat ({hodd} k) sp) •',
        f'      {fam} 0 j k + {fam} 1 j k =',
        f'      ({Cw} k * reducePoly {R} {n} (1 * unitPair sp j)) • {gq} + reduceMatrix {R} {n} 1 {m} (eF k j) := by',
        f'    intro k j',
        f'    rw [{links}]',
        f'    exact heF k j',
        f'  have zj : ∀ j : Fin S, (MxxIR.selectInt (Int.ofNat (j : Nat)) [{blist}]).toNat = (j : Nat) % {Rp} := by decide',
        f'  have ha : ∀ k, 0 ≤ famAt {aext} k := famAt_nonneg _ (fun c ↦ ({hv_proj(C, aext)[0]} c).1)',
        f'  have hzf : ∀ f : Fin {Fm}, (({zN} f : Nat) : Int) = {z} (f : Nat) := by',
        f'    intro f',
        f'    have hs : ∀ (b : Fin {B}) (k : Fin {Wb}), ({hz_rows} : Fin {B} → Nat → Int) b (f * {Wb} + k) = if k = sel{grp} f b then 1 else 0 := by',
        f'      intro b k',
        f'      fin_cases b',
        *hs_cases,
        *plk,
        f'    show _ = GenFinal.prfZB {Wz} (famAt {aext}) ![{srow}] ![{trow}] f',
        f'    rw [GenFinal.prfZB_onehot _ _ _ f (sel{grp} f) hs]',
        f'    have hrow : ∀ b : Fin {B}, ({trow_v} : Fin {B} → Nat → Int) b (f * {Wb} + sel{grp} f b) = ((({kv} : Fin {B} → Nat) b : Nat) : Int) := by',
        f'      intro b',
        f'      fin_cases b',
        *[f'      · exact (hk{kk}).symm' for kk in kk_of_b],
        f'    refine Eq.trans ?_ (Finset.sum_congr rfl (fun b _ ↦ (congrArg (fun x : Int ↦ famAt {aext} x.toNat) (hrow b)).symm))',
        f'    simp only [Int.toNat_natCast, Fin.sum_univ_succ, Fin.sum_univ_zero, Matrix.cons_val_zero, Matrix.cons_val_succ, add_zero]',
        f'    push_cast [Int.toNat_of_nonneg (ha _)]',
        f'    ring',
        f'  obtain ⟨Kf, ef, hef, hf, hKf⟩ := RefreshTopProof.GenOnlineFresh.fresh_inst (iS : Nat) (hashModel := hashModel) (params := {ON_PARAMS}) hsm {sigma} {hodd}',
        f'    ({fam}) {mu} (fun f k ↦ polyNorm_substitution_Xe_gen (by norm_num) _ _) r_col (gadgetOf_runs {cert}) hsw hbF',
        f'    {nu}',
        f'    (fun f ↦ (reducePoly_centeredIntLift {gtR} (by norm_num) _).symm)',
        f'    {val} {z}',
        f'    (fun f ↦ by',
        f'      have hv := sampler_value_gen (s := {fcs["s"]}) (log := {fcs["L"]}) (P := {Psv}) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)',
        f'        (GenFinal.listTable {glist}) ({fcs["name"]}_hinv (iS : Nat)) {fcs["idx"]}',
        f'        {hodd} {perm_of(fcs["idx"], fcs["L"])} (by intro i; revert i; decide) ({fcs["name"]}_hC (iS : Nat)) (μ := {mu} f) (z := _) rfl',
        f'        (ν := centeredIntLift _) (reducePoly_centeredIntLift {gtR} (by norm_num) _).symm',
        f'      rw [hzf f] at hv',
        f'      exact hv)',
        f'    {" ".join(arefs)}',
    ]
    # conclusion
    fi = split_header(theorem_text(open(f'{C.pdir}/Gen_Online_Fresh.lean').read(), 'fresh_inst'))
    concl0 = fi[1].replace('i0', '(iS : Nat)')
    kq = re.search(r' ∧\n\s*Kf = (fun k _ l ↦ .*?)\s*$', concl0, re.S)
    if not kq:
        raise Fail('fresh_inst: no key witness')
    concl0 = concl0[:kq.start()]
    kfterm = re.sub(r'(?<![\w.])(w_\d+_\d+)\b', lambda m_: f'({O}.{m_.group(1)} ctx)', ' '.join(kq.group(1).split()))
    mb = re.search(r'∃ \(Kf : (.*?)\) \(ef : (.*?)\),\n\s*CoeffBound ef .*? ∧\n\s*fresh = (.*)$', concl0, re.S)
    eq = mb.group(3).replace('freshErr ', 'freshErr ').replace(' val z)', f' {val} {z})')
    outw = re.search(r'IterRuns .*\) \(Int\.toNat \(\d+\)\) w_\d+_\d+ fresh\)', fi[0], re.S)
    h441 = atoms[-1]
    freshw = 'w_' + h441[1:]
    concl = (f'∃ (Kf : {mb.group(1)}) (ef : {mb.group(2)}) (Bf : Nat), CoeffBound ef Bf ∧\n'
             f'      {O}.{freshw} ctx = {eq} ∧\n      Kf = {kfterm}')
    if mb.group(3).count('freshErr') and ' val z)' not in mb.group(3):
        raise Fail('fresh: freshErr shape')
    lines += [f'  exact ⟨Kf, ef, _, hef, hf, hKf⟩']
    head = theorem_head(G, 'fresh_summand', '**The fresh summand** (`final_inst`\'s `hf`/`hef` with `E j = GenPart.errE external j`, `j = i mod R`): chain `' + ch.name + '` closed from the stage facts, the column sums and `fresh_inst`.')
    return head + '\n    ' + concl + ' := by\n  haveI : Fact ((iS : Nat) < S) := ⟨iS.isLt⟩\n' + '\n'.join(G.pre_lines + body + lines) + '\n'




def pin_bucket_dims(th):
    """Give every `bucket_of_cells`/`bucket_of_cells1` step its `M`/`W` explicitly (from its selector's
    `hsel…` statement): `t : Fin (M * W) → Int` is a wire of a literal `Fin` size, so `W` is otherwise
    solved only while checking `hsel`, by failing unfoldings of `message` (minutes per step at n = 2^17)."""
    dims = {m.group(1): (m.group(2), m.group(3))
            for m in re.finditer(r'have (hsel\w+) : ∀ \(o : Fin (\d+)\) \(j : Fin (\d+)\)', th)}

    def fix(m):
        sel = re.search(r'\(fun o ↦ (hsel\w+) o\) o\)', m.group(0))
        if not sel or sel.group(1) not in dims:
            return m.group(0)
        M, W = dims[sel.group(1)]
        return m.group(0).replace(m.group(1), f'{m.group(1)} (M := {M}) (W := {W})', 1)
    return re.sub(r'(bucket_of_cells1?) (?!\(M :=)[^\n]*\n[^\n]*', fix, th)


# --------------------------------------------------------------------------- the preimage bound

def emit_preimage(G):
    """`final_inst`'s `hP`/`hPd`: the online preimage is the preprocess `preimageRunsDispatched` output, a
    reduced matrix within the trapdoor's cutoff (`preimage_target_gen`)."""
    C = G.C
    pf = open(f'{C.gdir}/{PRE}_Facts.lean').read()
    fp = find_field(pf, r'^MxxRuntime\.preimageRunsDispatched ')
    if fp is None:
        raise Fail('no preimage fact')
    m = re.match(r'MxxRuntime\.preimageRunsDispatched backend \(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\) \(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\) \(\S+ ctx\) (\(Int\.toNat \(-?\d+\)\)) \(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\)$', fp[2])
    if not m:
        raise Fail(f'preimage fact shape: {fp[2][:160]}')
    A, T, cut, Pw = m.groups()
    ft = find_field(pf, r'^MxxRuntime\.trapdoorSample backend .* \(parallel_generatedRoot_\d+\.' + A + r' ctx\) \(parallel_generatedRoot_\d+\.' + T + r' ctx\)$')
    if ft is None:
        raise Fail('no trapdoor fact')
    ow = None
    for L in re.finditer(rf'^theorem link_{PRE}_' + Pw + rf'_{ON}_(w_\d+_\d+) ', C.links_text, re.M):
        ow = L.group(1)
    if ow is None:
        raise Fail(f'no online link of {Pw}')
    pfk = link_proof(C, ON, ow, PRE, Pw)
    ty = re.search(r'\(hP : ' + ow + r' = reduceMatrix _ (\d+) (\d+) (\d+) Pd\)', open(f'{C.pdir}/Gen_Final_Inst.lean').read())
    if not ty:
        raise Fail(f'no final_inst type of {ow}')
    nn, r_, c_ = ty.groups()
    head = theorem_head(G, 'preimage_summand', "**The preimage bound** (`final_inst`'s `hP`/`hPd`): the online preimage is the preprocess `preimageRunsDispatched` output (`preimage_target_gen`).")
    return (head + f'\n    ∃ Pd : ErrorMatrix {nn} {r_} {c_}, {ON}.{C.O.inst}.{ow} ctx = reduceMatrix _ {nn} {r_} {c_} Pd ∧ CoeffBound Pd {cut} := by\n'
            f'  obtain ⟨_, d, hd, hb⟩ := RefreshTopProof.Gen.preimage_target_gen FP.{ft[0]}.{ft[1]} FP.{fp[0]}.{fp[1]}\n'
            f'  exact ⟨d, {pfk}.trans hd, hb⟩\n')


# --------------------------------------------------------------------------- closed bounds

# (summand, hypothesis, bound variable) of the existential bounds stated as `B ≤ bound_<summand>_<B> i`
BOUND_VARS = [('mask_summand', 'heg', 'Beg'), ('fresh_summand', 'hef', 'Bf'),
              ('smudge_summand', 'hes', 'Bs'), ('smudge_summand', 'hM', 'BM')]

PROBE = r"""open Lean Meta Elab Tactic in
/-- Generator probe: logs the per-instance values of the bound `B` of `h : CoeffBound e B`, `B` closed
except for the instance `i`. -/
elab "probe_bound " tag:str h:ident i:term : tactic => withMainContext do
  let ty ← instantiateMVars (← inferType (← getLocalDeclFromUserName h.getId).toExpr)
  let B := ty.appArg!
  let iE ← Term.elabTerm i (some (mkConst ``Nat))
  let Babs ← kabstract B iE
  for k in [0:{S}] do
    let Bk := Babs.instantiate1 (mkNatLit k)
    if Bk.hasFVar || Bk.hasMVar then
      throwError "probe_bound {{tag.getString}}: open bound {{Bk}}"
    let v ← unsafe evalExpr Nat (mkConst ``Nat) Bk
    logInfo m!"BOUND {{tag.getString}} {{k}} {{v}}"
"""


def bound_tac():
    return '(by have hlt := iS.isLt; generalize (iS : Nat) = I at hlt ⊢; interval_cases I <;> decide)'


def apply_bounds(G, thms, mode, table):
    """Probe mode: log every bound before the final `exact`. Bounds mode: state `B ≤ bound_<s>_<B> i`."""
    out = []
    for th in thms:
        name = re.match(r'(?:/--.*?-/\n)?theorem (\S+)', th, re.S).group(1)
        vs = [(h, b) for s, h, b in BOUND_VARS if s == name]
        if not vs:
            out.append(th)
            continue
        lines = th.rstrip('\n').split('\n')
        k = max(i for i, ln in enumerate(lines) if ln.startswith('  exact ⟨'))
        if mode == 'probe':
            lines[k:k] = [f'  probe_bound "{name}.{b}" {h} (iS : Nat)' for h, b in vs]
        else:
            th2 = '\n'.join(lines)
            for h, b in vs:
                d = f'bound_{name.replace("_summand", "")}_{b}'
                if d not in table:
                    raise Fail(f'no probed values of {d}')
                pat = re.compile(r'CoeffBound (\w+) ' + b + r'( ∧| :=)')
                if not pat.search(th2):
                    raise Fail(f'{name}: no CoeffBound … {b}')
                th2 = pat.sub(lambda m_: f'CoeffBound {m_.group(1)} {b} ∧ {b} ≤ {d} (iS : Nat){m_.group(2)}', th2, count=1)
            lines = th2.split('\n')
            ex = lines[k]
            for h, b in vs:
                ex = re.sub(r'(?<=[⟨, ])' + h + r'(?=[,⟩])', f'{h}, {bound_tac()}', ex, count=1)
            lines[k] = ex
        out.append('\n'.join(lines) + '\n')
    return out


def bound_defs(table):
    return [f'/-- Closed per-instance value of `{d}` (probed by the generator; checked per instance by `decide`). -/\n'
            f'def {d} (i : Nat) : Nat := [{", ".join(str(v) for v in vals)}].getD i 0\n' for d, vals in sorted(table.items())]


def read_probe(logs):
    table = {}
    for m in re.finditer(r'BOUND (\w+)\.(\w+) (\d+) (\d+)', ''.join(open(x).read() for x in logs)):
        d = f'bound_{m.group(1).replace("_summand", "")}_{m.group(2)}'
        table.setdefault(d, {})[int(m.group(3))] = int(m.group(4))
    return {d: [v[i] for i in sorted(v)] for d, v in table.items()}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    pdir = args[0]
    out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else (
        f'{pdir}/Gen_Summands_Sigma.lean' if SIGMA else f'{pdir}/Gen_Summands.lean')
    C = Ctx(pdir)
    G = Gen(C)
    names = re.findall(r'^theorem (chain_h\d+)', C.chains, re.M)
    thms = []
    for nm in names:
        ch = Chain(G, nm)
        last = ch.nh[-1][1]
        mseg = re.match(r'seg_(h\d+)_\d+', last)
        role = online_role(C, mseg.group(1)[1:]) if mseg else None
        G.pre_lines, G.done = [], set()
        if SIGMA:
            # the stages of `σ_0 G` evaluate one PRF chain, the mask
            body, missing = emit_chain(G, ch)
            thms.append(mask_summand(G, ch, body))
            continue
        if role == 'smudge.member':
            body, missing = emit_chain(G, ch)
            thms.append(smudge_summand(G, ch, body))
        elif mseg is None and 'place_rel' in last:
            try:
                body, missing = emit_chain(G, ch)
                thms.append(fresh_summand(G, ch, body))
                continue
            except Fail as e:
                G.notes.append(f'fresh: {e}')
        if '--test-chains' in sys.argv and role != 'smudge.member':
            try:
                body, missing = emit_chain(G, ch)
            except Fail as e:
                G.notes.append(str(e))
                continue
            head = theorem_head(G, f'test_{nm}', f'Test: the binders of `{nm}` discharged.')
            thms.append(head + '\n    True := by\n  haveI : Fact ((iS : Nat) < S) := ⟨iS.isLt⟩\n' + '\n'.join(G.pre_lines + body) + '\n  trivial\n')
    for f in (() if SIGMA else (emit_hint, emit_pre, emit_keys, emit_preimage)):
        try:
            thms.append(f(G))
        except Fail as e:
            G.notes.append(f'{f.__name__}: {e}')
    thms = [pin_bucket_dims(th) for th in thms]
    probe = '--probe' in sys.argv
    table = read_probe(sys.argv[sys.argv.index('--bounds') + 1].split(',')) if '--bounds' in sys.argv else None
    extra = []
    if probe:
        thms = apply_bounds(G, thms, 'probe', None)
        extra = [PROBE.format(S=C.S)]
    elif table is not None:
        thms = apply_bounds(G, thms, 'bounds', table)
        extra = bound_defs(table)
    ns = f'RefreshTopProof.GenInst.{stage_ns(ON)}'
    opens = [
        f'open {ON} RefreshTopProof.GenInst RefreshTopProof.GenIface RefreshTopProof.GenIface.BrLut RefreshTopProof.Gen',
        f'open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.GenFinal GeneratedClaim {PRE_KEYS}',
        'open RefreshTopProof.GenIface.BrLutKey' + ('' if SIGMA else ' RefreshTopProof.GenOnlineSmudge') + ' Mxx.ExponentLut.Refresh',
        'open scoped BigOperators', '']
    opts = ['set_option maxHeartbeats 64000000', 'set_option maxRecDepth 65536', '']
    base = os.path.splitext(out)[0]
    stem = os.path.basename(base)
    outd = os.path.dirname(out) or '.'
    # one module per summand (each elaborates in its own process; one module of all of them exceeds 20 GB)
    imports = ([f'import Gen_Wiring_Sigma', f'import Gen_Inst_{ON}', f'import Gen_Inst_{ENC}', 'import Gen_Requirement',
                'import Gen_Online_Fresh'] if SIGMA else
               ['import Gen_Wiring', f'import Gen_Inst_{ON}', f'import Gen_Inst_{ENC}', 'import Gen_Online_Values',
                'import Gen_Requirement', 'import Gen_Inst_Keys', 'import Gen_Online_Fresh', 'import Gen_Final_Inst'])
    pre = '\n'.join([
        *imports, 'import Mathlib.Tactic.IntervalCases', '',
        '/-! Generated by `generator/gen_summands.py` (WP F): lemmas, certificates and closed bounds shared by the',
        f'summand modules `{stem}_*` (see the generator\'s docstring). -/', '',
        *opts, f'namespace {ns}', '', *opens,
        LEMMAS, '' if SIGMA else LEMMAS_COLUMNS, '', *extra, *G.cert, *G.top,
        *(f'-- note: {x}' for x in G.notes), '',
        f'end {ns}', ''])
    files = {f'{outd}/{stem}_Pre.lean': pre}
    mods = []
    for th in thms:
        name = re.search(r'^theorem (\w+)', th, re.M).group(1)
        mod = f'{stem}_{"".join(w.capitalize() for w in name.split("_"))}'
        mods.append(mod)
        files[f'{outd}/{mod}.lean'] = '\n'.join([
            f'import {stem}_Pre', '', f'/-! Generated by `generator/gen_summands.py` (WP F): `{name}`. -/', '',
            *opts, f'namespace {ns}', '', *opens, th, f'end {ns}', ''])
    files[out] = '\n'.join([*(f'import {m}' for m in mods), '',
                            '/-! Generated by `generator/gen_summands.py` (WP F): the closed summand facts of `final_inst` at a',
                            'symbolic instance, one module per summand (see the generator\'s docstring). -/', ''])
    for f, txt in files.items():
        open(f, 'w').write(txt)
    nth = sum(len(re.findall(r'^theorem ', x, re.M)) for x in files.values())
    print(f'{out}: {len(mods)} summand modules ({" ".join(mods)}), {nth} theorems, {len(G.notes)} notes')
    for x in G.notes:
        print('  note:', x)


if __name__ == '__main__':
    main()
