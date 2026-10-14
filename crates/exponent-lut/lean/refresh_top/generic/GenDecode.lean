import TopCrtGeneric
import ExponentIterate
import ExponentGadget
import GenIface

/-!
# The looped decode stage, for any parameter set

The decode stage of the refresh to the top level reads a family `T : Fin S → M_Q(1 × m)` of CRT
terms. Member `t R + j` is the term of CRT prime `q_t` and output component `j < R`: the centered
lift to `Q = ∏ q_t` of the residue of `y_j + e'_j` modulo `q_t`, times the CRT basis element `K_t`
(the online evaluation's CRT-term role). For each component the stage

* sums the members `j, R + j, 2 R + j, …` (a static read and a sequential loop over strided
  members),
* subtracts the expected row `σ A'_0 + A'_1 - (Δ r) (c X^e) G'`, with `σ` built from its
  coefficients and `A'` component `j`'s column block of the caller's matrix (`colBlock`),
* and runs a sequential loop over the `m` columns whose body exports the column's coefficients
  and runs an inner sequential loop adding `((2 v + ⌊Q/2⌋) / Q) mod 2` per coefficient `v`.

Every loop is handled here by one lemma about `IterRuns` with a `∀`-quantified step hypothesis
(`IterRuns.invariant`), so nothing depends on the slot count, the column count or the ring
dimension. The per-scope adapters turning a generated child scope into the step shapes
`StrideStep`, `ColumnStep` and `RoundStep` are written by `generator/gen_decode_loops.py`.

* `termOf_of_rebase`: the online CRT-term role (`centeredRebase` of a residue, times `K_t`).
* `crt_terms_sum`: the terms of the residues of `v` sum to `v`.
* `strided_family_sum`: the strided member sum.
* `round_small`, `round_loop_const`, `column_loop_zero`: the column and coefficient loops add
  nothing for a small difference.
* `decode_component_zero`: one component, end to end.
* Bridges to the role interface `GenIface`: `decodeColumn_sem` (the proof of
  `GenIface.DecodeColumn.sem`), `roundSum_small`, `column_loop_zero_spec`,
  `strided_of_memberSum1` and `decode_component_zero_iface` (the same component theorem from the
  interface's `MemberSum1.Spec` and `DecodeColumn.Spec`).
-/

namespace GenDecode

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof
open scoped BigOperators

variable {n : Nat}

/-! ## CRT terms -/

/-- `term` is a CRT term of `v` for prime `q` and basis element `K`: the centered lift to `Q` of
the residue of `v` modulo `q`, times `K`. -/
def TermOf {Q m : Nat} (q : Nat) (K : Int) (term v : ExactMatrix Q n 1 m) : Prop :=
  ∃ l : ExactMatrix Q n 1 m, LiftOf q l v ∧ term = matrixMulScalarRight l (matrixPolynomial [K])

/-- The online CRT-term role: a residue `r` of `v`, its centered rebase `l` and `l K`. -/
theorem termOf_of_rebase {q Q m : Nat} {r : ExactMatrix q n 1 m} {l term v : ExactMatrix Q n 1 m}
    {K : Int} (hr : ResidueAt r v) (hl : centeredRebaseRuns r l)
    (ht : term = matrixMulScalarRight l (matrixPolynomial [K])) : TermOf q K term v :=
  ⟨l, ⟨r, hr, hl⟩, ht⟩

/-- The CRT terms of `v` for pairwise coprime primes `q_t` of `Q = ∏ q_t` and a CRT basis `K`
sum to `v`. -/
theorem crt_terms_sum {Q m k : Nat} (hn : 0 < n) (hQ : 1 < Q) (q : Fin k → Nat)
    (hq : ∀ i, 1 < q i) (K : Fin k → Int)
    (hK : ∀ i j, K i ≡ (if i = j then 1 else 0) [ZMOD (q j : Int)])
    (hcop : ∀ i j, i ≠ j → Nat.Coprime (q i) (q j)) (hprod : Q = ∏ i, q i)
    {v : ExactMatrix Q n 1 m} {T : Fin k → ExactMatrix Q n 1 m}
    (hT : ∀ i, TermOf (q i) (K i) (T i) v) : ∑ i, T i = v := by
  haveI : NeZero Q := ⟨by omega⟩
  haveI : Fact (1 < Q) := ⟨hQ⟩
  choose l hl hTl using hT
  funext row c
  obtain rfl : row = 0 := Subsingleton.elim _ _
  apply negacyclic_ext hn
  intro j
  have hs : (∑ i, T i) 0 c = ∑ i, T i 0 c := by
    rw [Finset.sum_apply, Finset.sum_apply]
  rw [hs, Negacyclic.coeff_sum]
  have hterm : ∀ i, (T i 0 c).coeff j = (K i : ZMod Q) * (l i 0 c).coeff j := by
    intro i
    rw [hTl i]
    simp only [matrixMulScalarRight, MxxRuntime.matrixPolynomial_single]
    rw [mul_comm, coeff_intCast_mul_gen]
  simp only [hterm]
  choose x hx hxq using fun i ↦ liftOf_coeff hn (hq i) (hl i) c j
  simp only [hx]
  have e := crt_int_gen q K hK hcop hxq
  rw [← Nat.cast_prod, ← hprod, ← ZMod.intCast_eq_intCast_iff] at e
  push_cast at e
  rw [ZMod.natCast_zmod_val] at e
  rw [← e]
  exact Finset.sum_congr rfl fun i _ ↦ mul_comm _ _

/-- Member `t R + j` of a strided family has CRT index `t` and component `j`. -/
theorem stride_div {t R j : Nat} (hj : j < R) : (t * R + j) / R = t := by
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hj, Nat.zero_add]

theorem stride_mod {t R j : Nat} (hj : j < R) : (t * R + j) % R = j := by
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj]

theorem stride_lt {S K R j t : Nat} (hS : K * R ≤ S) (hj : j < R) (ht : t < K) : t * R + j < S := by
  have : t * R + R ≤ K * R := by
    have := Nat.mul_le_mul_right R (show t + 1 ≤ K by omega)
    simpa [Nat.add_mul] using this
  omega

/-! ## The strided member sum -/

/-- One iteration `i` of the strided member sum: read member `(i + 1) R + j` and add it. -/
def StrideStep {α : Type} [Add α] {S : Nat} (f : Fin S → α) (R j : Int) (i : Nat) (current next : α) :
    Prop :=
  ∃ x : α, familyGetDynamic f ((Int.ofNat i + 1) * R + j) x ∧ next = current + x

/-- The strided member sum: a static read of member `j` and `K - 1` iterations each adding member
`(i + 1) R + j` compute `Σ_{t < K} f (t R + j)`. -/
theorem strided_family_sum {α : Type} [AddCommMonoid α] {S K Rn jn : Nat} (f : Fin S → α)
    (hidx : ∀ t, t < K → t * Rn + jn < S) {R j j0 : Int} (hR : R = (Rn : Int)) (hj : j = (jn : Int))
    (hj0 : j0 = (jn : Int)) {body : Nat → α → α → Prop} {count : Nat} {init final : α}
    (hcount : count + 1 = K) (hinit : familyGetStatic f j0 init)
    (hstep : ∀ i current next, body i current next → StrideStep f R j i current next)
    (h : MxxIR.IterRuns body count init final) :
    final = ∑ t : Fin K, f ⟨t * Rn + jn, hidx t t.isLt⟩ := by
  classical
  subst hR hj hj0 hcount
  let g : Nat → α := fun t ↦ if h : t * Rn + jn < S then f ⟨_, h⟩ else 0
  have hinv := MxxIR.IterRuns.invariant
    (Invariant := fun i (s : α) ↦ s = ∑ t ∈ Finset.range (i + 1), g t) ?_ ?_ h
  · rw [hinv, Finset.sum_range (fun t ↦ g t)]
    refine Finset.sum_congr rfl fun t _ ↦ ?_
    simp only [g, dif_pos (hidx t t.isLt)]
  · obtain ⟨p, hp, rfl⟩ := hinit
    have hp' : p.val = jn := by exact_mod_cast hp
    simp only [Finset.sum_range_one, g, Nat.zero_mul, Nat.zero_add, dif_pos (hp' ▸ p.isLt)]
    congr 1
    exact Fin.ext hp'
  · intro i current next hcur hb
    obtain ⟨x, ⟨p, hp, rfl⟩, rfl⟩ := hstep i current next hb
    have hp' : p.val = (i + 1) * Rn + jn := by
      have : ((p.val : Nat) : Int) = (((i + 1) * Rn + jn : Nat) : Int) := by
        rw [hp]; push_cast; rfl
      exact_mod_cast this
    rw [hcur, Finset.sum_range_succ (n := i + 1)]
    congr 1
    simp only [g, dif_pos (hp' ▸ p.isLt)]
    congr 1
    exact Fin.ext hp'

/-! ## The coefficient and column loops -/

/-- A canonical residue `v` of a small `e` rounds to an even multiple:
`((2 v + ⌊q/2⌋) / q) mod 2 = 0` when `4 |e| + 4 < q`. -/
theorem round_small {q : Nat} (hq : 1 < q) (e : Int) (he : 4 * |e| + 4 < (q : Int)) :
    ((((e : ZMod q).val : Int) * 2 + ((q / 2 : Nat) : Int)) / (q : Int)) % 2 = 0 := by
  haveI : NeZero q := ⟨by omega⟩
  rw [ZMod.val_intCast]
  have hq0 : (0 : Int) < q := by exact_mod_cast (show 0 < q by omega)
  have hh : (((q / 2 : Nat) : Int)) * 2 ≤ q ∧ (q : Int) < ((q / 2 : Nat) : Int) * 2 + 2 := by
    constructor <;> push_cast <;> omega
  rcases le_or_gt 0 e with h0 | h0
  · rw [abs_of_nonneg h0] at he
    rw [Int.emod_eq_of_lt h0 (by omega), Int.ediv_eq_zero_of_lt (by omega) (by omega)]
    rfl
  · rw [abs_of_neg h0] at he
    have hm : e % (q : Int) = e + q := by
      rw [show e = (e + q) + q * (-1) by ring, Int.add_mul_emod_self_left,
        Int.emod_eq_of_lt (by omega) (by omega)]
      ring
    rw [hm, show (e + q) * 2 + ((q / 2 : Nat) : Int) =
        ((e + q) * 2 + ((q / 2 : Nat) : Int) - 2 * q) + 2 * q by ring,
      Int.add_mul_ediv_right _ _ (by omega), Int.ediv_eq_zero_of_lt (by omega) (by omega)]
    rfl

/-- One iteration `k` of a column's coefficient loop: read coefficient value `k` and add its
rounding bit. -/
def RoundStep (vals : Fin n → Int) (half q : Int) (k : Nat) (current next : Int) : Prop :=
  ∃ v : Int, familyGetDynamic vals (Int.ofNat k) v ∧ next = current + (v * 2 + half) / q % 2

/-- A coefficient loop over the canonical values of a reduced small polynomial adds nothing. -/
theorem round_loop_const {q : Nat} (hn : 0 < n) (hq : 1 < q) {x : ExactMatrix q n 1 1}
    {e : ErrorPoly n} (hx : x 0 0 = reducePoly q n e) (he : ∀ i, 4 * |e.coeff i| + 4 < (q : Int))
    {vals : Fin n → Int} (hv : polynomialValues false x vals) {half qc : Int}
    (hh : half = ((q / 2 : Nat) : Int)) (hqc : qc = q) {inner : Nat → Int → Int → Prop}
    (hstep : ∀ k current next, inner k current next → RoundStep vals half qc k current next)
    {count : Nat} {init final : Int} (h : MxxIR.IterRuns inner count init final) :
    final = init := by
  simp only [polynomialValues, Bool.false_eq_true, if_false] at hv
  refine MxxIR.IterRuns.invariant (Invariant := fun _ s ↦ s = init) rfl ?_ h
  intro k current next hcur hb
  obtain ⟨v, ⟨p, -, rfl⟩, rfl⟩ := hstep k current next hb
  rw [hv p, hx, reducePoly_coeff_eq hq hn, hh, hqc, round_small hq _ (he p), add_zero, hcur]

/-- One iteration `i` of the column loop of a difference `D`: slice column `i`, export its values
and run a coefficient loop of `RoundStep`s from the current state to the next. -/
def ColumnStep {q m : Nat} (D : ExactMatrix q n 1 m) (half qc : Int) (i : Nat) (current next : Int) :
    Prop :=
  ∃ (x : ExactMatrix q n 1 1) (vals : Fin n → Int) (count : Nat) (inner : Nat → Int → Int → Prop),
    sliceMatrix D 0 1 (Int.ofNat i) (1 + Int.ofNat i) x ∧ polynomialValues false x vals ∧
    (∀ k current next, inner k current next → RoundStep vals half qc k current next) ∧
    MxxIR.IterRuns inner count current next

/-- The column loop of a small difference adds nothing, for any column count. -/
theorem column_loop_zero {q m : Nat} (hn : 0 < n) (hq : 1 < q) (E : ErrorMatrix n 1 m)
    (hE : ∀ c i, 4 * |(E 0 c).coeff i| + 4 < (q : Int)) {half qc : Int}
    (hh : half = ((q / 2 : Nat) : Int)) (hqc : qc = q) {body : Nat → Int → Int → Prop}
    (hstep : ∀ i current next, body i current next →
      ColumnStep (reduceMatrix q n 1 m E) half qc i current next)
    {count : Nat} {init final : Int} (h : MxxIR.IterRuns body count init final) :
    final = init := by
  refine MxxIR.IterRuns.invariant (Invariant := fun _ s ↦ s = init) rfl ?_ h
  intro i current next hcur hb
  obtain ⟨x, vals, cnt, inner, hs, hv, hr, hrun⟩ := hstep i current next hb
  have ha : (Int.ofNat i).toNat < m := by
    obtain ⟨_, _, _, _, _, hle, _⟩ := hs
    show i < m
    have : ((1 + i : Nat) : Int) ≤ (m : Int) := by push_cast; simpa using hle
    omega
  rw [← hcur]
  exact round_loop_const hn hq (slice_entry_gen hs ha) (fun k ↦ hE _ k) hv hh hqc hr hrun

/-! ## Rows, σ and the expected row -/

theorem slice_row_gen {q rows columns : Nat} {A : ExactMatrix q n rows columns}
    {x : ExactMatrix q n 1 columns} {r r' c c' : Int}
    (h : sliceMatrix A r r' c c' x) (hc : c = 0) (hr : r.toNat < rows) :
    x = fun _ k ↦ A ⟨r.toNat, hr⟩ k := by
  obtain ⟨_, _, _, _, _, _, _, _, hx⟩ := h
  funext row k
  obtain rfl : row = 0 := Subsingleton.elim _ _
  rw [hx 0 k (by simp [hr]) (by subst hc; simp)]
  subst hc
  simp

/-- Columns `[s, s + m)` of `A` (zero past its columns): component `j`'s `A'` is the column block
`m j` of the caller's `[A'_(0) | A'_(1)]`. -/
noncomputable def colBlock {q rows columns : Nat} (m : Nat) (A : ExactMatrix q n rows columns) (s : Nat) :
    ExactMatrix q n rows m :=
  fun r k ↦ if h : s + k.val < columns then A r ⟨s + k.val, h⟩ else 0

/-- A slice of every row of `A` is a column block. -/
theorem slice_colBlock {q rows columns m : Nat} {A : ExactMatrix q n rows columns}
    {x : ExactMatrix q n rows m} {r' c c' : Int}
    (h : sliceMatrix A 0 r' c c' x) : x = colBlock m A c.toNat := by
  obtain ⟨-, -, -, hc0, -, hc', -, hcols, hx⟩ := h
  funext r k
  have hk : c.toNat + k.val < columns := by
    have := k.isLt
    simp only [Int.ofNat_eq_coe] at hc' hcols
    omega
  rw [hx r k (by simp) hk]
  simp [colBlock, hk]

/-- σ imported from its coefficient values. -/
theorem sigma_of_values {q : Nat} {bits : Fin n → Int} {σ : ExactMatrix q n 1 1}
    (h : polynomialFromValues false bits σ) : σ = fun _ _ ↦ polynomialOfCoefficients bits := by
  simpa [polynomialFromValues] using h

/-- The expected row `σ A'_0 + A'_1 - (Δ r) (c X^e) G'` of one component, as the stage builds it
(`G'` the centered rebase of the gadget `G`). -/
noncomputable def decodeExpected {Q Qg m : Nat} (A : ExactMatrix Q n 2 m) (σ : ExactMatrix Q n 1 1)
    (Δ : Int) (r : ExactMatrix Q n 1 1) (c e : Int) (G : ExactMatrix Qg n 1 m) : ExactMatrix Q n 1 m :=
  matrixSub (matrixAdd (matrixMulScalarLeft σ (fun _ k ↦ A 0 k)) (fun _ k ↦ A 1 k))
    (matrixMulScalarLeft (matrixMulScalarLeft (matrixMulScalarLeft (matrixPolynomial [Δ]) r)
      (multiplyMonomial (liftInteger c) e)) (centeredRebase G))

/-! ## One component -/

/-- One component `j` of the decode stage outputs its initial count. The CRT terms `f i` are
indexed `i = t R + j` (prime `t`, component `j`), and each is the term of the value `V (i mod R)`
for the prime `qN (i / R)` and basis element `KN (i / R)`. When `V j = Y + e'` for the expected
row `Y` and a small `e'`, the strided sum `s` of component `j` is `V j`, and the column loop over
`D = s - Y` adds nothing. -/
theorem decode_component_zero {Q S K Rn jn m : Nat} (hn : 0 < n) (hQ : 1 < Q)
    (qN : Nat → Nat) (KN : Nat → Int) (hq : ∀ i : Fin K, 1 < qN i)
    (hK : ∀ i j : Fin K, KN i ≡ (if i = j then 1 else 0) [ZMOD (qN j : Int)])
    (hcop : ∀ i j : Fin K, i ≠ j → Nat.Coprime (qN i) (qN j)) (hprod : Q = ∏ i : Fin K, qN i)
    (hS : K * Rn ≤ S) (hjn : jn < Rn)
    (f : Fin S → ExactMatrix Q n 1 m) (V : Nat → ExactMatrix Q n 1 m)
    (hT : ∀ i : Fin S, TermOf (qN (i / Rn)) (KN (i / Rn)) (f i) (V (i % Rn)))
    (Y : ExactMatrix Q n 1 m) (E : ErrorMatrix n 1 m) (hV : V jn = Y + reduceMatrix Q n 1 m E)
    (hE : ∀ c i, 4 * |(E 0 c).coeff i| + 4 < (Q : Int))
    {R j j0 : Int} (hR : R = (Rn : Int)) (hj : j = (jn : Int)) (hj0 : j0 = (jn : Int))
    {sbody : Nat → ExactMatrix Q n 1 m → ExactMatrix Q n 1 m → Prop} {scount : Nat}
    {sinit sfinal : ExactMatrix Q n 1 m}
    (hscount : scount + 1 = K) (hsinit : familyGetStatic f j0 sinit)
    (hsstep : ∀ i current next, sbody i current next → StrideStep f R j i current next)
    (hs : MxxIR.IterRuns sbody scount sinit sfinal)
    {D : ExactMatrix Q n 1 m} (hD : D = matrixSub sfinal Y)
    {half qc : Int} (hh : half = ((Q / 2 : Nat) : Int)) (hqc : qc = Q)
    {cbody : Nat → Int → Int → Prop}
    (hcstep : ∀ i current next, cbody i current next → ColumnStep D half qc i current next)
    {ccount : Nat} {cinit cfinal : Int} (hc : MxxIR.IterRuns cbody ccount cinit cfinal) :
    cfinal = cinit := by
  have hidx : ∀ t, t < K → t * Rn + jn < S := fun t ht ↦ stride_lt hS hjn ht
  have hsum := strided_family_sum f hidx hR hj hj0 hscount hsinit hsstep hs
  have hv : sfinal = Y + reduceMatrix Q n 1 m E := by
    rw [hsum, ← hV]
    refine crt_terms_sum hn hQ (fun t : Fin K ↦ qN t) hq (fun t : Fin K ↦ KN t) hK hcop hprod
      (fun t ↦ ?_)
    have h := hT ⟨t * Rn + jn, hidx t t.isLt⟩
    simp only [stride_div hjn, stride_mod hjn] at h
    exact h
  have hD' : D = reduceMatrix Q n 1 m E := by
    rw [hD, hv, matrixSub]
    exact add_sub_cancel_left _ _
  rw [hD'] at hcstep
  exact column_loop_zero hn hQ E hE hh hqc hcstep hc

/-! ## Bridges to the role interface `GenIface` -/

open RefreshTopProof.GenIface in
/-- The proof of `GenIface.DecodeColumn.sem` (same statement): a column's coefficient loop over all
`n` values adds their rounded bits. -/
theorem decodeColumn_sem {q n C : Nat} {x : ExactMatrix q n 1 C} {h d : Int} (i : Nat) (hi : i < C)
    {cur next : Int} {col : ExactMatrix q n 1 1} {vals : Fin n → Int}
    (hcol : sliceMatrix x 0 1 (i : Int) (1 + i) col) (hvals : polynomialValues false col vals)
    {body : Nat → Int → Int → Prop}
    (hstep : ∀ j c c', body j c c' → RoundStep.Spec vals h d j c c')
    (hloop : MxxIR.IterRuns body n cur next) : DecodeColumn.Spec x h d i cur next := by
  classical
  refine ⟨hi, col, vals, hcol, hvals, ?_⟩
  have hs := seq_sum (fun j ↦ if hj : j < n then (vals ⟨j, hj⟩ * 2 + h) / d % 2 else 0)
    (fun j c c' hb ↦ by
      obtain ⟨hj, -, rfl⟩ := hstep j c c' hb
      simp [hj]) hloop
  rw [hs, roundSum, Finset.sum_range (fun j ↦ if hj : j < n then (vals ⟨j, hj⟩ * 2 + h) / d % 2 else 0)]
  congr 1
  exact Finset.sum_congr rfl fun j _ ↦ by simp

/-- The rounded bits of the canonical values of a reduced small polynomial are all zero. -/
theorem roundSum_small {q : Nat} (hn : 0 < n) (hq : 1 < q) {x : ExactMatrix q n 1 1}
    {e : ErrorPoly n} (hx : x 0 0 = reducePoly q n e) (he : ∀ i, 4 * |e.coeff i| + 4 < (q : Int))
    {vals : Fin n → Int} (hv : polynomialValues false x vals) :
    RefreshTopProof.GenIface.roundSum vals ((q / 2 : Nat) : Int) q = 0 := by
  simp only [polynomialValues, Bool.false_eq_true, if_false] at hv
  unfold RefreshTopProof.GenIface.roundSum
  refine Finset.sum_eq_zero fun i _ ↦ ?_
  rw [hv i, hx, reducePoly_coeff_eq hq hn]
  exact round_small hq _ (he i)

open RefreshTopProof.GenIface in
/-- The column loop of a small difference adds nothing, from the interface's column spec. -/
theorem column_loop_zero_spec {q m : Nat} (hn : 0 < n) (hq : 1 < q) (E : ErrorMatrix n 1 m)
    (hE : ∀ c i, 4 * |(E 0 c).coeff i| + 4 < (q : Int)) {half qc : Int}
    (hh : half = ((q / 2 : Nat) : Int)) (hqc : qc = q) {body : Nat → Int → Int → Prop}
    (hstep : ∀ i current next, body i current next →
      DecodeColumn.Spec (reduceMatrix q n 1 m E) half qc i current next)
    {count : Nat} {init final : Int} (h : MxxIR.IterRuns body count init final) :
    final = init := by
  subst hh hqc
  refine MxxIR.IterRuns.invariant (Invariant := fun _ s ↦ s = init) rfl ?_ h
  intro i current next hcur hb
  obtain ⟨hi, col, vals, hs, hv, rfl⟩ := hstep i current next hb
  rw [roundSum_small hn hq (slice_entry_gen hs (by simpa using hi)) (fun k ↦ hE _ k) hv, add_zero,
    hcur]

open RefreshTopProof.GenIface in
/-- The interface's strided member sum (`idx i = (i + 1) R + j` after a first read of member
`j`) is the sum of members `t R + j`, `t < K`. -/
theorem strided_of_memberSum1 {α : Type} [AddCommMonoid α] {S K Rn jn : Nat} (f : Fin S → α)
    (hidx : ∀ t, t < K → t * Rn + jn < S) {idx : Nat → Int}
    (hi : ∀ i, idx i = ((i + 1) * Rn + jn : Nat)) {count : Nat} {init final : α}
    (hcount : count + 1 = K) (hinit : init = f ⟨jn, by have := hidx 0 (by omega); simpa using this⟩)
    (h : MemberSum1.Spec f idx count init final) :
    final = ∑ t : Fin K, f ⟨t * Rn + jn, hidx t t.isLt⟩ := by
  classical
  subst hcount
  obtain ⟨-, hs⟩ := h
  rw [hs, hinit, Fin.sum_univ_succ]
  congr 1
  · congr 1
    exact Fin.ext (by simp)
  · rw [Finset.sum_range]
    refine Finset.sum_congr rfl fun t _ ↦ ?_
    rw [hi, Int.toNat_natCast, famAt_of_lt _ (hidx _ (by simp))]
    exact congrArg f (Fin.ext (by simp))

open RefreshTopProof.GenIface in
/-- `decode_component_zero` from the interface's specs: the strided sum as `MemberSum1.Spec` and
the column loop's steps as `DecodeColumn.Spec`. -/
theorem decode_component_zero_iface {Q S K Rn jn m : Nat} (hn : 0 < n) (hQ : 1 < Q)
    (qN : Nat → Nat) (KN : Nat → Int) (hq : ∀ i : Fin K, 1 < qN i)
    (hK : ∀ i j : Fin K, KN i ≡ (if i = j then 1 else 0) [ZMOD (qN j : Int)])
    (hcop : ∀ i j : Fin K, i ≠ j → Nat.Coprime (qN i) (qN j)) (hprod : Q = ∏ i : Fin K, qN i)
    (hS : K * Rn ≤ S) (hjn : jn < Rn)
    (f : Fin S → ExactMatrix Q n 1 m) (V : Nat → ExactMatrix Q n 1 m)
    (hT : ∀ i : Fin S, TermOf (qN (i / Rn)) (KN (i / Rn)) (f i) (V (i % Rn)))
    (Y : ExactMatrix Q n 1 m) (E : ErrorMatrix n 1 m) (hV : V jn = Y + reduceMatrix Q n 1 m E)
    (hE : ∀ c i, 4 * |(E 0 c).coeff i| + 4 < (Q : Int))
    {idx : Nat → Int} (hi : ∀ i, idx i = ((i + 1) * Rn + jn : Nat)) {scount : Nat}
    {sinit sfinal : ExactMatrix Q n 1 m} (hscount : scount + 1 = K)
    (hsinit : sinit = f ⟨jn, by have := stride_lt hS hjn (show 0 < K by omega); simpa using this⟩)
    (hs : MemberSum1.Spec f idx scount sinit sfinal)
    {D : ExactMatrix Q n 1 m} (hD : D = matrixSub sfinal Y)
    {half qc : Int} (hh : half = ((Q / 2 : Nat) : Int)) (hqc : qc = Q)
    {cbody : Nat → Int → Int → Prop}
    (hcstep : ∀ i current next, cbody i current next → DecodeColumn.Spec D half qc i current next)
    {ccount : Nat} {cinit cfinal : Int} (hc : MxxIR.IterRuns cbody ccount cinit cfinal) :
    cfinal = cinit := by
  have hidx : ∀ t, t < K → t * Rn + jn < S := fun t ht ↦ stride_lt hS hjn ht
  have hsum := strided_of_memberSum1 f hidx hi hscount hsinit hs
  have hv : sfinal = Y + reduceMatrix Q n 1 m E := by
    rw [hsum, ← hV]
    refine crt_terms_sum hn hQ (fun t : Fin K ↦ qN t) hq (fun t : Fin K ↦ KN t) hK hcop hprod
      (fun t ↦ ?_)
    have h := hT ⟨t * Rn + jn, hidx t t.isLt⟩
    simp only [stride_div hjn, stride_mod hjn] at h
    exact h
  have hD' : D = reduceMatrix Q n 1 m E := by
    rw [hD, hv, matrixSub]
    exact add_sub_cancel_left _ _
  rw [hD'] at hcstep
  exact column_loop_zero_spec hn hQ E hE hh hqc hcstep hc

end GenDecode
