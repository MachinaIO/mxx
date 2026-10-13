import PeSetup

/-!
# Online evaluation

Each step multiplies the state encoding by the Dynamic RHS of the attribute symbol it reads and
evaluates its LUT. The generated loops of step `0` compute the ciphertext action `c Dec(C)` column
by column (`sequential_generatedRoot_23`), the companion keys and row of the RHS
(`sequential_generatedRoot_33`), and the LUT branches.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The digits of column `jj` of the joined RHS ciphertext `(top, bot)`. -/
noncomputable abbrev rhsDigits (top bot : ExactMatrix Q N 1 88) (jj : Fin 88) :
    ExactMatrix Q N (44 + 44) 1 :=
  columnDigits Q (44 + 44) Backend.backend 65536 44 32768 (top 0 jj) (bot 0 jj)

/-! ## The companion row of the product -/

/-- The state of the inner companion loop of the product: the block of the companion row and the
carried expanded companions. -/
abbrev MulBlock := ExactMatrix Q N 1 88 × ExactMatrix Q N 1 44 × ExactMatrix Q N 1 44 × Unit

/-- The index of the companion of member `x`, private digit row `a` (digit row `44 + a`), and
column `jj` in the encryption's companion family. -/
def mulIndex (x : Fin 4) (a : Fin 44) (jj : Fin 88) : Fin 15488 :=
  ⟨3872 * x + 88 * a + jj, by omega⟩

/-- The digit-row sum `z_0 u_0 + z_1 u_1` of the product's companions at column `jj` and private
digit row `a`: the digits are those of key entry `44 + a`. -/
noncomputable def mulCompanionSum (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 44) (x : Fin 4)
    (z0 z1 : Fin 15488 → ExactMatrix Q N 1 44) (jj : Fin 88) (a : Fin 44) : ExactPoly Q N :=
  (z0 (mulIndex x a jj) * keyDigits A ⟨44 + a, by omega⟩ 0 +
    z1 (mulIndex x a jj) * keyDigits A ⟨44 + a, by omega⟩ 1) 0 0

/-- The expansions of a compact family. -/
noncomputable abbrev expandFam {k : Nat} (zc : Fin k → ExactMatrix PQ N 1 1) :
    Fin k → ExactMatrix Q N 1 44 :=
  fun i ↦ expand (zc i)

/-! ## LUT coefficients -/

/-- Term `j` of the LUT coefficient of automorphism `k` for the table at `offset` of the program
data: `X^(t_(offset + j)) σ_k(X^(2N - j))`, as the generated loop computes it. -/
noncomputable def coefTerm (tables : Fin 512 → Int) (offset k : Nat) (j : Fin 128) (h : offset + 128 ≤ 512) :
    ExactMatrix Q N 1 1 :=
  matrixMulScalarLeft (multiplyMonomial (1 : ExactMatrix Q N 1 1) (tables ⟨offset + j, by omega⟩ * 1))
    (ringAutomorphism k (multiplyMonomial (1 : ExactMatrix Q N 1 1) ((131072 - (j : Int)) * 1)))

/-- Term `j` of the LUT coefficient of the last step, which reads its table as constants:
`t_(offset + j) σ_k(X^(2N - j))`. -/
noncomputable def coefTermC (tables : Fin 512 → Int) (offset k : Nat) (j : Fin 128) (h : offset + 128 ≤ 512) :
    ExactMatrix Q N 1 1 :=
  matrixMulScalarLeft (liftInteger (tables ⟨offset + j, by omega⟩))
    (ringAutomorphism k (multiplyMonomial (1 : ExactMatrix Q N 1 1) ((131072 - (j : Int)) * 1)))

/-! ## Column actions -/

/-- The runtime facts of one column `jj` of a column action: its decomposition runs and is
`rhsDigits top bot jj`. -/
def ColumnRuns (top bot : ExactMatrix Q N 1 88) (jj : Fin 88) : Prop :=
  ∃ D' : ExactMatrix Q N (44 + 44) 1,
    gadgetDecomposeRuns Backend.backend 65536 44 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
      compactCenteredRebaseRuns 32768 D' (rhsDigits top bot jj)

set_option maxHeartbeats 4000000 in
/-- A column action loop over `88` columns whose iteration `i` adds `(x Dec(top, bot)_i) e_i` to
each of its `r` rows `x = rows k`: it ends with the action, column by column, and every column
decomposition runs. -/
theorem colAction {σ : Type} {body : Nat → σ → σ → Prop} {r : Nat} (proj : Fin r → σ → ExactMatrix Q N 1 88)
    {rows : Fin r → ExactMatrix Q N 1 88} {top bot : ExactMatrix Q N 1 88} {init final : σ}
    (hinit : ∀ k, proj k init = 0)
    (hstep : ∀ i s t (hi : i < 88), body i s t →
      (∀ k, proj k t = proj k s + (rows k * rhsDigits top bot ⟨i, hi⟩) 0 0 •
        (unitRow ((i : Nat) : Int) : ExactMatrix Q N 1 88)) ∧ ColumnRuns top bot ⟨i, hi⟩)
    (h : MxxIR.IterRuns body 88 init final) (jj : Fin 88) :
    (∀ k, proj k final 0 jj = (rows k * rhsDigits top bot jj) 0 0) ∧ ColumnRuns top bot jj := by
  have hrun : ∀ k, proj k final 0 jj = (rows k * rhsDigits top bot jj) 0 0 := by
    intro k
    have h' : MxxIR.IterRuns body (88 * 1) init final := h
    let F : Fin 88 → Fin 1 → ExactPoly Q N := fun jj _ ↦ (rows k * rhsDigits top bot jj) 0 0
    have hacc := IterRuns.accumulate (proj k)
      (fun i ↦ if h : i < 88 * 1 then
        F ⟨i / 1, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
          ⟨i % 1, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
        (unitRow (((i / 1 : Nat) : Int)) : ExactMatrix Q N 1 88) else 0) (fun _ _ ↦ True) trivial
      (fun i s t hi _ hb ↦ ⟨by
        have h1 := (hstep i s t (by omega) hb).1 k
        simp only [dif_pos hi, Nat.div_one]
        exact h1, trivial⟩) h'
    obtain ⟨hsum, -⟩ := hacc
    dsimp only at hsum
    rw [hsum, hinit k, zero_add, sum_unitRow_entry F jj]
    simp [F]
  refine ⟨fun k ↦ hrun k, ?_⟩
  let Q' : Nat → σ → Prop := fun i _ ↦ ∀ jj : Fin 88, jj.val < i → ColumnRuns top bot jj
  have hacc := IterRuns.accumulate (fun _ : σ ↦ (0 : ExactMatrix Q N 1 88)) (fun _ ↦ 0) Q'
    (fun jj h ↦ absurd h (Nat.not_lt_zero _)) (fun i s t hi hq hb ↦ ⟨by simp, fun jj hjj ↦ by
      rcases Nat.lt_succ_iff_lt_or_eq.mp hjj with hlt | heq
      · exact hq jj hlt
      · have : jj = ⟨i, hi⟩ := Fin.ext heq
        subst this
        exact (hstep i s t hi hb).2⟩) h
  exact hacc.2 jj jj.isLt

/-! ## The companion rows of the first LUT depth -/

/-- The switching automorphisms of the first LUT depth, in the order of their packages:
`g^a` for `1 ≤ a < 16` and `g^(16 a)` for `1 ≤ a < 8`, `g = 1025`. -/
abbrev switchList : List Int := [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)]

/-- The package members of the first-depth switches. -/
abbrev switchMembers : List Int := [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17), (18), (19), (20), (21)]

/-- The automorphism of first-depth branch `b`. -/
def kSw (b : Nat) : Nat := (MxxIR.selectInt (Int.ofNat b) switchList).toNat

theorem kSw_odd (b : Fin 22) : Odd (kSw b) := by
  fin_cases b <;> decide

theorem switchMembers_eq (b : Nat) (hb : b < 22) :
    MxxIR.selectInt (Int.ofNat b) switchMembers = (b : Int) := by
  interval_cases b <;> rfl

/-- The key `A` routed by `σ_k`. -/
noncomputable def routedKey (k : Nat) (hk : Odd k) (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 44) :
    Fin 2 → Fin 2 → ExactMatrix Q N 1 44 :=
  fun r j ↦ substituteMatrix k hk (A r j)

/-- The index of the companion of first-depth branch `b`, private digit row `a`, and column `jj`. -/
def lutIndex (b : Fin 22) (a : Fin 44) (jj : Fin 88) : Fin 85184 :=
  ⟨3872 * b + 88 * a + jj, by omega⟩

/-- The digit-row sum `z_0 u_0 + z_1 u_1` of the companions of first-depth branch `b` at column
`jj` and private digit row `a`, the digits of entry `44 + a` of the routed key. -/
noncomputable def lutCompanionSum (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 44)
    (z0 z1 : Fin 85184 → ExactMatrix Q N 1 44) (b : Fin 22) (jj : Fin 88) (a : Fin 44) :
    ExactPoly Q N :=
  (z0 (lutIndex b a jj) * keyDigits (routedKey (kSw b) (kSw_odd b) A) ⟨44 + a, by omega⟩ 0 +
    z1 (lutIndex b a jj) * keyDigits (routedKey (kSw b) (kSw_odd b) A) ⟨44 + a, by omega⟩ 1) 0 0

/-! ## Joined rows and their halves -/

noncomputable def actionDigits (top bot : ExactMatrix Q N 1 88) : Fin 2 → ExactMatrix Q N (44 + 44) 44 :=
  fun j a l ↦ rhsDigits top bot ⟨44 * (j : Nat) + l, by omega⟩ a 0

/-- A column loop of a joined row `c = [row₀ | row₁]` by the joined matrix `(top, bot)` with
components `tk`, `bk`: component `j` of the result is `row` times the digits of the columns of
component `j`, and every column decomposition runs. -/
theorem colActionSplit {c : ExactMatrix Q N 1 88} {row : Fin 2 → ExactMatrix Q N 1 44}
    (hc : concatColumns (row 0) (row 1) c) {top bot : ExactMatrix Q N 1 88}
    {tk bk : Fin 2 → ExactMatrix Q N 1 44} (htop : ∀ p : Fin 88, top 0 p = joinedEntry tk p)
    (hbot : ∀ p : Fin 88, bot 0 p = joinedEntry bk p) {acc : ExactMatrix Q N 1 88}
    (hacc : ∀ jj : Fin 88, acc 0 jj = (c * rhsDigits top bot jj) 0 0 ∧
      ∃ D' : ExactMatrix Q N (44 + 44) 1,
        gadgetDecomposeRuns Backend.backend 65536 44 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
          compactCenteredRebaseRuns 32768 D' (rhsDigits top bot jj)) :
    (∀ j : Fin 2, rowWindow 44 (44 * j) (by omega) acc = pairMul row (actionDigits top bot j)) ∧
      ∀ (j : Fin 2) (l : Fin 44), ∃ D' : ExactMatrix Q N (44 + 44) 1,
        gadgetDecomposeRuns Backend.backend 65536 44 (pairColumn (tk j 0 l) (bk j 0 l)) D' ∧
          compactCenteredRebaseRuns 32768 D' (fun a _ ↦ actionDigits top bot j a l) := by
  refine ⟨fun j ↦ ?_, fun j l ↦ ?_⟩
  · rw [window_mul_columns (m := 44) (k := 44) hc (fun p ↦ (hacc p).1) j (by omega)]
    rfl
  · obtain ⟨D', h1, h2⟩ := (hacc ⟨44 * (j : Nat) + l, by omega⟩).2
    rw [htop, hbot, joinedEntry_pair, joinedEntry_pair] at h1
    refine ⟨D', h1, ?_⟩
    have hcol : rhsDigits top bot ⟨44 * (j : Nat) + l, by omega⟩ =
        fun a _ ↦ actionDigits top bot j a l := by
      funext a b
      rw [Subsingleton.elim b 0]
      rfl
    rwa [hcol] at h2

/-- The two halves of a row, concatenated, are the row. -/
theorem concat_windows {x y : ExactMatrix Q N 1 88}
    (h : concatColumns (rowWindow 44 0 (by omega) x) (rowWindow 44 44 (by omega) x) y) : y = x := by
  funext r p
  rw [Subsingleton.elim r 0, h 0 p]
  split
  · simp only [rowWindow]
    congr 1
    ext
    simp
  · simp only [rowWindow]
    congr 1
    ext
    simp
    omega

/-- The joined row `[r₀ | r₁]`. -/
def joinRow (r : Fin 2 → ExactMatrix Q N 1 44) : ExactMatrix Q N 1 88 := fun _ p ↦ joinedEntry r p

theorem concat_joinRow {r : Fin 2 → ExactMatrix Q N 1 44} {y : ExactMatrix Q N 1 88}
    (h : concatColumns (r 0) (r 1) y) : y = joinRow r := by
  funext a p
  rw [Subsingleton.elim a 0, concatColumns_entry h, pair_eta]
  rfl

/-- The halves of a joined row. -/
def halves (x : ExactMatrix Q N 1 88) : Fin 2 → ExactMatrix Q N 1 44 :=
  fun j ↦ rowWindow 44 (44 * j) (by omega) x

/-- Entry `p` of a joined row, by the half it lies in. -/
theorem joinedEntry_split (r : Fin 2 → ExactMatrix Q N 1 44) (p : Fin 88) :
    joinedEntry r p = if h : p.val < 44 then r 0 0 ⟨p.val, h⟩ else r 1 0 ⟨p.val - 44, by omega⟩ := by
  unfold joinedEntry Fin.addCases
  split <;> simp_all <;> rfl

theorem concatColumns_joinRow (r : Fin 2 → ExactMatrix Q N 1 44) :
    concatColumns (r 0) (r 1) (joinRow r) := by
  intro a p
  rw [Subsingleton.elim a 0]
  simp only [joinRow, joinedEntry_split]

theorem joinRow_halves (x : ExactMatrix Q N 1 88) : joinRow (halves x) = x := by
  funext a p
  rw [Subsingleton.elim a 0]
  simp only [joinRow, joinedEntry_split, halves, rowWindow]
  split
  · simp
  · congr 1
    ext
    simp
    omega

theorem entry_halves (x : ExactMatrix Q N 1 88) (p : Fin 88) : x 0 p = joinedEntry (halves x) p := by
  conv_lhs => rw [← joinRow_halves x]
  rfl

theorem rowWindow_full (x : ExactMatrix Q N 1 88) : rowWindow 88 0 (by omega) x = x := by
  funext a p
  rw [Subsingleton.elim a 0]
  simp [rowWindow]


/-! ## The internal branches of the first LUT depth -/

/-- The inverses of the first-depth switches. -/
abbrev switchInvList : List Int := [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)]

/-- The helper members of the inverses of the first-depth switches. -/
abbrev switchInvMembers : List Int := [(36), (35), (34), (33), (32), (31), (30), (29), (28), (27), (26), (25), (24), (23), (22), (21), (20), (19), (18), (17), (16), (15)]

theorem switchInvMembers_eq (b : Nat) (hb : b < 22) :
    MxxIR.selectInt (Int.ofNat b) switchInvMembers = ((36 - b : Nat) : Int) := by
  interval_cases b <;> rfl

/-- The inverse of the automorphism of first-depth branch `b`. -/
def kInv (b : Nat) : Nat := (MxxIR.selectInt (Int.ofNat b) switchInvList).toNat

theorem kInv_odd (b : Fin 22) : Odd (kInv b) := by
  fin_cases b <;> decide

theorem kSw_kInv (b : Fin 22) : kSw b * kInv b % (2 * N) = 1 := by
  fin_cases b <;> decide

theorem kSw_coprime (b : Fin 22) : N.Coprime (kSw b) := by
  fin_cases b <;> decide

/-- The `88` columns of block `b` of a row of blocks. -/
def rowBlock {W : Nat} (x : ExactMatrix Q N 1 W) (b : Nat) (h : 88 * b + 88 ≤ W) :
    ExactMatrix Q N 1 88 :=
  rowWindow 88 (88 * b) h x


theorem kSw_cast (b : Nat) (hb : b < 22) :
    ((kSw b : Nat) : Int) = MxxIR.selectInt (Int.ofNat b) switchList := by
  interval_cases b <;> rfl

theorem kInv_cast (b : Nat) (hb : b < 22) :
    ((kInv b : Nat) : Int) = MxxIR.selectInt (Int.ofNat b) switchInvList := by
  interval_cases b <;> rfl

/-- A routing by the automorphism of first-depth branch `b` is the substitution of `kSw b`. -/
theorem routing_kSw {b : Nat} (hb : b < 22) {rows columns : Nat} {x y : ExactMatrix Q N rows columns}
    (h : ringAutomorphismRuns (MxxIR.selectInt (Int.ofNat b) switchList) x y) :
    y = substituteMatrix (kSw b) (kSw_odd ⟨b, hb⟩) x := by
  obtain ⟨hodd, hy⟩ := ringAutomorphismRuns_eq hQ hN h
  rw [hy]
  rfl

/-- A hashed block of the helper key of `k` is its block. -/
theorem helperKey_read {model : HashModel} {key : ByteArray} {k : Int} {c : Nat} (hc : c < 4)
    {y : ExactMatrix Q N 1 44}
    (h : hashSample model dynamicPrefix [.bytes schemaBytes, .bytes helperKeyBytes, .u64Le k,
      .bytes componentBytes, .u64Le (c : Int)] key y) :
    y = helperKey model key k ⟨c / 2, by omega⟩ ⟨c % 2, by omega⟩ := by
  rw [hashSample_eq h, helperKey]
  interval_cases c <;> rfl

/-! ## Coefficient loops -/

set_option maxHeartbeats 4000000 in
/-- A coefficient loop over the `128` table entries whose iteration `i` adds `coefTerm … i` sums
the coefficient terms. -/
theorem coefAccum {body : Nat → ExactMatrix Q N 1 1 → ExactMatrix Q N 1 1 → Prop}
    {tables : Fin 512 → Int} {offset k : Nat} (ho : offset + 128 ≤ 512) {final : ExactMatrix Q N 1 1}
    (hstep : ∀ i s t (hi : i < 128), body i s t → t = s + coefTerm tables offset k ⟨i, hi⟩ ho)
    (h : MxxIR.IterRuns body 128 0 final) :
    final = ∑ j : Fin 128, coefTerm tables offset k j ho := by
  have hacc := IterRuns.accumulate (fun s : ExactMatrix Q N 1 1 ↦ s)
    (fun i ↦ if hi : i < 128 then coefTerm tables offset k ⟨i, hi⟩ ho else 0)
    (fun _ _ ↦ True) trivial (fun i s t hi _ hb ↦ ⟨by
      simp only [dif_pos hi]
      exact hstep i s t hi hb, trivial⟩) h
  obtain ⟨hsum, -⟩ := hacc
  dsimp only at hsum
  rw [hsum, zero_add, ← Fin.sum_univ_eq_sum_range (fun i ↦ if hi : i < 128 then
    coefTerm tables offset k ⟨i, hi⟩ ho else 0)]
  exact Finset.sum_congr rfl fun j _ ↦ by simp [j.isLt]

set_option maxHeartbeats 4000000 in
/-- The constant-coefficient form of `coefAccum`, for the last step.

A coefficient loop over the `128` table entries whose iteration `i` adds `coefTermC … i` sums
the coefficient terms. -/
theorem coefAccumC {body : Nat → ExactMatrix Q N 1 1 → ExactMatrix Q N 1 1 → Prop}
    {tables : Fin 512 → Int} {offset k : Nat} (ho : offset + 128 ≤ 512) {final : ExactMatrix Q N 1 1}
    (hstep : ∀ i s t (hi : i < 128), body i s t → t = s + coefTermC tables offset k ⟨i, hi⟩ ho)
    (h : MxxIR.IterRuns body 128 0 final) :
    final = ∑ j : Fin 128, coefTermC tables offset k j ho := by
  have hacc := IterRuns.accumulate (fun s : ExactMatrix Q N 1 1 ↦ s)
    (fun i ↦ if hi : i < 128 then coefTermC tables offset k ⟨i, hi⟩ ho else 0)
    (fun _ _ ↦ True) trivial (fun i s t hi _ hb ↦ ⟨by
      simp only [dif_pos hi]
      exact hstep i s t hi hb, trivial⟩) h
  obtain ⟨hsum, -⟩ := hacc
  dsimp only at hsum
  rw [hsum, zero_add, ← Fin.sum_univ_eq_sum_range (fun i ↦ if hi : i < 128 then
    coefTermC tables offset k ⟨i, hi⟩ ho else 0)]
  exact Finset.sum_congr rfl fun j _ ↦ by simp [j.isLt]

/-! ## The scalar actions of the LUT branches -/

/-- `W⁻¹` as the graph's constant polynomial. -/
noncomputable def winvPoly : ExactMatrix Q N 1 1 := matrixPolynomial [(77469527647209439214259235281224611949048149835750629526602555268330821212381343721480441750241968331304616539815191276988146174287613644747658263498964462003172995682625109900434768370549872016137303635774952449)]

/-- The public scalar action `V = Dec(W⁻¹ α g)` of a branch with the coefficient sum `α`. -/
noncomputable def scalarV (coef : ExactMatrix Q N 1 1) : ExactMatrix Q N 44 44 :=
  Classical.epsilon fun V ↦ gadgetDecomposeRuns Backend.backend 65536 44
    (matrixMulScalarLeft (matrixMulScalarLeft winvPoly coef) g0) V

theorem scalarV_eq {coef : ExactMatrix Q N 1 1} {g : ExactMatrix Q N 1 44} {V : ExactMatrix Q N 44 44}
    (hg : gadgetMatrixRuns Backend.backend 65536 44 g)
    (hV : gadgetDecomposeRuns Backend.backend 65536 44
      (matrixMulScalarLeft (matrixMulScalarLeft winvPoly coef) g) V) : V = scalarV coef := by
  rw [gadgetMatrixRuns_unique hg g0_runs] at hV
  exact gadgetDecomposeRuns_unique hV (Classical.epsilon_spec ⟨V, hV⟩)

/-- The coefficient sum of the branch of `k` for the table at `offset`. -/
noncomputable def branchCoef (tables : Fin 512 → Int) (offset k : Nat) (ho : offset + 128 ≤ 512) :
    ExactMatrix Q N 1 1 :=
  ∑ j : Fin 128, coefTerm tables offset k j ho

/-- The coefficient sum of the branch of `k` in the last step. -/
noncomputable def branchCoefC (tables : Fin 512 → Int) (offset k : Nat) (ho : offset + 128 ≤ 512) :
    ExactMatrix Q N 1 1 :=
  ∑ j : Fin 128, coefTermC tables offset k j ho

/-- The six rows of a LUT branch sum: the two row components and the four key blocks. -/
abbrev SumState := ExactMatrix Q N 1 44 × ExactMatrix Q N 1 44 × ExactMatrix Q N 1 44 ×
  ExactMatrix Q N 1 44 × ExactMatrix Q N 1 44 × ExactMatrix Q N 1 44 × Unit

/-! ## First-depth branch rows -/

/-- The data a first-depth branch reads: the helpers (two components, by helper member), the
imported branch keys `T` (two rows of blocks), the hash key, the state rows, and the companion
row of every first-depth branch. -/
structure BranchData where
  hf0 : Fin 37 → ExactMatrix Q N 1 44
  hf1 : Fin 37 → ExactMatrix Q N 1 44
  T0 : ExactMatrix Q N 1 11176
  T1 : ExactMatrix Q N 1 11176
  key : ByteArray
  c0 : ExactMatrix Q N 1 44
  c1 : ExactMatrix Q N 1 44
  E : ExactMatrix Q N 1 1936

namespace BranchData

variable (d : BranchData) (model : HashModel) (b : Nat) (hb : b < 22)

/-- The key `T_k` of branch `b`. -/
def Tb0 : ExactMatrix Q N 1 88 := rowBlock d.T0 b (by omega)
def Tb1 : ExactMatrix Q N 1 88 := rowBlock d.T1 b (by omega)

/-- The helper of `k`. -/
def hrow : Fin 2 → ExactMatrix Q N 1 44 := ![d.hf0 ⟨b, by omega⟩, d.hf1 ⟨b, by omega⟩]

/-- The helper of `k⁻¹`. -/
def hinv : Fin 2 → ExactMatrix Q N 1 44 := ![d.hf0 ⟨36 - b, by omega⟩, d.hf1 ⟨36 - b, by omega⟩]

/-- The rows of the switching ciphertext `C_k`. -/
noncomputable def top : Fin 2 → ExactMatrix Q N 1 44 := fun j ↦
  substituteMatrix (kSw b) (kSw_odd ⟨b, hb⟩) (helperKey model d.key (kInv b) 0 j)
noncomputable def bot : Fin 2 → ExactMatrix Q N 1 44 := fun j ↦
  substituteMatrix (kSw b) (kSw_odd ⟨b, hb⟩) (helperKey model d.key (kInv b) 1 j) -
    substituteMatrix (kSw b) (kSw_odd ⟨b, hb⟩) (d.hinv b hb j)

/-- The routed state rows. -/
noncomputable def σc : Fin 2 → ExactMatrix Q N 1 44 :=
  ![substituteMatrix (kSw b) (kSw_odd ⟨b, hb⟩) d.c0, substituteMatrix (kSw b) (kSw_odd ⟨b, hb⟩) d.c1]

/-- The companion row of branch `b`. -/
def Eb : ExactMatrix Q N 1 88 := rowBlock d.E b (by omega)

/-- Component `j` of the row of branch `b` before its scalar action:
`h_k Dec(T_k) + σ_k(c) Dec(C_k) + E_b`. -/
noncomputable def row (j : Fin 2) : ExactMatrix Q N 1 44 :=
  pairMul (d.hrow b hb) (actionDigits (d.Tb0 b hb) (d.Tb1 b hb) j) +
    (pairMul (d.σc b hb) (actionDigits (joinRow (d.top model b hb)) (joinRow (d.bot model b hb)) j) +
      halves (d.Eb b hb) j)

/-- The decompositions of branch `b`: the columns of `T_k` and of `C_k`. -/
def Runs : Prop :=
  (∀ (j : Fin 2) (l : Fin 44), ∃ D' : ExactMatrix Q N (44 + 44) 1,
    gadgetDecomposeRuns Backend.backend 65536 44
      (pairColumn (halves (d.Tb0 b hb) j 0 l) (halves (d.Tb1 b hb) j 0 l)) D' ∧
      compactCenteredRebaseRuns 32768 D' (fun a _ ↦ actionDigits (d.Tb0 b hb) (d.Tb1 b hb) j a l)) ∧
  (∀ (j : Fin 2) (l : Fin 44), ∃ D' : ExactMatrix Q N (44 + 44) 1,
    gadgetDecomposeRuns Backend.backend 65536 44
      (pairColumn (d.top model b hb j 0 l) (d.bot model b hb j 0 l)) D' ∧
      compactCenteredRebaseRuns 32768 D'
        (fun a _ ↦ actionDigits (joinRow (d.top model b hb)) (joinRow (d.bot model b hb)) j a l))

end BranchData

/-! ## The companion rows of the second LUT depth

Second-depth branch `d < 105` switches the keys of internal branch `d % 15` by the first-depth
leaf automorphism of member `15 + d / 15`, and so reuses the companions of that member. -/

/-- The companion members of the second-depth branches. -/
abbrev memberList2 : List Int := [(15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21)]

/-- The internal parents of the second-depth branches. -/
abbrev parentList2 : List Int := [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)]

/-- The switching automorphisms of the second-depth branches. -/
abbrev switchList2 : List Int := [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)]

theorem memberList2_eq (d : Nat) (hd : d < 105) :
    MxxIR.selectInt (Int.ofNat d) memberList2 = ((15 + d / 15 : Nat) : Int) := by
  interval_cases d <;> rfl

theorem parentList2_eq (d : Nat) (hd : d < 105) :
    MxxIR.selectInt (Int.ofNat d) parentList2 = Int.ofNat (d % 15) := by
  interval_cases d <;> rfl

theorem switchList2_eq (d : Nat) (hd : d < 105) :
    MxxIR.selectInt (Int.ofNat d) switchList2 = MxxIR.selectInt (Int.ofNat (15 + d / 15)) switchList := by
  interval_cases d <;> rfl

/-- The first-depth package of second-depth branch `d`. -/
def member2 (d : Nat) (hd : d < 105) : Fin 22 := ⟨15 + d / 15, by omega⟩

/-- The keys of internal branch `e`, as the four blocks `A r j`. -/
def parentKeys (K00 K01 K10 K11 : Fin 15 → ExactMatrix Q N 1 44) (e : Fin 15) :
    Fin 2 → Fin 2 → ExactMatrix Q N 1 44 :=
  ![![K00 e, K01 e], ![K10 e, K11 e]]

/-! ## Second-depth leaf rows -/

/-- The inverse automorphisms of the second-depth switches. -/
abbrev invList2 : List Int := [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)]

/-- The helper members of the inverse second-depth switches. -/
abbrev invMembers2 : List Int := [(21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15)]

/-- The automorphisms of the second-depth leaves: parent switch composed with the leaf switch. -/
abbrev coefList2 : List Int := [(17409), (18433), (19457), (20481), (21505), (22529), (23553), (24577), (25601), (26625), (27649), (28673), (29697), (30721), (31745), (33793), (34817), (35841), (36865), (37889), (38913), (39937), (40961), (41985), (43009), (44033), (45057), (46081), (47105), (48129), (50177), (51201), (52225), (53249), (54273), (55297), (56321), (57345), (58369), (59393), (60417), (61441), (62465), (63489), (64513), (66561), (67585), (68609), (69633), (70657), (71681), (72705), (73729), (74753), (75777), (76801), (77825), (78849), (79873), (80897), (82945), (83969), (84993), (86017), (87041), (88065), (89089), (90113), (91137), (92161), (93185), (94209), (95233), (96257), (97281), (99329), (100353), (101377), (102401), (103425), (104449), (105473), (106497), (107521), (108545), (109569), (110593), (111617), (112641), (113665), (115713), (116737), (117761), (118785), (119809), (120833), (121857), (122881), (123905), (124929), (125953), (126977), (128001), (129025), (130049)]

theorem invList2_eq (d : Nat) (hd : d < 105) :
    MxxIR.selectInt (Int.ofNat d) invList2 = ((kInv (15 + d / 15) : Nat) : Int) := by
  interval_cases d <;> rfl

theorem invMembers2_eq (d : Nat) (hd : d < 105) :
    MxxIR.selectInt (Int.ofNat d) invMembers2 = Int.ofNat (36 - (15 + d / 15)) := by
  interval_cases d <;> rfl

/-- The automorphism of the coefficient of second-depth leaf `d`. -/
def kLeaf2 (d : Nat) : Nat := (MxxIR.selectInt (Int.ofNat d) coefList2).toNat

/-- The data a second-depth leaf reads: the helpers, the imported branch keys `T`, the hash key,
the state rows of the internal branches, and the companion row of every second-depth branch. -/
structure Leaf2Data where
  hf0 : Fin 37 → ExactMatrix Q N 1 44
  hf1 : Fin 37 → ExactMatrix Q N 1 44
  T0 : ExactMatrix Q N 1 11176
  T1 : ExactMatrix Q N 1 11176
  key : ByteArray
  c0 : Fin 15 → ExactMatrix Q N 1 44
  c1 : Fin 15 → ExactMatrix Q N 1 44
  E : ExactMatrix Q N 1 9240

namespace Leaf2Data

variable (d : Leaf2Data) (model : HashModel) (l : Nat) (hl : l < 105)

/-- The key `T` of leaf `l`, block `22 + l`. -/
def Tb0 : ExactMatrix Q N 1 88 := rowBlock d.T0 (22 + l) (by omega)
def Tb1 : ExactMatrix Q N 1 88 := rowBlock d.T1 (22 + l) (by omega)

/-- The helper of the switch `k` of leaf `l`, member `15 + l / 15`. -/
def hrow : Fin 2 → ExactMatrix Q N 1 44 :=
  ![d.hf0 ⟨15 + l / 15, by omega⟩, d.hf1 ⟨15 + l / 15, by omega⟩]

/-- The helper of `k⁻¹`. -/
def hinv : Fin 2 → ExactMatrix Q N 1 44 :=
  ![d.hf0 ⟨36 - (15 + l / 15), by omega⟩, d.hf1 ⟨36 - (15 + l / 15), by omega⟩]

/-- The rows of the switching ciphertext `C_k`. -/
noncomputable def top : Fin 2 → ExactMatrix Q N 1 44 := fun j ↦
  substituteMatrix (kSw (15 + l / 15)) (kSw_odd (member2 l hl))
    (helperKey model d.key (kInv (15 + l / 15)) 0 j)
noncomputable def bot : Fin 2 → ExactMatrix Q N 1 44 := fun j ↦
  substituteMatrix (kSw (15 + l / 15)) (kSw_odd (member2 l hl))
      (helperKey model d.key (kInv (15 + l / 15)) 1 j) -
    substituteMatrix (kSw (15 + l / 15)) (kSw_odd (member2 l hl)) (d.hinv l hl j)

/-- The routed state rows of the parent `l % 15`. -/
noncomputable def σc : Fin 2 → ExactMatrix Q N 1 44 :=
  ![substituteMatrix (kSw (15 + l / 15)) (kSw_odd (member2 l hl)) (d.c0 ⟨l % 15, by omega⟩),
    substituteMatrix (kSw (15 + l / 15)) (kSw_odd (member2 l hl)) (d.c1 ⟨l % 15, by omega⟩)]

/-- The companion row of leaf `l`. -/
def Eb : ExactMatrix Q N 1 88 := rowBlock d.E l (by omega)

/-- Component `j` of the row of leaf `l` before its scalar action. -/
noncomputable def row (j : Fin 2) : ExactMatrix Q N 1 44 :=
  pairMul (d.hrow l hl) (actionDigits (d.Tb0 l hl) (d.Tb1 l hl) j) +
    (pairMul (d.σc l hl) (actionDigits (joinRow (d.top model l hl)) (joinRow (d.bot model l hl)) j) +
      halves (d.Eb l hl) j)

/-- The decompositions of leaf `l`: the columns of `T` and of `C_k`. -/
def Runs : Prop :=
  (∀ (j : Fin 2) (a : Fin 44), ∃ D' : ExactMatrix Q N (44 + 44) 1,
    gadgetDecomposeRuns Backend.backend 65536 44
      (pairColumn (halves (d.Tb0 l hl) j 0 a) (halves (d.Tb1 l hl) j 0 a)) D' ∧
      compactCenteredRebaseRuns 32768 D' (fun r _ ↦ actionDigits (d.Tb0 l hl) (d.Tb1 l hl) j r a)) ∧
  (∀ (j : Fin 2) (a : Fin 44), ∃ D' : ExactMatrix Q N (44 + 44) 1,
    gadgetDecomposeRuns Backend.backend 65536 44
      (pairColumn (d.top model l hl j 0 a) (d.bot model l hl j 0 a)) D' ∧
      compactCenteredRebaseRuns 32768 D'
        (fun r _ ↦ actionDigits (joinRow (d.top model l hl)) (joinRow (d.bot model l hl)) j r a))

end Leaf2Data

end PeProof
