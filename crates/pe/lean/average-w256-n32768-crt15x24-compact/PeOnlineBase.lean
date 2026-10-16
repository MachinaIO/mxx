import PeSetup

/-!
# Online evaluation

Each step multiplies the state encoding by the Dynamic RHS of the attribute symbol it reads and
evaluates its LUT. The generated loops of step `0` compute the ciphertext action `c Dec(C)` column
by column (`sequential_generatedRoot_23`), the companion keys and row of the RHS
(`sequential_generatedRoot_33`), and the LUT branches.
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The digits of column `jj` of the joined RHS ciphertext `(top, bot)`. -/
noncomputable abbrev rhsDigits (top bot : ExactMatrix Q N 1 60) (jj : Fin 60) :
    ExactMatrix Q N (30 + 30) 1 :=
  columnDigits Q (30 + 30) Backend.backend 4096 30 2048 (top 0 jj) (bot 0 jj)

/-! ## The companion row of the product -/

/-- The state of the inner companion loop of the product: the block of the companion row and the
carried expanded companions. -/
abbrev MulBlock := ExactMatrix Q N 1 60 × ExactMatrix Q N 1 30 × ExactMatrix Q N 1 30 × Unit

/-- The index of the companion of member `x`, private digit row `a` (digit row `30 + a`), and
column `jj` in the encryption's companion family. -/
def mulIndex (x : Fin 2) (a : Fin 30) (jj : Fin 60) : Fin 3600 :=
  ⟨1800 * x + 60 * a + jj, by omega⟩

/-- The digit-row sum `z_0 u_0 + z_1 u_1` of the product's companions at column `jj` and private
digit row `a`: the digits are those of key entry `30 + a`. -/
noncomputable def mulCompanionSum (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 30) (x : Fin 2)
    (z0 z1 : Fin 3600 → ExactMatrix Q N 1 30) (jj : Fin 60) (a : Fin 30) : ExactPoly Q N :=
  (z0 (mulIndex x a jj) * keyDigits A ⟨30 + a, by omega⟩ 0 +
    z1 (mulIndex x a jj) * keyDigits A ⟨30 + a, by omega⟩ 1) 0 0

/-- The expansions of a compact family. -/
noncomputable abbrev expandFam {k : Nat} (zc : Fin k → ExactMatrix PQ N 1 1) :
    Fin k → ExactMatrix Q N 1 30 :=
  fun i ↦ expand (zc i)

/-! ## LUT coefficients -/

/-- Term `j` of the LUT coefficient of automorphism `k` for the table at `offset` of the program
data: `X^(t_(offset + j)) σ_k(X^(2N - j))`, as the generated loop computes it. -/
noncomputable def coefTerm (tables : Fin 512 → Int) (offset k : Nat) (j : Fin 256) (h : offset + 256 ≤ 512) :
    ExactMatrix Q N 1 1 :=
  matrixMulScalarLeft (multiplyMonomial (1 : ExactMatrix Q N 1 1) (tables ⟨offset + j, by omega⟩ * 1))
    (ringAutomorphism k (multiplyMonomial (1 : ExactMatrix Q N 1 1) ((65536 - (j : Int)) * 1)))

/-- Term `j` of the LUT coefficient of the last step, which reads its table as constants:
`t_(offset + j) σ_k(X^(2N - j))`. -/
noncomputable def coefTermC (tables : Fin 512 → Int) (offset k : Nat) (j : Fin 256) (h : offset + 256 ≤ 512) :
    ExactMatrix Q N 1 1 :=
  matrixMulScalarLeft (liftInteger (tables ⟨offset + j, by omega⟩))
    (ringAutomorphism k (multiplyMonomial (1 : ExactMatrix Q N 1 1) ((65536 - (j : Int)) * 1)))

/-! ## Column actions -/

/-- The runtime facts of one column `jj` of a column action: its decomposition runs and is
`rhsDigits top bot jj`. -/
def ColumnRuns (top bot : ExactMatrix Q N 1 60) (jj : Fin 60) : Prop :=
  ∃ D' : ExactMatrix Q N (30 + 30) 1,
    gadgetDecomposeRuns Backend.backend 4096 30 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
      compactCenteredRebaseRuns 2048 D' (rhsDigits top bot jj)

set_option maxHeartbeats 4000000 in
/-- A column action loop over `60` columns whose iteration `i` adds `(x Dec(top, bot)_i) e_i` to
each of its `r` rows `x = rows k`: it ends with the action, column by column, and every column
decomposition runs. -/
theorem colAction {σ : Type} {body : Nat → σ → σ → Prop} {r : Nat} (proj : Fin r → σ → ExactMatrix Q N 1 60)
    {rows : Fin r → ExactMatrix Q N 1 60} {top bot : ExactMatrix Q N 1 60} {init final : σ}
    (hinit : ∀ k, proj k init = 0)
    (hstep : ∀ i s t (hi : i < 60), body i s t →
      (∀ k, proj k t = proj k s + (rows k * rhsDigits top bot ⟨i, hi⟩) 0 0 •
        (unitRow ((i : Nat) : Int) : ExactMatrix Q N 1 60)) ∧ ColumnRuns top bot ⟨i, hi⟩)
    (h : MxxIR.IterRuns body 60 init final) (jj : Fin 60) :
    (∀ k, proj k final 0 jj = (rows k * rhsDigits top bot jj) 0 0) ∧ ColumnRuns top bot jj := by
  have hrun : ∀ k, proj k final 0 jj = (rows k * rhsDigits top bot jj) 0 0 := by
    intro k
    have h' : MxxIR.IterRuns body (60 * 1) init final := h
    let F : Fin 60 → Fin 1 → ExactPoly Q N := fun jj _ ↦ (rows k * rhsDigits top bot jj) 0 0
    have hacc := IterRuns.accumulate (proj k)
      (fun i ↦ if h : i < 60 * 1 then
        F ⟨i / 1, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
          ⟨i % 1, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
        (unitRow (((i / 1 : Nat) : Int)) : ExactMatrix Q N 1 60) else 0) (fun _ _ ↦ True) trivial
      (fun i s t hi _ hb ↦ ⟨by
        have h1 := (hstep i s t (by omega) hb).1 k
        simp only [dif_pos hi, Nat.div_one]
        exact h1, trivial⟩) h'
    obtain ⟨hsum, -⟩ := hacc
    dsimp only at hsum
    rw [hsum, hinit k, zero_add, sum_unitRow_entry F jj]
    simp [F]
  refine ⟨fun k ↦ hrun k, ?_⟩
  let Q' : Nat → σ → Prop := fun i _ ↦ ∀ jj : Fin 60, jj.val < i → ColumnRuns top bot jj
  have hacc := IterRuns.accumulate (fun _ : σ ↦ (0 : ExactMatrix Q N 1 60)) (fun _ ↦ 0) Q'
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
abbrev switchList : List Int := [257, 513, 769, 1025, 1281, 1537, 1793, 2049, 2305, 2561, 2817, 3073, 3329, 3585, 3841, 4097, 8193, 12289, 16385, 20481, 24577, 28673, 32769, 36865, 40961, 45057, 49153, 53249, 57345, 61441]

/-- The package members of the first-depth switches. -/
abbrev switchMembers : List Int := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

/-- The automorphism of first-depth branch `b`. -/
def kSw (b : Nat) : Nat := (MxxIR.selectInt (Int.ofNat b) switchList).toNat

theorem kSw_odd (b : Fin 30) : Odd (kSw b) := by
  fin_cases b <;> decide

theorem switchMembers_eq (b : Nat) (hb : b < 30) :
    MxxIR.selectInt (Int.ofNat b) switchMembers = (b : Int) := by
  interval_cases b <;> rfl

/-- The key `A` routed by `σ_k`. -/
noncomputable def routedKey (k : Nat) (hk : Odd k) (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 30) :
    Fin 2 → Fin 2 → ExactMatrix Q N 1 30 :=
  fun r j ↦ substituteMatrix k hk (A r j)

/-- The index of the companion of first-depth branch `b`, private digit row `a`, and column `jj`. -/
def lutIndex (b : Fin 30) (a : Fin 30) (jj : Fin 60) : Fin 54000 :=
  ⟨1800 * b + 60 * a + jj, by omega⟩

/-- The digit-row sum `z_0 u_0 + z_1 u_1` of the companions of first-depth branch `b` at column
`jj` and private digit row `a`, the digits of entry `30 + a` of the routed key. -/
noncomputable def lutCompanionSum (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 30)
    (z0 z1 : Fin 54000 → ExactMatrix Q N 1 30) (b : Fin 30) (jj : Fin 60) (a : Fin 30) :
    ExactPoly Q N :=
  (z0 (lutIndex b a jj) * keyDigits (routedKey (kSw b) (kSw_odd b) A) ⟨30 + a, by omega⟩ 0 +
    z1 (lutIndex b a jj) * keyDigits (routedKey (kSw b) (kSw_odd b) A) ⟨30 + a, by omega⟩ 1) 0 0

/-- The companion row `E_b` of first-depth branch `b`: at column `jj`, the sum over the private
digit rows of its companion digit-row sums. -/
noncomputable def lutCompRow (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 30)
    (z0 z1 : Fin 54000 → ExactMatrix Q N 1 30) (b : Fin 30) : ExactMatrix Q N 1 60 :=
  fun _ jj ↦ ∑ a, lutCompanionSum A z0 z1 b jj a

/-! ## Joined rows and their halves -/

noncomputable def actionDigits (top bot : ExactMatrix Q N 1 60) : Fin 2 → ExactMatrix Q N (30 + 30) 30 :=
  fun j a l ↦ rhsDigits top bot ⟨30 * (j : Nat) + l, by omega⟩ a 0

/-- A column loop of a joined row `c = [row₀ | row₁]` by the joined matrix `(top, bot)` with
components `tk`, `bk`: component `j` of the result is `row` times the digits of the columns of
component `j`, and every column decomposition runs. -/
theorem colActionSplit {c : ExactMatrix Q N 1 60} {row : Fin 2 → ExactMatrix Q N 1 30}
    (hc : concatColumns (row 0) (row 1) c) {top bot : ExactMatrix Q N 1 60}
    {tk bk : Fin 2 → ExactMatrix Q N 1 30} (htop : ∀ p : Fin 60, top 0 p = joinedEntry tk p)
    (hbot : ∀ p : Fin 60, bot 0 p = joinedEntry bk p) {acc : ExactMatrix Q N 1 60}
    (hacc : ∀ jj : Fin 60, acc 0 jj = (c * rhsDigits top bot jj) 0 0 ∧
      ∃ D' : ExactMatrix Q N (30 + 30) 1,
        gadgetDecomposeRuns Backend.backend 4096 30 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
          compactCenteredRebaseRuns 2048 D' (rhsDigits top bot jj)) :
    (∀ j : Fin 2, rowWindow 30 (30 * j) (by omega) acc = pairMul row (actionDigits top bot j)) ∧
      ∀ (j : Fin 2) (l : Fin 30), ∃ D' : ExactMatrix Q N (30 + 30) 1,
        gadgetDecomposeRuns Backend.backend 4096 30 (pairColumn (tk j 0 l) (bk j 0 l)) D' ∧
          compactCenteredRebaseRuns 2048 D' (fun a _ ↦ actionDigits top bot j a l) := by
  refine ⟨fun j ↦ ?_, fun j l ↦ ?_⟩
  · rw [window_mul_columns (m := 30) (k := 30) hc (fun p ↦ (hacc p).1) j (by omega)]
    rfl
  · obtain ⟨D', h1, h2⟩ := (hacc ⟨30 * (j : Nat) + l, by omega⟩).2
    rw [htop, hbot, joinedEntry_pair, joinedEntry_pair] at h1
    refine ⟨D', h1, ?_⟩
    have hcol : rhsDigits top bot ⟨30 * (j : Nat) + l, by omega⟩ =
        fun a _ ↦ actionDigits top bot j a l := by
      funext a b
      rw [Subsingleton.elim b 0]
      rfl
    rwa [hcol] at h2

/-- The two halves of a row, concatenated, are the row. -/
theorem concat_windows {x y : ExactMatrix Q N 1 60}
    (h : concatColumns (rowWindow 30 0 (by omega) x) (rowWindow 30 30 (by omega) x) y) : y = x := by
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
def joinRow (r : Fin 2 → ExactMatrix Q N 1 30) : ExactMatrix Q N 1 60 := fun _ p ↦ joinedEntry r p

theorem concat_joinRow {r : Fin 2 → ExactMatrix Q N 1 30} {y : ExactMatrix Q N 1 60}
    (h : concatColumns (r 0) (r 1) y) : y = joinRow r := by
  funext a p
  rw [Subsingleton.elim a 0, concatColumns_entry h, pair_eta]
  rfl

/-- The halves of a joined row. -/
def halves (x : ExactMatrix Q N 1 60) : Fin 2 → ExactMatrix Q N 1 30 :=
  fun j ↦ rowWindow 30 (30 * j) (by omega) x

/-- Entry `p` of a joined row, by the half it lies in. -/
theorem joinedEntry_split (r : Fin 2 → ExactMatrix Q N 1 30) (p : Fin 60) :
    joinedEntry r p = if h : p.val < 30 then r 0 0 ⟨p.val, h⟩ else r 1 0 ⟨p.val - 30, by omega⟩ := by
  unfold joinedEntry Fin.addCases
  split <;> simp_all <;> rfl

theorem concatColumns_joinRow (r : Fin 2 → ExactMatrix Q N 1 30) :
    concatColumns (r 0) (r 1) (joinRow r) := by
  intro a p
  rw [Subsingleton.elim a 0]
  simp only [joinRow, joinedEntry_split]

theorem joinRow_halves (x : ExactMatrix Q N 1 60) : joinRow (halves x) = x := by
  funext a p
  rw [Subsingleton.elim a 0]
  simp only [joinRow, joinedEntry_split, halves, rowWindow]
  split
  · simp
  · congr 1
    ext
    simp
    omega

theorem entry_halves (x : ExactMatrix Q N 1 60) (p : Fin 60) : x 0 p = joinedEntry (halves x) p := by
  conv_lhs => rw [← joinRow_halves x]
  rfl

theorem rowWindow_full (x : ExactMatrix Q N 1 60) : rowWindow 60 0 (by omega) x = x := by
  funext a p
  rw [Subsingleton.elim a 0]
  simp [rowWindow]


/-! ## The internal branches of the first LUT depth -/

/-- The inverses of the first-depth switches. -/
abbrev switchInvList : List Int := [65281, 65025, 64769, 64513, 64257, 64001, 63745, 63489, 63233, 62977, 62721, 62465, 62209, 61953, 61697, 61441, 57345, 53249, 49153, 45057, 40961, 36865, 32769, 28673, 24577, 20481, 16385, 12289, 8193, 4097]

/-- The helper members of the inverses of the first-depth switches. -/
abbrev switchInvMembers : List Int := [44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15]

theorem switchInvMembers_eq (b : Nat) (hb : b < 30) :
    MxxIR.selectInt (Int.ofNat b) switchInvMembers = ((44 - b : Nat) : Int) := by
  interval_cases b <;> rfl

/-- The inverse of the automorphism of first-depth branch `b`. -/
def kInv (b : Nat) : Nat := (MxxIR.selectInt (Int.ofNat b) switchInvList).toNat

theorem kInv_odd (b : Fin 30) : Odd (kInv b) := by
  fin_cases b <;> decide

theorem kSw_kInv (b : Fin 30) : kSw b * kInv b % (2 * N) = 1 := by
  fin_cases b <;> decide

theorem kSw_coprime (b : Fin 30) : N.Coprime (kSw b) := by
  fin_cases b <;> decide

/-- The `60` columns of block `b` of a row of blocks. -/
def rowBlock {W : Nat} (x : ExactMatrix Q N 1 W) (b : Nat) (h : 60 * b + 60 ≤ W) :
    ExactMatrix Q N 1 60 :=
  rowWindow 60 (60 * b) h x


theorem kSw_cast (b : Nat) (hb : b < 30) :
    ((kSw b : Nat) : Int) = MxxIR.selectInt (Int.ofNat b) switchList := by
  interval_cases b <;> rfl

theorem kInv_cast (b : Nat) (hb : b < 30) :
    ((kInv b : Nat) : Int) = MxxIR.selectInt (Int.ofNat b) switchInvList := by
  interval_cases b <;> rfl

/-- A routing by the automorphism of first-depth branch `b` is the substitution of `kSw b`. -/
theorem routing_kSw {b : Nat} (hb : b < 30) {rows columns : Nat} {x y : ExactMatrix Q N rows columns}
    (h : ringAutomorphismRuns (MxxIR.selectInt (Int.ofNat b) switchList) x y) :
    y = substituteMatrix (kSw b) (kSw_odd ⟨b, hb⟩) x := by
  obtain ⟨hodd, hy⟩ := ringAutomorphismRuns_eq hQ hN h
  rw [hy]
  rfl

/-- A hashed block of the helper key of `k` is its block. -/
theorem helperKey_read {model : HashModel} {key : ByteArray} {k : Int} {c : Nat} (hc : c < 4)
    {y : ExactMatrix Q N 1 30}
    (h : hashSample model dynamicPrefix [.bytes schemaBytes, .bytes helperKeyBytes, .u64Le k,
      .bytes componentBytes, .u64Le (c : Int)] key y) :
    y = helperKey model key k ⟨c / 2, by omega⟩ ⟨c % 2, by omega⟩ := by
  rw [hashSample_eq h, helperKey]
  interval_cases c <;> rfl

/-! ## Coefficient loops -/

set_option maxHeartbeats 4000000 in
/-- A coefficient loop over the `256` table entries whose iteration `i` adds `coefTerm … i` sums
the coefficient terms. -/
theorem coefAccum {body : Nat → ExactMatrix Q N 1 1 → ExactMatrix Q N 1 1 → Prop}
    {tables : Fin 512 → Int} {offset k : Nat} (ho : offset + 256 ≤ 512) {final : ExactMatrix Q N 1 1}
    (hstep : ∀ i s t (hi : i < 256), body i s t → t = s + coefTerm tables offset k ⟨i, hi⟩ ho)
    (h : MxxIR.IterRuns body 256 0 final) :
    final = ∑ j : Fin 256, coefTerm tables offset k j ho := by
  have hacc := IterRuns.accumulate (fun s : ExactMatrix Q N 1 1 ↦ s)
    (fun i ↦ if hi : i < 256 then coefTerm tables offset k ⟨i, hi⟩ ho else 0)
    (fun _ _ ↦ True) trivial (fun i s t hi _ hb ↦ ⟨by
      simp only [dif_pos hi]
      exact hstep i s t hi hb, trivial⟩) h
  obtain ⟨hsum, -⟩ := hacc
  dsimp only at hsum
  rw [hsum, zero_add, ← Fin.sum_univ_eq_sum_range (fun i ↦ if hi : i < 256 then
    coefTerm tables offset k ⟨i, hi⟩ ho else 0)]
  exact Finset.sum_congr rfl fun j _ ↦ by simp [j.isLt]

set_option maxHeartbeats 4000000 in
/-- The constant-coefficient form of `coefAccum`, for the last step.

A coefficient loop over the `256` table entries whose iteration `i` adds `coefTermC … i` sums
the coefficient terms. -/
theorem coefAccumC {body : Nat → ExactMatrix Q N 1 1 → ExactMatrix Q N 1 1 → Prop}
    {tables : Fin 512 → Int} {offset k : Nat} (ho : offset + 256 ≤ 512) {final : ExactMatrix Q N 1 1}
    (hstep : ∀ i s t (hi : i < 256), body i s t → t = s + coefTermC tables offset k ⟨i, hi⟩ ho)
    (h : MxxIR.IterRuns body 256 0 final) :
    final = ∑ j : Fin 256, coefTermC tables offset k j ho := by
  have hacc := IterRuns.accumulate (fun s : ExactMatrix Q N 1 1 ↦ s)
    (fun i ↦ if hi : i < 256 then coefTermC tables offset k ⟨i, hi⟩ ho else 0)
    (fun _ _ ↦ True) trivial (fun i s t hi _ hb ↦ ⟨by
      simp only [dif_pos hi]
      exact hstep i s t hi hb, trivial⟩) h
  obtain ⟨hsum, -⟩ := hacc
  dsimp only at hsum
  rw [hsum, zero_add, ← Fin.sum_univ_eq_sum_range (fun i ↦ if hi : i < 256 then
    coefTermC tables offset k ⟨i, hi⟩ ho else 0)]
  exact Finset.sum_congr rfl fun j _ ↦ by simp [j.isLt]

/-! ## The scalar actions of the LUT branches -/

/-- `W⁻¹` as the graph's constant polynomial. -/
noncomputable def winvPoly : ExactMatrix Q N 1 1 := matrixPolynomial [(71413264965178608184357752453475574199477050220518708989523601544437844709191912168638523104835678117098241)]

/-- The public scalar action `V = Dec(W⁻¹ α g)` of a branch with the coefficient sum `α`. -/
noncomputable def scalarV (coef : ExactMatrix Q N 1 1) : ExactMatrix Q N 30 30 :=
  Classical.epsilon fun V ↦ gadgetDecomposeRuns Backend.backend 4096 30
    (matrixMulScalarLeft (matrixMulScalarLeft winvPoly coef) g0) V

theorem scalarV_eq {coef : ExactMatrix Q N 1 1} {g : ExactMatrix Q N 1 30} {V : ExactMatrix Q N 30 30}
    (hg : gadgetMatrixRuns Backend.backend 4096 30 g)
    (hV : gadgetDecomposeRuns Backend.backend 4096 30
      (matrixMulScalarLeft (matrixMulScalarLeft winvPoly coef) g) V) : V = scalarV coef := by
  rw [gadgetMatrixRuns_unique hg g0_runs] at hV
  exact gadgetDecomposeRuns_unique hV (Classical.epsilon_spec ⟨V, hV⟩)

/-- The coefficient sum of the branch of `k` for the table at `offset`. -/
noncomputable def branchCoef (tables : Fin 512 → Int) (offset k : Nat) (ho : offset + 256 ≤ 512) :
    ExactMatrix Q N 1 1 :=
  ∑ j : Fin 256, coefTerm tables offset k j ho

/-- The coefficient sum of the branch of `k` in the last step. -/
noncomputable def branchCoefC (tables : Fin 512 → Int) (offset k : Nat) (ho : offset + 256 ≤ 512) :
    ExactMatrix Q N 1 1 :=
  ∑ j : Fin 256, coefTermC tables offset k j ho

/-- The six rows of a LUT branch sum: the two row components and the four key blocks. -/
abbrev SumState := ExactMatrix Q N 1 30 × ExactMatrix Q N 1 30 × ExactMatrix Q N 1 30 ×
  ExactMatrix Q N 1 30 × ExactMatrix Q N 1 30 × ExactMatrix Q N 1 30 × Unit

/-! ## First-depth branch rows -/

/-- The data a first-depth branch reads: the helpers (two components, by helper member), the
imported branch keys `T` (the two rows of each internal branch `b < 15` and of each leaf
`15 + i`, as preprocessing publishes them), the hash key, the state rows, and the companion row of
every first-depth branch. -/
structure BranchData where
  hf0 : Fin 45 → ExactMatrix Q N 1 30
  hf1 : Fin 45 → ExactMatrix Q N 1 30
  TI0 : Fin 15 → ExactMatrix Q N 1 60
  TI1 : Fin 15 → ExactMatrix Q N 1 60
  TL0 : Fin 15 → ExactMatrix Q N 1 60
  TL1 : Fin 15 → ExactMatrix Q N 1 60
  key : ByteArray
  c0 : ExactMatrix Q N 1 30
  c1 : ExactMatrix Q N 1 30
  E : Fin 30 → ExactMatrix Q N 1 60

namespace BranchData

variable (d : BranchData) (model : HashModel) (b : Nat) (hb : b < 30)

/-- The key `T_k` of branch `b`. -/
def Tb0 : ExactMatrix Q N 1 60 := if h : b < 15 then d.TI0 ⟨b, h⟩ else d.TL0 ⟨b - 15, by omega⟩
def Tb1 : ExactMatrix Q N 1 60 := if h : b < 15 then d.TI1 ⟨b, h⟩ else d.TL1 ⟨b - 15, by omega⟩

/-- The helper of `k`. -/
def hrow : Fin 2 → ExactMatrix Q N 1 30 := ![d.hf0 ⟨b, by omega⟩, d.hf1 ⟨b, by omega⟩]

/-- The helper of `k⁻¹`. -/
def hinv : Fin 2 → ExactMatrix Q N 1 30 := ![d.hf0 ⟨44 - b, by omega⟩, d.hf1 ⟨44 - b, by omega⟩]

/-- The rows of the switching ciphertext `C_k`. -/
noncomputable def top : Fin 2 → ExactMatrix Q N 1 30 := fun j ↦
  substituteMatrix (kSw b) (kSw_odd ⟨b, hb⟩) (helperKey model d.key (kInv b) 0 j)
noncomputable def bot : Fin 2 → ExactMatrix Q N 1 30 := fun j ↦
  substituteMatrix (kSw b) (kSw_odd ⟨b, hb⟩) (helperKey model d.key (kInv b) 1 j) -
    substituteMatrix (kSw b) (kSw_odd ⟨b, hb⟩) (d.hinv b hb j)

/-- The routed state rows. -/
noncomputable def σc : Fin 2 → ExactMatrix Q N 1 30 :=
  ![substituteMatrix (kSw b) (kSw_odd ⟨b, hb⟩) d.c0, substituteMatrix (kSw b) (kSw_odd ⟨b, hb⟩) d.c1]

/-- The companion row of branch `b`. -/
def Eb : ExactMatrix Q N 1 60 := d.E ⟨b, hb⟩

/-- Component `j` of the row of branch `b` before its scalar action:
`h_k Dec(T_k) + σ_k(c) Dec(C_k) + E_b`. -/
noncomputable def row (j : Fin 2) : ExactMatrix Q N 1 30 :=
  pairMul (d.hrow b hb) (actionDigits (d.Tb0 b hb) (d.Tb1 b hb) j) +
    (pairMul (d.σc b hb) (actionDigits (joinRow (d.top model b hb)) (joinRow (d.bot model b hb)) j) +
      halves (d.Eb b hb) j)

/-- The decompositions of branch `b`: the columns of `T_k` and of `C_k`. -/
def Runs : Prop :=
  (∀ (j : Fin 2) (l : Fin 30), ∃ D' : ExactMatrix Q N (30 + 30) 1,
    gadgetDecomposeRuns Backend.backend 4096 30
      (pairColumn (halves (d.Tb0 b hb) j 0 l) (halves (d.Tb1 b hb) j 0 l)) D' ∧
      compactCenteredRebaseRuns 2048 D' (fun a _ ↦ actionDigits (d.Tb0 b hb) (d.Tb1 b hb) j a l)) ∧
  (∀ (j : Fin 2) (l : Fin 30), ∃ D' : ExactMatrix Q N (30 + 30) 1,
    gadgetDecomposeRuns Backend.backend 4096 30
      (pairColumn (d.top model b hb j 0 l) (d.bot model b hb j 0 l)) D' ∧
      compactCenteredRebaseRuns 2048 D'
        (fun a _ ↦ actionDigits (joinRow (d.top model b hb)) (joinRow (d.bot model b hb)) j a l))

end BranchData

theorem BranchData.Tb0_internal (d : BranchData) {b : Nat} (hb : b < 15) :
    d.Tb0 b (by omega) = d.TI0 ⟨b, hb⟩ := by
  unfold BranchData.Tb0; rw [dif_pos hb]

theorem BranchData.Tb1_internal (d : BranchData) {b : Nat} (hb : b < 15) :
    d.Tb1 b (by omega) = d.TI1 ⟨b, hb⟩ := by
  unfold BranchData.Tb1; rw [dif_pos hb]

theorem BranchData.Tb0_leaf (d : BranchData) {i : Nat} (hi : i < 15) :
    d.Tb0 (15 + i) (by omega) = d.TL0 ⟨i, hi⟩ := by
  unfold BranchData.Tb0; rw [dif_neg (by omega)]; congr 2; omega

theorem BranchData.Tb1_leaf (d : BranchData) {i : Nat} (hi : i < 15) :
    d.Tb1 (15 + i) (by omega) = d.TL1 ⟨i, hi⟩ := by
  unfold BranchData.Tb1; rw [dif_neg (by omega)]; congr 2; omega

/-! ## The companion rows of the second LUT depth

Second-depth branch `d < 225` switches the keys of internal branch `d % 15` by the first-depth
leaf automorphism of member `15 + d / 15`, and so reuses the companions of that member. -/

/-- The companion members of the second-depth branches. -/
abbrev memberList2 : List Int := [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29]

/-- The internal parents of the second-depth branches. -/
abbrev parentList2 : List Int := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

/-- The switching automorphisms of the second-depth branches. -/
abbrev switchList2 : List Int := [4097, 4097, 4097, 4097, 4097, 4097, 4097, 4097, 4097, 4097, 4097, 4097, 4097, 4097, 4097, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 12289, 12289, 12289, 12289, 12289, 12289, 12289, 12289, 12289, 12289, 12289, 12289, 12289, 12289, 12289, 16385, 16385, 16385, 16385, 16385, 16385, 16385, 16385, 16385, 16385, 16385, 16385, 16385, 16385, 16385, 20481, 20481, 20481, 20481, 20481, 20481, 20481, 20481, 20481, 20481, 20481, 20481, 20481, 20481, 20481, 24577, 24577, 24577, 24577, 24577, 24577, 24577, 24577, 24577, 24577, 24577, 24577, 24577, 24577, 24577, 28673, 28673, 28673, 28673, 28673, 28673, 28673, 28673, 28673, 28673, 28673, 28673, 28673, 28673, 28673, 32769, 32769, 32769, 32769, 32769, 32769, 32769, 32769, 32769, 32769, 32769, 32769, 32769, 32769, 32769, 36865, 36865, 36865, 36865, 36865, 36865, 36865, 36865, 36865, 36865, 36865, 36865, 36865, 36865, 36865, 40961, 40961, 40961, 40961, 40961, 40961, 40961, 40961, 40961, 40961, 40961, 40961, 40961, 40961, 40961, 45057, 45057, 45057, 45057, 45057, 45057, 45057, 45057, 45057, 45057, 45057, 45057, 45057, 45057, 45057, 49153, 49153, 49153, 49153, 49153, 49153, 49153, 49153, 49153, 49153, 49153, 49153, 49153, 49153, 49153, 53249, 53249, 53249, 53249, 53249, 53249, 53249, 53249, 53249, 53249, 53249, 53249, 53249, 53249, 53249, 57345, 57345, 57345, 57345, 57345, 57345, 57345, 57345, 57345, 57345, 57345, 57345, 57345, 57345, 57345, 61441, 61441, 61441, 61441, 61441, 61441, 61441, 61441, 61441, 61441, 61441, 61441, 61441, 61441, 61441]

theorem memberList2_eq (d : Nat) (hd : d < 225) :
    MxxIR.selectInt (Int.ofNat d) memberList2 = ((15 + d / 15 : Nat) : Int) := by
  interval_cases d <;> rfl

theorem parentList2_eq (d : Nat) (hd : d < 225) :
    MxxIR.selectInt (Int.ofNat d) parentList2 = Int.ofNat (d % 15) := by
  interval_cases d <;> rfl

theorem switchList2_eq (d : Nat) (hd : d < 225) :
    MxxIR.selectInt (Int.ofNat d) switchList2 = MxxIR.selectInt (Int.ofNat (15 + d / 15)) switchList := by
  interval_cases d <;> rfl

/-- The first-depth package of second-depth branch `d`. -/
def member2 (d : Nat) (hd : d < 225) : Fin 30 := ⟨15 + d / 15, by omega⟩

/-- The keys of internal branch `e`, as the four blocks `A r j`. -/
def parentKeys (K00 K01 K10 K11 : Fin 15 → ExactMatrix Q N 1 30) (e : Fin 15) :
    Fin 2 → Fin 2 → ExactMatrix Q N 1 30 :=
  ![![K00 e, K01 e], ![K10 e, K11 e]]

/-! ## Second-depth leaf rows -/

/-- The inverse automorphisms of the second-depth switches. -/
abbrev invList2 : List Int := [61441, 61441, 61441, 61441, 61441, 61441, 61441, 61441, 61441, 61441, 61441, 61441, 61441, 61441, 61441, 57345, 57345, 57345, 57345, 57345, 57345, 57345, 57345, 57345, 57345, 57345, 57345, 57345, 57345, 57345, 53249, 53249, 53249, 53249, 53249, 53249, 53249, 53249, 53249, 53249, 53249, 53249, 53249, 53249, 53249, 49153, 49153, 49153, 49153, 49153, 49153, 49153, 49153, 49153, 49153, 49153, 49153, 49153, 49153, 49153, 45057, 45057, 45057, 45057, 45057, 45057, 45057, 45057, 45057, 45057, 45057, 45057, 45057, 45057, 45057, 40961, 40961, 40961, 40961, 40961, 40961, 40961, 40961, 40961, 40961, 40961, 40961, 40961, 40961, 40961, 36865, 36865, 36865, 36865, 36865, 36865, 36865, 36865, 36865, 36865, 36865, 36865, 36865, 36865, 36865, 32769, 32769, 32769, 32769, 32769, 32769, 32769, 32769, 32769, 32769, 32769, 32769, 32769, 32769, 32769, 28673, 28673, 28673, 28673, 28673, 28673, 28673, 28673, 28673, 28673, 28673, 28673, 28673, 28673, 28673, 24577, 24577, 24577, 24577, 24577, 24577, 24577, 24577, 24577, 24577, 24577, 24577, 24577, 24577, 24577, 20481, 20481, 20481, 20481, 20481, 20481, 20481, 20481, 20481, 20481, 20481, 20481, 20481, 20481, 20481, 16385, 16385, 16385, 16385, 16385, 16385, 16385, 16385, 16385, 16385, 16385, 16385, 16385, 16385, 16385, 12289, 12289, 12289, 12289, 12289, 12289, 12289, 12289, 12289, 12289, 12289, 12289, 12289, 12289, 12289, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 4097, 4097, 4097, 4097, 4097, 4097, 4097, 4097, 4097, 4097, 4097, 4097, 4097, 4097, 4097]

/-- The helper members of the inverse second-depth switches. -/
abbrev invMembers2 : List Int := [29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]

/-- The automorphisms of the second-depth leaves: parent switch composed with the leaf switch. -/
abbrev coefList2 : List Int := [4353, 4609, 4865, 5121, 5377, 5633, 5889, 6145, 6401, 6657, 6913, 7169, 7425, 7681, 7937, 8449, 8705, 8961, 9217, 9473, 9729, 9985, 10241, 10497, 10753, 11009, 11265, 11521, 11777, 12033, 12545, 12801, 13057, 13313, 13569, 13825, 14081, 14337, 14593, 14849, 15105, 15361, 15617, 15873, 16129, 16641, 16897, 17153, 17409, 17665, 17921, 18177, 18433, 18689, 18945, 19201, 19457, 19713, 19969, 20225, 20737, 20993, 21249, 21505, 21761, 22017, 22273, 22529, 22785, 23041, 23297, 23553, 23809, 24065, 24321, 24833, 25089, 25345, 25601, 25857, 26113, 26369, 26625, 26881, 27137, 27393, 27649, 27905, 28161, 28417, 28929, 29185, 29441, 29697, 29953, 30209, 30465, 30721, 30977, 31233, 31489, 31745, 32001, 32257, 32513, 33025, 33281, 33537, 33793, 34049, 34305, 34561, 34817, 35073, 35329, 35585, 35841, 36097, 36353, 36609, 37121, 37377, 37633, 37889, 38145, 38401, 38657, 38913, 39169, 39425, 39681, 39937, 40193, 40449, 40705, 41217, 41473, 41729, 41985, 42241, 42497, 42753, 43009, 43265, 43521, 43777, 44033, 44289, 44545, 44801, 45313, 45569, 45825, 46081, 46337, 46593, 46849, 47105, 47361, 47617, 47873, 48129, 48385, 48641, 48897, 49409, 49665, 49921, 50177, 50433, 50689, 50945, 51201, 51457, 51713, 51969, 52225, 52481, 52737, 52993, 53505, 53761, 54017, 54273, 54529, 54785, 55041, 55297, 55553, 55809, 56065, 56321, 56577, 56833, 57089, 57601, 57857, 58113, 58369, 58625, 58881, 59137, 59393, 59649, 59905, 60161, 60417, 60673, 60929, 61185, 61697, 61953, 62209, 62465, 62721, 62977, 63233, 63489, 63745, 64001, 64257, 64513, 64769, 65025, 65281]

theorem invList2_eq (d : Nat) (hd : d < 225) :
    MxxIR.selectInt (Int.ofNat d) invList2 = ((kInv (15 + d / 15) : Nat) : Int) := by
  interval_cases d <;> rfl

theorem invMembers2_eq (d : Nat) (hd : d < 225) :
    MxxIR.selectInt (Int.ofNat d) invMembers2 = Int.ofNat (44 - (15 + d / 15)) := by
  interval_cases d <;> rfl

/-- The automorphism of the coefficient of second-depth leaf `d`. -/
def kLeaf2 (d : Nat) : Nat := (MxxIR.selectInt (Int.ofNat d) coefList2).toNat

/-- The data a second-depth leaf reads: the helpers, the imported branch keys `T`, the hash key,
the state rows of the internal branches, and the companion row of every second-depth branch. -/
structure Leaf2Data where
  hf0 : Fin 45 → ExactMatrix Q N 1 30
  hf1 : Fin 45 → ExactMatrix Q N 1 30
  T0 : Fin 225 → ExactMatrix Q N 1 60
  T1 : Fin 225 → ExactMatrix Q N 1 60
  key : ByteArray
  c0 : Fin 15 → ExactMatrix Q N 1 30
  c1 : Fin 15 → ExactMatrix Q N 1 30
  E : Fin 225 → ExactMatrix Q N 1 60

namespace Leaf2Data

variable (d : Leaf2Data) (model : HashModel) (l : Nat) (hl : l < 225)

/-- The key `T` of leaf `l`. -/
def Tb0 : ExactMatrix Q N 1 60 := d.T0 ⟨l, hl⟩
def Tb1 : ExactMatrix Q N 1 60 := d.T1 ⟨l, hl⟩

/-- The helper of the switch `k` of leaf `l`, member `15 + l / 15`. -/
def hrow : Fin 2 → ExactMatrix Q N 1 30 :=
  ![d.hf0 ⟨15 + l / 15, by omega⟩, d.hf1 ⟨15 + l / 15, by omega⟩]

/-- The helper of `k⁻¹`. -/
def hinv : Fin 2 → ExactMatrix Q N 1 30 :=
  ![d.hf0 ⟨44 - (15 + l / 15), by omega⟩, d.hf1 ⟨44 - (15 + l / 15), by omega⟩]

/-- The rows of the switching ciphertext `C_k`. -/
noncomputable def top : Fin 2 → ExactMatrix Q N 1 30 := fun j ↦
  substituteMatrix (kSw (15 + l / 15)) (kSw_odd (member2 l hl))
    (helperKey model d.key (kInv (15 + l / 15)) 0 j)
noncomputable def bot : Fin 2 → ExactMatrix Q N 1 30 := fun j ↦
  substituteMatrix (kSw (15 + l / 15)) (kSw_odd (member2 l hl))
      (helperKey model d.key (kInv (15 + l / 15)) 1 j) -
    substituteMatrix (kSw (15 + l / 15)) (kSw_odd (member2 l hl)) (d.hinv l hl j)

/-- The routed state rows of the parent `l % 15`. -/
noncomputable def σc : Fin 2 → ExactMatrix Q N 1 30 :=
  ![substituteMatrix (kSw (15 + l / 15)) (kSw_odd (member2 l hl)) (d.c0 ⟨l % 15, by omega⟩),
    substituteMatrix (kSw (15 + l / 15)) (kSw_odd (member2 l hl)) (d.c1 ⟨l % 15, by omega⟩)]

/-- The companion row of leaf `l`. -/
def Eb : ExactMatrix Q N 1 60 := d.E ⟨l, hl⟩

/-- Component `j` of the row of leaf `l` before its scalar action. -/
noncomputable def row (j : Fin 2) : ExactMatrix Q N 1 30 :=
  pairMul (d.hrow l hl) (actionDigits (d.Tb0 l hl) (d.Tb1 l hl) j) +
    (pairMul (d.σc l hl) (actionDigits (joinRow (d.top model l hl)) (joinRow (d.bot model l hl)) j) +
      halves (d.Eb l hl) j)

/-- The decompositions of leaf `l`: the columns of `T` and of `C_k`. -/
def Runs : Prop :=
  (∀ (j : Fin 2) (a : Fin 30), ∃ D' : ExactMatrix Q N (30 + 30) 1,
    gadgetDecomposeRuns Backend.backend 4096 30
      (pairColumn (halves (d.Tb0 l hl) j 0 a) (halves (d.Tb1 l hl) j 0 a)) D' ∧
      compactCenteredRebaseRuns 2048 D' (fun r _ ↦ actionDigits (d.Tb0 l hl) (d.Tb1 l hl) j r a)) ∧
  (∀ (j : Fin 2) (a : Fin 30), ∃ D' : ExactMatrix Q N (30 + 30) 1,
    gadgetDecomposeRuns Backend.backend 4096 30
      (pairColumn (d.top model l hl j 0 a) (d.bot model l hl j 0 a)) D' ∧
      compactCenteredRebaseRuns 2048 D'
        (fun r _ ↦ actionDigits (joinRow (d.top model l hl)) (joinRow (d.bot model l hl)) j r a))

end Leaf2Data

abbrev LeafState := ExactMatrix Q N 1 30 × ExactMatrix Q N 1 30 × ExactMatrix Q N 1 60 ×
  ExactMatrix Q N 1 60 × Unit

/-- The leaf family address is `(group, slot, member)`. First-depth leaves use slot zero. -/
def leafLocation1 (member : Fin 15) : Fin 4 × Fin 4 × Fin 15 :=
  (0, 0, member)

/-- Leaf `l` uses group `(1 + l % 15) / 4`, slot `(1 + l % 15) % 4`, member `l / 15`. -/
def leafLocation2 (leaf : Fin 225) : Fin 4 × Fin 4 × Fin 15 :=
  (⟨(1 + leaf.val % 15) / 4, by omega⟩,
    ⟨(1 + leaf.val % 15) % 4, by omega⟩, ⟨leaf.val / 15, by omega⟩)

/-- The original slot index of a four-slot group lane. -/
def chunkSlot (group slot : Fin 4) : Fin 16 := ⟨4 * group.val + slot.val, by omega⟩

theorem chunkSlot_address (slot : Fin 16) :
    chunkSlot ⟨slot.val / 4, by omega⟩ ⟨slot.val % 4, by omega⟩ = slot := by
  apply Fin.ext
  dsimp only [chunkSlot]
  omega

theorem forall16 {pred : Fin 16 → Prop}
    (h0 : pred 0) (h1 : pred 1) (h2 : pred 2) (h3 : pred 3)
    (h4 : pred 4) (h5 : pred 5) (h6 : pred 6) (h7 : pred 7)
    (h8 : pred 8) (h9 : pred 9) (h10 : pred 10) (h11 : pred 11)
    (h12 : pred 12) (h13 : pred 13) (h14 : pred 14) (h15 : pred 15) : ∀ slot, pred slot := by
  intro slot
  fin_cases slot
  all_goals assumption

theorem sum_chunks {Value : Type} [AddCommMonoid Value] (term : Fin 16 → Value) :
    ∑ slot, term slot =
      (∑ slot : Fin 4, term (chunkSlot 0 slot)) +
      (∑ slot : Fin 4, term (chunkSlot 1 slot)) +
      (∑ slot : Fin 4, term (chunkSlot 2 slot)) +
      (∑ slot : Fin 4, term (chunkSlot 3 slot)) := by
  simp only [Fin.sum_univ_succ, Fin.sum_univ_zero]
  abel

theorem sum_member_chunks {Value : Type} [AddCommMonoid Value]
    (term : Fin 15 → Fin 16 → Value) :
    (∑ member, ∑ slot, term member slot) =
      (∑ member, ∑ slot : Fin 4, term member (chunkSlot 0 slot)) +
      (∑ member, ∑ slot : Fin 4, term member (chunkSlot 1 slot)) +
      (∑ member, ∑ slot : Fin 4, term member (chunkSlot 2 slot)) +
      (∑ member, ∑ slot : Fin 4, term member (chunkSlot 3 slot)) := by
  simp_rw [sum_chunks]
  simp only [Finset.sum_add_distrib]

theorem add_chunk_sums {Value : Type} [AddSemigroup Value] (base : Value)
    {first second third fourth left right : Value}
    (h : first + second + third + fourth = left + right) :
    base + first + second + third + fourth = base + left + right := by
  calc
    _ = base + (first + second + third + fourth) := by simp only [add_assoc]
    _ = base + left + right := by rw [h, add_assoc]

/-- Slot zero is the first-depth coefficient; slot `1 + p` is parent `p`'s coefficient. -/
def kSlot (slot member : Nat) : Nat := kSw (15 + member) + 256 * slot

theorem leafLocation2_reconstruct (leaf : Fin 225) :
    15 * (leafLocation2 leaf).2.2.val +
      (4 * (leafLocation2 leaf).1.val + (leafLocation2 leaf).2.1.val - 1) = leaf.val := by
  simp only [leafLocation2]
  omega

abbrev leafSwitches : List Int := [4097, 8193, 12289, 16385, 20481, 24577, 28673, 32769, 36865, 40961, 45057, 49153, 53249, 57345, 61441]
abbrev leafInvSwitches : List Int := [61441, 57345, 53249, 49153, 45057, 40961, 36865, 32769, 28673, 24577, 20481, 16385, 12289, 8193, 4097]

theorem leafSwitchSelect (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member) leafSwitches = Int.ofNat (kSw (15 + member)) := by
  interval_cases member <;> rfl

theorem leafInvSwitchSelect (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member) leafInvSwitches = Int.ofNat (kInv (15 + member)) := by
  interval_cases member <;> rfl

theorem leafMemberSelect (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member) [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29] =
      Int.ofNat (15 + member) := by
  interval_cases member <;> rfl

theorem leafInvMemberSelect (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member) [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15] =
      Int.ofNat (44 - (15 + member)) := by
  interval_cases member <;> rfl

theorem slotSelect0 (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member) [4097, 8193, 12289, 16385, 20481, 24577, 28673, 32769, 36865, 40961, 45057, 49153, 53249, 57345, 61441] =
      Int.ofNat (kSlot 0 member) := by
  interval_cases member <;> rfl

theorem slotSelect1 (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member) [4353, 8449, 12545, 16641, 20737, 24833, 28929, 33025, 37121, 41217, 45313, 49409, 53505, 57601, 61697] =
      Int.ofNat (kSlot 1 member) := by
  interval_cases member <;> rfl

theorem slotSelect2 (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member) [4609, 8705, 12801, 16897, 20993, 25089, 29185, 33281, 37377, 41473, 45569, 49665, 53761, 57857, 61953] =
      Int.ofNat (kSlot 2 member) := by
  interval_cases member <;> rfl

theorem slotSelect3 (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member) [4865, 8961, 13057, 17153, 21249, 25345, 29441, 33537, 37633, 41729, 45825, 49921, 54017, 58113, 62209] =
      Int.ofNat (kSlot 3 member) := by
  interval_cases member <;> rfl

theorem slotSelect4 (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member) [5121, 9217, 13313, 17409, 21505, 25601, 29697, 33793, 37889, 41985, 46081, 50177, 54273, 58369, 62465] =
      Int.ofNat (kSlot 4 member) := by
  interval_cases member <;> rfl

theorem slotSelect5 (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member) [5377, 9473, 13569, 17665, 21761, 25857, 29953, 34049, 38145, 42241, 46337, 50433, 54529, 58625, 62721] =
      Int.ofNat (kSlot 5 member) := by
  interval_cases member <;> rfl

theorem slotSelect6 (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member) [5633, 9729, 13825, 17921, 22017, 26113, 30209, 34305, 38401, 42497, 46593, 50689, 54785, 58881, 62977] =
      Int.ofNat (kSlot 6 member) := by
  interval_cases member <;> rfl

theorem slotSelect7 (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member) [5889, 9985, 14081, 18177, 22273, 26369, 30465, 34561, 38657, 42753, 46849, 50945, 55041, 59137, 63233] =
      Int.ofNat (kSlot 7 member) := by
  interval_cases member <;> rfl

theorem slotSelect8 (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member) [6145, 10241, 14337, 18433, 22529, 26625, 30721, 34817, 38913, 43009, 47105, 51201, 55297, 59393, 63489] =
      Int.ofNat (kSlot 8 member) := by
  interval_cases member <;> rfl

theorem slotSelect9 (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member) [6401, 10497, 14593, 18689, 22785, 26881, 30977, 35073, 39169, 43265, 47361, 51457, 55553, 59649, 63745] =
      Int.ofNat (kSlot 9 member) := by
  interval_cases member <;> rfl

theorem slotSelect10 (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member) [6657, 10753, 14849, 18945, 23041, 27137, 31233, 35329, 39425, 43521, 47617, 51713, 55809, 59905, 64001] =
      Int.ofNat (kSlot 10 member) := by
  interval_cases member <;> rfl

theorem slotSelect11 (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member) [6913, 11009, 15105, 19201, 23297, 27393, 31489, 35585, 39681, 43777, 47873, 51969, 56065, 60161, 64257] =
      Int.ofNat (kSlot 11 member) := by
  interval_cases member <;> rfl

theorem slotSelect12 (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member) [7169, 11265, 15361, 19457, 23553, 27649, 31745, 35841, 39937, 44033, 48129, 52225, 56321, 60417, 64513] =
      Int.ofNat (kSlot 12 member) := by
  interval_cases member <;> rfl

theorem slotSelect13 (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member) [7425, 11521, 15617, 19713, 23809, 27905, 32001, 36097, 40193, 44289, 48385, 52481, 56577, 60673, 64769] =
      Int.ofNat (kSlot 13 member) := by
  interval_cases member <;> rfl

theorem slotSelect14 (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member) [7681, 11777, 15873, 19969, 24065, 28161, 32257, 36353, 40449, 44545, 48641, 52737, 56833, 60929, 65025] =
      Int.ofNat (kSlot 14 member) := by
  interval_cases member <;> rfl

theorem slotSelect15 (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member) [7937, 12033, 16129, 20225, 24321, 28417, 32513, 36609, 40705, 44801, 48897, 52993, 57089, 61185, 65281] =
      Int.ofNat (kSlot 15 member) := by
  interval_cases member <;> rfl

/-- The second-depth leaf in member `member` and slot `parent + 1`. -/
def leafIndex (parent : Fin 15) (member : Fin 15) : Fin 225 :=
  ⟨15 * member + parent, by omega⟩

@[simp] theorem leafIndex_div (parent : Fin 15) (member : Fin 15) :
    (leafIndex parent member).val / 15 = member.val := by
  dsimp only [leafIndex]
  omega

@[simp] theorem leafIndex_mod (parent : Fin 15) (member : Fin 15) :
    (leafIndex parent member).val % 15 = parent.val := by
  dsimp only [leafIndex]
  omega

@[simp] theorem kSlot_zero (member : Nat) : kSlot 0 member = kSw (15 + member) := by
  simp only [kSlot, Nat.mul_zero, Nat.add_zero]

set_option maxRecDepth 10000 in
theorem kSlot_leaf (parent : Fin 15) (member : Fin 15) :
    kSlot (parent.val + 1) member = kLeaf2 (leafIndex parent member) := by
  obtain ⟨parent, hp⟩ := parent
  obtain ⟨member, hm⟩ := member
  interval_cases parent <;> interval_cases member <;> rfl

/-- View the leaves of one internal parent as the fifteen first-depth leaf members. -/
noncomputable def Leaf2Data.memberBranch (d : Leaf2Data) (parent : Fin 15) : BranchData :=
  ⟨d.hf0, d.hf1, fun _ ↦ 0, fun _ ↦ 0,
    fun member ↦ d.T0 (leafIndex parent member), fun member ↦ d.T1 (leafIndex parent member),
    d.key, d.c0 parent, d.c1 parent,
    fun branch ↦ d.E ⟨15 * (branch.val - 15) + parent, by omega⟩⟩

theorem Leaf2Data.memberBranch_E (d : Leaf2Data) (parent : Fin 15) (member : Fin 15) :
    (d.memberBranch parent).Eb (15 + member) (by omega) = d.E (leafIndex parent member) := by
  simp only [memberBranch, BranchData.Eb, leafIndex, Nat.add_sub_cancel_left]

theorem leafIndex_member (parent : Fin 15) (member : Fin 15) :
    member2 (leafIndex parent member) (leafIndex parent member).isLt =
      ⟨15 + member, by omega⟩ := by
  apply Fin.ext
  exact congrArg (15 + ·) (leafIndex_div parent member)

theorem leafIndex_parent (parent : Fin 15) (member : Fin 15) :
    (⟨(leafIndex parent member).val % 15, by omega⟩ : Fin 15) = parent := by
  exact Fin.ext (leafIndex_mod parent member)

set_option maxRecDepth 10000 in
theorem Leaf2Data.memberBranch_top (d : Leaf2Data) (model : HashModel)
    (parent : Fin 15) (member : Fin 15) :
    (d.memberBranch parent).top model (15 + member) (by omega) =
      d.top model (leafIndex parent member) (leafIndex parent member).isLt := by
  let top (branch : Fin 30) : Fin 2 → ExactMatrix Q N 1 30 := fun port ↦
    substituteMatrix (kSw branch) (kSw_odd branch) (helperKey model d.key (kInv branch) 0 port)
  change top ⟨15 + member, by omega⟩ = top (member2 (leafIndex parent member) (leafIndex parent member).isLt)
  rw [leafIndex_member]

set_option maxRecDepth 10000 in
theorem Leaf2Data.memberBranch_bot (d : Leaf2Data) (model : HashModel)
    (parent : Fin 15) (member : Fin 15) :
    (d.memberBranch parent).bot model (15 + member) (by omega) =
      d.bot model (leafIndex parent member) (leafIndex parent member).isLt := by
  let bot (branch : Fin 30) : Fin 2 → ExactMatrix Q N 1 30 := fun port ↦
    substituteMatrix (kSw branch) (kSw_odd branch) (helperKey model d.key (kInv branch) 1 port) -
      substituteMatrix (kSw branch) (kSw_odd branch)
        (![d.hf0 ⟨44 - branch, by omega⟩, d.hf1 ⟨44 - branch, by omega⟩] port)
  change bot ⟨15 + member, by omega⟩ = bot (member2 (leafIndex parent member) (leafIndex parent member).isLt)
  rw [leafIndex_member]

theorem Leaf2Data.memberBranch_hrow (d : Leaf2Data) (parent : Fin 15) (member : Fin 15) :
    (d.memberBranch parent).hrow (15 + member) (by omega) =
      d.hrow (leafIndex parent member) (leafIndex parent member).isLt := by
  let rows (branch : Fin 30) : Fin 2 → ExactMatrix Q N 1 30 :=
    ![d.hf0 ⟨branch, by omega⟩, d.hf1 ⟨branch, by omega⟩]
  change rows ⟨15 + member, by omega⟩ = rows (member2 (leafIndex parent member) (leafIndex parent member).isLt)
  rw [leafIndex_member]

theorem Leaf2Data.memberBranch_σc (d : Leaf2Data) (parent : Fin 15) (member : Fin 15) :
    (d.memberBranch parent).σc (15 + member) (by omega) =
      d.σc (leafIndex parent member) (leafIndex parent member).isLt := by
  let rows (parent : Fin 15) (branch : Fin 30) : Fin 2 → ExactMatrix Q N 1 30 :=
    ![substituteMatrix (kSw branch) (kSw_odd branch) (d.c0 parent),
      substituteMatrix (kSw branch) (kSw_odd branch) (d.c1 parent)]
  change rows parent ⟨15 + member, by omega⟩ =
    rows ⟨(leafIndex parent member).val % 15, by omega⟩
      (member2 (leafIndex parent member) (leafIndex parent member).isLt)
  rw [leafIndex_parent, leafIndex_member]

set_option maxRecDepth 10000 in
theorem Leaf2Data.memberBranch_row (d : Leaf2Data) (model : HashModel)
    (parent : Fin 15) (member : Fin 15) (port : Fin 2) :
    (d.memberBranch parent).row model (15 + member) (by omega) port =
      d.row model (leafIndex parent member) (leafIndex parent member).isLt port := by
  unfold BranchData.row Leaf2Data.row
  rw [BranchData.Tb0_leaf, BranchData.Tb1_leaf, memberBranch_top, memberBranch_bot,
    memberBranch_E, memberBranch_hrow, memberBranch_σc]
  rfl

set_option maxRecDepth 10000 in
theorem Leaf2Data.memberBranch_runs (d : Leaf2Data) (model : HashModel)
    (parent : Fin 15) (member : Fin 15) :
    (d.memberBranch parent).Runs model (15 + member) (by omega) ↔
      d.Runs model (leafIndex parent member) (leafIndex parent member).isLt := by
  unfold BranchData.Runs Leaf2Data.Runs
  rw [BranchData.Tb0_leaf, BranchData.Tb1_leaf, memberBranch_top, memberBranch_bot]
  rfl

/-- Regroup all second-depth leaves by member, without changing their sum. -/
theorem leaf_sum_members {α : Type} [AddCommMonoid α] (term : Fin 225 → α) :
    ∑ leaf, term leaf = ∑ member : Fin 15, ∑ parent : Fin 15, term (leafIndex parent member) := by
  rw [← (finProdFinEquiv : Fin 15 × Fin 15 ≃ Fin 225).sum_comp, Fintype.sum_prod_type]
  apply Finset.sum_congr rfl
  intro member _
  apply Finset.sum_congr rfl
  intro parent _
  congr 1
  apply Fin.ext
  change parent.val + 15 * member.val = 15 * member.val + parent.val
  omega

/-- Slot zero is the first-depth leaf; successor slots select one internal parent. -/
noncomputable def leafSlotData (first : BranchData) (second : Leaf2Data) : Fin 16 → BranchData :=
  Fin.cases first second.memberBranch

/-- A weighted leaf term in the member-major traversal. -/
noncomputable def memberTerm (model : HashModel) (first : BranchData) (second : Leaf2Data)
    (coef : Nat → ExactMatrix Q N 1 1) (member : Fin 15) (slot : Fin 16) (port : Fin 2) :
    ExactMatrix Q N 1 30 :=
  (leafSlotData first second slot).row model (15 + member) (by omega) port *
    scalarV (coef (kSlot slot member))

theorem memberTerm_zero (model : HashModel) (first : BranchData) (second : Leaf2Data)
    (coef : Nat → ExactMatrix Q N 1 1) (member : Fin 15) (port : Fin 2) :
    memberTerm model first second coef member 0 port =
      first.row model (15 + member) (by omega) port * scalarV (coef (kSw (15 + member))) := by
  simp only [memberTerm, leafSlotData, Fin.cases_zero, Fin.val_zero, kSlot_zero]

theorem memberTerm_succ (model : HashModel) (first : BranchData) (second : Leaf2Data)
    (coef : Nat → ExactMatrix Q N 1 1) (member : Fin 15) (parent : Fin 15) (port : Fin 2) :
    memberTerm model first second coef member parent.succ port =
      second.row model (leafIndex parent member) (leafIndex parent member).isLt port *
        scalarV (coef (kLeaf2 (leafIndex parent member))) := by
  simp only [memberTerm, leafSlotData, Fin.cases_succ, Fin.val_succ, kSlot_leaf,
    Leaf2Data.memberBranch_row]

set_option maxRecDepth 10000 in
/-- Member-major accumulation is the original first- and second-depth branch sum. -/
theorem memberTerms_sum (model : HashModel) (first : BranchData) (second : Leaf2Data)
    (coef : Nat → ExactMatrix Q N 1 1) (port : Fin 2) :
    (∑ member : Fin 15, ∑ slot : Fin 16, memberTerm model first second coef member slot port) =
      (∑ member : Fin 15, first.row model (15 + member) (by omega) port *
        scalarV (coef (kSw (15 + member)))) +
      ∑ leaf : Fin 225, second.row model leaf leaf.isLt port * scalarV (coef (kLeaf2 leaf)) := by
  simp only [Fin.sum_univ_succ (n := 15), memberTerm_zero, memberTerm_succ, Finset.sum_add_distrib]
  rw [leaf_sum_members]

theorem slotIndexSelect (parent : Fin 15) (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (Int.ofNat member)
        (List.ofFn fun lane : Fin 15 ↦ Int.ofNat (15 * lane + parent)) =
      Int.ofNat (15 * member + parent) := by
  interval_cases member <;> rfl

theorem slotParentSelect (parent : Fin 15) (member : Nat) (hm : member < 15) :
    MxxIR.selectInt (MxxIR.selectInt (Int.ofNat member)
      (List.ofFn fun lane : Fin 15 ↦ Int.ofNat (15 * lane + parent))) parentList2 =
      Int.ofNat parent := by
  rw [slotIndexSelect parent member hm, parentList2_eq _ (by omega)]
  congr 1
  omega

end PeProof
