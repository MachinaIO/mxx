import MxxIR
import ExponentLut
import ExponentPublic
import ExponentFunctional
import ExponentStacking
import ExponentScale
import ExponentRuntime
import ExponentDynamic
import ExponentAutomorphism
import GenPbc

/-!
# The preprocessing's keys, for any parameter set

The key-side counterpart of `GenRel`: nothing here names the ring dimension `n`, the label length,
the cells per member, the members, the columns, a modulus, a CRT basis, a gadget base or a digit
count. Every loop lemma takes a `∀`-quantified step hypothesis and runs `IterRuns.invariant`;
nothing is unrolled.

* Canonical digits (`decG`, `rdecG`): the decomposition a `gadgetDecomposeRuns` determines and its
  centered rebase; `blockModSwitchG`, the block modulus switch as a function.
* Key forms. `pubKeyG` (a public multiplication), `privKeyG` (a private multiplication moved down by
  a block switch), `alignKeyG` (a label key aligned to a lower modulus), `branchKeyG` (one LUT
  branch: routed by an automorphism, switched, multiplied by the helper keys), `cellKeyG` (a PRF
  cell: `privKeyG ∘ pubKeyG`), `lutKeysG` (the branches of a LUT). Each has an `_of_pre` lemma (the
  preprocessing's two-row stacks with one-row decompositions) and an `_of_stacks` lemma (the online
  evaluation's three-row stacks with two-row decompositions): both sides compute the same function
  of the same inputs, which is every key link between the stages.
* Hash-sampled keys: `labelKeyG` (label public matrices, a `par` over the label members),
  `rhsTopG` (selector tops), `lutAG` (LUT helper keys), `lutTopG` (LUT branch tops).
* Switching ciphertexts: `selector_rhs_gen` (`s top_j + bot_j = p μ (s, 1)_j G + e_j`) and
  `branch_rhs_gen` (`σ(s) top_j + bot_j = τ (s, 1)_j G + e_j`).
* Loops: `iter_add_gen` (a `seq` loop that adds), `rowsum_gen` (the `W` cells of a member),
  `lut_keys_loop_gen` (the branch loop of a LUT's keys).
* Trapdoor: `preimage_sampled_gen`.
-/

namespace RefreshTopProof.Gen

open Mxx.Primitives MxxRuntime MxxExponentLut
open scoped BigOperators

variable {n : Nat}

/-! ## Block modulus switch as a function -/

/-- The block modulus switch as a function (its output does not depend on the side conditions). -/
noncomputable def blockModSwitchG {q p n rows columns : Nat} (basis : List Nat) (plaintext : Int)
    (input : ExactMatrix q n rows columns) : ExactMatrix p n rows columns :=
  fun row column ↦ polynomialOfCoefficients (fun i ↦
    let auxiliary := (basis.filter (fun prime ↦ p % prime != 0)).prod
    let z := Int.ofNat ((input row column).coeff i).val
    let correction := rnsCentered auxiliary (z * rnsInverse auxiliary plaintext)
    ((z - plaintext * correction) / Int.ofNat auxiliary) % Int.ofNat p)

/-- A block modulus switch that runs on one input runs on every input of the same moduli. -/
theorem blockModSwitchRuns_fnG {q p rows columns rows' columns' : Nat} {basis : List Nat}
    {plaintext : Int} {x : ExactMatrix q n rows columns} {y : ExactMatrix p n rows columns}
    (h : blockModSwitchRuns basis plaintext x y) (x' : ExactMatrix q n rows' columns') :
    blockModSwitchRuns basis plaintext x' (blockModSwitchG basis plaintext x' : ExactMatrix p n _ _) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, -⟩ := h
  exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, fun _ _ ↦ rfl⟩

theorem blockModSwitchRuns_eqG {q p rows columns : Nat} {basis : List Nat} {plaintext : Int}
    {x : ExactMatrix q n rows columns} {y : ExactMatrix p n rows columns}
    (h : blockModSwitchRuns basis plaintext x y) : y = blockModSwitchG basis plaintext x :=
  blockModSwitchRuns_unique h (blockModSwitchRuns_fnG h x)

/-! ## Rows, slices and rebases -/

theorem rowOf_addG {q rows columns : Nat} (x y : ExactMatrix q n rows columns) (k : Fin rows) :
    rowOf (x + y) k = rowOf x k + rowOf y k := by
  funext r c
  simp [rowOf]

/-- A one-row slice is the row. -/
theorem slice_rowOfG {q rows columns : Nat} {x : ExactMatrix q n rows columns}
    {out : ExactMatrix q n 1 columns} {r : Nat} (hr : r < rows)
    (h : sliceMatrix x r (r + 1) 0 columns out) : out = rowOf x ⟨r, hr⟩ :=
  sliceMatrix_row (k := ⟨r, hr⟩) h

/-- Row `k` of a two-row stack times `D`. -/
theorem stack2_mul_rowsG {q columns m' : Nat} {x y : ExactMatrix q n 1 columns}
    {z : ExactMatrix q n (1 + 1) columns} (hz : concatRows x y z) (D : ExactMatrix q n columns m') :
    rowOf (z * D) 0 = x * D ∧ rowOf (z * D) 1 = y * D := by
  obtain ⟨h0, h1⟩ := concatRows_two hz
  exact ⟨by rw [rowOf_mul, h0], by rw [rowOf_mul, h1]⟩

/-- A compact rebase within one modulus is the identity. -/
theorem rebase_selfG (hn : 0 < n) {q rows columns bound : Nat} (hq : 1 < q)
    {D' D : ExactMatrix q n rows columns} (hD : compactCenteredRebaseRuns bound D' D) : D = D' := by
  rw [centeredRebaseRuns_reduce hD.2.2.1]
  funext r c
  exact reducePoly_centeredIntLift hq hn _

/-- The compact rebase of a stack is the stack of the rebases. -/
theorem rebase_topG {q p m columns bound : Nat} {D' : ExactMatrix q n (m + m) columns}
    {D : ExactMatrix p n (m + m) columns} (hD : compactCenteredRebaseRuns bound D' D) :
    topRows D = centeredRebase (topRows D') ∧ bottomRows D = centeredRebase (bottomRows D') := by
  rw [hD.2.2.1.2]
  exact ⟨rfl, rfl⟩

/-! ## Canonical digits -/

theorem preimageWithin_zeroG {q rows columns : Nat} (bound : Nat) :
    PreimageWithin (0 : ExactMatrix q n rows columns) bound :=
  ⟨0, by funext r c; simp [reduceMatrix], coeffBound_mono coeffBound_zero (Nat.zero_le _)⟩

/-- Every target has its regular decomposition at an exact layout. -/
theorem gadgetDecomposeRuns_existsG (hn : 0 < n) {q rows columns inner : Nat}
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    (hq : 1 < q) (hbase : base = layout.base) (hdigits : digits = layout.digitCount)
    (hinner : inner = rows * layout.digitCount) (X : ExactMatrix q n rows columns) :
    ∃ D : ExactMatrix q n inner columns, gadgetDecomposeRuns backend base digits X D := by
  refine ⟨castMatrixRows hinner.symm (regularDecomposeMatrix layout X),
    layout, hlayout, hbase, hdigits, hinner, rfl, ?_⟩
  rw [regularGadgetMatrix_reconstruct layout X hq hn hexact, sub_self]
  exact preimageWithin_zeroG _

/-- The decomposition the runtime relation determines. -/
noncomputable def decG {q rows columns inner : Nat} (backend : BackendContext) (base digits : Int)
    (X : ExactMatrix q n rows columns) : ExactMatrix q n inner columns :=
  Classical.epsilon (fun D ↦ gadgetDecomposeRuns backend base digits X D)

theorem decG_eq {q rows columns inner : Nat} {backend : BackendContext} {base digits : Int}
    {X : ExactMatrix q n rows columns} {D : ExactMatrix q n inner columns}
    (h : gadgetDecomposeRuns backend base digits X D) : D = decG backend base digits X :=
  gadgetDecomposeRuns_unique h (Classical.epsilon_spec ⟨D, h⟩)

/-- The rebased digits of a target at `q'`, read at `q`. -/
noncomputable def rdecG {q' q rows columns inner : Nat} (backend : BackendContext) (base digits : Int)
    (X : ExactMatrix q' n rows columns) : ExactMatrix q n inner columns :=
  centeredRebase (decG backend base digits X : ExactMatrix q' n inner columns)

/-- A decomposition followed by a compact rebase gives the rebased canonical digits. -/
theorem rdecG_eq {q' q rows columns inner bound : Nat} {backend : BackendContext}
    {base digits : Int} {X : ExactMatrix q' n rows columns} {d : ExactMatrix q' n inner columns}
    {r : ExactMatrix q n inner columns} (hd : gadgetDecomposeRuns backend base digits X d)
    (hr : compactCenteredRebaseRuns bound d r) : r = rdecG backend base digits X := by
  rw [hr.2.2.1.2, decG_eq hd]
  rfl

/-- Given one decomposition at a modulus, every target of that modulus has its canonical digits. -/
theorem decG_of_runs (hn : 0 < n) {q rows rows' columns columns' inner inner' : Nat}
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    (hq : 1 < q) {Y : ExactMatrix q n rows' columns'} {DY : ExactMatrix q n inner' columns'}
    (hY : gadgetDecomposeRuns backend base digits Y DY) (hinner : inner = rows * layout.digitCount)
    (X : ExactMatrix q n rows columns) :
    gadgetDecomposeRuns backend base digits X (decG (inner := inner) backend base digits X) := by
  obtain ⟨l, hl, hb, hd, _⟩ := hY
  rw [hlayout] at hl
  cases hl
  obtain ⟨D, hD⟩ := gadgetDecomposeRuns_existsG hn layout hlayout hexact hq hb hd hinner X
  rw [← decG_eq hD]
  exact hD

/-! ## Row locality of the decomposition -/

theorem castExactMatrix_entry_congrG {q q' rows rows' columns : Nat} (h : q = q')
    {M : ExactMatrix q n rows columns} {M' : ExactMatrix q n rows' columns} {i : Fin rows}
    {i' : Fin rows'} {j j' : Fin columns} (he : M i j = M' i' j') :
    castExactMatrix h M i j = castExactMatrix h M' i' j' := by
  subst h
  exact he

theorem regularColumnIndex_pairG {q rows : Nat} (layout : RegularLayout q) (r : Fin rows)
    (d : Fin layout.digitCount) :
    regularColumnIndex layout (finProdFinEquiv (r, d)) =
      finProdFinEquiv (r, Fin.castLE (Nat.mul_le_mul_right layout.digitsPerTower
        (Nat.sub_le layout.crtModuli.length layout.droppedModuli)) d) := by
  simp [regularColumnIndex]

theorem regularIndexEquiv_symm_pairG {q rows : Nat} (layout : RegularLayout q) (r : Fin rows)
    (x : Fin (layout.crtModuli.length * layout.digitsPerTower)) :
    (regularIndexEquiv layout rows).symm (finProdFinEquiv (r, x)) =
      (r, ((Equiv.sigmaEquivProd _ _).trans finProdFinEquiv).symm x) := by
  simp [regularIndexEquiv]

/-- The digits of a row depend only on the row. -/
theorem regularDecompose_row_localG {q rows rows' columns : Nat} (layout : RegularLayout q)
    (hexact : layout.droppedModuli = 0) (T : ExactMatrix q n rows columns)
    (T' : ExactMatrix q n rows' columns) (r : Fin rows) (r' : Fin rows')
    (hrow : ∀ c, T r c = T' r' c) (d : Fin layout.digitCount) (c : Fin columns) :
    regularDecomposeMatrix layout T (finProdFinEquiv (r, d)) c =
      regularDecomposeMatrix layout T' (finProdFinEquiv (r', d)) c := by
  unfold regularDecomposeMatrix regularCorrectedTarget
  simp only [hexact, if_true]
  rw [regularColumnIndex_pairG, regularColumnIndex_pairG]
  unfold fullDecomposeMatrix
  apply castExactMatrix_entry_congrG
  rw [regularIndexEquiv_symm_pairG, regularIndexEquiv_symm_pairG]
  simp only [regularDigitsUnflattened]
  congr 1
  exact castExactMatrix_entry_congrG _ (hrow c)

/-- The decomposition of two stacked rows is the stack of their decompositions. -/
theorem decompose_stackG {q m columns : Nat} {backend : BackendContext} {base digits : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0)
    {x y : ExactMatrix q n 1 columns} {z : ExactMatrix q n (1 + 1) columns} (hz : concatRows x y z)
    {Dz : ExactMatrix q n (m + m) columns} {Dx Dy : ExactMatrix q n m columns}
    (hDz : gadgetDecomposeRuns backend base digits z Dz)
    (hDx : gadgetDecomposeRuns backend base digits x Dx)
    (hDy : gadgetDecomposeRuns backend base digits y Dy) :
    topRows Dz = Dx ∧ bottomRows Dz = Dy := by
  obtain ⟨l₁, hl₁, _, _, hw₁, hD₁, _⟩ := hDz
  obtain ⟨l₂, hl₂, _, _, hw₂, hD₂, _⟩ := hDx
  obtain ⟨l₃, hl₃, _, _, hw₃, hD₃, _⟩ := hDy
  rw [hlayout] at hl₁ hl₂ hl₃
  cases hl₁
  cases hl₂
  cases hl₃
  have hm : m = layout.digitCount := by simpa using hw₂
  subst hD₁ hD₂ hD₃
  refine ⟨?_, ?_⟩
  · funext i c
    simp only [topRows, Matrix.submatrix_apply, id, castMatrixRows_apply]
    have hi : (Fin.cast hw₁ (Fin.castAdd m i) : Fin ((1 + 1) * layout.digitCount)) =
        finProdFinEquiv ((0 : Fin (1 + 1)), (Fin.cast hm i)) := by
      ext
      simp [finProdFinEquiv]
    have hi' : (Fin.cast hw₂ i : Fin (1 * layout.digitCount)) =
        finProdFinEquiv ((0 : Fin 1), (Fin.cast hm i)) := by
      ext
      simp [finProdFinEquiv]
    rw [hi, hi']
    exact regularDecompose_row_localG layout hexact z x 0 0
      (fun c ↦ by rw [hz 0 c]; simp) _ c
  · funext i c
    simp only [bottomRows, Matrix.submatrix_apply, id, castMatrixRows_apply]
    have hi : (Fin.cast hw₁ (Fin.natAdd m i) : Fin ((1 + 1) * layout.digitCount)) =
        finProdFinEquiv ((1 : Fin (1 + 1)), (Fin.cast hm i)) := by
      ext
      simp [finProdFinEquiv]
      omega
    have hi' : (Fin.cast hw₃ i : Fin (1 * layout.digitCount)) =
        finProdFinEquiv ((0 : Fin 1), (Fin.cast hm i)) := by
      ext
      simp [finProdFinEquiv]
    rw [hi, hi']
    exact regularDecompose_row_localG layout hexact z y 1 0
      (fun c ↦ by rw [hz 1 c]; simp) _ c

/-- The digits of a two-row stack (decomposed at `q'`, rebased to `q`) are the rebased canonical
digits of its rows. -/
theorem stack_rdecG (hn : 0 < n) {q' q m columns bound : Nat} {backend : BackendContext}
    {base digits : Int} (hq' : 1 < q') (layout : RegularLayout q')
    (hlayout : backend.regularLayout q' n = some layout) (hexact : layout.droppedModuli = 0)
    (hm : m = 1 * layout.digitCount)
    {x y : ExactMatrix q' n 1 columns} {z : ExactMatrix q' n (1 + 1) columns} (hz : concatRows x y z)
    {D' : ExactMatrix q' n (m + m) columns} (hD' : gadgetDecomposeRuns backend base digits z D')
    {D : ExactMatrix q n (m + m) columns} (hD : compactCenteredRebaseRuns bound D' D) :
    topRows D = rdecG (inner := m) backend base digits x ∧
      bottomRows D = rdecG (inner := m) backend base digits y := by
  have hdec := fun (X : ExactMatrix q' n 1 columns) ↦
    decG_of_runs (inner := m) hn layout hlayout hexact hq' hD' hm X
  obtain ⟨t, b⟩ := decompose_stackG layout hlayout hexact hz hD' (hdec x) (hdec y)
  obtain ⟨t', b'⟩ := rebase_topG hD
  exact ⟨by rw [t', t]; rfl, by rw [b', b]; rfl⟩

/-! ## Key forms -/

/-- The four keys `(A₀₀, A₀₁, A₁₀, A₁₁)` of an encoding, as the export's tuple. -/
abbrev KeysG (n q m : Nat) :=
  ExactMatrix q n 1 m × ExactMatrix q n 1 m × ExactMatrix q n 1 m × ExactMatrix q n 1 m × Unit

/-- The four keys of a pair-indexed key matrix as a tuple. -/
def keyTupleG {q m : Nat} (K : Fin 2 → Fin 2 → ExactMatrix q n 1 m) : KeysG n q m :=
  (K 0 0, K 0 1, K 1 0, K 1 1, ())

theorem keyTupleG_add {q m : Nat} (K L : Fin 2 → Fin 2 → ExactMatrix q n 1 m) :
    keyTupleG (K + L) = keyTupleG K + keyTupleG L := rfl

theorem keyTupleG_sum {q m : Nat} (s : Finset Nat) (K : Nat → Fin 2 → Fin 2 → ExactMatrix q n 1 m) :
    keyTupleG (∑ i ∈ s, K i) = ∑ i ∈ s, keyTupleG (K i) := by
  induction s using Finset.induction_on with
  | empty => rfl
  | insert a s ha ih => rw [Finset.sum_insert ha, Finset.sum_insert ha, keyTupleG_add, ih]

/-- The keys of a public multiplication: the keys `A` act on the stacked keys `As` of the
multiplicand through the canonical digits of each column `j`. -/
noncomputable def pubKeyG {q m m' : Nat} (backend : BackendContext) (base digits : Int)
    (A : Fin 2 → Fin 2 → ExactMatrix q n 1 m) (As : Fin 2 → Fin 2 → ExactMatrix q n 1 m') :
    Fin 2 → Fin 2 → ExactMatrix q n 1 m' :=
  fun k j ↦ A k 0 * decG (inner := m) backend base digits (As 0 j) +
    A k 1 * decG (inner := m) backend base digits (As 1 j)

/-- The keys after a private multiplication by an RHS with tops `top` and bottoms `bot` (at `q'`),
moved down to `q'` by a block switch. -/
noncomputable def privKeyG {q q' m m' : Nat} (backend : BackendContext) (base digits : Int)
    (basis : List Nat) (A : Fin 2 → Fin 2 → ExactMatrix q n 1 m)
    (top bot : Fin 2 → ExactMatrix q' n 1 m') : Fin 2 → Fin 2 → ExactMatrix q' n 1 m' :=
  fun k j ↦ blockModSwitchG basis 1
    (A k 0 * rdecG (inner := m) backend base digits (top j) +
      A k 1 * rdecG (inner := m) backend base digits (bot j))

/-- The keys of a PRF cell: the label keys `A` act publicly on the state keys `As`, then the
product is multiplied privately by the selector RHS and moved down. -/
noncomputable def cellKeyG {q q' m m' m'' : Nat} (backend : BackendContext)
    (base digits base' digits' : Int) (basis : List Nat)
    (A : Fin 2 → Fin 2 → ExactMatrix q n 1 m) (As : Fin 2 → Fin 2 → ExactMatrix q n 1 m')
    (top bot : Fin 2 → ExactMatrix q' n 1 m'') : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'' :=
  privKeyG backend base' digits' basis (pubKeyG backend base digits A As) top bot

/-- A label key aligned to a lower modulus: each key times the canonical digits of `X` (the
scaled gadget), block-switched down. -/
noncomputable def alignKeyG {Q q m m' rows : Nat} (backend : BackendContext) (base digits : Int)
    (basis : List Nat) (A : Fin 2 → Fin 2 → ExactMatrix Q n 1 m) (X : ExactMatrix Q n rows m') :
    Fin 2 → Fin 2 → ExactMatrix q n 1 m' :=
  fun k j ↦ blockModSwitchG basis 1 (A k j * decG (inner := m) backend base digits X)

/-- The keys of a LUT branch with automorphism `σ`: the input keys `A` routed by `σ` and
switched by the branch's ciphertexts `sw` to the next modulus, then multiplied by the helper keys
`Ad`. -/
noncomputable def branchKeyG {q q' m m' : Nat} (backend : BackendContext) (base digits : Int)
    (basis : List Nat) (σ : Nat) (A Ad : Fin 2 → Fin 2 → ExactMatrix q n 1 m)
    (sw : Fin 2 → Fin 2 → ExactMatrix q' n 1 m') : Fin 2 → Fin 2 → ExactMatrix q n 1 m' :=
  fun k j ↦
    let Asw : Fin 2 → ExactMatrix q' n 1 m' := fun k' ↦ blockModSwitchG basis 1
      (ringAutomorphism σ (A k' 0) * rdecG (inner := m) backend base digits (sw 0 j) +
        ringAutomorphism σ (A k' 1) * rdecG (inner := m) backend base digits (sw 1 j))
    Ad k 0 * rdecG (inner := m) backend base digits (Asw 0) +
      Ad k 1 * rdecG (inner := m) backend base digits (Asw 1)

/-! ### Public multiplication -/

set_option maxHeartbeats 4000000 in
/-- The keys of a public multiplication as the preprocessing computes them: two-row stacks of the
keys times one-row decompositions of the multiplicand's keys (no rebase: one modulus). -/
theorem pubKeyG_of_pre {q m m' : Nat} {backend : BackendContext} {base digits : Int}
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {S0 S1 : ExactMatrix q n (1 + 1) m}
    (hS0 : concatRows (A 0 0) (A 1 0) S0) (hS1 : concatRows (A 0 1) (A 1 1) S1)
    {As : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {d0 d1 : Fin 2 → ExactMatrix q n m m'}
    (hd0 : ∀ j, gadgetDecomposeRuns backend base digits (As 0 j) (d0 j))
    (hd1 : ∀ j, gadgetDecomposeRuns backend base digits (As 1 j) (d1 j))
    {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixAdd (matrixMul S0 (d0 j)) (matrixMul S1 (d1 j))) 0 1 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixAdd (matrixMul S0 (d0 j)) (matrixMul S1 (d1 j))) 1 2 0 m' (M 1 j)) :
    M = pubKeyG backend base digits A As := by
  funext k j
  obtain ⟨s0a, s0b⟩ := stack2_mul_rowsG hS0 (d0 j)
  obtain ⟨s1a, s1b⟩ := stack2_mul_rowsG hS1 (d1 j)
  simp only [pubKeyG]
  fin_cases k
  · show M 0 j = A 0 0 * _ + A 0 1 * _
    rw [slice_rowOfG (by norm_num) (hM0 j)]
    show rowOf (S0 * d0 j + S1 * d1 j) 0 = _
    rw [rowOf_addG, s0a, s1a, ← decG_eq (hd0 j), ← decG_eq (hd1 j)]
  · show M 1 j = A 1 0 * _ + A 1 1 * _
    rw [slice_rowOfG (by norm_num) (hM1 j)]
    show rowOf (S0 * d0 j + S1 * d1 j) 1 = _
    rw [rowOf_addG, s0b, s1b, ← decG_eq (hd0 j), ← decG_eq (hd1 j)]

set_option maxHeartbeats 4000000 in
/-- The keys of a public multiplication as the online evaluation computes them: the encoding's
three-row stacks times the decomposition of the multiplicand's key stack (rebased within its
modulus), rows `1` and `2`. -/
theorem pubKeyG_of_stacks (hn : 0 < n) {q m m' bound : Nat} {backend : BackendContext}
    {base digits : Int} (hq : 1 < q) (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    (hm : m = 1 * layout.digitCount)
    {x0 x1 : ExactMatrix q n 1 m} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows x0 (A 0 0) l01) (hl : concatRows l01 (A 1 0) l)
    (hr01 : concatRows x1 (A 0 1) r01) (hr : concatRows r01 (A 1 1) r)
    (hboth : concatColumns l r both)
    {As : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {Xs : Fin 2 → ExactMatrix q n (1 + 1) m'}
    (hXs : ∀ j, concatRows (As 0 j) (As 1 j) (Xs j))
    {D' : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xs j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (M 1 j)) :
    M = pubKeyG backend base digits A As := by
  have hdec := fun (X : ExactMatrix q n 1 m') ↦
    decG_of_runs (inner := m) hn layout hlayout hexact hq (hD' 0) hm X
  funext k j
  obtain ⟨t, b⟩ := decompose_stackG layout hlayout hexact (hXs j) (hD' j)
    (hdec (As 0 j)) (hdec (As 1 j))
  have hDj : D j = D' j := rebase_selfG hn hq (hD j)
  obtain ⟨-, s1, s2⟩ := stacked_row hl01 hl hr01 hr hboth (D j)
  simp only [pubKeyG]
  fin_cases k
  · show M 0 j = A 0 0 * _ + A 0 1 * _
    rw [slice_rowOfG (by norm_num) (hM0 j)]
    show rowOf (both * D j) 1 = _
    rw [s1, pairMul, hDj, t, b]
    rfl
  · show M 1 j = A 1 0 * _ + A 1 1 * _
    rw [slice_rowOfG (by norm_num) (hM1 j)]
    show rowOf (both * D j) 2 = _
    rw [s2, pairMul, hDj, t, b]
    rfl

/-! ### Private multiplication -/

set_option maxHeartbeats 4000000 in
/-- The keys of a private multiplication as the preprocessing computes them. -/
theorem privKeyG_of_pre {q q' m m' bound : Nat} {backend : BackendContext} {base digits : Int}
    {basis : List Nat}
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {S0 S1 : ExactMatrix q n (1 + 1) m}
    (hS0 : concatRows (A 0 0) (A 1 0) S0) (hS1 : concatRows (A 0 1) (A 1 1) S1)
    {top bot : Fin 2 → ExactMatrix q' n 1 m'}
    {dT dB : Fin 2 → ExactMatrix q' n m m'} {rT rB : Fin 2 → ExactMatrix q n m m'}
    (hdT : ∀ j, gadgetDecomposeRuns backend base digits (top j) (dT j))
    (hrT : ∀ j, compactCenteredRebaseRuns bound (dT j) (rT j))
    (hdB : ∀ j, gadgetDecomposeRuns backend base digits (bot j) (dB j))
    (hrB : ∀ j, compactCenteredRebaseRuns bound (dB j) (rB j))
    {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixAdd (matrixMul S0 (rT j)) (matrixMul S1 (rB j))) 0 1 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixAdd (matrixMul S0 (rT j)) (matrixMul S1 (rB j))) 1 2 0 m' (M 1 j))
    {A' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    (hA' : ∀ k j, blockModSwitchRuns basis 1 (M k j) (A' k j)) :
    A' = privKeyG backend base digits basis A top bot := by
  funext k j
  obtain ⟨s0a, s0b⟩ := stack2_mul_rowsG hS0 (rT j)
  obtain ⟨s1a, s1b⟩ := stack2_mul_rowsG hS1 (rB j)
  rw [blockModSwitchRuns_eqG (hA' k j)]
  simp only [privKeyG]
  congr 1
  fin_cases k
  · show M 0 j = A 0 0 * _ + A 0 1 * _
    rw [slice_rowOfG (by norm_num) (hM0 j)]
    show rowOf (S0 * rT j + S1 * rB j) 0 = _
    rw [rowOf_addG, s0a, s1a, rdecG_eq (hdT j) (hrT j), rdecG_eq (hdB j) (hrB j)]
  · show M 1 j = A 1 0 * _ + A 1 1 * _
    rw [slice_rowOfG (by norm_num) (hM1 j)]
    show rowOf (S0 * rT j + S1 * rB j) 1 = _
    rw [rowOf_addG, s0b, s1b, rdecG_eq (hdT j) (hrT j), rdecG_eq (hdB j) (hrB j)]

set_option maxHeartbeats 4000000 in
/-- The keys of a private multiplication as the online evaluation computes them. -/
theorem privKeyG_of_stacks (hn : 0 < n) {q q' m m' bound : Nat} {backend : BackendContext}
    {base digits : Int} (hq' : 1 < q') (layout : RegularLayout q')
    (hlayout : backend.regularLayout q' n = some layout) (hexact : layout.droppedModuli = 0)
    (hm : m = 1 * layout.digitCount) {basis : List Nat}
    {c : Fin 2 → ExactMatrix q n 1 m} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows (c 0) (A 0 0) l01) (hl : concatRows l01 (A 1 0) l)
    (hr01 : concatRows (c 1) (A 0 1) r01) (hr : concatRows r01 (A 1 1) r)
    (hboth : concatColumns l r both)
    {top bot : Fin 2 → ExactMatrix q' n 1 m'} {Xb : Fin 2 → ExactMatrix q' n (1 + 1) m'}
    (hXb : ∀ j, concatRows (top j) (bot j) (Xb j))
    {D' : Fin 2 → ExactMatrix q' n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xb j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (M 1 j))
    {A' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    (hA' : ∀ k j, blockModSwitchRuns basis 1 (M k j) (A' k j)) :
    A' = privKeyG backend base digits basis A top bot := by
  funext k j
  obtain ⟨t, b⟩ := stack_rdecG hn hq' layout hlayout hexact hm (hXb j) (hD' j) (hD j)
  obtain ⟨-, s1, s2⟩ := stacked_row hl01 hl hr01 hr hboth (D j)
  rw [blockModSwitchRuns_eqG (hA' k j)]
  simp only [privKeyG]
  congr 1
  fin_cases k
  · show M 0 j = A 0 0 * _ + A 0 1 * _
    rw [slice_rowOfG (by norm_num) (hM0 j)]
    show rowOf (both * D j) 1 = _
    rw [s1, pairMul, t, b]
    rfl
  · show M 1 j = A 1 0 * _ + A 1 1 * _
    rw [slice_rowOfG (by norm_num) (hM1 j)]
    show rowOf (both * D j) 2 = _
    rw [s2, pairMul, t, b]
    rfl

/-! ### Aligned label keys -/

/-- The aligned label keys: each hashed key times the digits of `X`, block-switched down. -/
theorem alignKeyG_of_runs {Q q m m' rows : Nat} {backend : BackendContext} {base digits : Int}
    {basis : List Nat} {A : Fin 2 → Fin 2 → ExactMatrix Q n 1 m} {X : ExactMatrix Q n rows m'}
    {d : ExactMatrix Q n m m'} (hd : gadgetDecomposeRuns backend base digits X d)
    {A' : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hA' : ∀ k j, blockModSwitchRuns basis 1 (matrixMul (A k j) d) (A' k j)) :
    A' = alignKeyG backend base digits basis A X := by
  funext k j
  rw [blockModSwitchRuns_eqG (hA' k j)]
  simp only [alignKeyG]
  rw [← decG_eq hd]
  rfl

/-! ### LUT branches -/

set_option maxHeartbeats 4000000 in
/-- The keys of a LUT branch as the preprocessing computes them (two-row stacks, one-row
decompositions). -/
theorem branchKeyG_of_pre {q q' m m' bound : Nat} {backend : BackendContext} {base digits : Int}
    {basis : List Nat} {index : Int}
    {A Ad : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {a00 a10 a01 a11 : ExactMatrix q n 1 m}
    (hσ00 : ringAutomorphismRuns index (A 0 0) a00) (hσ10 : ringAutomorphismRuns index (A 1 0) a10)
    (hσ01 : ringAutomorphismRuns index (A 0 1) a01) (hσ11 : ringAutomorphismRuns index (A 1 1) a11)
    {S0 S1 T0 T1 : ExactMatrix q n (1 + 1) m}
    (hS0 : concatRows a00 a10 S0) (hS1 : concatRows a01 a11 S1)
    (hT0 : concatRows (Ad 0 0) (Ad 1 0) T0) (hT1 : concatRows (Ad 0 1) (Ad 1 1) T1)
    {sw : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    {dA dB dC dE : Fin 2 → ExactMatrix q' n m m'} {rA rB rC rE : Fin 2 → ExactMatrix q n m m'}
    (hdA : ∀ j, gadgetDecomposeRuns backend base digits (sw 0 j) (dA j))
    (hrA : ∀ j, compactCenteredRebaseRuns bound (dA j) (rA j))
    (hdB : ∀ j, gadgetDecomposeRuns backend base digits (sw 1 j) (dB j))
    (hrB : ∀ j, compactCenteredRebaseRuns bound (dB j) (rB j))
    {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixAdd (matrixMul S0 (rA j)) (matrixMul S1 (rB j))) 0 1 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixAdd (matrixMul S0 (rA j)) (matrixMul S1 (rB j))) 1 2 0 m' (M 1 j))
    {Asw : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    (hAsw : ∀ k j, blockModSwitchRuns basis 1 (M k j) (Asw k j))
    (hdC : ∀ j, gadgetDecomposeRuns backend base digits (Asw 0 j) (dC j))
    (hrC : ∀ j, compactCenteredRebaseRuns bound (dC j) (rC j))
    (hdE : ∀ j, gadgetDecomposeRuns backend base digits (Asw 1 j) (dE j))
    (hrE : ∀ j, compactCenteredRebaseRuns bound (dE j) (rE j))
    {K : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hK0 : ∀ j, sliceMatrix (matrixAdd (matrixMul T0 (rC j)) (matrixMul T1 (rE j))) 0 1 0 m' (K 0 j))
    (hK1 : ∀ j, sliceMatrix (matrixAdd (matrixMul T0 (rC j)) (matrixMul T1 (rE j))) 1 2 0 m' (K 1 j)) :
    K = branchKeyG backend base digits basis index.toNat A Ad sw := by
  have hσ : ∀ {x y : ExactMatrix q n 1 m}, ringAutomorphismRuns index x y →
      y = ringAutomorphism index.toNat x := fun hxy ↦ hxy.2.2.2
  have hA : ∀ k j, Asw k j = blockModSwitchG basis 1 (M k j) := fun k j ↦
    blockModSwitchRuns_eqG (hAsw k j)
  have hM : ∀ k j, M k j =
      ringAutomorphism index.toNat (A k 0) * rdecG (inner := m) backend base digits (sw 0 j) +
      ringAutomorphism index.toNat (A k 1) * rdecG (inner := m) backend base digits (sw 1 j) := by
    intro k j
    obtain ⟨s0a, s0b⟩ := stack2_mul_rowsG hS0 (rA j)
    obtain ⟨s1a, s1b⟩ := stack2_mul_rowsG hS1 (rB j)
    fin_cases k
    · show M 0 j = ringAutomorphism index.toNat (A 0 0) * _ + ringAutomorphism index.toNat (A 0 1) * _
      rw [slice_rowOfG (by norm_num) (hM0 j)]
      show rowOf (S0 * rA j + S1 * rB j) 0 = _
      rw [rowOf_addG, s0a, s1a, rdecG_eq (hdA j) (hrA j), rdecG_eq (hdB j) (hrB j), hσ hσ00,
        hσ hσ01]
    · show M 1 j = ringAutomorphism index.toNat (A 1 0) * _ + ringAutomorphism index.toNat (A 1 1) * _
      rw [slice_rowOfG (by norm_num) (hM1 j)]
      show rowOf (S0 * rA j + S1 * rB j) 1 = _
      rw [rowOf_addG, s0b, s1b, rdecG_eq (hdA j) (hrA j), rdecG_eq (hdB j) (hrB j), hσ hσ10,
        hσ hσ11]
  funext k j
  obtain ⟨t0a, t0b⟩ := stack2_mul_rowsG hT0 (rC j)
  obtain ⟨t1a, t1b⟩ := stack2_mul_rowsG hT1 (rE j)
  simp only [branchKeyG]
  fin_cases k
  · show K 0 j = Ad 0 0 * _ + Ad 0 1 * _
    rw [slice_rowOfG (by norm_num) (hK0 j)]
    show rowOf (T0 * rC j + T1 * rE j) 0 = _
    rw [rowOf_addG, t0a, t1a, rdecG_eq (hdC j) (hrC j), rdecG_eq (hdE j) (hrE j), hA, hA, hM, hM]
  · show K 1 j = Ad 1 0 * _ + Ad 1 1 * _
    rw [slice_rowOfG (by norm_num) (hK1 j)]
    show rowOf (T0 * rC j + T1 * rE j) 1 = _
    rw [rowOf_addG, t0b, t1b, rdecG_eq (hdC j) (hrC j), rdecG_eq (hdE j) (hrE j), hA, hA, hM, hM]

set_option maxHeartbeats 4000000 in
/-- The keys of a LUT branch as the online evaluation computes them (three-row stacks). -/
theorem branchKeyG_of_stacks (hn : 0 < n) {q q' m m' bound : Nat} {backend : BackendContext}
    {base digits : Int} (hq' : 1 < q') (layout : RegularLayout q')
    (hlayout : backend.regularLayout q' n = some layout) (hexact : layout.droppedModuli = 0)
    (hm : m = 1 * layout.digitCount)
    {c : Fin 2 → ExactMatrix q n 1 m} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {index : Int} {rc0 rA00 rA10 rc1 rA01 rA11 : ExactMatrix q n 1 m}
    (hr0 : ringAutomorphismRuns index (c 0) rc0) (hr1 : ringAutomorphismRuns index (A 0 0) rA00)
    (hr2 : ringAutomorphismRuns index (A 1 0) rA10) (hr3 : ringAutomorphismRuns index (c 1) rc1)
    (hr4 : ringAutomorphismRuns index (A 0 1) rA01) (hr5 : ringAutomorphismRuns index (A 1 1) rA11)
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows rc0 rA00 l01) (hl : concatRows l01 rA10 l)
    (hr01 : concatRows rc1 rA01 r01) (hr : concatRows r01 rA11 r) (hboth : concatColumns l r both)
    {sw : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'} {Xb : Fin 2 → ExactMatrix q' n (1 + 1) m'}
    (hXb : ∀ j, concatRows (sw 0 j) (sw 1 j) (Xb j))
    {D' : Fin 2 → ExactMatrix q' n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xb j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (M 1 j))
    {basis : List Nat} {Asw : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    (hAsw : ∀ k j, blockModSwitchRuns basis 1 (M k j) (Asw k j))
    {Z : Fin 2 → ExactMatrix q' n (1 + 1) m'} (hZ : ∀ j, concatRows (Asw 0 j) (Asw 1 j) (Z j))
    {E' : Fin 2 → ExactMatrix q' n (m + m) m'}
    (hE' : ∀ j, gadgetDecomposeRuns backend base digits (Z j) (E' j))
    {E : Fin 2 → ExactMatrix q n (m + m) m'} (hE : ∀ j, compactCenteredRebaseRuns bound (E' j) (E j))
    {h : Fin 2 → ExactMatrix q n 1 m} {Ad : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {hl01' hr01' : ExactMatrix q n (1 + 1) m} {hl' hr' : ExactMatrix q n (1 + 1 + 1) m}
    {H : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hhl01 : concatRows (h 0) (Ad 0 0) hl01') (hhl : concatRows hl01' (Ad 1 0) hl')
    (hhr01 : concatRows (h 1) (Ad 0 1) hr01') (hhr : concatRows hr01' (Ad 1 1) hr')
    (hH : concatColumns hl' hr' H)
    {K : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hK0 : ∀ j, sliceMatrix (matrixMul H (E j)) 1 2 0 m' (K 0 j))
    (hK1 : ∀ j, sliceMatrix (matrixMul H (E j)) 2 3 0 m' (K 1 j)) :
    K = branchKeyG backend base digits basis index.toNat A Ad sw := by
  have hσ : ∀ {x y : ExactMatrix q n 1 m}, ringAutomorphismRuns index x y →
      y = ringAutomorphism index.toNat x := fun hxy ↦ hxy.2.2.2
  have hDtop := fun j ↦ stack_rdecG hn hq' layout hlayout hexact hm (hXb j) (hD' j) (hD j)
  have hEtop := fun j ↦ stack_rdecG hn hq' layout hlayout hexact hm (hZ j) (hE' j) (hE j)
  have hM : ∀ k j, M k j =
      ringAutomorphism index.toNat (A k 0) * rdecG (inner := m) backend base digits (sw 0 j) +
      ringAutomorphism index.toNat (A k 1) * rdecG (inner := m) backend base digits (sw 1 j) := by
    intro k j
    obtain ⟨-, s1, s2⟩ := stacked_row hl01 hl hr01 hr hboth (D j)
    obtain ⟨dt, db⟩ := hDtop j
    fin_cases k
    · show M 0 j = ringAutomorphism index.toNat (A 0 0) * _ + ringAutomorphism index.toNat (A 0 1) * _
      rw [slice_rowOfG (by norm_num) (hM0 j)]
      show rowOf (both * D j) 1 = _
      rw [s1, pairMul, dt, db, hσ hr1, hσ hr4]
      rfl
    · show M 1 j = ringAutomorphism index.toNat (A 1 0) * _ + ringAutomorphism index.toNat (A 1 1) * _
      rw [slice_rowOfG (by norm_num) (hM1 j)]
      show rowOf (both * D j) 2 = _
      rw [s2, pairMul, dt, db, hσ hr2, hσ hr5]
      rfl
  have hA : ∀ k j, Asw k j = blockModSwitchG basis 1 (M k j) := fun k j ↦
    blockModSwitchRuns_eqG (hAsw k j)
  funext k j
  obtain ⟨-, s1, s2⟩ := stacked_row hhl01 hhl hhr01 hhr hH (E j)
  obtain ⟨et, eb⟩ := hEtop j
  simp only [branchKeyG]
  fin_cases k
  · show K 0 j = Ad 0 0 * _ + Ad 0 1 * _
    rw [slice_rowOfG (by norm_num) (hK0 j)]
    show rowOf (H * E j) 1 = _
    rw [s1, pairMul, et, eb, hA, hA, hM, hM]
    rfl
  · show K 1 j = Ad 1 0 * _ + Ad 1 1 * _
    rw [slice_rowOfG (by norm_num) (hK1 j)]
    show rowOf (H * E j) 2 = _
    rw [s2, pairMul, et, eb, hA, hA, hM, hM]
    rfl

/-! ### Composite key forms -/

/-- The keys of a PRF cell from its public and private parts (both stages). -/
theorem cellKeyG_of_parts {q q' m m' m'' : Nat} {backend : BackendContext}
    {base digits base' digits' : Int} {basis : List Nat}
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {As : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    {top bot : Fin 2 → ExactMatrix q' n 1 m''} {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    {A' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m''}
    (hM : M = pubKeyG backend base digits A As)
    (hA' : A' = privKeyG backend base' digits' basis M top bot) :
    A' = cellKeyG backend base digits base' digits' basis A As top bot := by
  rw [hA', hM]
  rfl

/-- **The key link.** Two stages that compute keys by the same key form from equal inputs compute
equal keys (`F` is any of the forms above, applied to the stage's own inputs). -/
theorem key_link {α β : Type} (F : α → β) {x x' : α} {K K' : β} (hK : K = F x) (hK' : K' = F x')
    (hx : x = x') : K = K' := by
  rw [hK, hK', hx]

/-- The four keys of a tuple read back. -/
theorem keyTupleG_inj {q m : Nat} {K L : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    (h : keyTupleG K = keyTupleG L) : K = L := by
  simp only [keyTupleG, Prod.mk.injEq] at h
  obtain ⟨h00, h01, h10, h11, -⟩ := h
  funext k j
  fin_cases k <;> fin_cases j
  · exact h00
  · exact h01
  · exact h10
  · exact h11

/-! ## Hash-sampled keys -/

/-- `member`, the tag component naming a family member. -/
abbrev memberTagG : Blob := [109, 101, 109, 98, 101, 114]
/-- `component`, the tag component naming a block. -/
abbrev componentTagG : Blob := [99, 111, 109, 112, 111, 110, 101, 110, 116]

/-- The `2 × 2` public matrix of `1 × m` blocks sampled under `tagPrefix` and the leading
components `pre`; block `(k, j)` has the trailing components `component, 2 k + j` (definitionally
`GenEnc.hashPair`). -/
noncomputable def keyPairG (model : HashModel) (key : ByteArray) (q m : Nat) (tagPrefix : Blob)
    (pre : List HashTagComponent) : Fin 2 → Fin 2 → ExactMatrix q n 1 m :=
  fun k j ↦ model.sample q n 1 m key (completeHashTag tagPrefix
    (pre ++ [.bytes componentTagG, .u64Le (2 * (k : Int) + (j : Int))]))

/-- The label keys of member `i` (`rns-bgg/input/…` with `member, i`). -/
noncomputable def labelKeysG (model : HashModel) (key : ByteArray) (q m : Nat) (tagPrefix : Blob)
    (i : Int) : Fin 2 → Fin 2 → ExactMatrix q n 1 m :=
  keyPairG model key q m tagPrefix [.bytes memberTagG, .u64Le i]

/-- The LUT helper keys `A_σ` at `level` (`rns-bgg/lut-a` with `level, σ`). -/
noncomputable def lutAG (model : HashModel) (key : ByteArray) (q m : Nat) (tagPrefix : Blob)
    (level σ : Int) : Fin 2 → Fin 2 → ExactMatrix q n 1 m :=
  keyPairG model key q m tagPrefix [.u64Le level, .integer σ]

/-- The pair of tops of member `i` of a selector or branch family (`… member, i, component, j`). -/
noncomputable def topPairG (model : HashModel) (key : ByteArray) (q m : Nat) (tagPrefix : Blob)
    (pre : List HashTagComponent) : Fin 2 → ExactMatrix q n 1 m :=
  fun j ↦ model.sample q n 1 m key (completeHashTag tagPrefix
    (pre ++ [.bytes componentTagG, .u64Le (j : Int)]))

/-- A sample under the components `pre ++ [component, 2 k + j]` is block `(k, j)`. -/
theorem keyPairG_of_hash {q m : Nat} {model : HashModel} {key : ByteArray} {tagPrefix : Blob}
    {pre : List HashTagComponent} (k j : Fin 2) {c : Int} (hc : c = 2 * (k : Int) + (j : Int))
    {A : ExactMatrix q n 1 m}
    (h : hashSample model tagPrefix (pre ++ [.bytes componentTagG, .u64Le c]) key A) :
    A = keyPairG model key q m tagPrefix pre k j := by
  rw [hashSample_eq h, hc]
  rfl

/-- The four samples of a public matrix are its blocks. -/
theorem keyPairG_of_samples {q m : Nat} {model : HashModel} {key : ByteArray} {tagPrefix : Blob}
    {pre : List HashTagComponent} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    (h : ∀ k j : Fin 2, hashSample model tagPrefix
      (pre ++ [.bytes componentTagG, .u64Le (2 * (k : Int) + (j : Int))]) key (A k j)) :
    A = keyPairG model key q m tagPrefix pre :=
  funext fun k ↦ funext fun j ↦ keyPairG_of_hash k j rfl (h k j)

/-- A sample under the components `pre ++ [component, j]` is top `j`. -/
theorem topPairG_of_hash {q m : Nat} {model : HashModel} {key : ByteArray} {tagPrefix : Blob}
    {pre : List HashTagComponent} (j : Fin 2) {c : Int} (hc : c = (j : Int))
    {t : ExactMatrix q n 1 m}
    (h : hashSample model tagPrefix (pre ++ [.bytes componentTagG, .u64Le c]) key t) :
    t = topPairG model key q m tagPrefix pre j := by
  rw [hashSample_eq h, hc]
  rfl

/-- **The label family** (a `par` over the members): member `i` of a family sampled with
`member, i, component, c` is block `c` of the label keys of member `i`. -/
theorem label_family_gen {q m count : Nat} {model : HashModel} {key : ByteArray}
    {tagPrefix : Blob} {c : Int} {out : Fin count → ExactMatrix q n 1 m}
    (h : ∀ i : Fin count, hashSample model tagPrefix
      [.bytes memberTagG, .u64Le (Int.ofNat i), .bytes componentTagG, .u64Le c] key (out i)) :
    out = fun i : Fin count ↦ model.sample q n 1 m key (completeHashTag tagPrefix
      [.bytes memberTagG, .u64Le (Int.ofNat i), .bytes componentTagG, .u64Le c]) :=
  funext fun i ↦ hashSample_eq (h i)

/-- The label families of the four blocks are the label keys of each member. -/
theorem label_families_gen {q m count : Nat} {model : HashModel} {key : ByteArray}
    {tagPrefix : Blob} {out : Fin 2 → Fin 2 → Fin count → ExactMatrix q n 1 m}
    (h : ∀ (k j : Fin 2) (i : Fin count), hashSample model tagPrefix
      [.bytes memberTagG, .u64Le (Int.ofNat i), .bytes componentTagG,
        .u64Le (2 * (k : Int) + (j : Int))] key (out k j i)) :
    ∀ i : Fin count, (fun k j ↦ out k j i) = labelKeysG model key q m tagPrefix (Int.ofNat i) :=
  fun i ↦ funext fun k ↦ funext fun j ↦ keyPairG_of_hash k j rfl (h k j i)

/-! ## Switching ciphertexts -/

theorem reduce_one_matrixG {q : Nat} : (1 : ExactMatrix q n 1 1) = fun _ _ ↦ reducePoly q n 1 := by
  funext r c
  obtain rfl : r = 0 := Subsingleton.elim _ _
  obtain rfl : c = 0 := Subsingleton.elim _ _
  simp

theorem matrixPolynomial_reduceG {q : Nat} (k : Int) :
    (matrixPolynomial [k] : ExactMatrix q n 1 1) = fun _ _ ↦ reducePoly q n (k : ErrorPoly n) := by
  funext r c
  simp [matrixPolynomial]

set_option maxHeartbeats 4000000 in
/-- **A selector RHS** (one column `j`, as the preprocessing computes it):
`bot = u G p μ - s top + e`, so `s top + bot = p μ u G + e` for the column's unit `u ∈ {s, 1}`.
The scalar `p`, the message `μ` and the gadget `G` are arbitrary. -/
theorem selector_rhs_gen {q m : Nat} {sp μ u : ErrorPoly n} {p : Int}
    {s uM msg : ExactMatrix q n 1 1} (hs : s = fun _ _ ↦ reducePoly q n sp)
    (hu : uM = fun _ _ ↦ reducePoly q n u) (hmsg : msg = fun _ _ ↦ reducePoly q n μ)
    {G top e : ExactMatrix q n 1 m} {sigma : Rat} {cutoff : Int} (he : gaussianSample sigma cutoff e) :
    ∃ eI : ErrorMatrix n 1 m, CoeffBound eI cutoff.toNat ∧
      reducePoly q n sp • top + matrixAdd (matrixSub (matrixMulScalarRight (matrixMulScalarRight
          (matrixMulScalarLeft uM G) (matrixPolynomial [p])) msg) (matrixMulScalarLeft s top)) e =
        reducePoly q n ((p : ErrorPoly n) * μ * u) • G + reduceMatrix q n 1 m eI := by
  obtain ⟨E, hE, hb⟩ := gaussian_bound he
  refine ⟨E, hb, ?_⟩
  subst hs hu hmsg hE
  simp only [matrixMulScalarRight_eq, matrixMulScalarLeft_eq, matrixPolynomial_reduceG]
  funext r c
  simp only [matrixAdd, matrixSub, Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply,
    smul_eq_mul, map_mul]
  ring

/-- The selector RHS of both columns: `s top_j + bot_j = p μ (s, 1)_j G + e_j`. -/
theorem selector_rhs_pair_gen {q m : Nat} {sp μ : ErrorPoly n} {p : Int}
    {s one msg : ExactMatrix q n 1 1} (hs : s = fun _ _ ↦ reducePoly q n sp) (hone : one = 1)
    (hmsg : msg = fun _ _ ↦ reducePoly q n μ)
    {G : ExactMatrix q n 1 m} {top bot e : Fin 2 → ExactMatrix q n 1 m} {sigma : Rat} {cutoff : Int}
    (he : ∀ j, gaussianSample sigma cutoff (e j))
    (hbot0 : bot 0 = matrixAdd (matrixSub (matrixMulScalarRight (matrixMulScalarRight
          (matrixMulScalarLeft s G) (matrixPolynomial [p])) msg) (matrixMulScalarLeft s (top 0))) (e 0))
    (hbot1 : bot 1 = matrixAdd (matrixSub (matrixMulScalarRight (matrixMulScalarRight
          (matrixMulScalarLeft one G) (matrixPolynomial [p])) msg) (matrixMulScalarLeft s (top 1))) (e 1)) :
    ∃ eB : Fin 2 → ErrorMatrix n 1 m, (∀ j, CoeffBound (eB j) cutoff.toNat) ∧ ∀ j,
      reducePoly q n sp • top j + bot j =
        reducePoly q n ((p : ErrorPoly n) * μ * unitPair sp j) • G + reduceMatrix q n 1 m (eB j) := by
  obtain ⟨E0, hE0, h0⟩ := selector_rhs_gen (u := sp) (p := p) (top := top 0) hs hs hmsg (he 0)
  obtain ⟨E1, hE1, h1⟩ := selector_rhs_gen (u := 1) (p := p) (top := top 1) hs
    (hone.trans reduce_one_matrixG) hmsg (he 1)
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · simpa [unitPair, hbot0] using h0
  · simpa [unitPair, hbot1] using h1

set_option maxHeartbeats 4000000 in
/-- **A LUT branch's switching ciphertext** (one column `j`, as the preprocessing computes it):
`bot = -σ(s) top + (τ u) G + e`, so `σ(s) top + bot = τ u G + e` for the column's unit `u` and the
branch table polynomial `τ`. -/
theorem branch_rhs_gen {q m : Nat} {σsp τ u : ErrorPoly n} {sσ T uM : ExactMatrix q n 1 1}
    (hsσ : sσ = fun _ _ ↦ reducePoly q n σsp) (hT : T = fun _ _ ↦ reducePoly q n τ)
    (hu : uM = fun _ _ ↦ reducePoly q n u)
    {G top e : ExactMatrix q n 1 m} {sigma : Rat} {cutoff : Int} (he : gaussianSample sigma cutoff e) :
    ∃ eI : ErrorMatrix n 1 m, CoeffBound eI cutoff.toNat ∧
      reducePoly q n σsp • top + matrixAdd (matrixAdd (matrixMulScalarLeft (matrixNeg sσ) top)
          (matrixMulScalarLeft (matrixMulScalarLeft T uM) G)) e =
        reducePoly q n (τ * u) • G + reduceMatrix q n 1 m eI := by
  obtain ⟨E, hE, hb⟩ := gaussian_bound he
  refine ⟨E, hb, ?_⟩
  subst hsσ hT hu hE
  simp only [matrixMulScalarLeft_eq, matrixNeg]
  funext r c
  simp only [matrixAdd, Matrix.add_apply, Matrix.smul_apply, Matrix.neg_apply, smul_eq_mul, map_mul]
  ring

set_option maxHeartbeats 4000000 in
/-- `branch_rhs_gen` with the branch table left as the runtime's `1 × 1` value `T` (its value is
the LUT's business): `σ(s) top + bot = (T u) G + e`. -/
theorem branch_rhs_raw_gen {q m : Nat} {σsp u : ErrorPoly n} {sσ T uM : ExactMatrix q n 1 1}
    (hsσ : sσ = fun _ _ ↦ reducePoly q n σsp) (hu : uM = fun _ _ ↦ reducePoly q n u)
    {G top e : ExactMatrix q n 1 m} {sigma : Rat} {cutoff : Int} (he : gaussianSample sigma cutoff e) :
    ∃ eI : ErrorMatrix n 1 m, CoeffBound eI cutoff.toNat ∧
      reducePoly q n σsp • top + matrixAdd (matrixAdd (matrixMulScalarLeft (matrixNeg sσ) top)
          (matrixMulScalarLeft (matrixMulScalarLeft T uM) G)) e =
        (T 0 0 * reducePoly q n u) • G + reduceMatrix q n 1 m eI := by
  obtain ⟨E, hE, hb⟩ := gaussian_bound he
  refine ⟨E, hb, ?_⟩
  subst hsσ hu hE
  simp only [matrixMulScalarLeft_eq, matrixNeg]
  funext r c
  simp only [matrixAdd, Matrix.add_apply, Matrix.smul_apply, Matrix.neg_apply, smul_eq_mul]
  ring

/-- An automorphism of a reduced polynomial is the reduced substitution (definitionally
`GenEnc.ringAutomorphism_poly`). -/
theorem ringAutomorphism_reduceG {q : Nat} (hq : 1 < q) (hn : 0 < n) {index : Int}
    {s : ErrorPoly n} {out : ExactMatrix q n 1 1}
    (h : ringAutomorphismRuns index (fun _ _ ↦ reducePoly q n s : ExactMatrix q n 1 1) out) :
    ∃ hodd : Odd index.toNat, out = fun _ _ ↦ reducePoly q n (substitution index.toNat hodd s) := by
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq hq hn h
  refine ⟨hodd, ?_⟩
  rw [hsub]
  funext r c
  exact (reducePoly_substitution _ _ _).symm

/-! ## Loops -/

/-- **A `seq` loop that adds.** Iteration `i < count` adds `f i`. -/
theorem iter_add_gen {α : Type} [AddCommMonoid α] {body : Nat → α → α → Prop} {count : Nat}
    {init final : α} (f : Nat → α)
    (hstep : ∀ i current next, i < count → body i current next → next = current + f i)
    (h : MxxIR.IterRuns body count init final) :
    final = init + ∑ i ∈ Finset.range count, f i :=
  MxxIR.IterRuns.invariant
    (Invariant := fun k (s : α) ↦ k ≤ count → s = init + ∑ i ∈ Finset.range k, f i)
    (fun _ ↦ by simp) (fun i current next hcur hbody hk ↦ by
      rw [hstep i current next (by omega) hbody, hcur (by omega), Finset.sum_range_succ,
        add_assoc]) h le_rfl

/-- A family read at `c + i`. -/
theorem familyGet_addG {α : Type} {count : Nat} {f : Fin count → α} (c i : Nat) {out : α}
    (h : familyGetDynamic f ((c : Int) + Int.ofNat i) out) :
    ∃ hk : c + i < count, out = f ⟨c + i, hk⟩ := by
  obtain ⟨pos, hpos, hout⟩ := h
  have hv : pos.val = c + i := by simp only [Int.ofNat_eq_natCast] at hpos; omega
  exact ⟨hv ▸ pos.isLt, by rw [hout]; exact congrArg f (Fin.ext hv)⟩

/-- A family read at a constant index `k`. -/
theorem familyGet_atG {α : Type} {count : Nat} {f : Fin count → α} (k : Nat) {out : α}
    (h : familyGetDynamic f (k : Int) out) : ∃ hk : k < count, out = f ⟨k, hk⟩ := by
  obtain ⟨pos, hpos, hout⟩ := h
  have hv : pos.val = k := by omega
  exact ⟨hv ▸ pos.isLt, by rw [hout]; exact congrArg f (Fin.ext hv)⟩

/-- The unpacked row of member `o`: reads at `W o + i` are the member's cells. -/
theorem unpack_cells_gen {α : Type} {M W : Nat} {F : Fin (M * W) → α} (o : Fin M)
    {u : Fin W → α} (h : ∀ i : Fin W, familyGetDynamic F ((W : Int) * Int.ofNat o + Int.ofNat i) (u i)) :
    u = fun i ↦ F (cellAt o i) := by
  funext i
  obtain ⟨pos, hpos, hout⟩ := h i
  rw [hout]
  congr 1
  ext
  simp only [cellAt, Int.ofNat_eq_natCast] at hpos ⊢
  exact_mod_cast hpos

/-- **The row sum of a member** (keys or encodings alike): cells `0` and `1` before the loop,
iteration `i` adds cell `2 + i` (`W = w + 2`). -/
theorem rowsum_gen {α : Type} [AddCommMonoid α] {w : Nat} (u : Fin (w + 2) → α)
    {body : Nat → α → α → Prop} {init final : α} (hinit : init = u 0 + u 1)
    (hstep : ∀ i current next (hi : i < w), body i current next →
      next = current + u ⟨2 + i, by omega⟩)
    (h : MxxIR.IterRuns body w init final) : final = ∑ k, u k := by
  classical
  have hsum := iter_add_gen (fun i ↦ if hi : 2 + i < w + 2 then u ⟨2 + i, hi⟩ else 0)
    (fun i current next hi hbody ↦ by
      rw [hstep i current next hi hbody]
      simp only [dif_pos (show 2 + i < w + 2 by omega)]) h
  rw [hsum, hinit, Fin.sum_univ_succ, Fin.sum_univ_succ, ← add_assoc]
  congr 1
  rw [← Fin.sum_univ_eq_sum_range]
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  rw [dif_pos (by omega)]
  congr 1
  ext
  simp [Fin.succ]
  omega

/-! ### LUT keys -/

/-- The keys of branch `i` of a width-`W` LUT with automorphisms `σf`, helper keys under
`tagPrefix` at `level`, and switching families `fam` (zero past the branches). -/
noncomputable def lutKeysG {q q' m m' W : Nat} (backend : BackendContext) (base digits : Int)
    (basis : List Nat) (model : HashModel) (key : ByteArray) (tagPrefix : Blob) (level : Int)
    (σf : Nat → Int) (A : Fin 2 → Fin 2 → ExactMatrix q n 1 m)
    (fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q' n 1 m') (i : Nat) : KeysG n q m' :=
  if h : i < W then
    keyTupleG (branchKeyG backend base digits basis (σf i).toNat A
      (lutAG model key q m tagPrefix level (σf i)) (fun k j ↦ fam k j ⟨i, h⟩))
  else 0

/-- **The branch loop of a LUT's keys** (`seq[W]`): iteration `i` adds the keys of branch `i`. -/
theorem lut_keys_loop_gen {q q' m m' W : Nat} {backend : BackendContext} {base digits : Int}
    {basis : List Nat} {model : HashModel} {key : ByteArray} {tagPrefix : Blob} {level : Int}
    {σf : Nat → Int} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q' n 1 m'}
    {body : Nat → KeysG n q m' → KeysG n q m' → Prop} {init final : KeysG n q m'}
    (hstep : ∀ i current next, i < W → body i current next →
      next = current + lutKeysG backend base digits basis model key tagPrefix level σf A fam i)
    (h : MxxIR.IterRuns body W init final) :
    final = init + ∑ i ∈ Finset.range W,
      lutKeysG backend base digits basis model key tagPrefix level σf A fam i :=
  iter_add_gen _ hstep h

/-- One step of the branch loop from `branchKeyG_of_pre`: the step's keys are those of branch `i`
once its switching ciphertexts are the families' members `i` and its helper keys the samples. -/
theorem lut_keys_step_gen {q q' m m' W : Nat} {backend : BackendContext} {base digits : Int}
    {basis : List Nat} {model : HashModel} {key : ByteArray} {tagPrefix : Blob} {level : Int}
    {σf : Nat → Int} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q' n 1 m'} {i : Nat} (hi : i < W)
    {sw : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'} (hsw : ∀ k j, sw k j = fam k j ⟨i, hi⟩)
    {Ad : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    (hAd : ∀ k j : Fin 2, hashSample model tagPrefix
      [.u64Le level, .integer (σf i), .bytes componentTagG, .u64Le (2 * (k : Int) + (j : Int))] key
        (Ad k j))
    {K : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hK : K = branchKeyG backend base digits basis (σf i).toNat A Ad sw)
    {current next : KeysG n q m'} (hnext : next = current + keyTupleG K) :
    next = current + lutKeysG backend base digits basis model key tagPrefix level σf A fam i := by
  have hA : Ad = lutAG model key q m tagPrefix level (σf i) :=
    keyPairG_of_samples (pre := [.u64Le level, .integer (σf i)]) hAd
  rw [hnext, hK, hA, lutKeysG, dif_pos hi]
  congr 3
  funext k j
  exact hsw k j

/-- A switching family read at the branch index: member `i` of the family. -/
theorem familyGet_ofNatG {α : Type} {count : Nat} {f : Fin count → α} (i : Nat) {out : α}
    (h : familyGetDynamic f (Int.ofNat i) out) : ∃ hi : i < count, out = f ⟨i, hi⟩ := by
  obtain ⟨pos, hpos, hout⟩ := h
  have hv : pos.val = i := by simp only [Int.ofNat_eq_natCast] at hpos; omega
  exact ⟨hv ▸ pos.isLt, by rw [hout]; exact congrArg f (Fin.ext hv)⟩

/-- **The carried-state LUT loop** (the online form): iteration `i` adds `G i` of the carried
switching state and loads member `i + 1` of the families into it. Started on member `0`, the loop
adds `G i (fam i)` for every `i < count` and ends on member `count`. With
`G i sw = keyTupleG (branchKeyG … (σf i) A (lutAG … (σf i)) sw)` this is the preprocessing's
`∑ lutKeysG` (`lut_keys_loop_gen`) on the same families: the LUT key link. -/
theorem carried_loop_gen {S K V : Type} [AddCommMonoid K] (key : S → K) (sw : S → V)
    (fam : Nat → V) (G : Nat → V → K) {body : Nat → S → S → Prop} {count : Nat} {init final : S}
    (hinit : sw init = fam 0)
    (hstep : ∀ i current next, i < count → body i current next →
      key next = key current + G i (sw current) ∧ sw next = fam (i + 1))
    (h : MxxIR.IterRuns body count init final) :
    key final = key init + ∑ i ∈ Finset.range count, G i (fam i) ∧ sw final = fam count :=
  MxxIR.IterRuns.invariant
    (Invariant := fun k (s : S) ↦ k ≤ count →
      key s = key init + ∑ i ∈ Finset.range k, G i (fam i) ∧ sw s = fam k)
    (fun _ ↦ by simp [hinit]) (fun i current next hcur hbody hk ↦ by
      obtain ⟨hc1, hc2⟩ := hcur (by omega)
      obtain ⟨hn1, hn2⟩ := hstep i current next (by omega) hbody
      refine ⟨?_, hn2⟩
      rw [hn1, hc1, hc2, Finset.sum_range_succ, add_assoc]) h le_rfl

/-- **The deterministic-loop link.** Two loops whose steps both compute `F i` of the projected
state (`p₁`, `p₂`: e.g. the keys of an online state carrying rows and keys, and the keys of a
preprocessing state) end on equal projections when they start on equal projections. -/
theorem iter_link_gen {S₁ S₂ K : Type} (p₁ : S₁ → K) (p₂ : S₂ → K) (F : Nat → K → K)
    {body₁ : Nat → S₁ → S₁ → Prop} {body₂ : Nat → S₂ → S₂ → Prop} {count : Nat}
    {init₁ final₁ : S₁} {init₂ final₂ : S₂}
    (hs₁ : ∀ i current next, i < count → body₁ i current next → p₁ next = F i (p₁ current))
    (hs₂ : ∀ i current next, i < count → body₂ i current next → p₂ next = F i (p₂ current))
    (h₁ : MxxIR.IterRuns body₁ count init₁ final₁) (h₂ : MxxIR.IterRuns body₂ count init₂ final₂)
    (h0 : p₁ init₁ = p₂ init₂) : p₁ final₁ = p₂ final₂ := by
  let it : Nat → K := fun k ↦ Nat.rec (p₁ init₁) (fun i x ↦ F i x) k
  have inv₁ := MxxIR.IterRuns.invariant (Invariant := fun k (s : S₁) ↦ k ≤ count → p₁ s = it k)
    (fun _ ↦ rfl) (fun i current next hcur hbody hk ↦ by
      rw [hs₁ i current next (by omega) hbody, hcur (by omega)]) h₁ le_rfl
  have inv₂ := MxxIR.IterRuns.invariant (Invariant := fun k (s : S₂) ↦ k ≤ count → p₂ s = it k)
    (fun _ ↦ h0.symm) (fun i current next hcur hbody hk ↦ by
      rw [hs₂ i current next (by omega) hbody, hcur (by omega)]) h₂ le_rfl
  rw [inv₁, inv₂]

/-- The LUT key link: the carried online loop over the families' first `count ≤ W` branches adds
the preprocessing's branch keys `lutKeysG`. -/
theorem lut_keys_link_gen {q q' m m' W : Nat} {backend : BackendContext} {base digits : Int}
    {basis : List Nat} {model : HashModel} {key : ByteArray} {tagPrefix : Blob} {level : Int}
    {σf : Nat → Int} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q' n 1 m'} {count : Nat} (hcount : count ≤ W)
    (sw : Nat → Fin 2 → Fin 2 → ExactMatrix q' n 1 m')
    (hsw : ∀ i (hi : i < W), sw i = fun k j ↦ fam k j ⟨i, hi⟩) :
    ∑ i ∈ Finset.range count, keyTupleG (branchKeyG backend base digits basis (σf i).toNat A
        (lutAG model key q m tagPrefix level (σf i)) (sw i)) =
      ∑ i ∈ Finset.range count,
        lutKeysG backend base digits basis model key tagPrefix level σf A fam i := by
  refine Finset.sum_congr rfl fun i hi ↦ ?_
  have hiW : i < W := by simp at hi; omega
  rw [lutKeysG, dif_pos hiW, hsw i hiW]

/-! ### Column loops of keys -/

/-- A `1 × 1` value placed at column `c` of a `1 × C` row (definitionally `GenLoop.placeCol`). -/
noncomputable def placeColG {q : Nat} (C c : Nat) (x : ExactMatrix q n 1 1) : ExactMatrix q n 1 C :=
  fun _ w ↦ if (w : Nat) = c then x 0 0 else 0

/-- The four keys placed at column `c`. -/
noncomputable def keyPlaceG {q : Nat} (C c : Nat) (x : KeysG n q 1) : KeysG n q C :=
  (placeColG C c x.1, placeColG C c x.2.1, placeColG C c x.2.2.1, placeColG C c x.2.2.2.1, ())

theorem tensor_placeG {q C : Nat} {c : Nat} {x : ExactMatrix q n 1 1}
    {out : ExactMatrix q n (1 * 1) (C * 1)}
    (h : tensorRuns (unitRow (c : Int) : ExactMatrix q n 1 C) x out) :
    out = placeColG (C * 1) c x := by
  funext r w
  obtain rfl : r = 0 := Subsingleton.elim (α := Fin 1) _ _
  rw [tensorRuns_unitRow h w]
  simp only [Nat.div_one, placeColG]
  split_ifs
  · congr 1
    exact Subsingleton.elim (α := Fin 1) _ _
  · rfl

/-- **The step of a key column loop** (`Step::Sum` on keys): each key of the next state is the
current one plus the tensor of `unitRow c` with the step's value (widths `C * 1`, which the
export's literal widths meet definitionally). -/
theorem key_column_step_gen {q C : Nat} (c : Nat) {current next : KeysG n q (C * 1)}
    {x : KeysG n q 1} {o0 o1 o2 o3 : ExactMatrix q n (1 * 1) (C * 1)}
    (h0 : tensorRuns (unitRow (c : Int) : ExactMatrix q n 1 C) x.1 o0)
    (h1 : tensorRuns (unitRow (c : Int) : ExactMatrix q n 1 C) x.2.1 o1)
    (h2 : tensorRuns (unitRow (c : Int) : ExactMatrix q n 1 C) x.2.2.1 o2)
    (h3 : tensorRuns (unitRow (c : Int) : ExactMatrix q n 1 C) x.2.2.2.1 o3)
    (hnext : next = (current.1 + o0, current.2.1 + o1, current.2.2.1 + o2, current.2.2.2.1 + o3, ())) :
    next = current + keyPlaceG (C * 1) c x := by
  rw [hnext, tensor_placeG h0, tensor_placeG h1, tensor_placeG h2, tensor_placeG h3]
  rfl

/-- **A key column loop**: iteration `i` places the value `x i` at column `c₀ + i`. -/
theorem iter_key_column_gen {q C : Nat} (c₀ : Nat) {count : Nat}
    {body : Nat → KeysG n q C → KeysG n q C → Prop} {init final : KeysG n q C}
    (x : Nat → KeysG n q 1)
    (hstep : ∀ i current next, i < count → body i current next →
      next = current + keyPlaceG C (c₀ + i) (x i))
    (h : MxxIR.IterRuns body count init final) :
    final = init + ∑ i ∈ Finset.range count, keyPlaceG C (c₀ + i) (x i) :=
  iter_add_gen _ hstep h

/-! ## Trapdoor and decoder preimage -/

/-- A preimage under a sampled trapdoor solves the target within the cutoff. -/
theorem preimage_sampled_gen {q rows inner cols : Nat} {Token : Type} {backend : BackendContext}
    {sigma : Rat} {base digits cutoff : Int} {A : ExactMatrix q n rows inner}
    {T : TrapdoorValue (ExactMatrix q n rows inner) Token}
    (hT : trapdoorSample backend sigma base digits cutoff A T)
    {target : ExactMatrix q n rows cols} {cut : Nat} {P : ExactMatrix q n inner cols}
    (hP : preimageRunsDispatched backend A T target cut P) :
    A * P = target ∧ PreimageWithin P cut := by
  rcases hP.2 with h | h
  · exact ⟨preimageRuns_equation h, preimageRuns_bounded h⟩
  · exact absurd (h.1.symm.trans (trapdoorSample_sampled hT)) (by decide)

/-- The decoder's preimage of a target `t` (the generated `matrixMul` form). -/
theorem preimage_target_gen {q rows inner cols : Nat} {Token : Type} {backend : BackendContext}
    {sigma : Rat} {base digits cutoff : Int} {A : ExactMatrix q n rows inner}
    {T : TrapdoorValue (ExactMatrix q n rows inner) Token}
    (hT : trapdoorSample backend sigma base digits cutoff A T)
    {target : ExactMatrix q n rows cols} {cut : Nat} {P : ExactMatrix q n inner cols}
    (hP : preimageRunsDispatched backend A T target cut P) :
    matrixMul A P = target ∧ ∃ d : ErrorMatrix n inner cols,
      P = reduceMatrix q n inner cols d ∧ CoeffBound d cut :=
  preimage_sampled_gen hT hP

end RefreshTopProof.Gen
