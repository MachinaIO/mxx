import MxxPrimitives
import PrimitivesCrtRadix
import PrimitivesCrtDecomposition

namespace MxxRuntime

universe u v

open Mxx.Primitives
open scoped BigOperators

/- A trapdoor value keeps the public matrix and the private sampler token in one value.  The
token is intentionally abstract: its contents are not a second public-matrix interpretation. -/
inductive TrapdoorKind where
  | sampledSecret
  | publicGadget
  deriving DecidableEq, Repr

inductive GadgetMode where
  | regular
  | compact
  deriving DecidableEq, Repr

structure GadgetLayoutMetadata where
  crtModuli : List Nat
  baseBits : Nat
  digitsPerTower : Nat
  droppedModuli : Nat
  deriving DecidableEq, Repr

structure TrapdoorValue (Public : Type u) (Token : Type v) where
  publicMatrix : Public
  privateToken : Token
  kind : TrapdoorKind
  sigma : Rat
  gadgetBase : Int
  digitCount : Int
  mode : GadgetMode
  layout : GadgetLayoutMetadata

/- The two ports emitted by `TrapdoorSample` are coupled by this relation.  In particular, a
public port from one invocation cannot be paired with a token from another invocation unless the
stored values themselves establish that equality. -/
def trapdoorRuns {Public : Type u} {Token : Type v} (publicMatrix : Public)
    (trapdoor : TrapdoorValue Public Token) : Prop :=
  trapdoor.publicMatrix = publicMatrix

def sampledTrapdoorRuns {Public : Type u} {Token : Type v} (publicMatrix : Public)
    (trapdoor : TrapdoorValue Public Token) : Prop :=
  trapdoor.kind = .sampledSecret ∧ trapdoorRuns publicMatrix trapdoor

def publicGadgetTrapdoorRuns {Public : Type u} {Token : Type v} (publicMatrix : Public)
    (trapdoor : TrapdoorValue Public Token) : Prop :=
  trapdoor.kind = .publicGadget ∧ trapdoorRuns publicMatrix trapdoor

theorem trapdoorRuns_public {Public : Type u} {Token : Type v} (publicMatrix : Public)
    (trapdoor : TrapdoorValue Public Token) (h : trapdoorRuns publicMatrix trapdoor) :
    trapdoor.publicMatrix = publicMatrix :=
  h

theorem sampledTrapdoorRuns_public {Public : Type u} {Token : Type v} (publicMatrix : Public)
    (trapdoor : TrapdoorValue Public Token) (h : sampledTrapdoorRuns publicMatrix trapdoor) :
    trapdoor.publicMatrix = publicMatrix :=
  h.2

theorem publicGadgetTrapdoorRuns_public {Public : Type u} {Token : Type v} (publicMatrix : Public)
    (trapdoor : TrapdoorValue Public Token) (h : publicGadgetTrapdoorRuns publicMatrix trapdoor) :
    trapdoor.publicMatrix = publicMatrix :=
  h.2

/- A regular layout fixes the ordered CRT towers and a uniform number of balanced base digits per
   tower.  The residual proof is part of the concrete layout witness; runtime does not infer it
   from a declared cutoff. -/
structure RegularLayout (q : Nat) where
  crtModuli : List Nat
  crtModuli_nonempty : crtModuli ≠ []
  droppedModuli : Nat
  dropped_lt : droppedModuli < crtModuli.length
  modulus_pos : ∀ tower : Fin crtModuli.length, 0 < crtModuli.get tower
  pairwise_coprime : Pairwise (fun left right : Fin crtModuli.length =>
    Nat.Coprime (crtModuli.get left) (crtModuli.get right))
  product_eq : q = ∏ tower : Fin crtModuli.length, crtModuli.get tower
  baseBits : Nat
  base : Nat
  base_eq : base = 2 ^ baseBits
  base_gt_one : 1 < base
  base_even : Even base
  digitsPerTower : Nat
  digits_pos : 0 < digitsPerTower
  capacity : ∀ (tower : Fin crtModuli.length),
    crtModuli.get tower ≤ base ^ digitsPerTower

noncomputable def RegularLayout.ordered {q : Nat} (layout : RegularLayout q) : OrderedCrtLayout :=
  OrderedCrtLayout.ofCapacity (q := q) (Tower := Fin layout.crtModuli.length)
    (modulus := fun tower => layout.crtModuli.get tower) layout.modulus_pos
    layout.pairwise_coprime layout.product_eq layout.baseBits layout.base layout.base_eq
    layout.base_gt_one layout.base_even (fun _ => layout.digitsPerTower)
    (fun _ => layout.digits_pos) layout.capacity

abbrev RegularLimb (layout : RegularLayout q) :=
  Sigma (fun (_ : Fin layout.crtModuli.length) ↦ Fin layout.digitsPerTower)

def castExactMatrix {q r n rows columns : Nat} (h : q = r)
    (value : ExactMatrix q n rows columns) : ExactMatrix r n rows columns :=
  Eq.mp (congrArg (fun modulus : Nat => ExactMatrix modulus n rows columns) h) value

def castMatrixRows {q n rows rows' columns : Nat} (h : rows = rows')
    (value : ExactMatrix q n rows columns) : ExactMatrix q n rows' columns :=
  Eq.mp (congrArg (fun count : Nat => ExactMatrix q n count columns) h) value

def castMatrixColumns {q n rows columns columns' : Nat} (h : columns = columns')
    (value : ExactMatrix q n rows columns) : ExactMatrix q n rows columns' :=
  Eq.mp (congrArg (fun count : Nat => ExactMatrix q n rows count) h) value

noncomputable def regularDigitCoefficient {q n : Nat} (layout : RegularLayout q)
    (value : ExactPoly layout.ordered.modulusProduct n)
    (limb : RegularLimb layout) (coefficient : Fin n) : Int :=
    balancedDigitAt layout.base
    (centeredLift (layout.crtModuli.get limb.1)
      (layout.ordered.crtEquiv (value.coeff coefficient) limb.1)) limb.2.val

noncomputable def regularDigitPoly {q n : Nat} (layout : RegularLayout q)
    (value : ExactPoly layout.ordered.modulusProduct n)
    (limb : RegularLimb layout) : ErrorPoly n :=
  ∑ coefficient : Fin n,
    scaledBasis (regularDigitCoefficient layout value limb coefficient) coefficient

/-- Runtime's row, tower, digit order as an equivalence, without enumerating indices. -/
def regularIndexEquiv {q : Nat} (layout : RegularLayout q) (rows : Nat) :
    Fin rows × RegularLimb layout ≃
      Fin (rows * (layout.crtModuli.length * layout.digitsPerTower)) :=
  (Equiv.prodCongr (Equiv.refl _)
    ((Equiv.sigmaEquivProd _ _).trans finProdFinEquiv)).trans finProdFinEquiv

theorem regularIndexEquiv_val {q rows : Nat} (layout : RegularLayout q)
    (row : Fin rows) (tower : Fin layout.crtModuli.length)
    (digit : Fin layout.digitsPerTower) :
    (regularIndexEquiv layout rows (row, ⟨tower, digit⟩)).val =
      row.val * (layout.crtModuli.length * layout.digitsPerTower) +
        tower.val * layout.digitsPerTower + digit.val := by
  simp [regularIndexEquiv, finProdFinEquiv, Equiv.sigmaEquivProd]
  ring

noncomputable def regularGadgetUnflattened {q n rows : Nat} (layout : RegularLayout q) :
    Matrix (Fin rows) (Fin rows × RegularLimb layout)
      (ExactPoly layout.ordered.modulusProduct n) :=
  fun row index ↦ if row = index.1 then
    algebraMap (ZMod layout.ordered.modulusProduct) _
      (CrtDecomposition.regularGadgetWeight layout.ordered index.2) else 0

noncomputable def regularDigitsUnflattened {q n rows columns : Nat}
    (layout : RegularLayout q)
    (target : ExactMatrix layout.ordered.modulusProduct n rows columns) :
    Matrix (Fin rows × RegularLimb layout) (Fin columns)
      (ExactPoly layout.ordered.modulusProduct n) :=
  fun index column ↦ CrtDecomposition.regularDigitExactPoly
    layout.ordered (target index.1 column) index.2

theorem regularUnflattened_reconstruct {q n rows columns : Nat}
    (layout : RegularLayout q)
    (target : ExactMatrix layout.ordered.modulusProduct n rows columns)
    (hq : 1 < layout.ordered.modulusProduct) (hn : 0 < n) :
    regularGadgetUnflattened layout * regularDigitsUnflattened layout target = target := by
  classical
  funext row column
  simp only [Matrix.mul_apply, Fintype.sum_prod_type,
    regularGadgetUnflattened, regularDigitsUnflattened, ite_mul, zero_mul]
  rw [Finset.sum_eq_single row]
  · have h := congrFun (congrFun
      (CrtDecomposition.regularGadgetRow_mul_regularDecompositionColumn
        layout.ordered (target row column) hq hn) ()) ()
    simpa [Matrix.mul_apply, CrtDecomposition.regularGadgetRow,
      CrtDecomposition.regularDecompositionColumn] using h
  · intro other _ hne
    simp [Ne.symm hne]
  · simp

noncomputable def fullGadgetMatrix {q n rows : Nat} (layout : RegularLayout q) :
    ExactMatrix q n rows (rows * (layout.crtModuli.length * layout.digitsPerTower)) :=
  castExactMatrix layout.product_eq.symm
    (fun row column ↦ regularGadgetUnflattened layout row
      ((regularIndexEquiv layout rows).symm column))

noncomputable def fullDecomposeMatrix {q n rows columns : Nat}
    (layout : RegularLayout q) (target : ExactMatrix q n rows columns) :
    ExactMatrix q n (rows * (layout.crtModuli.length * layout.digitsPerTower)) columns :=
  castExactMatrix layout.product_eq.symm
    (fun row column ↦ regularDigitsUnflattened layout
      (castExactMatrix layout.product_eq target)
      ((regularIndexEquiv layout rows).symm row) column)

theorem castExactMatrix_mul {q q' n rows inner columns : Nat} (h : q = q')
    (left : ExactMatrix q n rows inner) (right : ExactMatrix q n inner columns) :
    castExactMatrix h (left * right) =
      castExactMatrix h left * castExactMatrix h right := by
  cases h
  rfl

theorem castExactMatrix_cancel {q q' n rows columns : Nat} (h : q = q')
    (value : ExactMatrix q n rows columns) :
    castExactMatrix h.symm (castExactMatrix h value) = value := by
  cases h
  rfl

theorem castExactMatrix_bounded {q q' n rows columns : Nat} (h : q = q')
    (value : ExactMatrix q n rows columns) (bound : Nat)
    (hbound : PreimageWithin value bound) :
    PreimageWithin (castExactMatrix h value) bound := by
  cases h
  exact hbound

theorem fullDecomposeMatrix_bounded {q n rows columns : Nat}
    (layout : RegularLayout q) (target : ExactMatrix q n rows columns)
    (hq : 1 < q) (hn : 0 < n) :
    PreimageWithin (fullDecomposeMatrix layout target) (layout.base / 2) := by
  unfold fullDecomposeMatrix
  apply castExactMatrix_bounded
  let value := castExactMatrix layout.product_eq target
  let witness : ErrorMatrix n
      (rows * (layout.crtModuli.length * layout.digitsPerTower)) columns :=
    fun row column ↦ CrtDecomposition.regularDigitPoly layout.ordered
      (value ((regularIndexEquiv layout rows).symm row).1 column)
      ((regularIndexEquiv layout rows).symm row).2
  refine ⟨witness, ?_, ?_⟩
  · funext row column
    exact CrtDecomposition.regularDigitExactPoly_eq_reduce layout.ordered _ _
      (by rw [layout.ordered.modulusProduct_eq_q]; exact hq) hn
  · intro row column coefficient
    exact (coeff_natAbs_le_polyNorm _ coefficient).trans
      (CrtDecomposition.regularDigitPoly_bound layout.ordered hn _ _)

theorem fullGadgetMatrix_reconstruct {q n rows columns : Nat}
    (layout : RegularLayout q) (target : ExactMatrix q n rows columns)
    (hq : 1 < q) (hn : 0 < n) :
    fullGadgetMatrix layout * fullDecomposeMatrix layout target = target := by
  classical
  unfold fullGadgetMatrix fullDecomposeMatrix
  rw [← castExactMatrix_mul]
  let g : ExactMatrix layout.ordered.modulusProduct n rows
      (rows * (layout.crtModuli.length * layout.digitsPerTower)) :=
    fun row column ↦ regularGadgetUnflattened layout row
      ((regularIndexEquiv layout rows).symm column)
  let d : ExactMatrix layout.ordered.modulusProduct n
      (rows * (layout.crtModuli.length * layout.digitsPerTower)) columns :=
    fun row column ↦ regularDigitsUnflattened layout
        (castExactMatrix layout.product_eq target)
        ((regularIndexEquiv layout rows).symm row) column
  have hinner : g * d =
      castExactMatrix layout.product_eq target := by
    rw [← regularUnflattened_reconstruct layout
      (castExactMatrix layout.product_eq target)
      (by rw [layout.ordered.modulusProduct_eq_q]; exact hq) hn]
    funext row column
    simp only [Matrix.mul_apply]
    exact (regularIndexEquiv layout rows).symm.sum_comp
      (fun index : Fin rows × RegularLimb layout ↦
        regularGadgetUnflattened (n := n) layout row index *
          regularDigitsUnflattened layout (castExactMatrix layout.product_eq target) index column)
  rw [hinner, castExactMatrix_cancel]

/-- The same trailing low-part selection used by the concrete CRT parameters. -/
def RegularLayout.retainedTowers {q : Nat} (layout : RegularLayout q) : Nat :=
  layout.crtModuli.length - layout.droppedModuli

def RegularLayout.digitCount {q : Nat} (layout : RegularLayout q) : Nat :=
  layout.retainedTowers * layout.digitsPerTower

def RegularLayout.lowProduct {q : Nat} (layout : RegularLayout q) : Nat :=
  (layout.crtModuli.drop layout.retainedTowers).prod

/-- Inclusive bound of Proposition 1; adding one gives a strict integer bound. -/
def RegularLayout.errorBound {q : Nat} (layout : RegularLayout q) : Nat :=
  layout.droppedModuli * (layout.lowProduct / 2)

def regularColumnIndex {q rows : Nat} (layout : RegularLayout q)
    (index : Fin (rows * layout.digitCount)) :
    Fin (rows * (layout.crtModuli.length * layout.digitsPerTower)) :=
  finProdFinEquiv
    ((finProdFinEquiv.symm index).1,
      Fin.castLE (Nat.mul_le_mul_right layout.digitsPerTower
        (Nat.sub_le layout.crtModuli.length layout.droppedModuli))
        (finProdFinEquiv.symm index).2)

noncomputable def regularLowCoefficient {q n : Nat} (layout : RegularLayout q)
    (value : ExactPoly q n) (coefficient : Fin n) : Int :=
  ((layout.crtModuli.drop layout.retainedTowers).map fun (p : Nat) ↦
    ((layout.lowProduct / p : Nat) : Int) *
      centeredLift p (((value.coeff coefficient).val : ZMod p) *
        ((layout.lowProduct / p : Nat) : ZMod p)⁻¹)).sum

noncomputable def regularLowResidual {q n rows columns : Nat} (layout : RegularLayout q)
    (target : ExactMatrix q n rows columns) : ErrorMatrix n rows columns :=
  fun row column ↦ ∑ coefficient : Fin n,
    scaledBasis (regularLowCoefficient layout (target row column) coefficient) coefficient

noncomputable def regularCorrectedTarget {q n rows columns : Nat} (layout : RegularLayout q)
    (target : ExactMatrix q n rows columns) : ExactMatrix q n rows columns :=
  if layout.droppedModuli = 0 then target
  else target - reduceMatrix q n rows columns (regularLowResidual layout target)

noncomputable def regularGadgetMatrix {q n rows : Nat} (layout : RegularLayout q) :
    ExactMatrix q n rows (rows * layout.digitCount) :=
  fun row column ↦ fullGadgetMatrix layout row (regularColumnIndex layout column)

noncomputable def regularDecomposeMatrix {q n rows columns : Nat}
    (layout : RegularLayout q) (target : ExactMatrix q n rows columns) :
    ExactMatrix q n (rows * layout.digitCount) columns :=
  fun row column ↦ fullDecomposeMatrix layout (regularCorrectedTarget layout target)
    (regularColumnIndex layout row) column

theorem regularDecomposeMatrix_bounded {q n rows columns : Nat}
    (layout : RegularLayout q) (target : ExactMatrix q n rows columns)
    (hq : 1 < q) (hn : 0 < n) :
    PreimageWithin (regularDecomposeMatrix layout target) (layout.base / 2) := by
  obtain ⟨witness, heq, hbound⟩ :=
    fullDecomposeMatrix_bounded layout (regularCorrectedTarget layout target) hq hn
  refine ⟨(fun row column ↦ witness (regularColumnIndex layout row) column), ?_, ?_⟩
  · funext row column
    exact congrFun (congrFun heq (regularColumnIndex layout row)) column
  · intro row column coefficient
    exact hbound (regularColumnIndex layout row) column coefficient

theorem regularGadgetMatrix_reconstruct {q n rows columns : Nat}
    (layout : RegularLayout q) (target : ExactMatrix q n rows columns)
    (hq : 1 < q) (hn : 0 < n) (hexact : layout.droppedModuli = 0) :
    regularGadgetMatrix layout * regularDecomposeMatrix layout target = target := by
  have hindex (index : Fin (rows * layout.digitCount)) :
      (regularColumnIndex layout index).val = index.val := by
    simp [regularColumnIndex, RegularLayout.digitCount, RegularLayout.retainedTowers, hexact,
      finProdFinEquiv, Nat.mod_add_div]
  have hcount : layout.digitCount = layout.crtModuli.length * layout.digitsPerTower := by
    simp [RegularLayout.digitCount, RegularLayout.retainedTowers, hexact]
  have hsum := fullGadgetMatrix_reconstruct layout target hq hn
  funext row column
  rw [Matrix.mul_apply]
  have hg (index : Fin (rows * layout.digitCount)) :
      regularColumnIndex layout index = Fin.cast (congrArg (rows * ·) hcount) index :=
    Fin.ext (hindex index)
  simp only [regularGadgetMatrix, regularDecomposeMatrix, regularCorrectedTarget, hexact,
    ↓reduceIte, hg]
  calc
    _ = ∑ index : Fin (rows * (layout.crtModuli.length * layout.digitsPerTower)),
        fullGadgetMatrix layout row index * fullDecomposeMatrix layout target index column :=
      (finCongr (congrArg (rows * ·) hcount)).sum_comp _
    _ = target row column := congrFun (congrFun hsum row) column

/-- Approximation is an invocation-local primitive premise. The equation and the bound
refer to the same concrete target and deterministic digits, never to independent limb errors. -/
theorem regularGadgetMatrix_approximate {q n rows columns : Nat}
    (layout : RegularLayout q) (target : ExactMatrix q n rows columns)
    (hresidual : PreimageWithin
      (target - regularGadgetMatrix layout * regularDecomposeMatrix layout target)
      layout.errorBound) :
    Approx target (regularGadgetMatrix layout * regularDecomposeMatrix layout target)
      layout.errorBound := by
  obtain ⟨error, heq, hbound⟩ := hresidual
  exact ⟨error, (sub_eq_iff_eq_add.mp heq).trans (add_comm _ _), hbound⟩

theorem regularGadgetMatrix_residual_exact {q n rows columns : Nat}
    (layout : RegularLayout q) (target : ExactMatrix q n rows columns)
    (hq : 1 < q) (hn : 0 < n) (hexact : layout.droppedModuli = 0) :
    PreimageWithin (target - regularGadgetMatrix layout * regularDecomposeMatrix layout target)
      layout.errorBound := by
  rw [regularGadgetMatrix_reconstruct layout target hq hn hexact, sub_self]
  refine ⟨0, ?_, ?_⟩
  · funext row column
    exact (map_zero (reducePoly q n)).symm
  · simpa [RegularLayout.errorBound, hexact] using
      (coeffBound_zero (n := n) (rows := rows) (columns := columns))

def publicGadgetPreimageRuns {q n sourceRows inner targetColumns : Nat} {Token : Type v}
    (publicMatrix : ExactMatrix q n sourceRows inner)
    (trapdoor : TrapdoorValue (ExactMatrix q n sourceRows inner) Token)
    (target : ExactMatrix q n sourceRows targetColumns)
    (preimage : ExactMatrix q n inner targetColumns) : Prop :=
  trapdoor.kind = .publicGadget ∧ trapdoor.publicMatrix = publicMatrix ∧
    trapdoor.mode = .regular ∧
    ∃ layout : RegularLayout q,
      layout.droppedModuli = 0 ∧
      layout.crtModuli = trapdoor.layout.crtModuli ∧
      layout.baseBits = trapdoor.layout.baseBits ∧
      layout.digitsPerTower = trapdoor.layout.digitsPerTower ∧
      trapdoor.gadgetBase = layout.base ∧
      trapdoor.digitCount = layout.digitCount ∧
      ∃ hwidth : inner = sourceRows * (layout.digitCount),
        publicMatrix = castMatrixColumns hwidth.symm (regularGadgetMatrix layout) ∧
          preimage = castMatrixRows hwidth.symm (regularDecomposeMatrix layout target)

/-- The public gadget constructor fixes its matrix from the same backend layout as decomposition. -/
noncomputable def regularGadgetTrapdoor {q n rows : Nat} (layout : RegularLayout q)
    (sigma : Rat) : TrapdoorValue
      (ExactMatrix q n rows (rows * (layout.digitCount))) Unit :=
  { publicMatrix := regularGadgetMatrix layout
    privateToken := ()
    kind := .publicGadget
    sigma := sigma
    gadgetBase := layout.base
    digitCount := layout.digitCount
    mode := .regular
    layout :=
      { crtModuli := layout.crtModuli
        baseBits := layout.baseBits
        digitsPerTower := layout.digitsPerTower
        droppedModuli := layout.droppedModuli } }

theorem regularGadgetTrapdoor_preimage {q n rows columns : Nat}
    (layout : RegularLayout q) (sigma : Rat) (target : ExactMatrix q n rows columns)
    (hexact : layout.droppedModuli = 0) :
    publicGadgetPreimageRuns (regularGadgetMatrix layout)
      (regularGadgetTrapdoor layout sigma) target (regularDecomposeMatrix layout target) := by
  exact ⟨rfl, rfl, rfl, layout, hexact, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem publicGadgetPreimageRuns_equation {q n rows inner columns : Nat} {Token : Type v}
    {publicMatrix : ExactMatrix q n rows inner}
    {trapdoor : TrapdoorValue (ExactMatrix q n rows inner) Token}
    {target : ExactMatrix q n rows columns} {preimage : ExactMatrix q n inner columns}
    (hq : 1 < q) (hn : 0 < n)
    (h : publicGadgetPreimageRuns publicMatrix trapdoor target preimage) :
    publicMatrix * preimage = target := by
  rcases h with ⟨_, _, _, layout, hexact, _, _, _, _, _, hwidth, hpublic, hpreimage⟩
  subst inner
  simp only [castMatrixColumns, castMatrixRows] at hpublic hpreimage
  rw [hpublic, hpreimage]
  exact regularGadgetMatrix_reconstruct layout target hq hn hexact

/-- Fixed backend parameters, selected by the same modulus/ring-dimension key as execution. -/
structure BackendContext where
  regularLayout : (q n : Nat) → Option (RegularLayout q)

def trapdoorLayoutMatches {q n rows columns : Nat} {Token : Type v}
    (backend : BackendContext)
    (trapdoor : TrapdoorValue (ExactMatrix q n rows columns) Token) : Prop :=
  ∃ layout, backend.regularLayout q n = some layout ∧
    trapdoor.layout.crtModuli = layout.crtModuli ∧
    trapdoor.layout.baseBits = layout.baseBits ∧
    trapdoor.layout.digitsPerTower = layout.digitsPerTower ∧
    trapdoor.layout.droppedModuli = layout.droppedModuli ∧
    trapdoor.gadgetBase = layout.base ∧
    trapdoor.digitCount = layout.digitCount

/- A sampled trapdoor draw exposes all payload fields used by the IR node while coupling its two
   output ports through the same trapdoor value.  The private token remains abstract: this relation
   records successful sampler outputs and makes no termination or distribution claim. -/
def trapdoorSample {q n rows columns : Nat} {Token : Type v}
    (backend : BackendContext)
    (sigma : Rat) (gadgetBase digitCount cutoff : Int)
    (publicMatrix : ExactMatrix q n rows columns)
    (trapdoor : TrapdoorValue (ExactMatrix q n rows columns) Token) : Prop :=
  sampledTrapdoorRuns publicMatrix trapdoor ∧
    trapdoor.sigma = sigma ∧ trapdoor.gadgetBase = gadgetBase ∧
    trapdoor.digitCount = digitCount ∧ 0 < gadgetBase ∧ 0 < digitCount ∧ 0 ≤ cutoff ∧
    trapdoorLayoutMatches backend trapdoor

theorem trapdoorSample_public {q n rows columns : Nat} {Token : Type v}
    {backend : BackendContext}
    {sigma : Rat} {gadgetBase digitCount cutoff : Int}
    {publicMatrix : ExactMatrix q n rows columns}
    {trapdoor : TrapdoorValue (ExactMatrix q n rows columns) Token}
    (h : trapdoorSample backend sigma gadgetBase digitCount cutoff publicMatrix trapdoor) :
    trapdoor.publicMatrix = publicMatrix :=
  h.1.2

theorem trapdoorSample_sampled {q n rows columns : Nat} {Token : Type v}
    {backend : BackendContext}
    {sigma : Rat} {gadgetBase digitCount cutoff : Int}
    {publicMatrix : ExactMatrix q n rows columns}
    {trapdoor : TrapdoorValue (ExactMatrix q n rows columns) Token}
    (h : trapdoorSample backend sigma gadgetBase digitCount cutoff publicMatrix trapdoor) :
    trapdoor.kind = .sampledSecret :=
  h.1.1

/- The preimage relation has the exact IR argument order: explicit public matrix, trapdoor,
target, cutoff, and returned preimage.  Both the public identity and the equation mention the
same invocation-local values.  The cutoff is a successful-sample bound, not a termination claim. -/
def sampledPreimageRuns {q n sourceRows inner targetColumns : Nat} {Token : Type v}
    (publicMatrix : ExactMatrix q n sourceRows inner)
    (trapdoor : TrapdoorValue (ExactMatrix q n sourceRows inner) Token)
    (target : ExactMatrix q n sourceRows targetColumns) (cutoff : Nat)
    (preimage : ExactMatrix q n inner targetColumns) : Prop :=
  trapdoor.kind = .sampledSecret ∧ trapdoor.publicMatrix = publicMatrix ∧
    RightPreimage publicMatrix preimage target ∧ PreimageWithin preimage cutoff

def preimageRuns {q n sourceRows inner targetColumns : Nat} {Token : Type v}
    (publicMatrix : ExactMatrix q n sourceRows inner)
    (trapdoor : TrapdoorValue (ExactMatrix q n sourceRows inner) Token)
    (target : ExactMatrix q n sourceRows targetColumns) (cutoff : Nat)
    (preimage : ExactMatrix q n inner targetColumns) : Prop :=
  sampledPreimageRuns publicMatrix trapdoor target cutoff preimage

/- The exporter-facing dispatch relation keeps the sampled relation stable while adding the
   deterministic public-gadget branch.  Public execution deliberately has no cutoff conjunct;
   its supported layout derives the digit bound from the concrete balanced decomposition. -/
def preimageRunsDispatched {q n sourceRows inner targetColumns : Nat} {Token : Type v}
    (backend : BackendContext)
    (publicMatrix : ExactMatrix q n sourceRows inner)
    (trapdoor : TrapdoorValue (ExactMatrix q n sourceRows inner) Token)
    (target : ExactMatrix q n sourceRows targetColumns) (cutoff : Nat)
    (preimage : ExactMatrix q n inner targetColumns) : Prop :=
  trapdoorLayoutMatches backend trapdoor ∧
    (preimageRuns publicMatrix trapdoor target cutoff preimage ∨
      publicGadgetPreimageRuns publicMatrix trapdoor target preimage)

theorem preimageRuns_sampled {q n sourceRows inner targetColumns : Nat} {Token : Type v}
    {publicMatrix : ExactMatrix q n sourceRows inner}
    {trapdoor : TrapdoorValue (ExactMatrix q n sourceRows inner) Token}
    {target : ExactMatrix q n sourceRows targetColumns} {cutoff : Nat}
    {preimage : ExactMatrix q n inner targetColumns}
    (h : preimageRuns publicMatrix trapdoor target cutoff preimage) :
    trapdoor.kind = .sampledSecret :=
  h.1

theorem preimageRuns_public {q n sourceRows inner targetColumns : Nat} {Token : Type v}
    {publicMatrix : ExactMatrix q n sourceRows inner}
    {trapdoor : TrapdoorValue (ExactMatrix q n sourceRows inner) Token}
    {target : ExactMatrix q n sourceRows targetColumns} {cutoff : Nat}
    {preimage : ExactMatrix q n inner targetColumns}
    (h : preimageRuns publicMatrix trapdoor target cutoff preimage) :
    trapdoor.publicMatrix = publicMatrix :=
  h.2.1

theorem preimageRuns_equation {q n sourceRows inner targetColumns : Nat} {Token : Type v}
    {publicMatrix : ExactMatrix q n sourceRows inner}
    {trapdoor : TrapdoorValue (ExactMatrix q n sourceRows inner) Token}
    {target : ExactMatrix q n sourceRows targetColumns} {cutoff : Nat}
    {preimage : ExactMatrix q n inner targetColumns}
    (h : preimageRuns publicMatrix trapdoor target cutoff preimage) :
    publicMatrix * preimage = target :=
  h.2.2.1.equation

theorem preimageRuns_bounded {q n sourceRows inner targetColumns : Nat} {Token : Type v}
    {publicMatrix : ExactMatrix q n sourceRows inner}
    {trapdoor : TrapdoorValue (ExactMatrix q n sourceRows inner) Token}
    {target : ExactMatrix q n sourceRows targetColumns} {cutoff : Nat}
    {preimage : ExactMatrix q n inner targetColumns}
    (h : preimageRuns publicMatrix trapdoor target cutoff preimage) :
    PreimageWithin preimage cutoff :=
  h.2.2.2

theorem preimageRuns_rejects_public_mismatch {q n sourceRows inner targetColumns : Nat}
    {Token : Type v} {publicMatrix : ExactMatrix q n sourceRows inner}
    {trapdoor : TrapdoorValue (ExactMatrix q n sourceRows inner) Token}
    {target : ExactMatrix q n sourceRows targetColumns} {cutoff : Nat}
    {preimage : ExactMatrix q n inner targetColumns}
    (mismatch : trapdoor.publicMatrix ≠ publicMatrix) :
    ¬preimageRuns publicMatrix trapdoor target cutoff preimage := by
  intro h
  exact mismatch h.2.1

theorem preimageRunsDispatched_public {q n sourceRows inner targetColumns : Nat} {Token : Type v}
    {backend : BackendContext}
    {publicMatrix : ExactMatrix q n sourceRows inner}
    {trapdoor : TrapdoorValue (ExactMatrix q n sourceRows inner) Token}
    {target : ExactMatrix q n sourceRows targetColumns} {cutoff : Nat}
    {preimage : ExactMatrix q n inner targetColumns}
    (h : preimageRunsDispatched backend publicMatrix trapdoor target cutoff preimage) :
    trapdoor.publicMatrix = publicMatrix := by
  rcases h.2 with h | h
  · exact preimageRuns_public h
  · exact h.2.1

theorem preimageRunsDispatched_equation {q n sourceRows inner targetColumns : Nat}
    {backend : BackendContext}
    {Token : Type v} {publicMatrix : ExactMatrix q n sourceRows inner}
    {trapdoor : TrapdoorValue (ExactMatrix q n sourceRows inner) Token}
    {target : ExactMatrix q n sourceRows targetColumns} {cutoff : Nat}
    {preimage : ExactMatrix q n inner targetColumns}
    (hq : 1 < q) (hn : 0 < n)
    (h : preimageRunsDispatched backend publicMatrix trapdoor target cutoff preimage) :
    publicMatrix * preimage = target := by
  rcases h.2 with h | h
  · exact preimageRuns_equation h
  · exact publicGadgetPreimageRuns_equation hq hn h
/- Fixed-layout regular gadget semantics. A missing backend ring has no successful run;
   the Rust export boundary separately rejects missing layouts before theorem checking. -/
def gadgetDecomposeRuns {q n rows columns inner : Nat}
    (backend : BackendContext) (base digits : Int)
    (target : ExactMatrix q n rows columns)
    (decomposition : ExactMatrix q n inner columns) : Prop :=
  ∃ layout, backend.regularLayout q n = some layout ∧
    base = layout.base ∧ digits = layout.digitCount ∧
    ∃ hwidth : inner = rows * (layout.digitCount),
      decomposition = castMatrixRows hwidth.symm (regularDecomposeMatrix layout target) ∧
        PreimageWithin (target - regularGadgetMatrix layout * regularDecomposeMatrix layout target)
          layout.errorBound

theorem gadgetDecomposeRuns_deterministic {q n rows columns inner : Nat}
    {backend : BackendContext} {base digits : Int}
    {target : ExactMatrix q n rows columns}
    {first second : ExactMatrix q n inner columns}
    (firstRuns : gadgetDecomposeRuns backend base digits target first)
    (secondRuns : gadgetDecomposeRuns backend base digits target second) :
    first = second := by
  rcases firstRuns with ⟨left, hl, _, _, hw, hf, _⟩
  rcases secondRuns with ⟨right, hr, _, _, hw', hs, _⟩
  have heq : left = right := Option.some.inj (hl.symm.trans hr)
  subst right
  exact hf.trans hs.symm

def gadgetMatrixRuns {q n rows columns : Nat}
    (backend : BackendContext) (base digits : Int)
    (output : ExactMatrix q n rows columns) : Prop :=
  ∃ layout, backend.regularLayout q n = some layout ∧
    base = layout.base ∧ digits = layout.digitCount ∧
    ∃ hwidth : columns = rows * (layout.digitCount),
      output = castMatrixColumns hwidth.symm (regularGadgetMatrix layout)

theorem gadgetDecomposeRuns_approximate {q n rows columns inner : Nat}
    {backend : BackendContext} {base digits : Int}
    {target : ExactMatrix q n rows columns}
    {decomposition : ExactMatrix q n inner columns}
    {gadget : ExactMatrix q n rows inner}
    (hg : gadgetMatrixRuns backend base digits gadget)
    (hd : gadgetDecomposeRuns backend base digits target decomposition) :
    ∃ layout, backend.regularLayout q n = some layout ∧
      Approx target (gadget * decomposition) layout.errorBound := by
  rcases hg with ⟨layout, hl, _, _, hw, hg⟩
  rcases hd with ⟨other, ho, _, _, hw', hd, he⟩
  have hsame : other = layout := Option.some.inj (ho.symm.trans hl)
  subst other
  subst inner
  simp only [castMatrixColumns, castMatrixRows] at hg hd
  refine ⟨layout, hl, ?_⟩
  rw [hg, hd]
  exact regularGadgetMatrix_approximate layout target he

def gadgetTrapdoorRuns {q n rows columns : Nat} {Token : Type v}
    (backend : BackendContext) (sigma : Rat) (base digits cutoff : Int)
    (trapdoor : TrapdoorValue (ExactMatrix q n rows columns) Token) : Prop :=
  trapdoor.kind = .publicGadget ∧ trapdoor.mode = .regular ∧
    trapdoor.sigma = sigma ∧ trapdoor.gadgetBase = base ∧ trapdoor.digitCount = digits ∧
    0 ≤ cutoff ∧ trapdoorLayoutMatches backend trapdoor ∧
    gadgetMatrixRuns backend base digits trapdoor.publicMatrix

/- The successful bounded relation is intentionally existential only over the returned sample.  It
does not assert that the sampler terminates or that a cutoff admits a preimage. -/
def boundedSampleRuns {q n rows columns : Nat}
    (value : ExactMatrix q n rows columns) (cutoff : Nat) : Prop :=
  PreimageWithin value cutoff

theorem boundedSampleRuns_iff {q n rows columns : Nat}
    (value : ExactMatrix q n rows columns) (cutoff : Nat) :
    boundedSampleRuns value cutoff ↔ PreimageWithin value cutoff :=
  Iff.rfl

end MxxRuntime
