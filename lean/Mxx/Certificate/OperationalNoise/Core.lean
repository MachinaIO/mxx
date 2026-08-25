import Lean.Elab.Tactic.Omega
import Mxx.Certificate.OperationalNoise.Replay

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise

inductive RowTable (α : Type) where
  | empty
  | node (id : Nat) (value : α) (left right : RowTable α)
deriving Repr

def RowTable.lookup {α : Type} (table : RowTable α) (wanted : Nat) : Option α :=
  match table with
  | .empty => none
  | .node id value left right =>
      if wanted = id then some value
      else if wanted < id then left.lookup wanted
      else right.lookup wanted

def RowTable.inorder {α : Type} (table : RowTable α) : List (Nat × α) :=
  match table with
  | .empty => []
  | .node id value left right => left.inorder ++ (id, value) :: right.inorder

def RowTable.height {α : Type} (table : RowTable α) : Nat :=
  match table with
  | .empty => 0
  | .node _ _ left right => Nat.max left.height right.height + 1

def RowTable.orderedFrom {α : Type} : Option Nat → Option Nat → RowTable α → Bool
  | _, _, .empty => true
  | lower, upper, .node id _ left right =>
      let aboveLower := match lower with | none => true | some bound => bound < id
      let belowUpper := match upper with | none => true | some bound => id < bound
      aboveLower && belowUpper && orderedFrom lower (some id) left &&
        orderedFrom (some id) upper right

def RowTable.balanced {α : Type} (table : RowTable α) : Bool :=
  match table with
  | .empty => true
  | .node _ _ left right =>
      let leftHeight := left.height
      let rightHeight := right.height
      let close := leftHeight ≤ rightHeight + 1 && rightHeight ≤ leftHeight + 1
      close && left.balanced && right.balanced

def RowTable.wellFormed {α : Type} (table : RowTable α) : Bool :=
  table.orderedFrom none none && table.balanced

def RowTable.AllFrom {α : Type} (predicate : Nat → α → Prop) : RowTable α → Prop
  | .empty => True
  | .node id value left right =>
      predicate id value ∧ left.AllFrom predicate ∧ right.AllFrom predicate

def RowTable.allBool {α : Type} (predicate : Nat → α → Bool) : RowTable α → Bool
  | .empty => true
  | .node id value left right =>
      predicate id value && left.allBool predicate && right.allBool predicate

structure FamilyDomain where
  lower : Nat
  upper : Nat
deriving DecidableEq, Repr

def FamilyDomain.Contains (domain : FamilyDomain) (selector : Nat) : Prop :=
  domain.lower ≤ selector ∧ selector < domain.upper

def FamilyDomain.u64Representable (endpoint : Nat) : Bool := endpoint ≤ 18446744073709551615

def FamilyDomain.valid (domain : FamilyDomain) : Bool :=
  domain.lower < domain.upper && u64Representable domain.upper

structure SourceId where
  value : Nat
deriving DecidableEq, Repr

structure EventId where
  value : Nat
deriving DecidableEq, Repr

structure ExprId where
  value : Nat
deriving DecidableEq, Repr

structure ProgramId where
  value : Nat
deriving DecidableEq, Repr

structure MatrixShape where
  modulus : Nat
  ringDimension : Nat
  rows : Nat
  columns : Nat
deriving DecidableEq, Repr

def MatrixShape.valid (shape : MatrixShape) : Bool :=
  shape.modulus > 0 && shape.ringDimension > 0 && shape.rows > 0 && shape.columns > 0

def MatrixShape.matchesCertificate (shape : MatrixShape) (modulus ringDimension : Nat) : Bool :=
  shape.valid && shape.modulus = modulus && shape.ringDimension = ringDimension

def MatrixShape.coefficientCount (shape : MatrixShape) : Nat :=
  shape.ringDimension * shape.rows * shape.columns

structure Matrix where
  shape : MatrixShape
  coefficients : List Int
deriving DecidableEq, Repr

def Matrix.valid (matrix : Matrix) (modulus ringDimension : Nat) : Bool :=
  matrix.shape.matchesCertificate modulus ringDimension &&
    matrix.coefficients.length = matrix.shape.coefficientCount

def centeredCoefficient (modulus : Nat) (coefficient : Int) : Int :=
  if modulus = 0 then coefficient
  else
    let remainder := coefficient % Int.ofNat modulus
    if 2 * remainder ≤ Int.ofNat modulus then remainder else remainder - Int.ofNat modulus

def centeredNorm (modulus : Nat) (coefficient : Int) : Nat :=
  (centeredCoefficient modulus coefficient).natAbs

def maxNatList : List Nat → Nat
  | [] => 0
  | value :: values => Nat.max value (maxNatList values)

def Matrix.maxCenteredCoefficientNorm (matrix : Matrix) (modulus : Nat) : Nat :=
  maxNatList (matrix.coefficients.map (centeredNorm modulus))

def Matrix.CoefficientBound (matrix : Matrix) (modulus bound : Nat) : Prop :=
  ∀ coefficient, coefficient ∈ matrix.coefficients → centeredNorm modulus coefficient ≤ bound

theorem maxNatList_le_of_forall {values : List Nat} {bound : Nat}
    (hbound : ∀ value, value ∈ values → value ≤ bound) : maxNatList values ≤ bound := by
  induction values with
  | nil => simp [maxNatList]
  | cons value values ih =>
      simp only [maxNatList]
      rw [Nat.max_le]
      constructor
      · exact hbound value (by simp)
      · apply ih
        intro next hnext
        exact hbound next (by simp [hnext])

theorem Matrix.maxCenteredCoefficientNorm_le_of_coefficientBound
    {matrix : Matrix} {modulus bound : Nat} (hbound : matrix.CoefficientBound modulus bound) :
    matrix.maxCenteredCoefficientNorm modulus ≤ bound := by
  have mappedBound : ∀ (coefficients : List Int),
      (∀ coefficient, coefficient ∈ coefficients → centeredNorm modulus coefficient ≤ bound) →
        maxNatList (coefficients.map (centeredNorm modulus)) ≤ bound := by
    intro coefficients
    induction coefficients with
    | nil =>
        intro
        simp [maxNatList]
    | cons coefficient coefficients ih =>
        intro coefficientBound
        have headBound := coefficientBound coefficient (List.Mem.head coefficients)
        have tailBound : ∀ next, next ∈ coefficients → centeredNorm modulus next ≤ bound := by
          intro next hnext
          exact coefficientBound next (List.Mem.tail _ hnext)
        simp only [List.map_cons, maxNatList]
        exact Nat.max_le.mpr ⟨headBound, ih tailBound⟩
  exact mappedBound matrix.coefficients hbound

structure SourceAccess where
  source : SourceId
  selector : Option Nat
deriving DecidableEq, Repr

structure EventAccess where
  event : EventId
  selector : Option Nat
deriving DecidableEq, Repr

def SourceAccess.withSelector (access : SourceAccess) (selector : Option Nat) : SourceAccess :=
  { access with selector := match access.selector with | none => selector | some existing => some existing }

def EventAccess.withSelector (access : EventAccess) (selector : Option Nat) : EventAccess :=
  { access with selector := match access.selector with | none => selector | some existing => some existing }

inductive Value where
  | invalid
  | matrix (value : Matrix)
deriving DecidableEq, Repr

structure SourceRow where
  shape : MatrixShape
  coefficientBound : Nat
deriving DecidableEq, Repr

structure EventRow where
  shape : MatrixShape
  coefficientBound : Nat
deriving DecidableEq, Repr

inductive ExprRow where
  | constant (value : Matrix)
  | source (access : SourceAccess)
  | sampler (access : EventAccess)
deriving DecidableEq, Repr

structure ProgramRow where
  body : ExprId
deriving DecidableEq, Repr

inductive ResidualRoot where
  | closed (root : ExprId)
  | family (program : ProgramId) (domain : FamilyDomain)
deriving DecidableEq, Repr

abbrev InputAssignment := SourceAccess → Value
abbrev SamplerAssignment := EventAccess → Value

structure Cert where
  plaintextModulus : Nat
  ciphertextModulus : Nat
  ringDimension : Nat
  expressions : RowTable ExprRow
  programs : RowTable ProgramRow
  sources : RowTable SourceRow
  events : RowTable EventRow
  residualRoot : ResidualRoot

def sourceRowsWellFormed (cert : Cert) : Bool :=
  cert.sources.allBool (fun _ row => row.shape.matchesCertificate cert.ciphertextModulus cert.ringDimension)

def eventRowsWellFormed (cert : Cert) : Bool :=
  cert.events.allBool (fun _ row => row.shape.matchesCertificate cert.ciphertextModulus cert.ringDimension)

def optionIsSome {α : Type} (value : Option α) : Bool :=
  match value with
  | none => false
  | some _ => true

def expressionRowWellFormed (cert : Cert) : Nat → ExprRow → Bool
  | _, .constant matrix => matrix.valid cert.ciphertextModulus cert.ringDimension
  | _, .source access => optionIsSome (cert.sources.lookup access.source.value)
  | _, .sampler access => optionIsSome (cert.events.lookup access.event.value)

def programRowWellFormed (cert : Cert) : Nat → ProgramRow → Bool :=
  fun _ row => optionIsSome (cert.expressions.lookup row.body.value)

def rootWellFormed (cert : Cert) : Bool :=
  match cert.residualRoot with
  | .closed root => optionIsSome (cert.expressions.lookup root.value)
  | .family program domain => domain.valid && optionIsSome (cert.programs.lookup program.value)

def Cert.wellFormed (cert : Cert) : Bool :=
  cert.plaintextModulus > 0 && cert.ciphertextModulus > 0 && cert.ringDimension > 0 &&
    cert.expressions.wellFormed && cert.programs.wellFormed && cert.sources.wellFormed &&
    cert.events.wellFormed &&
    cert.expressions.allBool (expressionRowWellFormed cert) &&
    cert.programs.allBool (programRowWellFormed cert) && sourceRowsWellFormed cert &&
    eventRowsWellFormed cert && rootWellFormed cert

def Cert.Valid (cert : Cert) : Prop := cert.wellFormed = true

structure CheckedCert where
  val : Cert
  valid : val.Valid

theorem valid_reflects_wellFormed {cert : Cert} (hvalid : cert.Valid) : cert.wellFormed = true := hvalid

def evalExpr (fuel : Nat) (cert : Cert) (selector : Option Nat)
    (samplers : SamplerAssignment) (inputs : InputAssignment) (root : ExprId) : Option Matrix :=
  match fuel with
  | 0 => none
  | _ + 1 =>
      match cert.expressions.lookup root.value with
      | none => none
      | some (.constant matrix) =>
          if matrix.valid cert.ciphertextModulus cert.ringDimension then some matrix else none
      | some (.source access) =>
          match inputs (access.withSelector selector) with
          | .matrix matrix =>
              if matrix.valid cert.ciphertextModulus cert.ringDimension then some matrix else none
          | .invalid => none
      | some (.sampler access) =>
          match samplers (access.withSelector selector) with
          | .matrix matrix =>
              if matrix.valid cert.ciphertextModulus cert.ringDimension then some matrix else none
          | .invalid => none

def evalProgram (fuel : Nat) (cert : Cert) (selector : Nat)
    (samplers : SamplerAssignment) (inputs : InputAssignment) (program : ProgramId) : Option Matrix :=
  match cert.programs.lookup program.value with
  | none => none
  | some row => evalExpr fuel cert (some selector) samplers inputs row.body

def evalResidual (fuel : Nat) (cert : Cert) (selector : Option Nat)
    (samplers : SamplerAssignment) (inputs : InputAssignment) : Option Matrix :=
  match cert.residualRoot with
  | .closed root => evalExpr fuel cert selector samplers inputs root
  | .family program _ =>
      match selector with
      | none => none
      | some value => evalProgram fuel cert value samplers inputs program

def InputContract (cert : Cert) (inputs : InputAssignment) : Prop :=
  cert.sources.AllFrom (fun source row =>
    ∀ selector, ∃ matrix,
      inputs { source := ⟨source⟩, selector } = .matrix matrix ∧
      matrix.valid cert.ciphertextModulus cert.ringDimension = true ∧
      matrix.shape = row.shape ∧
      matrix.CoefficientBound cert.ciphertextModulus row.coefficientBound)

def SamplerContract (cert : Cert) (samplers : SamplerAssignment) : Prop :=
  cert.events.AllFrom (fun event row =>
    ∀ selector, ∃ matrix,
      samplers { event := ⟨event⟩, selector } = .matrix matrix ∧
      matrix.valid cert.ciphertextModulus cert.ringDimension = true ∧
      matrix.shape = row.shape ∧
      matrix.CoefficientBound cert.ciphertextModulus row.coefficientBound)

def OperationalClaim (cert : CheckedCert) : Prop :=
  ∀ (samplers : SamplerAssignment) (inputs : InputAssignment),
    InputContract cert.val inputs → SamplerContract cert.val samplers →
      match cert.val.residualRoot with
      | .closed _ =>
          match evalResidual 4 cert.val none samplers inputs with
          | none => False
          | some matrix =>
              2 * cert.val.plaintextModulus *
                matrix.maxCenteredCoefficientNorm cert.val.ciphertextModulus <
              cert.val.ciphertextModulus
      | .family _ domain =>
          ∀ selector, domain.Contains selector →
            match evalResidual 4 cert.val (some selector) samplers inputs with
            | none => False
            | some matrix =>
                2 * cert.val.plaintextModulus *
                  matrix.maxCenteredCoefficientNorm cert.val.ciphertextModulus <
                cert.val.ciphertextModulus

end Mxx.Certificate.OperationalNoise

namespace List

inductive Forall₂ {α β : Type} (relation : α → β → Prop) : List α → List β → Prop
  | nil : Forall₂ relation [] []
  | cons {head tail bound bounds} :
      relation head bound → Forall₂ relation tail bounds →
        Forall₂ relation (head :: tail) (bound :: bounds)

end List

namespace Mxx.Certificate.OperationalNoise.EventReplay

open Mxx.Certificate.OperationalNoise

/-! Pure finite algebra used by G0 replay. Central factors are canonicalized as an ordered
    multiset, while ordered factors retain their concatenation order. -/

def insertCentral (factor : Nat) : List Nat → List Nat
  | [] => [factor]
  | head :: tail =>
      if factor ≤ head then factor :: head :: tail else head :: insertCentral factor tail

def canonicalCentral : List Nat → List Nat
  | [] => []
  | head :: tail => insertCentral head (canonicalCentral tail)

/-! A representation policy for central factors. The Nat instance preserves the original
    canonicalization used by structural replay; semantic factor types may choose direct
    concatenation instead of introducing an artificial order. -/
class CentralNormalizer (Factor : Type) where
  normalize : List Factor → List Factor

instance : CentralNormalizer Nat where
  normalize := canonicalCentral

theorem canonicalCentral_nil : canonicalCentral [] = [] := by rfl

theorem insertCentral_mem (factor : Nat) (factors : List Nat) :
    factor ∈ insertCentral factor factors := by
  induction factors with
  | nil => simp [insertCentral]
  | cons head tail ih =>
      by_cases h : factor ≤ head
      · simp [insertCentral, h]
      · simp [insertCentral, h, ih]

theorem canonicalCentral_mem (factor : Nat) (factors : List Nat) :
    factor ∈ canonicalCentral (factor :: factors) := by
  simp [canonicalCentral, insertCentral_mem]

def MonomialKey.product {Factor : Type} [CentralNormalizer Factor]
    (left right : MonomialKey Factor) :
    MonomialKey Factor :=
  { centralFactors := CentralNormalizer.normalize (left.centralFactors ++ right.centralFactors)
    orderedFactors := left.orderedFactors ++ right.orderedFactors }

theorem product_central {Factor : Type} [CentralNormalizer Factor]
    (left right : MonomialKey Factor) :
    (MonomialKey.product left right).centralFactors =
      CentralNormalizer.normalize (left.centralFactors ++ right.centralFactors) := by rfl

theorem product_ordered {Factor : Type} [CentralNormalizer Factor]
    (left right : MonomialKey Factor) :
    (MonomialKey.product left right).orderedFactors =
      left.orderedFactors ++ right.orderedFactors := by rfl

structure MonomialContext (Factor : Type := Nat) where
  exteriorCentral : List Factor
  prefixFactors : List Factor
  suffixFactors : List Factor
deriving DecidableEq, Repr

def MonomialContext.plug {Factor : Type} [CentralNormalizer Factor]
    (context : MonomialContext Factor)
    (key : MonomialKey Factor) : MonomialKey Factor :=
  { centralFactors := CentralNormalizer.normalize (context.exteriorCentral ++ key.centralFactors)
    orderedFactors := context.prefixFactors ++ key.orderedFactors ++ context.suffixFactors }

theorem context_plug_central {Factor : Type} [CentralNormalizer Factor]
    (context : MonomialContext Factor)
    (key : MonomialKey Factor) :
    (context.plug key).centralFactors =
      CentralNormalizer.normalize (context.exteriorCentral ++ key.centralFactors) := by
  rfl

theorem context_plug_ordered {Factor : Type} [CentralNormalizer Factor]
    (context : MonomialContext Factor)
    (key : MonomialKey Factor) :
    (context.plug key).orderedFactors =
      context.prefixFactors ++ key.orderedFactors ++ context.suffixFactors := by
  rfl

def scalePolynomial {Factor : Type} (scalar : Int) (polynomial : Polynomial Factor) :
    Polynomial Factor :=
  polynomial.map (fun term => { term with coefficient := scalar * term.coefficient })

def contextualize {Factor : Type} [CentralNormalizer Factor] (context : MonomialContext Factor)
    (polynomial : Polynomial Factor) : Polynomial Factor :=
  polynomial.map (fun term => { term with key := context.plug term.key })

def relationReplacement {Factor : Type} [CentralNormalizer Factor]
    (context : MonomialContext Factor)
    (outerCoefficient : Int) (rhs : Polynomial Factor) : Polynomial Factor :=
  scalePolynomial outerCoefficient (contextualize context rhs)

theorem scalePolynomial_coefficient {Factor : Type} [DecidableEq Factor]
    (scalar : Int) (polynomial : Polynomial Factor)
    (key : MonomialKey Factor) :
    coefficient key (scalePolynomial scalar polynomial) = scalar * coefficient key polynomial := by
  induction polynomial with
  | nil => simp [scalePolynomial, coefficient]
  | cons term terms ih =>
      by_cases h : term.key = key
      · simp only [scalePolynomial, List.map, coefficient, if_pos h]
        change scalar * term.coefficient + coefficient key (scalePolynomial scalar terms) =
          scalar * (term.coefficient + coefficient key terms)
        rw [ih]
        exact (Int.mul_add _ _ _).symm
      · simp only [scalePolynomial, List.map, coefficient, if_neg h]
        change coefficient key (scalePolynomial scalar terms) = scalar * coefficient key terms
        exact ih

theorem relationReplacement_singleton {Factor : Type} [CentralNormalizer Factor]
    (context : MonomialContext Factor) (outerCoefficient : Int) (term : ExactTerm Factor) :
    relationReplacement context outerCoefficient [term] =
      [{ coefficient := outerCoefficient * term.coefficient, key := context.plug term.key }] := by
  rfl

def productMerge_contribution {Factor : Type} [CentralNormalizer Factor]
    (left right : ExactTerm Factor) :
    ExactTerm Factor :=
  { coefficient := left.coefficient * right.coefficient
    key := MonomialKey.product left.key right.key }

theorem productMerge_contribution_coefficient {Factor : Type} [CentralNormalizer Factor]
    (left right : ExactTerm Factor) :
    (productMerge_contribution left right).coefficient = left.coefficient * right.coefficient := by
  rfl

theorem productMerge_contribution_key {Factor : Type} [CentralNormalizer Factor]
    (left right : ExactTerm Factor) :
    (productMerge_contribution left right).key = MonomialKey.product left.key right.key := by
  rfl

/-- Reclassify every factor of a typed scalar monomial as commutative. -/
def scalarActionKey {Factor : Type} [CentralNormalizer Factor]
    (key : MonomialKey Factor) : MonomialKey Factor :=
  { centralFactors := CentralNormalizer.normalize (key.centralFactors ++ key.orderedFactors)
    orderedFactors := [] }

@[simp]
theorem scalarActionKey_central {Factor : Type} [CentralNormalizer Factor]
    (key : MonomialKey Factor) :
    (scalarActionKey key).centralFactors =
      CentralNormalizer.normalize (key.centralFactors ++ key.orderedFactors) := by
  rfl

@[simp]
theorem scalarActionKey_ordered {Factor : Type} [CentralNormalizer Factor]
    (key : MonomialKey Factor) :
    (scalarActionKey key).orderedFactors = [] := by
  rfl

theorem productMerge_left_scalar_key {Factor : Type} [CentralNormalizer Factor]
    (left right : ExactTerm Factor) :
    (productMerge_contribution { left with key := scalarActionKey left.key } right).key =
      MonomialKey.product (scalarActionKey left.key) right.key := by
  exact productMerge_contribution_key _ _

theorem productMerge_left_scalar_coefficient {Factor : Type} [CentralNormalizer Factor]
    (left right : ExactTerm Factor) :
    (productMerge_contribution { left with key := scalarActionKey left.key } right).coefficient =
      left.coefficient * right.coefficient := by
  simpa using productMerge_contribution_coefficient
    { left with key := scalarActionKey left.key } right

theorem productMerge_right_scalar_key {Factor : Type} [CentralNormalizer Factor]
    (left right : ExactTerm Factor) :
    (productMerge_contribution left { right with key := scalarActionKey right.key }).key =
      MonomialKey.product left.key (scalarActionKey right.key) := by
  exact productMerge_contribution_key _ _

theorem productMerge_right_scalar_coefficient {Factor : Type} [CentralNormalizer Factor]
    (left right : ExactTerm Factor) :
    (productMerge_contribution left { right with key := scalarActionKey right.key }).coefficient =
      left.coefficient * right.coefficient := by
  simpa using productMerge_contribution_coefficient
    left { right with key := scalarActionKey right.key }

/-- Reproduce the four typed scalar roles of the Rust product projector. Exactly one scalar
    operand is centralized before product-key construction; two scalars retain their ordered
    product so relation matching can run first. -/
def scalarProductKey {Factor : Type} [CentralNormalizer Factor]
    (left right : MonomialKey Factor)
    (leftScalar rightScalar : Bool) : MonomialKey Factor :=
  if leftScalar && !rightScalar then MonomialKey.product (scalarActionKey left) right
  else if rightScalar && !leftScalar then MonomialKey.product left (scalarActionKey right)
  else MonomialKey.product left right

/-- One exact operator-product contribution with the same coefficient multiplication and typed
    scalar-role key construction as the Rust projector. -/
def operatorProductContribution {Factor : Type} [CentralNormalizer Factor]
    (left right : ExactTerm Factor)
    (leftScalar rightScalar : Bool) : ExactTerm Factor :=
  { productMerge_contribution left right with
    key := scalarProductKey left.key right.key leftScalar rightScalar }

@[simp]
theorem operatorProductContribution_coefficient {Factor : Type} [CentralNormalizer Factor]
    (left right : ExactTerm Factor)
    (leftScalar rightScalar : Bool) :
    (operatorProductContribution left right leftScalar rightScalar).coefficient =
      left.coefficient * right.coefficient := by
  exact productMerge_contribution_coefficient left right

@[simp]
theorem operatorProductContribution_key {Factor : Type} [CentralNormalizer Factor]
    (left right : ExactTerm Factor)
    (leftScalar rightScalar : Bool) :
    (operatorProductContribution left right leftScalar rightScalar).key =
      scalarProductKey left.key right.key leftScalar rightScalar := by
  rfl

@[simp]
theorem operatorProductContribution_left_scalar {Factor : Type} [CentralNormalizer Factor]
    (left right : ExactTerm Factor) :
    operatorProductContribution left right true false =
      productMerge_contribution { left with key := scalarActionKey left.key } right := by
  rfl

@[simp]
theorem operatorProductContribution_right_scalar {Factor : Type} [CentralNormalizer Factor]
    (left right : ExactTerm Factor) :
    operatorProductContribution left right false true =
      productMerge_contribution left { right with key := scalarActionKey right.key } := by
  rfl

@[simp]
theorem operatorProductContribution_both_scalar {Factor : Type} [CentralNormalizer Factor]
    (left right : ExactTerm Factor) :
    operatorProductContribution left right true true = productMerge_contribution left right := by
  rfl

@[simp]
theorem operatorProductContribution_neither_scalar {Factor : Type} [CentralNormalizer Factor]
    (left right : ExactTerm Factor) :
    operatorProductContribution left right false false = productMerge_contribution left right := by
  rfl

theorem boundTransfer_sum {left right left' right' : Nat}
    (leftBound : left ≤ left') (rightBound : right ≤ right') :
    left + right ≤ left' + right' := by
  exact Nat.add_le_add leftBound rightBound

theorem boundTransfer_scale {scalar bound bound' : Nat} (boundTransfer : bound ≤ bound') :
    scalar * bound ≤ scalar * bound' := by
  exact Nat.mul_le_mul_left scalar boundTransfer

theorem boundTransfer_product {left right left' right' : Nat}
    (leftBound : left ≤ left') (rightBound : right ≤ right') :
    left * right ≤ left' * right' := by
  exact Nat.mul_le_mul leftBound rightBound

theorem operatorProductContribution_natAbs_le {Factor : Type} [CentralNormalizer Factor]
    (left right : ExactTerm Factor)
    (leftScalar rightScalar : Bool) (leftBound rightBound : Nat)
    (leftExactBound : left.coefficient.natAbs ≤ leftBound)
    (rightExactBound : right.coefficient.natAbs ≤ rightBound) :
    (operatorProductContribution left right leftScalar rightScalar).coefficient.natAbs ≤
      leftBound * rightBound := by
  rw [operatorProductContribution_coefficient, Int.natAbs_mul]
  exact boundTransfer_product leftExactBound rightExactBound

theorem boundTransfer_zero_product (bound : Nat) : 0 * bound = 0 := by simp

theorem boundTransfer_product_zero (bound : Nat) : bound * 0 = 0 := by simp

theorem List.sum_le_sum_forall₂ {left right : List Nat}
    (hbound : List.Forall₂ (fun value bound => value ≤ bound) left right) :
    left.sum ≤ right.sum := by
  induction hbound with
  | nil => simp
  | cons head tail ih => exact Nat.add_le_add head ih

theorem survivorFold_sound {contributions bounds : List Nat}
    (hbound : List.Forall₂ (fun value bound => value ≤ bound) contributions bounds) :
    contributions.sum ≤ bounds.sum := by
  exact List.sum_le_sum_forall₂ hbound

theorem preFold_to_invocationEnd {summaryActual summaryBound : Nat}
    {survivorContributions survivorBounds : List Nat}
    (summaryBoundProof : summaryActual ≤ summaryBound)
    (transferBounds : List.Forall₂ (fun value bound => value ≤ bound)
      survivorContributions survivorBounds) :
    summaryActual + survivorContributions.sum ≤ summaryBound + survivorBounds.sum := by
  exact Nat.add_le_add summaryBoundProof (survivorFold_sound transferBounds)

end Mxx.Certificate.OperationalNoise.EventReplay
