import Lean.Elab.Tactic.Omega

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
