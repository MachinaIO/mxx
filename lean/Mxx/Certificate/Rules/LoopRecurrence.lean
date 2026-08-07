import Mxx.Certificate.Semantics

namespace Mxx.Certificate

/-! # Sequential-loop recurrence schemas

This module defines the typed paths and projection operations used by the trace-bound recurrence
soundness layer. Executable trace evidence and dependent step derivations live in
`TraceBoundRecurrence`; no caller-provided invariant or preservation callback is exposed here.
-/

/-! ## Typed result paths

The recurrence result constructors are only safe after their path has been checked against the
fixed body-output schema.  Keeping this validation typed prevents a coefficient, noise, and total
bound from being confused merely because they use the same natural-number slot.
-/

def MatrixFactPath.carriedSlot : MatrixFactPath → Nat
  | .exactExpression slot | .affineCoefficient slot _ | .affineBasis slot _ |
      .familyElement slot _ _ => slot

private def MatrixFactPath.validAt
    (rootSlot : Nat) : MatrixFactPath → ValueFactSchema → Bool
  | .exactExpression slot, .matrix _ .exact _ _ => slot == rootSlot
  | .affineCoefficient slot term, .matrix _ (.affine terms) _ _
  | .affineBasis slot term, .matrix _ (.affine terms) _ _ =>
      slot == rootSlot && term < terms.length
  | .familyElement slot _ nested, .family _ element =>
      slot == rootSlot && nested.validAt rootSlot element
  | _, _ => false

private def MatrixFactPath.typeAt
    (rootSlot : Nat) : MatrixFactPath → ValueFactSchema → Option MatrixTypeExpr
  | .exactExpression slot, .matrix type .exact _ _ =>
      if slot == rootSlot then some type else none
  | .affineCoefficient slot term, .matrix _ (.affine terms) _ _ => do
      if slot != rootSlot then none else
      return (← terms[term]?).coefficientType
  | .affineBasis slot term, .matrix _ (.affine terms) _ _ => do
      if slot != rootSlot then none else
      return (← terms[term]?).basisType
  | .familyElement slot _ nested, .family _ element =>
      if slot == rootSlot then nested.typeAt rootSlot element else none
  | _, _ => none

private def MatrixFactPath.checkedType {arity : Nat}
    (path : MatrixFactPath)
    (schemas : Vector ValueFactTemplate arity) : Option MatrixTypeExpr := do
  let slot := path.carriedSlot
  let template ← schemas[slot]?
  path.typeAt slot template.schema

def MatrixFactPath.valid {arity : Nat}
    (path : MatrixFactPath)
    (schemas : Vector ValueFactTemplate arity) : Bool :=
  let slot := path.carriedSlot
  if slot < arity then
    match schemas[slot]? with
    | some template => path.validAt slot template.schema
    | none => false
  else false

def BoundFactPath.carriedSlot : BoundFactPath → Nat
  | .affineCoefficientBound slot _ | .affineNoiseBound slot | .matrixTotalBound slot |
      .familyElement slot _ _ => slot

private def BoundFactPath.validAt
    (rootSlot : Nat) : BoundFactPath → ValueFactSchema → Bool
  | .affineCoefficientBound slot term, .matrix _ (.affine terms) _ _ =>
      slot == rootSlot && term < terms.length
  | .affineNoiseBound slot, .matrix _ (.affine _) _ _ => slot == rootSlot
  | .matrixTotalBound slot, .matrix .. => slot == rootSlot
  | .familyElement slot _ nested, .family _ element =>
      slot == rootSlot && nested.validAt rootSlot element
  | _, _ => false

def BoundFactPath.valid {arity : Nat}
    (path : BoundFactPath)
    (schemas : Vector ValueFactTemplate arity) : Bool :=
  let slot := path.carriedSlot
  if slot < arity then
    match schemas[slot]? with
    | some template => path.validAt slot template.schema
    | none => false
  else false

def RuntimeFactPath.carriedSlot : {type : RuntimeScalarType} → RuntimeFactPath type → Nat
  | _, .integerValue slot | _, .booleanValue slot | _, .familyElement slot _ _ => slot

private def RuntimeFactPath.validAt
    (rootSlot : Nat) : {type : RuntimeScalarType} → RuntimeFactPath type → ValueFactSchema → Bool
  | _, .integerValue slot, .integer => slot == rootSlot
  | _, .booleanValue slot, .boolean => slot == rootSlot
  | _, .familyElement slot _ nested, .family _ element =>
      slot == rootSlot && nested.validAt rootSlot element
  | _, _, _ => false

def RuntimeFactPath.valid {arity : Nat} {type : RuntimeScalarType}
    (path : RuntimeFactPath type)
    (schemas : Vector ValueFactTemplate arity) : Bool :=
  let slot := path.carriedSlot
  if slot < arity then
    match schemas[slot]? with
    | some template => path.validAt slot template.schema
    | none => false
  else false

def IntBoundFactPath.carriedSlot : IntBoundFactPath → Nat
  | .lower slot | .upper slot | .familyElement slot _ _ => slot

private def IntBoundFactPath.validAt
    (rootSlot : Nat) : IntBoundFactPath → ValueFactSchema → Bool
  | .lower slot, .integer | .upper slot, .integer => slot == rootSlot
  | .familyElement slot _ nested, .family _ element =>
      slot == rootSlot && nested.validAt rootSlot element
  | _, _ => false

def IntBoundFactPath.valid {arity : Nat}
    (path : IntBoundFactPath)
    (schemas : Vector ValueFactTemplate arity) : Bool :=
  let slot := path.carriedSlot
  if slot < arity then
    match schemas[slot]? with
    | some template => path.validAt slot template.schema
    | none => false
  else false

theorem MatrixFactPath.valid_carriedSlot_lt
    {arity : Nat}
    {path : MatrixFactPath}
    {schemas : Vector ValueFactTemplate arity}
    (valid : path.valid schemas) :
    path.carriedSlot < arity := by
  simp only [MatrixFactPath.valid] at valid
  split at valid
  · assumption
  · simp at valid

theorem BoundFactPath.valid_carriedSlot_lt
    {arity : Nat}
    {path : BoundFactPath}
    {schemas : Vector ValueFactTemplate arity}
    (valid : path.valid schemas) :
    path.carriedSlot < arity := by
  simp only [BoundFactPath.valid] at valid
  split at valid
  · assumption
  · simp at valid

private def diamondCarriedFixtureType : MatrixTypeExpr where
  modulus := .constant 17
  ringDimension := .constant 4
  rows := .constant 1
  columns := .constant 1

private def diamondCarriedFixtureTemplates : Vector ValueFactTemplate 4 := ⟨#[
  {
    fact := .integer {
      expression := .carriedInput (.integerValue 0)
      lower := .carriedInput (.lower 0)
      upper := .carriedInput (.upper 0)
    }
    schema := .integer
  },
  {
    fact := .boolean { expression := .carriedInput (.booleanValue 1) }
    schema := .boolean
  },
  {
    fact := .family {
      aggregate := .carriedInput 2
      count := .constant 8
      elementSchema := .matrix diamondCarriedFixtureType .exact [] .unknown
    }
    schema := .family (.constant 8)
      (.matrix diamondCarriedFixtureType .exact [] .unknown)
  },
  {
    fact := .matrix {
      subject := .protocolInput ⟨"diamond-matrix-carried"⟩
      primary := .exact (.carriedInput diamondCarriedFixtureType (.exactExpression 3))
      relations := []
      totalNormBound := .carriedInput (.matrixTotalBound 3)
    }
    schema := .matrix diamondCarriedFixtureType .exact [] .unknown
  }
], rfl⟩

/-- The four carried shapes present in the generated Diamond workflow are accepted by the same
recursive schema checker: integer, Boolean, indexed matrix family, and matrix. -/
example : (RuntimeFactPath.integerValue 0).valid diamondCarriedFixtureTemplates = true := rfl
example : (RuntimeFactPath.booleanValue 1).valid diamondCarriedFixtureTemplates = true := rfl
example : (MatrixFactPath.familyElement 2 ⟨0⟩ (.exactExpression 2)).valid
    diamondCarriedFixtureTemplates = true := rfl
example : (MatrixFactPath.exactExpression 3).valid diamondCarriedFixtureTemplates = true := rfl
example : (MatrixFactPath.exactExpression 3).checkedType diamondCarriedFixtureTemplates =
    some diamondCarriedFixtureType := rfl

/-- A nested path cannot silently switch to a different carried slot. -/
example : (MatrixFactPath.familyElement 2 ⟨0⟩ (.exactExpression 3)).valid
    diamondCarriedFixtureTemplates = false := rfl


/-- Project one recurrence output without unrolling. Every carried path is checked against the
recursive body-output schema before it becomes an instance-qualified recurrence reference. -/
def projectRecurrenceOutput {arity : Nat}
    (recurrence : SequentialRecurrenceInstanceRef)
    (schemas : Vector ValueFactTemplate arity)
    (slot : Nat)
    (output : ValueInstanceRef) : Option ValueFact := do
  let template ← schemas[slot]?
  match template.fact with
  | .matrix fact => do
      if !fact.relations.isEmpty then none else
      let primary : MatrixPrimaryForm ← match fact.primary with
        | .exact _ =>
            let path := MatrixFactPath.exactExpression slot
            pure (.exact (.loopResult (← path.checkedType schemas) recurrence path))
        | .affine form => do
            let terms ← form.terms.zipIdx.mapM fun (term, termIndex) => do
              let coefficientPath := MatrixFactPath.affineCoefficient slot termIndex
              let basisPath := MatrixFactPath.affineBasis slot termIndex
              return {
                coefficient := {
                  expression := .loopResult (← coefficientPath.checkedType schemas)
                    recurrence coefficientPath
                  normBound := .recurrenceResult recurrence
                    (.affineCoefficientBound slot termIndex)
                }
                basis := .loopResult (← basisPath.checkedType schemas) recurrence basisPath
                mode := term.mode
              }
            pure (.affine {
              terms
              noiseBound := .recurrenceResult recurrence (.affineNoiseBound slot)
            })
      return .matrix {
        fact with
        subject := output
        primary
        relations := []
        totalNormBound := .recurrenceResult recurrence (.matrixTotalBound slot)
      }
  | .integer _ => return .integer {
      expression := .intWire (.recurrenceResult recurrence slot)
      lower := .recurrenceResult recurrence (.lower slot)
      upper := .recurrenceResult recurrence (.upper slot)
    }
  | .boolean _ => return .boolean {
      expression := .boolWire (.recurrenceResult recurrence slot)
    }
  | .family fact => return .family {
      fact with aggregate := .recurrenceResult recurrence.recurrence recurrence.path slot
    }
  | _ => none

private def compactProjectionFixtureRecurrence : SequentialRecurrenceInstanceRef := {
  recurrence := ⟨{
    stage := ⟨"fixture"⟩
    scope := ⟨[]⟩
    node := ⟨17⟩
  }⟩
  path := []
}

/-- Scalar recurrence outputs refer directly to the final carried slot, never to a rewritten
copy of the one-step body expression. -/
example : projectRecurrenceOutput compactProjectionFixtureRecurrence
    diamondCarriedFixtureTemplates 0 (.protocolInput ⟨"integer-output"⟩) = some (.integer {
      expression := .intWire (.recurrenceResult compactProjectionFixtureRecurrence 0)
      lower := .recurrenceResult compactProjectionFixtureRecurrence (.lower 0)
      upper := .recurrenceResult compactProjectionFixtureRecurrence (.upper 0)
    }) := rfl

/-- Matrix recurrence outputs are compact typed paths. In particular, the body-local carried
placeholder is not substituted into a second symbolic expression graph. -/
example : projectRecurrenceOutput compactProjectionFixtureRecurrence
    diamondCarriedFixtureTemplates 3 (.protocolInput ⟨"matrix-output"⟩) = some (.matrix {
      subject := .protocolInput ⟨"matrix-output"⟩
      primary := .exact
        (.loopResult diamondCarriedFixtureType compactProjectionFixtureRecurrence
          (.exactExpression 3))
      relations := []
      totalNormBound := .recurrenceResult compactProjectionFixtureRecurrence
        (.matrixTotalBound 3)
    }) := rfl

def FamilyAggregateRef.hasCarriedInput : FamilyAggregateRef → Bool
  | .carriedInput _ => true
  | .familyElement parent _ => parent.hasCarriedInput
  | _ => false

def ValueInstanceRef.hasCarriedInput : ValueInstanceRef → Bool
  | .familyElement aggregate _ => aggregate.hasCarriedInput
  | _ => false

def RuntimeExpr.hasCarriedInput : {type : RuntimeScalarType} → RuntimeExpr type → Bool
  | _, .intWire wire | _, .boolWire wire => wire.hasCarriedInput
  | _, .intBinary _ left right | _, .compare _ left right =>
      left.hasCarriedInput || right.hasCarriedInput
  | _, .bitExtract value _ | _, .boolToInt value => value.hasCarriedInput
  | _, .thresholdDecodeBool matrix .. => matrix.hasCarriedInput
  | _, .familyElement _ aggregate _ index =>
      aggregate.hasCarriedInput || index.hasCarriedInput
  | _, .select _ index _ => index.hasCarriedInput
  | _, .carriedInput _ => true
  | _, _ => false

mutual
  def MatrixExpr.hasCarriedInput : MatrixExpr → Bool
    | .wire reference => reference.value.hasCarriedInput
    | .gadget _ _ => false
    | .add left right | .multiply left right =>
        left.hasCarriedInput || right.hasCarriedInput
    | .negate value | .scalarMultiply _ value | .rowSlice value _ _ |
        .columnSlice value _ _ | .rowCoefficientEmbed _ _ value |
        .columnBasisEmbed _ _ value | .diagonalCoefficientEmbed _ _ value |
        .diagonalBasisEmbed _ _ value => value.hasCarriedInput
    | .rowConcat parts | .columnConcat parts | .diagonalConcat parts =>
        matrixExprListHasCarriedInput parts
    | .select index branches =>
        index.hasCarriedInput || matrixExprListHasCarriedInput branches
    | .carriedInput _ _ => true
    | _ => false

  def matrixExprListHasCarriedInput : List MatrixExpr → Bool
    | [] => false
    | expression :: tail => expression.hasCarriedInput || matrixExprListHasCarriedInput tail
end

def BoundExpr.hasCarriedInput : BoundExpr → Bool
  | .add left right | .multiply left right | .maximum left right | .minimum left right =>
      left.hasCarriedInput || right.hasCarriedInput
  | .floorDivide value _ => value.hasCarriedInput
  | .matrixProduct _ _ left right => left.hasCarriedInput || right.hasCarriedInput
  | .carriedInput _ => true
  | _ => false

def IntBoundExpr.hasCarriedInput : IntBoundExpr → Bool
  | .negate value => value.hasCarriedInput
  | .add left right | .subtract left right | .multiply left right | .divide left right |
      .minimum left right | .maximum left right =>
      left.hasCarriedInput || right.hasCarriedInput
  | .carriedInput _ => true
  | _ => false

private def MatrixRelation.hasCarriedInput : MatrixRelation → Bool
  | .preimage subject source target trapdoor =>
      subject.hasCarriedInput || source.value.hasCarriedInput ||
        target.value.hasCarriedInput || trapdoor.hasCarriedInput
  | .gadgetDecomposition subject target .. =>
      subject.hasCarriedInput || target.value.hasCarriedInput

def ValueFact.hasCarriedInput : ValueFact → Bool
  | .matrix fact =>
      let primary := match fact.primary with
        | .exact expression => expression.hasCarriedInput
        | .affine form => form.noiseBound.hasCarriedInput || form.terms.any fun term =>
            term.coefficient.expression.hasCarriedInput ||
              term.coefficient.normBound.hasCarriedInput || term.basis.hasCarriedInput
      fact.subject.hasCarriedInput || primary || fact.totalNormBound.hasCarriedInput ||
        fact.relations.any MatrixRelation.hasCarriedInput
  | .trapdoor fact => fact.privatePort.hasCarriedInput || fact.publicPort.hasCarriedInput ||
      fact.publicMatrix.hasCarriedInput
  | .integer fact => fact.expression.hasCarriedInput || fact.lower.hasCarriedInput ||
      fact.upper.hasCarriedInput
  | .boolean fact => fact.expression.hasCarriedInput
  | .bytes wire => wire.hasCarriedInput
  | .family fact => fact.aggregate.hasCarriedInput

/-! ## GGH15 preimage-chain hard-bound recurrence

These definitions are the closed hard-bound rule for the recurrence
`C_(i+1) = (s_i S_i) B_(i+1) + (s_i E_i + e_i K_i)`.  They evaluate only
the three numeric components and never expand the symbolic coefficient product.
-/

structure Ggh15UniformBounds where
  ciphertextModulus : Nat
  ringDimension : Nat
  secretRows : Nat
  publicColumns : Nat
  secretStep : Nat
  relationError : Nat
  preimage : Nat
  publicMatrix : Nat
  deriving BEq, DecidableEq, Repr

structure Ggh15RecurrenceBounds where
  coefficient : Nat
  noise : Nat
  total : Nat
  deriving BEq, DecidableEq, Repr

private def hardMatrixProduct (ringDimension innerDimension left right : Nat) : Nat :=
  ringDimension * innerDimension * left * right

private def centeredCap (ciphertextModulus value : Nat) : Nat :=
  min (ciphertextModulus / 2) value

def Ggh15RecurrenceBounds.step
    (uniform : Ggh15UniformBounds)
    (current : Ggh15RecurrenceBounds) : Ggh15RecurrenceBounds :=
  let coefficient := centeredCap uniform.ciphertextModulus
    (hardMatrixProduct uniform.ringDimension uniform.secretRows
      current.coefficient uniform.secretStep)
  let noise := centeredCap uniform.ciphertextModulus
    (hardMatrixProduct uniform.ringDimension uniform.secretRows
      current.coefficient uniform.relationError +
    hardMatrixProduct uniform.ringDimension uniform.publicColumns
      current.noise uniform.preimage)
  let total := centeredCap uniform.ciphertextModulus
    (hardMatrixProduct uniform.ringDimension uniform.secretRows
      coefficient uniform.publicMatrix + noise)
  { coefficient, noise, total }

@[simp] theorem Ggh15RecurrenceBounds.step_coefficient
    (uniform : Ggh15UniformBounds)
    (current : Ggh15RecurrenceBounds) :
    (current.step uniform).coefficient = centeredCap uniform.ciphertextModulus
      (hardMatrixProduct uniform.ringDimension uniform.secretRows
        current.coefficient uniform.secretStep) := rfl

@[simp] theorem Ggh15RecurrenceBounds.step_noise
    (uniform : Ggh15UniformBounds)
    (current : Ggh15RecurrenceBounds) :
    (current.step uniform).noise = centeredCap uniform.ciphertextModulus
      (hardMatrixProduct uniform.ringDimension uniform.secretRows
        current.coefficient uniform.relationError +
      hardMatrixProduct uniform.ringDimension uniform.publicColumns
        current.noise uniform.preimage) := rfl

@[simp] theorem Ggh15RecurrenceBounds.step_total
    (uniform : Ggh15UniformBounds)
    (current : Ggh15RecurrenceBounds) :
    (current.step uniform).total = centeredCap uniform.ciphertextModulus
      (hardMatrixProduct uniform.ringDimension uniform.secretRows
        (current.step uniform).coefficient uniform.publicMatrix +
      (current.step uniform).noise) := rfl

/-- Ordered numeric fold for the recurrence components.  The symbolic matrix expression remains
represented by `MatrixExpr.loopResult`; this function does not expand it. -/
def Ggh15RecurrenceBounds.after
    (uniform : Ggh15UniformBounds) : Nat → Ggh15RecurrenceBounds → Ggh15RecurrenceBounds
  | 0, initial => initial
  | count + 1, initial => Ggh15RecurrenceBounds.after uniform count (initial.step uniform)

@[simp] theorem Ggh15RecurrenceBounds.after_zero
    (uniform : Ggh15UniformBounds)
    (initial : Ggh15RecurrenceBounds) :
    initial.after uniform 0 = initial := rfl

@[simp] theorem Ggh15RecurrenceBounds.after_succ
    (uniform : Ggh15UniformBounds)
    (initial : Ggh15RecurrenceBounds)
    (count : Nat) :
    initial.after uniform (count + 1) = (initial.step uniform).after uniform count := rfl

theorem Ggh15RecurrenceBounds.after_add
    (uniform : Ggh15UniformBounds)
    (initial : Ggh15RecurrenceBounds)
    (left right : Nat) :
    initial.after uniform (left + right) = (initial.after uniform left).after uniform right := by
  induction left generalizing initial with
  | zero => simp
  | succ left induction =>
      simp only [Nat.succ_add, after_succ]
      exact induction (initial.step uniform)

def Ggh15RecurrenceBounds.resolvePath
    (bounds : Ggh15RecurrenceBounds)
    (path : BoundFactPath) : Option Nat :=
  match path with
  | .affineCoefficientBound 0 0 => some bounds.coefficient
  | .affineNoiseBound 0 => some bounds.noise
  | .matrixTotalBound 0 => some bounds.total
  | _ => none

example :
    ({ coefficient := 2, noise := 3, total := 0 } : Ggh15RecurrenceBounds).step {
      ciphertextModulus := 100000
      ringDimension := 4
      secretRows := 2
      publicColumns := 3
      secretStep := 5
      relationError := 7
      preimage := 11
      publicMatrix := 13
    } = { coefficient := 80, noise := 508, total := 8828 } := by
  decide

end Mxx.Certificate
