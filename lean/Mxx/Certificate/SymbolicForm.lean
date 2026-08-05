import Mxx.Certificate.ExpressionArena
import Mxx.Certificate.Typing

namespace Mxx.Certificate

/-! # Typed symbolic matrix forms

This module defines the analyzer-owned symbolic-form and bound-witness DAG syntax.  These DAGs
describe correctness facts; they are not executable IR and are never supplied by a protocol
certificate.  Each arena entry may refer only to an older entry.
-/

/-- Stable reference into the analyzer-owned symbolic-form arena. -/
structure SymbolicMatrixFormRef where
  id : Nat
  deriving BEq, DecidableEq, Repr

/-- Stable reference into the analyzer-owned bound-witness arena. -/
structure BoundWitnessRef where
  id : Nat
  deriving BEq, DecidableEq, Repr

/-- Stable identity of a checked preimage relation. -/
structure PreimageRelationRef where
  id : Nat
  deriving BEq, DecidableEq, Repr

/-- Stable identity of a checked gadget-decomposition relation. -/
structure GadgetDecompositionRelationRef where
  id : Nat
  deriving BEq, DecidableEq, Repr

/-- A compact nonempty collection used by selection and concatenation nodes. -/
structure NonemptyRefs (α : Type) where
  head : α
  tail : List α

def NonemptyRefs.toList {α : Type} (values : NonemptyRefs α) : List α :=
  values.head :: values.tail

def NonemptyRefs.all {α : Type} (values : NonemptyRefs α) (predicate : α → Bool) : Bool :=
  predicate values.head && values.tail.all predicate

/-- Whether a symbolic form contains an unbounded signal component. -/
inductive SignalPresence where
  | none
  | present
  deriving BEq, DecidableEq, Repr

def SignalPresence.combine : SignalPresence → SignalPresence → SignalPresence
  | .none, .none => .none
  | _, _ => .present

/-- Fixed-schema hard-bound summary carried across sequential recurrences. -/
structure MatrixBoundSummary where
  signal : SignalPresence
  coefficientL1Bound : BoundExpr
  noiseBound : BoundExpr
  totalBound : BoundExpr

namespace MatrixBoundSummary

/-- Summary for an exact value whose centered representative may fill the entire modulus. -/
def exactLarge (centeredRepresentativeBound : BoundExpr) : MatrixBoundSummary where
  signal := .present
  coefficientL1Bound := .constant 1
  noiseBound := .constant 0
  totalBound := centeredRepresentativeBound

/-- Summary for a bounded exact value. -/
def bounded (bound : BoundExpr) : MatrixBoundSummary where
  signal := .none
  coefficientL1Bound := .constant 0
  noiseBound := bound
  totalBound := bound

def negate (summary : MatrixBoundSummary) : MatrixBoundSummary := summary

/-- Deterministic hard-bound addition.  A signal-present result is capped by its centered
representative bound without replacing the more precise noise witness by that cap. -/
def add
    (centeredRepresentativeBound : BoundExpr)
    (left right : MatrixBoundSummary) : MatrixBoundSummary :=
  let signal := left.signal.combine right.signal
  {
    signal
    coefficientL1Bound := .add left.coefficientL1Bound right.coefficientL1Bound
    noiseBound := .add left.noiseBound right.noiseBound
    totalBound := match signal with
      | .none => .add left.totalBound right.totalBound
      | .present => .minimum (.add left.totalBound right.totalBound)
          centeredRepresentativeBound
  }

private def branchNoise
    (resultSignal : SignalPresence)
    (branch : MatrixBoundSummary) : BoundExpr :=
  match resultSignal, branch.signal with
  | .present, .none => branch.totalBound
  | _, _ => branch.noiseBound

/-- Whole-form dynamic selection.  Bounds are maxima, not sums.  If any branch contains a signal,
a bounded-only branch is interpreted as a zero-signal branch and its entire value is noise. -/
def select
    (head : MatrixBoundSummary)
    (tail : List MatrixBoundSummary) : MatrixBoundSummary :=
  let signal := tail.foldl (fun result branch => result.combine branch.signal) head.signal
  {
    signal
    coefficientL1Bound := tail.foldl
      (fun bound branch => .maximum bound branch.coefficientL1Bound)
      head.coefficientL1Bound
    noiseBound := tail.foldl
      (fun bound branch => .maximum bound (branchNoise signal branch))
      (branchNoise signal head)
    totalBound := tail.foldl
      (fun bound branch => .maximum bound branch.totalBound)
      head.totalBound
  }

end MatrixBoundSummary

/-- Closed semantic roles for bounds.  Protocols cannot add role names. -/
inductive BoundRole where
  | coefficient
  | matchedCoefficient
  | unmatchedCoefficient
  | relationError
  | noise
  | total
  deriving BEq, DecidableEq, Repr

/-- Closed multiplication rules used by bound witnesses. -/
inductive MatrixProductBoundRule where
  | coefficientProduct (mode : SignalProductMode)
  | leftNoiseProduct (mode : SignalProductMode)
  | rightNoiseProduct (mode : SignalProductMode)
  | totalProduct (mode : SignalProductMode)
  deriving BEq, DecidableEq, Repr

/-- Analyzer-derived evidence for every hard-bound summary component. -/
inductive BoundWitness where
  | atom (role : BoundRole) (bound : BoundExpr)
  | add (left right : BoundWitnessRef)
  | multiply
      (rule : MatrixProductBoundRule)
      (left right : BoundWitnessRef)
  | selectMax
      (index : RuntimeExprRef .integer)
      (branches : NonemptyRefs BoundWitnessRef)
  | preimageRewrite
      (relation : PreimageRelationRef)
      (matched unmatched : BoundWitnessRef)
  | gadgetRewrite
      (relation : GadgetDecompositionRelationRef)
      (matched unmatched : BoundWitnessRef)
  | recurrenceResult
      (recurrence : FactRecurrenceInstanceRef)
      (slot : Nat)
      (role : BoundRole)

/-- A witness entry records the role proved by its root node. -/
structure BoundWitnessEntry where
  role : BoundRole
  witness : BoundWitness

structure BoundWitnessWFContext where
  runtimeExpressionCount : Nat
  preimageRelationCount : Nat
  gadgetRelationCount : Nat
  recurrences : List (FactRecurrenceInstanceRef × Nat)
  allowCarriedBounds : Bool := false

private def boundHasCarriedInput : BoundExpr → Bool
  | .add left right | .multiply left right | .maximum left right | .minimum left right =>
      boundHasCarriedInput left || boundHasCarriedInput right
  | .floorDivide value _ => boundHasCarriedInput value
  | .matrixProduct _ _ left right =>
      boundHasCarriedInput left || boundHasCarriedInput right
  | .carriedInput _ => true
  | .constant _ | .parameter _ | .absolute _ | .recurrenceResult _ _ => false

private def recurrenceSlotIsValid
    (recurrences : List (FactRecurrenceInstanceRef × Nat))
    (recurrence : FactRecurrenceInstanceRef)
    (slot : Nat) : Bool :=
  recurrences.any fun registered => registered.1 == recurrence && slot < registered.2

private def witnessRefIsValid
    (priorEntries : List BoundWitnessEntry)
    (reference : BoundWitnessRef) : Bool :=
  reference.id < priorEntries.length

private def witnessRefHasRole
    (priorEntries : List BoundWitnessEntry)
    (reference : BoundWitnessRef)
    (role : BoundRole) : Bool :=
  match priorEntries[reference.id]? with
  | some entry => entry.role == role
  | none => false

def BoundWitnessEntry.refsBefore
    (context : BoundWitnessWFContext)
    (priorEntries : List BoundWitnessEntry)
    (entry : BoundWitnessEntry) : Bool :=
  match entry.witness with
  | .atom role bound =>
      role == entry.role && (context.allowCarriedBounds || !boundHasCarriedInput bound)
  | .add left right =>
      witnessRefHasRole priorEntries left entry.role &&
        witnessRefHasRole priorEntries right entry.role
  | .multiply _ left right =>
      witnessRefIsValid priorEntries left && witnessRefIsValid priorEntries right
  | .selectMax index branches =>
      index.id < context.runtimeExpressionCount &&
        branches.all (fun reference => witnessRefHasRole priorEntries reference entry.role)
  | .preimageRewrite relation matched unmatched =>
      relation.id < context.preimageRelationCount &&
        witnessRefIsValid priorEntries matched && witnessRefIsValid priorEntries unmatched
  | .gadgetRewrite relation matched unmatched =>
      relation.id < context.gadgetRelationCount &&
        witnessRefIsValid priorEntries matched && witnessRefIsValid priorEntries unmatched
  | .recurrenceResult recurrence slot role =>
      role == entry.role && recurrenceSlotIsValid context.recurrences recurrence slot

/-- Immutable append-only witness arena. -/
structure BoundWitnessArena where
  entries : List BoundWitnessEntry := []

def BoundWitnessArena.WF
    (context : BoundWitnessWFContext)
    (arena : BoundWitnessArena) : Bool :=
  arena.entries.zipIdx.all fun (entry, index) =>
    entry.refsBefore context (arena.entries.take index)

def BoundWitnessArena.intern
    (context : BoundWitnessWFContext)
    (arena : BoundWitnessArena)
    (entry : BoundWitnessEntry) : Option (BoundWitnessArena × BoundWitnessRef) :=
  if entry.refsBefore context arena.entries then
    some (⟨arena.entries ++ [entry]⟩, ⟨arena.entries.length⟩)
  else none

/-- Residual symbolic structure retained when eager affine normalization would duplicate a
dynamic selection or sequential recurrence. -/
inductive SymbolicMatrixForm where
  | signalAtom (expression : MatrixExprRef)
  | boundedAtom (expression : MatrixExprRef) (bound : BoundExpr)
  | affineLeaf (form : AffineForm)
  | add (left right : SymbolicMatrixFormRef)
  | negate (value : SymbolicMatrixFormRef)
  | multiply (left right : SymbolicMatrixFormRef)
  | select
      (index : RuntimeExprRef .integer)
      (branches : NonemptyRefs SymbolicMatrixFormRef)
  | rowSlice (value : SymbolicMatrixFormRef) (start stop : IntExpr)
  | columnSlice (value : SymbolicMatrixFormRef) (start stop : IntExpr)
  | rowConcat (parts : NonemptyRefs SymbolicMatrixFormRef)
  | columnConcat (parts : NonemptyRefs SymbolicMatrixFormRef)
  | diagonalConcat (parts : NonemptyRefs SymbolicMatrixFormRef)
  | reshape (value : SymbolicMatrixFormRef) (outputType : MatrixTypeExpr)
  | preimageRewrite
      (relation : PreimageRelationRef)
      (input : SymbolicMatrixFormRef)
  | gadgetRewrite
      (relation : GadgetDecompositionRelationRef)
      (input : SymbolicMatrixFormRef)
  | carriedInput (matrixType : MatrixTypeExpr) (slot : Nat)
  | recurrenceResult
      (matrixType : MatrixTypeExpr)
      (recurrence : FactRecurrenceInstanceRef)
      (slot : Nat)

/-- Every symbolic-form node carries its checked result type in its arena entry. -/
structure SymbolicMatrixFormEntry where
  matrixType : MatrixTypeExpr
  form : SymbolicMatrixForm

structure SymbolicFormWFContext where
  expressionArena : ExpressionArena
  preimageRelationCount : Nat
  gadgetRelationCount : Nat
  carriedArity : Nat
  recurrences : List (FactRecurrenceInstanceRef × Nat)
  allowCarriedInputs : Bool := false

private def symbolicRefHasType
    (priorEntries : List SymbolicMatrixFormEntry)
    (reference : SymbolicMatrixFormRef)
    (expected : MatrixTypeExpr) : Bool :=
  match priorEntries[reference.id]? with
  | some entry => entry.matrixType == expected
  | none => false

private def matrixRefHasType
    (context : SymbolicFormWFContext)
    (reference : MatrixExprRef)
    (expected : MatrixTypeExpr) : Bool :=
  match context.expressionArena.lookupMatrix reference with
  | some expression => expression.inferType == some expected
  | none => false

private def signalTermHasType (term : SignalTerm) (expected : MatrixTypeExpr) : Bool :=
  match term.coefficient.expression.inferType, term.basis.inferType with
  | some coefficientType, some basisType =>
      match inferMatrixProductType coefficientType basisType with
      | .ok product => product.output == expected && product.mode == term.mode
      | .error _ => false
  | _, _ => false

private def affineLeafHasType (form : AffineForm) (expected : MatrixTypeExpr) : Bool :=
  form.terms.all fun term => signalTermHasType term expected

private def symbolicRefsHaveType
    (priorEntries : List SymbolicMatrixFormEntry)
    (references : NonemptyRefs SymbolicMatrixFormRef)
    (expected : MatrixTypeExpr) : Bool :=
  references.all fun reference => symbolicRefHasType priorEntries reference expected

private def symbolicRefExists
    (priorEntries : List SymbolicMatrixFormEntry)
    (reference : SymbolicMatrixFormRef) : Bool :=
  reference.id < priorEntries.length

private def symbolicRefsExist
    (priorEntries : List SymbolicMatrixFormEntry)
    (references : NonemptyRefs SymbolicMatrixFormRef) : Bool :=
  references.all fun reference => symbolicRefExists priorEntries reference

def SymbolicMatrixFormEntry.refsBefore
    (context : SymbolicFormWFContext)
    (priorEntries : List SymbolicMatrixFormEntry)
    (entry : SymbolicMatrixFormEntry) : Bool :=
  match entry.form with
  | .signalAtom expression => matrixRefHasType context expression entry.matrixType
  | .boundedAtom expression bound =>
      matrixRefHasType context expression entry.matrixType &&
        (context.allowCarriedInputs || !boundHasCarriedInput bound)
  | .affineLeaf form => affineLeafHasType form entry.matrixType
  | .add left right =>
      symbolicRefHasType priorEntries left entry.matrixType &&
        symbolicRefHasType priorEntries right entry.matrixType
  | .negate value => symbolicRefHasType priorEntries value entry.matrixType
  | .multiply left right =>
      match priorEntries[left.id]?, priorEntries[right.id]? with
      | some leftEntry, some rightEntry =>
          match inferMatrixProductType leftEntry.matrixType rightEntry.matrixType with
          | .ok product => product.output == entry.matrixType
          | .error _ => false
      | _, _ => false
  | .select index branches =>
      (context.expressionArena.lookupInteger index).isSome &&
        symbolicRefsHaveType priorEntries branches entry.matrixType
  | .rowSlice value start stop =>
      match priorEntries[value.id]? with
      | some input =>
          entry.matrixType == { input.matrixType with rows := .subtract stop start }
      | none => false
  | .columnSlice value start stop =>
      match priorEntries[value.id]? with
      | some input =>
          entry.matrixType == { input.matrixType with columns := .subtract stop start }
      | none => false
  /- These constructors remain unavailable until their closed layout/relation tables are threaded
  through the WF context. Existence or an in-range relation ID alone is not semantic evidence. -/
  | .rowConcat _ | .columnConcat _ | .diagonalConcat _ | .reshape .. => false
  | .preimageRewrite .. | .gadgetRewrite .. => false
  | .carriedInput matrixType slot =>
      context.allowCarriedInputs && matrixType == entry.matrixType && slot < context.carriedArity
  | .recurrenceResult matrixType recurrence slot =>
      matrixType == entry.matrixType && recurrenceSlotIsValid context.recurrences recurrence slot

/-- Immutable append-only symbolic-form arena. -/
structure SymbolicMatrixFormArena where
  entries : List SymbolicMatrixFormEntry := []

def SymbolicMatrixFormArena.WF
    (context : SymbolicFormWFContext)
    (arena : SymbolicMatrixFormArena) : Bool :=
  arena.entries.zipIdx.all fun (entry, index) =>
    entry.refsBefore context (arena.entries.take index)

def SymbolicMatrixFormArena.intern
    (context : SymbolicFormWFContext)
    (arena : SymbolicMatrixFormArena)
    (entry : SymbolicMatrixFormEntry) : Option (SymbolicMatrixFormArena × SymbolicMatrixFormRef) :=
  if entry.refsBefore context arena.entries then
    some (⟨arena.entries ++ [entry]⟩, ⟨arena.entries.length⟩)
  else none

def SymbolicMatrixFormArena.lookup
    (arena : SymbolicMatrixFormArena)
    (reference : SymbolicMatrixFormRef) : Option SymbolicMatrixFormEntry :=
  arena.entries[reference.id]?

/-- Analyzer-owned matrix fact with immutable exact identity and an independently normalized
signal/noise decomposition. -/
structure MatrixSymbolicFact where
  subject : ValueInstanceRef
  matrixType : MatrixTypeExpr
  exactValue : MatrixExprRef
  decomposition : SymbolicMatrixFormRef
  bounds : MatrixBoundSummary
  boundWitness : BoundWitnessRef
  relations : List MatrixRelation
  coefficientRepresentation : CoefficientRepresentation

private def testMatrixType : MatrixTypeExpr where
  modulus := .constant 17
  ringDimension := .constant 4
  rows := .constant 2
  columns := .constant 2

private def testExpressionArena : ExpressionArena :=
  match ({} : ExpressionArena).internMatrix (.zero testMatrixType) with
  | some (arena, _) => arena
  | none => {}

private def testSymbolicContext : SymbolicFormWFContext where
  expressionArena := testExpressionArena
  preimageRelationCount := 0
  gadgetRelationCount := 0
  carriedArity := 1
  recurrences := []

example : SignalPresence.none.combine .present = .present := rfl

example :
    ((MatrixBoundSummary.select
      (.bounded (.constant 3))
      [.exactLarge (.constant 8)]).signal) = .present := rfl

example : SymbolicMatrixFormArena.WF testSymbolicContext {} = true := rfl

example :
    SymbolicMatrixFormArena.intern testSymbolicContext {} {
      matrixType := testMatrixType
      form := .carriedInput testMatrixType 0
    } = none := rfl

end Mxx.Certificate
