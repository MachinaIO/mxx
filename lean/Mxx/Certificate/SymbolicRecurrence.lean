import Mxx.Certificate.Rules.CoarseRecurrenceSchema

namespace Mxx.Certificate

/-- A symbolic body output whose type is fixed by the coarse carried-value schema. -/
inductive SymbolicCarriedOutput : CarriedValueSchema → Type where
  | matrix {matrixType representation}
      (form : SymbolicMatrixFormRef) :
      SymbolicCarriedOutput (.matrix matrixType representation)
  | integer (expression : RuntimeExprRef .integer) : SymbolicCarriedOutput .integer
  | boolean (expression : RuntimeExprRef .boolean) : SymbolicCarriedOutput .boolean
  | bytes (value : ValueInstanceRef) : SymbolicCarriedOutput .bytes
  | family {count element}
      (aggregate : FamilyAggregateRef) : SymbolicCarriedOutput (.family count element)

/-- Schema-indexed output vector. It is impossible to put a matrix reference into an integer
slot or silently change a family's coarse element schema. -/
inductive SymbolicCarriedOutputVector : List CarriedValueSchema → Type where
  | nil : SymbolicCarriedOutputVector []
  | cons {schema schemas}
      (head : SymbolicCarriedOutput schema)
      (tail : SymbolicCarriedOutputVector schemas) :
      SymbolicCarriedOutputVector (schema :: schemas)

inductive SymbolicRecurrenceError where
  | initialSchema (slot : Nat) (error : CoarseRecurrenceSchemaError)
  | bodySchema (slot : Nat) (error : CoarseRecurrenceSchemaError)
  | initialBodyMismatch (slot : Nat)
  | arityMismatch
  | suppliedSchemaMismatch
  | sourceMismatch
  deriving BEq, DecidableEq, Repr

private def validateRecurrenceSchemaLists :
    Nat → List ValueFactTemplate → List ValueFactTemplate →
      Except SymbolicRecurrenceError (List CarriedValueSchema)
  | _, [], [] => .ok []
  | slot, initial :: initials, body :: bodies => do
      let initialSchema ← initial.toCarriedValueSchema |>.mapError (.initialSchema slot)
      let bodySchema ← body.toCarriedValueSchema |>.mapError (.bodySchema slot)
      if initialSchema != bodySchema then throw (.initialBodyMismatch slot)
      return initialSchema :: (← validateRecurrenceSchemaLists (slot + 1) initials bodies)
  | _, _, _ => .error .arityMismatch

/-- Derive the stable, term-count-independent carried schema from the actual initial facts and the
analyzer-produced one-step body templates. Both sides pass through the relation-rejecting coarse
conversion. -/
def SequentialRecurrenceSource.validateCarriedSchemas
    (recurrence : SequentialRecurrenceSource) :
    Except SymbolicRecurrenceError (List CarriedValueSchema) :=
  validateRecurrenceSchemaLists 0 recurrence.initial.toList recurrence.bodyOutputs.toList

/-- Analyzer-owned closed recurrence transfer.  The recurrence boundary validates only stable
value type, matrix type, coefficient representation, and recursive family shape. Exact/affine
form and affine term count are body-local symbolic details and must not constrain later states. -/
structure SymbolicRecurrenceTransfer where
  identity : SequentialRecurrenceInstanceRef
  source : SequentialRecurrenceSource
  sourceIdentity : identity.recurrence.site = source.loop.site
  carriedSchemas : List CarriedValueSchema
  initialBounds : CarriedBoundTemplateVector
    (carriedSchemas.map CarriedValueSchema.boundSchema)
  /-- Whole symbolic forms for the concrete initial carried tuple.  Unlike the numeric
  `initialBounds`, this preserves whether a matrix begins exact or affine and retains all of
  its symbolic terms.  The recurrence boundary deliberately does not require these forms to
  have the same shape as `bodyOutputs`: a one-step body may turn an exact carried value into a
  multi-term affine value. -/
  initialOutputs : SymbolicCarriedOutputVector carriedSchemas
  bodyOutputs : SymbolicCarriedOutputVector carriedSchemas
  boundTransition : CarriedBoundTransitionVector
    (carriedSchemas.map CarriedValueSchema.boundSchema)
    (carriedSchemas.map CarriedValueSchema.boundSchema)
  stepRangeObligations : List (RecurrenceStepRangeObligation
    (carriedSchemas.map CarriedValueSchema.boundSchema)) := []
  schemaValidation : source.validateCarriedSchemas = .ok carriedSchemas

inductive ResolveSymbolicRecurrenceError where
  | count (error : IntEvalError)
  | negativeCount (value : Int)
  | initial (error : RecurrenceSchemaEvalError)
  | transition (error : RecurrenceSchemaEvalError)
  deriving BEq, DecidableEq, Repr

inductive ResolveSymbolicRecurrencesError where
  | duplicateIdentity (identity : SequentialRecurrenceInstanceRef)
  | recurrence
      (identity : SequentialRecurrenceInstanceRef)
      (error : ResolveSymbolicRecurrenceError)
  deriving BEq, DecidableEq, Repr

/-- Phase-B evaluation of one recurrence.  Symbolic matrices are never unrolled: only the fixed
numeric state vector is updated, and all output slots read the same immutable previous vector. -/
def SymbolicRecurrenceTransfer.resolveBounds
    (transfer : SymbolicRecurrenceTransfer)
    (environment : Mxx.Ir.ParamEnvironment)
    (priorStates : CheckedSymbolicRecurrenceStateTable) :
    Except ResolveSymbolicRecurrenceError CheckedSymbolicRecurrenceState := do
  let count ← evaluateIntExpr environment transfer.source.count |>.mapError .count
  if count < 0 then throw (.negativeCount count)
  let initial ← transfer.initialBounds.evaluate environment priorStates |>.mapError .initial
  let closedTransition ← transfer.boundTransition.closeExternalRecurrences environment priorStates
    |>.mapError .transition
  let closedRangeObligations ← transfer.stepRangeObligations.mapM
    (RecurrenceStepRangeObligation.closeExternalRecurrences environment priorStates)
    |>.mapError .transition
  let final ← iterateCarriedBoundTransitionWithRangeChecks environment closedTransition
    closedRangeObligations count.toNat initial
    |>.mapError .transition
  return {
    identity := transfer.identity
    schemas := transfer.carriedSchemas.map CarriedValueSchema.boundSchema
    values := final
  }

private def containsSymbolicRecurrenceIdentity
    (identity : SequentialRecurrenceInstanceRef) :
    List CheckedSymbolicRecurrenceState → Bool
  | [] => false
  | entry :: tail => entry.identity == identity ||
      containsSymbolicRecurrenceIdentity identity tail

private def resolveSymbolicRecurrencesFrom
    (environment : Mxx.Ir.ParamEnvironment) :
    List SymbolicRecurrenceTransfer → CheckedSymbolicRecurrenceStateTable →
      Except ResolveSymbolicRecurrencesError CheckedSymbolicRecurrenceStateTable
  | [], states => .ok states
  | transfer :: tail, states => do
      if containsSymbolicRecurrenceIdentity transfer.identity states.entries then
        throw (.duplicateIdentity transfer.identity)
      let resolved ← transfer.resolveBounds environment states
        |>.mapError (.recurrence transfer.identity)
      resolveSymbolicRecurrencesFrom environment tail {
        entries := resolved :: states.entries
      }

/-- Resolve analyzer-produced symbolic recurrences in dependency order. A transfer may read only
the checked prefix accumulated before it, so self, forward, and cyclic references fail closed
without a caller-supplied dependency graph or a general topological solver. -/
def resolveSymbolicRecurrences
    (environment : Mxx.Ir.ParamEnvironment)
    (transfers : List SymbolicRecurrenceTransfer) :
    Except ResolveSymbolicRecurrencesError CheckedSymbolicRecurrenceStateTable :=
  resolveSymbolicRecurrencesFrom environment transfers {}

/-- The only public constructor checks stable coarse equality and relation-freedom before
packaging already type-indexed symbolic outputs and bound transitions. -/
def SymbolicRecurrenceTransfer.build
    {schemas : List CarriedValueSchema}
    (identity : SequentialRecurrenceInstanceRef)
    (source : SequentialRecurrenceSource)
    (initialBounds : CarriedBoundTemplateVector
      (schemas.map CarriedValueSchema.boundSchema))
    (initialOutputs : SymbolicCarriedOutputVector schemas)
    (bodyOutputs : SymbolicCarriedOutputVector schemas)
    (boundTransition : CarriedBoundTransitionVector
      (schemas.map CarriedValueSchema.boundSchema)
      (schemas.map CarriedValueSchema.boundSchema))
    (stepRangeObligations : List (RecurrenceStepRangeObligation
      (schemas.map CarriedValueSchema.boundSchema))) :
    Except SymbolicRecurrenceError SymbolicRecurrenceTransfer :=
  match validation : source.validateCarriedSchemas with
  | .error error => .error error
  | .ok actual =>
      if sourceSame : identity.recurrence.site = source.loop.site then
        if same : actual = schemas then
          .ok {
            identity
            source
            sourceIdentity := sourceSame
            carriedSchemas := schemas
            initialBounds
            initialOutputs
            bodyOutputs
            boundTransition
            stepRangeObligations
            schemaValidation := by simpa [same] using validation
          }
        else .error .suppliedSchemaMismatch
      else .error .sourceMismatch

private def resolverFixtureSite : CoreNodeRef := {
  stage := ⟨"resolver-fixture"⟩
  scope := ⟨[]⟩
  node := ⟨0⟩
}

private def resolverFixtureSource : SequentialRecurrenceSource where
  loop := ⟨resolverFixtureSite⟩
  count := .constant 2
  carriedArity := 1
  initial := ⟨#[{
    fact := .integer {
      expression := .intConstant 3
      lower := .integer (.constant 3)
      upper := .integer (.constant 3)
    }
    schema := .integer
  }], rfl⟩
  bodyInputs := ⟨#[{
    definition := { stage := ⟨"resolver-fixture"⟩, name := "body" }
    bodyScope := ⟨[]⟩
    node := ⟨0⟩
    port := 0
  }], rfl⟩
  bodyOutputs := ⟨#[{
    fact := .integer {
      expression := .carriedInput (.integerValue 0)
      lower := .carriedInput (.lower 0)
      upper := .carriedInput (.upper 0)
    }
    schema := .integer
  }], rfl⟩
  invariantInputs := []
  iterationVariable := ⟨0⟩

private def resolverFixturePath :
    CarriedBoundStatePath [.integerInterval] .integerInterval :=
  .head .here

/-- Coarse schemas retain the scalar kind, so a body may not silently change a carried integer
into a Boolean. -/
private def exactSchemaMismatchFixtureSource : SequentialRecurrenceSource := {
  resolverFixtureSource with
  bodyOutputs := ⟨#[{
    fact := .boolean { expression := .boolConstant false }
    schema := .boolean
  }], rfl⟩
}

example : exactSchemaMismatchFixtureSource.validateCarriedSchemas =
    .error (.initialBodyMismatch 0) := by
  rfl

private def resolverFixtureTransfer : SymbolicRecurrenceTransfer where
  identity := { recurrence := ⟨resolverFixtureSite⟩, path := [] }
  source := resolverFixtureSource
  sourceIdentity := rfl
  carriedSchemas := [.integer]
  initialBounds := .cons
    (.integer (.integer (.constant 3)) (.integer (.constant 3))) .nil
  initialOutputs := .cons (.integer ⟨0⟩) .nil
  bodyOutputs := .cons (.integer ⟨0⟩) .nil
  boundTransition := .cons (.integer
    (.add (.previousState resolverFixturePath .lower) (.constant 1))
    (.add (.previousState resolverFixturePath .upper) (.constant 1))) .nil
  schemaValidation := rfl

example : resolverFixtureTransfer.resolveBounds [] {} = .ok {
    identity := resolverFixtureTransfer.identity
    schemas := [.integerInterval]
    values := .cons (.integer 5 5) .nil
  } := by
  rfl

private def zeroCountResolverFixtureSource : SequentialRecurrenceSource := {
  resolverFixtureSource with count := .constant 0
}

private def zeroCountResolverFixtureTransfer : SymbolicRecurrenceTransfer where
  identity := resolverFixtureTransfer.identity
  source := zeroCountResolverFixtureSource
  sourceIdentity := rfl
  carriedSchemas := [.integer]
  initialBounds := resolverFixtureTransfer.initialBounds
  initialOutputs := resolverFixtureTransfer.initialOutputs
  bodyOutputs := resolverFixtureTransfer.bodyOutputs
  boundTransition := resolverFixtureTransfer.boundTransition
  schemaValidation := rfl

example : zeroCountResolverFixtureTransfer.resolveBounds [] {} = .ok {
    identity := zeroCountResolverFixtureTransfer.identity
    schemas := [.integerInterval]
    values := .cons (.integer 3 3) .nil
  } := by
  rfl

private def simultaneousFixtureSite : CoreNodeRef := {
  stage := ⟨"simultaneous-fixture"⟩
  scope := ⟨[]⟩
  node := ⟨0⟩
}

private def simultaneousFixtureSource : SequentialRecurrenceSource where
  loop := ⟨simultaneousFixtureSite⟩
  count := .constant 1
  carriedArity := 2
  initial := ⟨#[
    {
      fact := .integer {
        expression := .intConstant 1
        lower := .integer (.constant 1)
        upper := .integer (.constant 2)
      }
      schema := .integer
    },
    {
      fact := .integer {
        expression := .intConstant 10
        lower := .integer (.constant 10)
        upper := .integer (.constant 20)
      }
      schema := .integer
    }
  ], rfl⟩
  bodyInputs := ⟨#[
    {
      definition := { stage := ⟨"simultaneous-fixture"⟩, name := "body" }
      bodyScope := ⟨[]⟩
      node := ⟨0⟩
      port := 0
    },
    {
      definition := { stage := ⟨"simultaneous-fixture"⟩, name := "body" }
      bodyScope := ⟨[]⟩
      node := ⟨0⟩
      port := 1
    }
  ], rfl⟩
  bodyOutputs := ⟨#[
    {
      fact := .integer {
        expression := .carriedInput (.integerValue 1)
        lower := .carriedInput (.lower 1)
        upper := .carriedInput (.upper 1)
      }
      schema := .integer
    },
    {
      fact := .integer {
        expression := .carriedInput (.integerValue 0)
        lower := .carriedInput (.lower 0)
        upper := .carriedInput (.upper 0)
      }
      schema := .integer
    }
  ], rfl⟩
  invariantInputs := []
  iterationVariable := ⟨0⟩

private abbrev simultaneousBoundSchemas : List CarriedBoundSchema :=
  [.integerInterval, .integerInterval]

private def simultaneousFirstPath :
    CarriedBoundStatePath simultaneousBoundSchemas .integerInterval :=
  .head .here

private def simultaneousSecondPath :
    CarriedBoundStatePath simultaneousBoundSchemas .integerInterval :=
  .tail (.head .here)

private def simultaneousFixtureTransfer : SymbolicRecurrenceTransfer where
  identity := { recurrence := ⟨simultaneousFixtureSite⟩, path := [] }
  source := simultaneousFixtureSource
  sourceIdentity := rfl
  carriedSchemas := [.integer, .integer]
  initialBounds := .cons
    (.integer (.integer (.constant 1)) (.integer (.constant 2)))
    (.cons (.integer (.integer (.constant 10)) (.integer (.constant 20))) .nil)
  initialOutputs := .cons (.integer ⟨0⟩) (.cons (.integer ⟨1⟩) .nil)
  bodyOutputs := .cons (.integer ⟨1⟩) (.cons (.integer ⟨0⟩) .nil)
  boundTransition := .cons
    (.integer
      (.previousState simultaneousSecondPath .lower)
      (.previousState simultaneousSecondPath .upper))
    (.cons
      (.integer
        (.previousState simultaneousFirstPath .lower)
        (.previousState simultaneousFirstPath .upper))
      .nil)
  schemaValidation := rfl

example : simultaneousFixtureTransfer.resolveBounds [] {} = .ok {
    identity := simultaneousFixtureTransfer.identity
    schemas := simultaneousBoundSchemas
    values := .cons (.integer 10 20) (.cons (.integer 1 2) .nil)
  } := by
  rfl

end Mxx.Certificate
