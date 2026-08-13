import Mxx.Certificate.OperationalBounds.DescriptorTransport
import Mxx.Certificate.OperationalBounds.Progress

namespace Mxx.Certificate

open Mxx.Ir

/-- The request arena stores only direct carrier values.  Graph nodes live in `OperationalScope`
and are not an auxiliary expression representation. -/
structure OperationalExprArena where
  direct : DirectOperationalIndexedArena := {}
  activeScope : Option ScopeTemplateKey := none
  activeNode : Option Nat := none
  deriving BEq

/-- Facts for one frozen scope and the request-owned direct carrier shared by its request. -/
structure OperationalScopeFacts where
  values : Array (Array OperationalFact) := #[]
  arena : OperationalExprArena := {}

/-- Promote a concrete fixed-assignment matrix result into the request-owned direct carrier. -/
def OperationalExprArena.promoteConcreteMatrixFact
    (arena : OperationalExprArena)
    (fact : OperationalMatrixFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let (fixed, reference) := arena.direct.fixed.pushMatrix fact
  let direct := { arena.direct with fixed }
  let (direct, value) ← match direct.pushShared emptyContext (.matrix fact.matrixType) reference with
    | some result => pure result
    | none => throw (.unsupportedOperationalExpr direct.values.size)
  pure ({ arena with direct }, {
    context := emptyContext, payload := .directValue value, storage := .sharedTemplate })

/-- Promote a selection-free scalar atom into the direct carrier. -/
def OperationalExprArena.promoteConcreteScalarFact
    (arena : OperationalExprArena)
    (fact : OperationalScalarFact) : Except OperationalError (OperationalExprArena × OperationalFact) := do
  let schema := operationalScalarSchema fact
  let (fixed, reference) := arena.direct.fixed.pushScalar fact
  let direct := { arena.direct with fixed }
  let (direct, value) ← match direct.pushShared emptyContext (.scalar schema) reference with
    | some result => pure result
    | none => throw (.unsupportedOperationalExpr direct.values.size)
  pure ({ arena with direct }, {
    context := emptyContext, payload := .directValue value, storage := .sharedTemplate })

/-- Relation producers consume the same direct carrier values used by all graph edges. -/
def OperationalExprArena.promoteDirectRelationOperand
    (arena : OperationalExprArena)
    (fact : OperationalFact) : Except OperationalError (OperationalExprArena × OperationalFact) :=
  pure (arena, fact)

def DynamicSelectionIdentity.fromDeclaredCount
    (origin : OperationalValueOrigin) (count : IntExpr) : DynamicSelectionIdentity := {
  index := origin
  expression := .variable {
    owner := {
      stage := ⟨s!"operational-selection:{reprStr origin}"⟩
      scope := ⟨[]⟩
      node := ⟨0⟩
    }
    slot := 0
    count
  }
}

/-- Construct an executable loop-lane selector without projecting its lexical index slot to the
synthetic selection slot.  Nested parallel loops may legitimately use slot one or above. -/
def DynamicSelectionIdentity.fromDeclaredCountAtSlot
    (origin : OperationalValueOrigin) (slot : Nat) (count : IntExpr) : DynamicSelectionIdentity := {
  index := origin
  expression := .variable {
    owner := {
      stage := ⟨s!"operational-selection:{reprStr origin}"⟩
      scope := ⟨[]⟩
      node := ⟨0⟩
    }
    slot
    count
  }
}

def DynamicSelectionIdentity.fromOrigin
    (origin : OperationalValueOrigin) (count : Nat) : DynamicSelectionIdentity :=
  .fromDeclaredCount origin (.constant count)

/-- The one exact free coordinate introduced by an executable parallel loop.  Inputs, body
outputs, and nested consumers all use this descriptor, so distinct arguments and output ports of
one loop remain aligned while an equal slot in a different scope or loop does not. -/
def parallelLoopLaneSelection
    (scope : ScopeTemplateKey)
    (node indexSlot : Nat)
    (count : IntExpr) : DynamicSelectionIdentity :=
  .fromDeclaredCountAtSlot
    (.loopInstance indexSlot (.constant 0) (.local scope { node, port := indexSlot })) indexSlot count

/-- The lexical coordinate for one sequential-body evaluation.  It is separate from a parallel
loop lane by its owning scope/node identity, while retaining the executable Graph-IR index slot. -/
def sequentialLoopLaneSelection
    (scope : ScopeTemplateKey)
    (node indexSlot : Nat)
    (count : IntExpr) : DynamicSelectionIdentity :=
  .fromDeclaredCountAtSlot
    (.loopInstance indexSlot (.constant 0) (.local scope { node, port := indexSlot })) indexSlot count

def parallelLoopFamilyBinder
    (scope : ScopeTemplateKey)
    (node indexSlot : Nat) : FamilyTemplateBinder := {
  owner := scope
  producerNode := node
  binderSlot := indexSlot
}

/-- The binder carried by `parallelLoopLaneSelection`.  Keeping this alongside the selection
constructor prevents the input and output paths from independently reconstructing its owner. -/
def parallelLoopLaneBinder
    (scope : ScopeTemplateKey)
    (node indexSlot : Nat)
    (count : IntExpr) : Except OperationalError IndexVariable := do
  match (parallelLoopLaneSelection scope node indexSlot count).expression with
  | .variable binder => pure binder
  | _ => throw (.loopInputModeMismatch node indexSlot)

inductive OperationalTransferClass where
  | input
  | scalar
  | matrix
  | structural
  deriving BEq, DecidableEq

def classifyIntBinary : IntBinaryOp → Unit
  | .add | .subtract | .multiply | .divide | .remainder => ()

def classifyIntCompare : IntCompareOp → Unit
  | .equal | .less | .lessEqual => ()

def classifyRealBinary : RealBinaryOp → Unit
  | .add | .subtract | .multiply | .divide => ()

def classifyConcatAxis : ConcatAxis → Unit
  | .rows | .columns | .diagonal => ()

def classifyHashVariant : Mxx.HashVariant → Unit
  | .plain | .decomposed | .smallDecomposed => ()

def classifyLoopInputMode : LoopInputMode → Unit
  | .broadcast | .zip | .zipOffset _ => ()

/-- Exhaustive compile-time inventory of the operational transfer surface. Nested operation enums
are themselves classified exhaustively, so extending (for example) `IntBinaryOp` or `ConcatAxis`
also requires an explicit operational-checker decision. -/
def operationalTransferClass : NodeKind → OperationalTransferClass
  | .input _ => .input
  | .constantInt _
  | .evaluateInt _
  | .constantReal _
  | .constantBool _
  | .boolToInt
  | .intToReal
  | .realSqrt
  | .bitExtract _
  | .extractCoefficient _ => .scalar
  | .intBinary operation =>
      let _ := classifyIntBinary operation
      .scalar
  | .intCompare operation =>
      let _ := classifyIntCompare operation
      .scalar
  | .realBinary operation =>
      let _ := classifyRealBinary operation
      .scalar
  | .zeroMatrix _
  | .identityMatrix _
  | .constantMatrix _ _
  | .unitRowMatrix _ _
  | .unitColumnMatrix _ _
  | .gadgetMatrix _ _
  | .smallGadgetMatrix _ _
  | .powerOfBaseMatrix _ _ _
  | .rotationMatrix _ _
  | .gadgetTrapdoor _ _
  | .uniformResidueSample _
  | .uniformIntervalSample _ _ _
  | .gaussianSample _ _
  | .gadgetDecompose _ _ _ _
  | .trapdoorSample _ _
  | .trapdoorPublic
  | .preimageSample _ _
  | .matrixAdd
  | .matrixSubtract
  | .matrixMultiply
  | .matrixNegate
  | .matrixScale _
  | .transpose
  | .slice _ _
  | .tensor
  | .liftIntegerToConstantPolynomial _
  | .thresholdDecodeBool _ _ _
  | .thresholdDecodeInt _ _ _
  | .crtRecompose _ _
  | .packPolynomialCoefficients _ _ => .matrix
  | .hashSample _ variant _ _ _ _ _ _ =>
      let _ := classifyHashVariant variant
      .matrix
  | .concat axis =>
      let _ := classifyConcatAxis axis
      .matrix
  | .select
  | .familyPack
  | .familyGetStatic _
  | .familyGetDynamic
  | .subgraphCall _ _
  | .sequentialLoop _ _ _ _ _ => .structural
  | .parallelLoop _ _ _ _ inputModes =>
      let _ := inputModes.map classifyLoopInputMode
      .structural


def absolute (value : Int) : Int := if value < 0 then -value else value

def capCentered (modulus value : Int) : Int :=
  if modulus ≤ 0 then 0 else min (modulus / 2) (absolute value)

def matrixCap (matrixType : MatrixTypeExpr) (environment : ParamEnvironment) : Option Int := do
  let modulus ← matrixType.modulus.evaluate environment
  if modulus ≤ 0 then none else some (modulus / 2)

def matrixRingDimension (matrixType : MatrixTypeExpr) (environment : ParamEnvironment) : Option Int := do
  let value ← matrixType.ringDimension.evaluate environment
  if value < 0 then none else some value

def resolveGadgetLayout
    (node : Nat)
    (layouts : List Mxx.GadgetLayoutDescriptor)
    (params : Mxx.SamplerParams) : Except OperationalError Mxx.GadgetLayoutDescriptor := do
  let candidates := layouts.filter fun descriptor => descriptor.matches params
  match candidates with
  | [descriptor] =>
      if descriptor.valid then pure descriptor else throw (.invalidGadgetLayout node)
  | [] => throw (.missingGadgetLayout node)
  | _ => throw (.ambiguousGadgetLayout node)

structure OperationalNumericSlot where
  matrixMaximum : Option Int := none
  integerLower : Option Int := none
  integerUpper : Option Int := none
  deriving Inhabited

abbrev OperationalNumericState := Array OperationalNumericSlot

def scalarFactClosedMaximum : OperationalScalarFact → Option Int
  | .trapdoor fact => match fact.maximum with
      | .closedOperational (.closedInt (.constant maximum)) => some maximum
      | _ => none
  | _ => none

def scalarFactNumericSlot : OperationalScalarFact → OperationalNumericSlot
  | .trapdoor fact => { matrixMaximum := scalarFactClosedMaximum (.trapdoor fact) }
  | .integer fact => { integerLower := some fact.lower, integerUpper := some fact.upper }
  | _ => {}

def scalarFactNumericExpressions
    (slot : Nat) : OperationalScalarFact → List (OperationalBoundPath × OperationalBoundExpr)
  | .trapdoor fact => match fact.maximum.closedOperational? with
      | some maximum => [(.matrixMaximum 0 slot, maximum)]
      | none => []
  | .integer fact => [
      (.integerLower 0 slot, fact.lowerExpression.closedOperational?.getD (.closedInt (.constant fact.lower))),
      (.integerUpper 0 slot, fact.upperExpression.closedOperational?.getD (.closedInt (.constant fact.upper)))
    ]
  | _ => []


def lookupPrevious
    (states : List OperationalNumericState) : OperationalBoundPath → Option Int
  | .matrixMaximum depth slot => states[depth]? >>= fun state => state[slot]? >>= (·.matrixMaximum)
  | .integerLower depth slot => states[depth]? >>= fun state => state[slot]? >>= (·.integerLower)
  | .integerUpper depth slot => states[depth]? >>= fun state => state[slot]? >>= (·.integerUpper)

def operationalBoundPathSlot : OperationalBoundPath → Nat
  | .matrixMaximum _ slot | .integerLower _ slot | .integerUpper _ slot => slot

def operationalBoundPathAtCurrentDepth : OperationalBoundPath → Bool
  | .matrixMaximum depth _ | .integerLower depth _ | .integerUpper depth _ => depth = 0

def numericStateFromComponents
    (paths : List OperationalBoundPath)
    (values : List Int) : Except OperationalError OperationalNumericState := do
  if paths.length != values.length then
    throw (.unsupportedOutputArity values.length paths.length)
  if paths.any fun path => !operationalBoundPathAtCurrentDepth path then
    throw (.invalidPreviousPath (paths.find? (fun path =>
      !operationalBoundPathAtCurrentDepth path) |>.getD (.matrixMaximum 0 0)))
  let size := paths.foldl (fun current path => max current (operationalBoundPathSlot path + 1)) 0
  let mut result : OperationalNumericState := Array.replicate size {}
  for (path, value) in paths.zip values do
    let slot := operationalBoundPathSlot path
    let previous := result[slot]!
    result := result.set! slot <| match path with
      | .matrixMaximum _ _ => { previous with matrixMaximum := some value }
      | .integerLower _ _ => { previous with integerLower := some value }
      | .integerUpper _ _ => { previous with integerUpper := some value }
  pure result

def replaceOperationalFactorHardBound
    (bound : OperationalBoundExpr)
    (factor : OperationalFactorKey) : OperationalFactorKey :=
  let update (summary : OperationalBoundedFactorSummary) := { summary with hardBound := bound }
  { factor with
    leaf := match factor.leaf with
      | .boundedSummary origin summary => .boundedSummary origin (update summary)
      | leaf => leaf
    boundedSummary := factor.boundedSummary.map update
  }

def abstractCarriedScalarMaximum
    (slot : Nat) : OperationalScalarFact → OperationalScalarFact
  | .trapdoor fact => .trapdoor { fact with maximum := .closed (.previous (.matrixMaximum 0 slot)) }
  | .integer fact => .integer {
      fact with
      lowerExpression := .closed (.previous (.integerLower 0 slot))
      upperExpression := .closed (.previous (.integerUpper 0 slot))
    }
  | fact => fact

structure CarriedFactorSchema where
  transforms : List OperationalFactorTransform
  inputType : MatrixTypeExpr
  outputType : MatrixTypeExpr
  role : OperationalFactorRole
  boundedType : Option MatrixTypeExpr
  boundedMetadata : Option OperationalMatrixMetadata
  protections : List OperationalCompressionProtection
  deriving BEq

structure CarriedSignalTermSchema where
  coefficient : Int
  factors : List CarriedFactorSchema
  modes : List OperationalProductMode
  outputType : MatrixTypeExpr
  deriving BEq

/-- The loop invariant records only normalized signal structure.  Bounded-only terms contribute
numeric recurrence slots, so their canonical encodings `[]` and `[0]` are deliberately identical.
For a signal term, however, the ordered factor roles, shapes, transforms, and matrix metadata stay
visible; changing any of them is a schema change even when the number of Large factors agrees. -/
def carriedSignalSignature
    (polynomial : OperationalPolynomial) : List CarriedSignalTermSchema :=
  polynomial.filterMap fun term =>
    if term.product.factors.any fun factor => factor.role == .large then
      some {
        coefficient := term.coefficient
        factors := term.product.factors.map fun factor => {
          transforms := factor.transforms
          inputType := factor.inputType
          outputType := factor.outputType
          role := factor.role
          boundedType := factor.boundedSummary.map (·.matrixType)
          boundedMetadata := factor.boundedSummary.map (·.metadata)
          protections := factor.protections
        }
        modes := term.product.modes
        outputType := term.product.outputType
      }
    else none

def scalarSchemaTag : OperationalScalarFact → Nat
  | .integer _ => 0
  | .boolean => 1
  | .real => 2
  | .trapdoor _ => 3
  | .bytes _ => 4
  | .typedBlob _ _ => 5
  | .unknown _ => 6

def sameCarriedMatrixFactSchema
    (left right : OperationalMatrixFact) : Bool :=
  left.matrixType == right.matrixType &&
  left.matrixParams.modulus == right.matrixParams.modulus &&
  left.matrixParams.ringDimension == right.matrixParams.ringDimension &&
  left.matrixParams.rows == right.matrixParams.rows &&
  left.matrixParams.columns == right.matrixParams.columns &&
  left.metadata == right.metadata && left.canonicalRange == right.canonicalRange &&
  left.identity.isNone && right.identity.isNone && left.relations.isEmpty &&
  right.relations.isEmpty &&
  carriedSignalSignature left.polynomial == carriedSignalSignature right.polynomial

def intExprIsClosed : IntExpr → Bool
  | .constant _ => true
  | .parameter _ => true
  | .loopIndex _ => false
  | .add left right | .subtract left right | .multiply left right | .divide left right |
      .roundDivide left right => intExprIsClosed left && intExprIsClosed right
  | .log2Ceil value => intExprIsClosed value

def intExprUsesLoop (slot : Nat) : IntExpr → Bool
  | .constant _ | .parameter _ => false
  | .loopIndex candidate => candidate == slot
  | .add left right | .subtract left right | .multiply left right | .divide left right |
      .roundDivide left right => intExprUsesLoop slot left || intExprUsesLoop slot right
  | .log2Ceil value => intExprUsesLoop slot value

def intExprUsesParameter (name : String) : IntExpr → Bool
  | .constant _ | .loopIndex _ => false
  | .parameter candidate => candidate == name
  | .add left right | .subtract left right | .multiply left right | .divide left right |
      .roundDivide left right => intExprUsesParameter name left || intExprUsesParameter name right
  | .log2Ceil value => intExprUsesParameter name value

def replaceLoopIndex
    (environment : ParamEnvironment) (slot : Nat) (value : Nat) : ParamEnvironment :=
  (ParamKey.loopIndex slot, ParamValue.integer (Int.ofNat value)) ::
    environment.filter fun entry => entry.1 != .loopIndex slot

def replaceParameter
    (environment : ParamEnvironment) (name : String) (value : Int) : ParamEnvironment :=
  (ParamKey.parameter name, ParamValue.integer value) ::
    environment.filter fun entry => entry.1 != .parameter name

/-- Evaluates only the numeric expression over the loop coordinates it actually references.
This never reevaluates an IR scope and allocates no lane facts. -/
def evaluateIntOverLoops
    (environment : ParamEnvironment)
    (domains : List OperationalParameterDomain)
    (expression : IntExpr) : Except OperationalError (List Int) := do
  let rec visit (environment : ParamEnvironment) : List OperationalParameterDomain →
      Except OperationalError (List Int)
    | [] => match expression.evaluate environment with
        | some value => pure [value]
        | none => throw .nonClosedExpression
    | .loopIndex slot count :: tail =>
        if !intExprUsesLoop slot expression || count = 0 then visit environment tail else
          return (← (List.range count).mapM fun index =>
            visit (replaceLoopIndex environment slot index) tail).flatten
    | .parameter name sourceEnvironment sourceDomains sourceExpression :: tail =>
        if !intExprUsesParameter name expression then visit environment tail else do
          let values ← evaluateIntOverLoops sourceEnvironment sourceDomains sourceExpression
          return (← values.mapM fun value =>
            visit (replaceParameter environment name value) tail).flatten
  visit environment domains

def evaluateIntMinimum
    (environment : ParamEnvironment) (domains : List OperationalParameterDomain)
    (expression : IntExpr) : Except OperationalError Int := do
  match ← evaluateIntOverLoops environment domains expression with
  | [] => throw .nonClosedExpression
  | first :: tail => pure (tail.foldl min first)

def evaluateIntMaximum
    (environment : ParamEnvironment) (domains : List OperationalParameterDomain)
    (expression : IntExpr) : Except OperationalError Int := do
  match ← evaluateIntOverLoops environment domains expression with
  | [] => throw .nonClosedExpression
  | first :: tail => pure (tail.foldl max first)

def evaluateIntMaximumAbsolute
    (environment : ParamEnvironment) (domains : List OperationalParameterDomain)
    (expression : IntExpr) : Except OperationalError Int := do
  let values ← evaluateIntOverLoops environment domains expression
  pure (values.foldl (fun maximum value => max maximum (absolute value)) 0)

/-- Check every contextual assignment, rather than only the cutoff maximum. -/
def validateContextualCutoffNonnegative
    (node : Nat)
    (environment : ParamEnvironment)
    (domains : List OperationalParameterDomain)
    (cutoff : IntExpr) : Except OperationalError OperationalBoundExpr := do
  let minimum ← evaluateIntMinimum environment domains cutoff
  if minimum < 0 then throw (.invalidBound node minimum)
  pure (.contextual .maximum environment domains cutoff)

/-- A gadget has no preimage-sampler cutoff contract; indexed and loop identities preserve this
distinction from sampled trapdoors. -/
def publicIdentityIsGadget : PublicMatrixIdentity → Bool
  | .gadget .. => true
  | .sampledTrapdoor .. => false
  | .indexed _ _ source | .loopInstance _ _ source => publicIdentityIsGadget source

def sameContextualDomainKey : OperationalParameterDomain → OperationalParameterDomain → Bool
  | .loopIndex left _, .loopIndex right _ => left == right
  | .parameter left _ _ _, .parameter right _ _ _ => left == right
  | _, _ => false

/-- Merge cutoff assignment domains without allowing an equal key to overwrite a different
definition.  The resulting enumeration visits each logical assignment exactly once. -/
def mergeContextualCutoffDomains
    (node : Nat)
    (left right : List OperationalParameterDomain) : Except OperationalError
    (List OperationalParameterDomain) := do
  let rec insert : List OperationalParameterDomain → OperationalParameterDomain →
      Except OperationalError (List OperationalParameterDomain)
    | [], candidate => pure [candidate]
    | head :: tail, candidate =>
        if sameContextualDomainKey head candidate then
          if head == candidate then pure (head :: tail) else throw (.preimageCutoffMismatch node)
        else return head :: (← insert tail candidate)
  right.foldlM insert left

/-- Preimage and trapdoor cutoffs must agree per merged assignment, not merely at extrema. -/
def validatePreimageCutoffAgreement
    (node : Nat)
    (environment : ParamEnvironment)
    (domains : List OperationalParameterDomain)
    (preimageCutoff : IntExpr)
    (publicIdentity : PublicMatrixIdentity)
    (trapdoorCutoff : Option OperationalBoundExpr) : Except OperationalError Unit := do
  match trapdoorCutoff with
  | none =>
      if publicIdentityIsGadget publicIdentity then pure () else throw (.missingPreimageCutoff node)
  | some (.contextual _ trapdoorEnvironment trapdoorDomains trapdoorExpression) =>
      let mergedDomains ← mergeContextualCutoffDomains node trapdoorDomains domains
      let trapdoorValues ← evaluateIntOverLoops trapdoorEnvironment mergedDomains trapdoorExpression
      let preimageValues ← evaluateIntOverLoops environment mergedDomains preimageCutoff
      if trapdoorValues != preimageValues then throw (.preimageCutoffMismatch node)
  /- A fully selected direct trapdoor has already materialized its gathered cutoff to a closed
  integer expression.  Preserve the same exact comparison for a relation with no remaining
  loop domain; rejecting this form would make valid gather-backed trapdoors unusable after the
  materialization boundary. -/
  | some (.closedInt trapdoorExpression) =>
      if !domains.isEmpty || trapdoorExpression != preimageCutoff then
        throw (.preimageCutoffMismatch node)
  | some _ => throw (.preimageCutoffMismatch node)

def evaluateIntInvariant
    (environment : ParamEnvironment) (domains : List OperationalParameterDomain)
    (expression : IntExpr) : Except OperationalError Int := do
  match ← evaluateIntOverLoops environment domains expression with
  | [] => throw .nonClosedExpression
  | first :: tail =>
      if tail.all (· == first) then pure first else throw .nonClosedExpression

def extendParameterDomains
    (environment : ParamEnvironment)
    (domains : List OperationalParameterDomain)
    (bindings : List (String × IntExpr)) : Except OperationalError (List OperationalParameterDomain) := do
  let mut result := domains
  for (name, expression) in bindings do
    result := .parameter name environment domains expression :: result.filter fun domain => match domain with
      | .parameter candidate _ _ _ => candidate != name
      | .loopIndex _ _ => true
  pure result

def instantiateParameterDomains (slot index : Nat) :
    List OperationalParameterDomain → List OperationalParameterDomain
  | [] => []
  | .loopIndex candidate count :: tail =>
      if candidate = slot then instantiateParameterDomains slot index tail
      else .loopIndex candidate count :: instantiateParameterDomains slot index tail
  | .parameter name environment domains expression :: tail =>
      .parameter name (replaceLoopIndex environment slot index)
          (instantiateParameterDomains slot index domains) expression ::
        instantiateParameterDomains slot index tail

def materializeInvariantParameters
    (environment : ParamEnvironment)
    (domains : List OperationalParameterDomain) : Except OperationalError ParamEnvironment := do
  let mut result := environment
  for domain in domains do
    match domain with
    | .loopIndex _ _ => pure ()
    | .parameter name sourceEnvironment sourceDomains sourceExpression =>
        let value ← evaluateIntInvariant sourceEnvironment sourceDomains sourceExpression
        result := replaceParameter result name value
  pure result

@[simp] theorem materializeInvariantParameters_nil (environment : ParamEnvironment) :
    materializeInvariantParameters environment [] = .ok environment := by
  rfl

def shiftPreviousDepthFrom (cutoff : Nat) : OperationalBoundExpr → OperationalBoundExpr
  | .closedInt value => .closedInt value
  | .contextual kind environment domains value => .contextual kind environment domains value
  | .previous (.matrixMaximum depth slot) =>
      .previous (.matrixMaximum (if cutoff ≤ depth then depth + 1 else depth) slot)
  | .previous (.integerLower depth slot) =>
      .previous (.integerLower (if cutoff ≤ depth then depth + 1 else depth) slot)
  | .previous (.integerUpper depth slot) =>
      .previous (.integerUpper (if cutoff ≤ depth then depth + 1 else depth) slot)
  | .negate value => .negate (shiftPreviousDepthFrom cutoff value)
  | .add left right => .add (shiftPreviousDepthFrom cutoff left)
      (shiftPreviousDepthFrom cutoff right)
  | .subtract left right => .subtract (shiftPreviousDepthFrom cutoff left)
      (shiftPreviousDepthFrom cutoff right)
  | .multiply left right => .multiply (shiftPreviousDepthFrom cutoff left)
      (shiftPreviousDepthFrom cutoff right)
  | .divide left right => .divide (shiftPreviousDepthFrom cutoff left)
      (shiftPreviousDepthFrom cutoff right)
  | .minimum left right => .minimum (shiftPreviousDepthFrom cutoff left)
      (shiftPreviousDepthFrom cutoff right)
  | .maximum left right => .maximum (shiftPreviousDepthFrom cutoff left)
      (shiftPreviousDepthFrom cutoff right)
  | .centeredCap modulus value => .centeredCap (shiftPreviousDepthFrom cutoff modulus)
      (shiftPreviousDepthFrom cutoff value)
  | .matrixProduct ringDimension innerDimension left right =>
      .matrixProduct (shiftPreviousDepthFrom cutoff ringDimension)
        (shiftPreviousDepthFrom cutoff innerDimension) (shiftPreviousDepthFrom cutoff left)
        (shiftPreviousDepthFrom cutoff right)
  | .recurrence count initial transition slot =>
      .recurrence count (initial.map (shiftPreviousDepthFrom cutoff))
        (transition.map (shiftPreviousDepthFrom (cutoff + 1))) slot
  | .recurrenceState count paths initial transition output =>
      .recurrenceState count paths (initial.map (shiftPreviousDepthFrom cutoff))
        (transition.map (shiftPreviousDepthFrom (cutoff + 1))) output

def shiftPreviousDepth := shiftPreviousDepthFrom 0

def OperationalBoundExpr.usesPrevious : OperationalBoundExpr → Bool
  | .closedInt _ | .contextual .. => false
  | .previous _ => true
  | .negate value => value.usesPrevious
  | .add left right | .subtract left right | .multiply left right | .divide left right |
      .minimum left right | .maximum left right | .centeredCap left right =>
      left.usesPrevious || right.usesPrevious
  | .matrixProduct ringDimension innerDimension left right =>
      ringDimension.usesPrevious || innerDimension.usesPrevious || left.usesPrevious ||
        right.usesPrevious
  | .recurrence .. | .recurrenceState .. => true

def OperationalBoundExpr.evaluateWithStates
    (environment : ParamEnvironment)
    (previousStates : List OperationalNumericState) : OperationalBoundExpr → Except OperationalError Int
  | .closedInt value => do
      if !intExprIsClosed value then throw .nonClosedExpression
      match value.evaluate environment with
      | some result => pure result
      | none => throw .nonClosedExpression
  | .contextual kind contextualEnvironment domains value =>
      match kind with
      | .minimum => evaluateIntMinimum contextualEnvironment domains value
      | .maximum => evaluateIntMaximum contextualEnvironment domains value
      | .maximumAbsolute => evaluateIntMaximumAbsolute contextualEnvironment domains value
  | .previous path =>
      match lookupPrevious previousStates path with
      | some result => pure result
      | none => throw (.invalidPreviousPath path)
  | .negate value => return -(← value.evaluateWithStates environment previousStates)
  | .add left right => return (← left.evaluateWithStates environment previousStates) +
      (← right.evaluateWithStates environment previousStates)
  | .subtract left right => return (← left.evaluateWithStates environment previousStates) -
      (← right.evaluateWithStates environment previousStates)
  | .multiply left right => return (← left.evaluateWithStates environment previousStates) *
      (← right.evaluateWithStates environment previousStates)
  | .divide left right => do
      let denominator ← right.evaluateWithStates environment previousStates
      if denominator = 0 then throw .divisionByZero
      if denominator < 0 then throw (.negativeDenominator denominator)
      return (← left.evaluateWithStates environment previousStates) / denominator
  | .minimum left right => do
      let left ← left.evaluateWithStates environment previousStates
      let right ← right.evaluateWithStates environment previousStates
      return min left right
  | .maximum left right => do
      let left ← left.evaluateWithStates environment previousStates
      let right ← right.evaluateWithStates environment previousStates
      return max left right
  | .centeredCap modulus value => do
      let modulus ← modulus.evaluateWithStates environment previousStates
      let value ← value.evaluateWithStates environment previousStates
      return capCentered modulus value
  | .matrixProduct ringDimension innerDimension left right => do
      let ringDimension ← ringDimension.evaluateWithStates environment previousStates
      let innerDimension ← innerDimension.evaluateWithStates environment previousStates
      let left ← left.evaluateWithStates environment previousStates
      let right ← right.evaluateWithStates environment previousStates
      return ringDimension * innerDimension * left * right
  | .recurrence count initial transition slot => do
      if initial.length != transition.length then
        throw (.unsupportedOutputArity transition.length initial.length)
      let initialValues ← initial.mapM (OperationalBoundExpr.evaluateWithStates environment previousStates)
      let initialState : OperationalNumericState := initialValues.map
        (fun value => { matrixMaximum := some value }) |>.toArray
      let rec iterate : Nat → OperationalNumericState → Except OperationalError OperationalNumericState
        | 0, state => pure state
        | remaining + 1, state => do
            let values ← transition.mapM
              (OperationalBoundExpr.evaluateWithStates environment (state :: previousStates))
            iterate remaining (values.map (fun value => { matrixMaximum := some value }) |>.toArray)
      let finalState ← iterate count initialState
      match finalState[slot]? >>= (·.matrixMaximum) with
      | some value => pure value
      | none => throw (.invalidPreviousPath (.matrixMaximum 0 slot))
  | .recurrenceState count paths initial transition output => do
      if paths.length != initial.length || initial.length != transition.length then
        throw (.unsupportedOutputArity transition.length initial.length)
      let initialValues ← initial.mapM
        (OperationalBoundExpr.evaluateWithStates environment previousStates)
      let initialState ← numericStateFromComponents paths initialValues
      let rec iterateState : Nat → OperationalNumericState →
          Except OperationalError OperationalNumericState
        | 0, state => pure state
        | remaining + 1, state => do
            let values ← transition.mapM
              (OperationalBoundExpr.evaluateWithStates environment (state :: previousStates))
            iterateState remaining (← numericStateFromComponents paths values)
      let finalState ← iterateState count initialState
      match lookupPrevious [finalState] output with
      | some value => pure value
      | none => throw (.invalidPreviousPath output)

@[simp] theorem OperationalBoundExpr.evaluateWithStates_closedConstant
    (environment : ParamEnvironment)
    (previousStates : List OperationalNumericState)
    (value : Int) :
    OperationalBoundExpr.evaluateWithStates environment previousStates
      (.closedInt (.constant value)) = .ok value := by
  simp [OperationalBoundExpr.evaluateWithStates, intExprIsClosed, IntExpr.evaluate]
  rfl

@[simp] theorem OperationalBoundExpr.evaluateWithStates_contextualMaximum_nil
    (environment : ParamEnvironment)
    (previousStates : List OperationalNumericState)
    (value : IntExpr)
    (result : Int)
    (evaluates : value.evaluate environment = some result) :
    OperationalBoundExpr.evaluateWithStates environment previousStates
      (.contextual .maximum environment [] value) = .ok result := by
  simp [OperationalBoundExpr.evaluateWithStates, evaluateIntMaximum, evaluateIntOverLoops,
    evaluateIntOverLoops.visit, evaluates]
  rfl

def OperationalBoundExpr.evaluate
    (environment : ParamEnvironment)
    (_previousState : OperationalState)
    (expression : OperationalBoundExpr) : Except OperationalError Int :=
  expression.evaluateWithStates environment []

def unconstrainedMatrixFact
    (node port : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment) : Except OperationalError OperationalMatrixFact := do
  let cap ← match matrixCap matrixType environment with
    | some cap => pure cap
    | none => throw (.invalidMatrixParameters node)
  let params ← match matrixType.evaluate environment (.constant cap) with
    | some params => pure params
    | none => throw (.invalidMatrixParameters node)
  let fact : OperationalMatrixFact := {
    subject := { node, port }
    origin := .value temporaryScope { node, port }
    matrixType
    matrixParams := params
    totalHardBound := .closedInt (.constant cap)
  }
  pure (fact.initializePrimitivePolynomial .large)

def defaultFact
    (node : Nat)
    (port : Nat)
    (wireType : WireTypeExpr)
    (environment : ParamEnvironment) : Except OperationalError OperationalMatrixFact :=
  match wireType with
  | .matrix matrixType => unconstrainedMatrixFact node port matrixType environment
  | .preimage matrixType => unconstrainedMatrixFact node port matrixType environment
  | _ => throw (.outputTypeMismatch node)

def defaultScalarFact
    (node port : Nat)
    (wireType : WireTypeExpr)
    (environment : ParamEnvironment)
    (domains : List OperationalParameterDomain := []) : Except OperationalError OperationalScalarFact :=
  match wireType with
  | .trapdoor matrixType sigma gadgetBase digitCount cutoff => do
      let cap ← match matrixCap matrixType environment with
        | some cap => pure cap
        | none => throw (.invalidMatrixParameters node)
      let params ← match matrixType.evaluate environment (.constant cap) with
        | some params => pure params
        | none => throw (.invalidMatrixParameters node)
      pure (.trapdoor {
        subject := { node, port }
        matrixType
        sigma
        gadgetBase
        digitCount
        preimageMaxCoefficientBound := cutoff
        matrixParams := params
        maximum := .closed (.closedInt (.constant cap))
        preimageCutoff := some (.closed (← validateContextualCutoffNonnegative node environment domains cutoff))
        publicIdentity := .sampledTrapdoor temporaryScope { node, port := 0 }
      })
  | .integer | .constantInt => pure (.integer {
      subject := { node, port }
      origin := .local temporaryScope { node, port }
      lower := 0
      upper := 0
      lowerExpression := .closed (.closedInt (.constant 0))
      upperExpression := .closed (.closedInt (.constant 0))
    })
  | .boolean | .constantBool => pure .boolean
  | .real | .constantReal => pure .real
  | .bytes length =>
      match length.evaluate environment with
      | some value => pure (.bytes {
          subject := { node, port }
          origin := .local temporaryScope { node, port }
          length := value
        })
      | none => throw (.invalidCount node 0)
  | .typedBlob typeName schemaHash => pure (.typedBlob typeName schemaHash)
  | _ => throw (.outputTypeMismatch node)

def lookupFact
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef) : Except OperationalError OperationalFact :=
  match facts.values[wire.node]?.bind fun outputs => outputs[wire.port]? with
  | some fact => pure fact
  | none => throw (.missingOperand node wire)

def integerFact
    (node port : Nat)
    (lower upper : Int) : Except OperationalError OperationalScalarFact := do
  if lower > upper then throw (.invalidBound node lower)
  pure (.integer {
    subject := { node, port }
    origin := .local temporaryScope { node, port }
    lower
    upper
    lowerExpression := .closed (.closedInt (.constant lower))
    upperExpression := .closed (.closedInt (.constant upper))
  })

def integerFactWithExpressions
    (node port : Nat)
    (lower upper : Int)
    (lowerExpression upperExpression : OperationalBoundExpr) :
    Except OperationalError OperationalScalarFact := do
  if lower > upper then throw (.invalidBound node lower)
  pure (.integer {
    subject := { node, port }
    origin := .local temporaryScope { node, port }
    lower
    upper
    lowerExpression := .closed lowerExpression
    upperExpression := .closed upperExpression
  })

structure OperationalIntegerInterval where
  lower : Int
  upper : Int
  lowerExpression : OperationalBoundExpr
  upperExpression : OperationalBoundExpr

/-- Integer arithmetic is invoked only after direct scalar reduction has materialized every
owner-aware leaf at its complete assignment.  Keeping this conversion at the primitive boundary
prevents the recursive interval algebra from reintroducing slot-only descriptors. -/
def requireMaterializedScalarBound
    (node : Nat) (bound : IndexedOperationalBoundExpr) : Except OperationalError OperationalBoundExpr :=
  match bound.closedOperational? with
  | some value => pure value
  | none => throw (.unsupportedOperationalExpr node)

def integerBinaryInterval
    (node : Nat)
    (operation : IntBinaryOp)
    (left right : OperationalIntegerFact) : Except OperationalError OperationalIntegerInterval := do
  let leftLower ← requireMaterializedScalarBound node left.lowerExpression
  let leftUpper ← requireMaterializedScalarBound node left.upperExpression
  let rightLower ← requireMaterializedScalarBound node right.lowerExpression
  let rightUpper ← requireMaterializedScalarBound node right.upperExpression
  match operation with
  | .add => pure {
      lower := left.lower + right.lower
      upper := left.upper + right.upper
      lowerExpression := .add leftLower rightLower
      upperExpression := .add leftUpper rightUpper
    }
  | .subtract => pure {
      lower := left.lower - right.upper
      upper := left.upper - right.lower
      lowerExpression := .subtract leftLower rightUpper
      upperExpression := .subtract leftUpper rightLower
    }
  | .multiply =>
      let values := [
        left.lower * right.lower,
        left.lower * right.upper,
        left.upper * right.lower,
        left.upper * right.upper
      ]
      match values with
      | [] => throw (.invalidBound node 0)
      | first :: tail =>
          let expressions := [
            OperationalBoundExpr.multiply leftLower rightLower,
            OperationalBoundExpr.multiply leftLower rightUpper,
            OperationalBoundExpr.multiply leftUpper rightLower,
            OperationalBoundExpr.multiply leftUpper rightUpper
          ]
          let firstExpression := expressions.headD (.closedInt (.constant first))
          pure {
            lower := tail.foldl min first
            upper := tail.foldl max first
            lowerExpression := expressions.drop 1 |>.foldl OperationalBoundExpr.minimum firstExpression
            upperExpression := expressions.drop 1 |>.foldl OperationalBoundExpr.maximum firstExpression
          }
  | .divide =>
      if right.lower ≤ 0 && 0 ≤ right.upper then throw .divisionByZero
      let values := [
        left.lower / right.lower,
        left.lower / right.upper,
        left.upper / right.lower,
        left.upper / right.upper
      ]
      match values with
      | [] => throw (.invalidBound node 0)
      | first :: tail =>
          let lower := tail.foldl min first
          let upper := tail.foldl max first
          pure {
            lower
            upper
            lowerExpression := .closedInt (.constant lower)
            upperExpression := .closedInt (.constant upper)
          }
  | .remainder =>
      if right.lower ≤ 0 && 0 ≤ right.upper then throw .divisionByZero
      let magnitude := max right.lower.natAbs right.upper.natAbs
      if magnitude = 0 then throw .divisionByZero
      pure {
        lower := 0
        upper := Int.ofNat (magnitude - 1)
        lowerExpression := .closedInt (.constant 0)
        upperExpression := .closedInt (.constant (Int.ofNat (magnitude - 1)))
      }

def cappedMatrixFact
    (nodeIndex : Nat)
    (outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (bound : Int) : Except OperationalError OperationalMatrixFact := do
  let cap ← match matrixCap matrixType environment with
    | some value => pure value
    | none => throw (.invalidMatrixParameters nodeIndex)
  if bound < 0 then throw (.invalidBound nodeIndex bound)
  let maximum := min cap bound
  let params ← match matrixType.evaluate environment (.constant maximum) with
    | some params => pure params
    | none => throw (.invalidMatrixParameters nodeIndex)
  let fact : OperationalMatrixFact := {
    subject := { node := nodeIndex, port := outputPort }
    origin := .value temporaryScope { node := nodeIndex, port := outputPort }
    matrixType
    matrixParams := params
    totalHardBound := .closedInt (.constant maximum)
  }
  pure (fact.initializePrimitivePolynomial .bounded)

def cappedMatrixFactExpr
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (bound : OperationalBoundExpr) : Except OperationalError OperationalMatrixFact := do
  let cap ← match matrixCap matrixType environment with
    | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
  let parameterBound ← if bound.usesPrevious then pure cap else do
    let maximum ← bound.evaluate environment #[]
    if maximum < 0 then throw (.invalidBound nodeIndex maximum)
    pure (min cap maximum)
  let params ← match matrixType.evaluate environment (.constant parameterBound) with
    | some params => pure params | none => throw (.invalidMatrixParameters nodeIndex)
  let fact : OperationalMatrixFact := {
    subject := { node := nodeIndex, port := outputPort }
    origin := .value temporaryScope { node := nodeIndex, port := outputPort }
    matrixType
    matrixParams := params
    totalHardBound := .minimum (.closedInt (.constant cap)) bound
  }
  pure (fact.initializePrimitivePolynomial .bounded)

def classifiedMatrixFact
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (bound : Int)
    (large : Bool)
    (canonicalRange : CanonicalRange := .unknown)
    (metadata : OperationalMatrixMetadata := {}) : Except OperationalError OperationalMatrixFact := do
  let fact ← cappedMatrixFact nodeIndex outputPort matrixType environment bound
  let role := if large then OperationalFactorRole.large else .bounded
  pure (({ fact with
    totalHardBound := .closedInt (.constant bound), canonicalRange, metadata
  }).initializePrimitivePolynomial role)

def classifiedMatrixFactExpr
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (bound : OperationalBoundExpr)
    (large : Bool)
    (canonicalRange : CanonicalRange := .unknown)
    (metadata : OperationalMatrixMetadata := {}) : Except OperationalError OperationalMatrixFact := do
  let fact ← cappedMatrixFactExpr nodeIndex outputPort matrixType environment bound
  let cap ← match matrixCap matrixType environment with
    | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
  let totalHardBound := .minimum (.closedInt (.constant cap)) bound
  let role := if large then OperationalFactorRole.large else .bounded
  pure (({ fact with totalHardBound, canonicalRange, metadata
    }).initializePrimitivePolynomial role)

def matrixTargetSummary (fact : OperationalMatrixFact) : RelationTargetSummary := {
  origin := fact.origin
  matrixType := fact.matrixType
  matrixParams := fact.matrixParams
  totalHardBound := fact.totalHardBound
  canonicalRange := fact.canonicalRange
  polynomial := relationSnapshotPolynomial fact.polynomial
}

def operationalProductFromFactors
    (factors : List OperationalFactorKey) : Except OperationalFlatError OperationalProductKey := do
  let first ← match factors.head? with
    | some factor => pure factor
    | none => throw .malformedProduct
  let rec visit
      (previousType : MatrixTypeExpr)
      (remaining : List OperationalFactorKey)
      (modes : List OperationalProductMode) :
      Except OperationalFlatError (List OperationalProductMode × MatrixTypeExpr) := do
    match remaining with
    | [] => pure (modes, previousType)
    | factor :: tail =>
        let (mode, outputType) ← inferOperationalProductMode previousType factor.inputType
        visit outputType tail (modes ++ [mode])
  let (modes, outputType) ← visit first.outputType (factors.drop 1) []
  pure { factors, modes, outputType }

def factorPublicIdentity? (factor : OperationalFactorKey) : Option PublicMatrixIdentity :=
  match factor.leaf with
  | .primitive (.publicMatrix identity) => some identity
  | _ => none

def factorPrimitiveOrigin? (factor : OperationalFactorKey) : Option MatrixOriginIdentity :=
  match factor.leaf with
  | .primitive (.matrix origin) => some origin
  | _ => none

def relationTargetOrigin : OperationalMatrixRelation → MatrixOriginIdentity
  | .decomposition relation => relation.inputOrigin
  | .preimage relation => relation.targetOrigin

def relationMatcherTarget : OperationalMatrixRelation → RelationTargetSummary
  | .decomposition relation => relation.inputSummary
  | .preimage relation => relation.targetSummary

def relationMatcherPublicIdentity : OperationalMatrixRelation → PublicMatrixIdentity
  | .decomposition relation => relation.publicIdentity
  | .preimage relation => relation.publicIdentity

/-- The relation boundary is exact: both adjacent primitive identities, the declared target
origin, and every concrete matrix shape reconstructed from the target snapshot must agree. -/
def exactAdjacentRelationMatches
    (environment : ParamEnvironment)
    (left right : OperationalFactorKey)
    (relation : OperationalMatrixRelation) : Bool :=
  let target := relationMatcherTarget relation
  let targetShape := target.matrixType.evaluate environment (.constant 0)
  let snapshotRebuilds := !target.polynomial.isEmpty && target.polynomial.all fun snapshotTerm =>
    match operationalPolynomialFromSnapshot [snapshotTerm] with
    | [{ product, .. }] => match operationalProductFromFactors product.factors with
      | .ok rebuilt =>
          rebuilt.modes == snapshotTerm.product.modes &&
            rebuilt.outputType == snapshotTerm.product.outputType &&
            match rebuilt.outputType.evaluate environment (.constant 0), targetShape with
            | some output, some expected =>
                output.modulus == expected.modulus && output.ringDimension == expected.ringDimension &&
                  output.rows == expected.rows && output.columns == expected.columns
            | _, _ => false
      | .error _ => false
    | _ => false
  let adjacentShapeMatches := match left.outputType.evaluate environment (.constant 0),
      right.inputType.evaluate environment (.constant 0), targetShape with
    | some leftShape, some rightShape, some expected =>
        leftShape.modulus == rightShape.modulus &&
          leftShape.ringDimension == rightShape.ringDimension &&
          leftShape.columns == rightShape.rows &&
          leftShape.rows == expected.rows && rightShape.columns == expected.columns &&
          expected.modulus == target.matrixParams.modulus &&
            expected.ringDimension == target.matrixParams.ringDimension &&
            expected.rows == target.matrixParams.rows && expected.columns == target.matrixParams.columns
    | _, _, _ => false
  factorPublicIdentity? left == some (relationMatcherPublicIdentity relation) &&
    factorPrimitiveOrigin? right == some (match relation with
      | .decomposition value => value.producer
      | .preimage value => value.producer) &&
    relationTargetOrigin relation == target.origin && adjacentShapeMatches && snapshotRebuilds

def matchingFactorRelation?
    (environment : ParamEnvironment)
    (left right : OperationalFactorKey) : Option OperationalMatrixRelation := do
  if !left.transforms.isEmpty || !right.transforms.isEmpty then none else pure ()
  right.relations.find? fun relation =>
    (match relation with
    | .decomposition value => value.status == ReconstructionStatus.available
    | .preimage _ => true) && exactAdjacentRelationMatches environment left right relation

def rewriteOperationalTermRelation?
    (node : Nat) (environment : ParamEnvironment)
    (term : OperationalTerm) : Except OperationalError (Option OperationalPolynomial) := do
  let rec visit
      (accumulated : List OperationalFactorKey) :
      List OperationalFactorKey → Except OperationalError (Option OperationalPolynomial)
    | left :: right :: tail =>
        match matchingFactorRelation? environment left right with
        | none => visit (accumulated ++ [left]) (right :: tail)
        | some relation => do
            let target := match relation with
              | .decomposition value => value.inputSummary
              | .preimage value => value.targetSummary
            let targetPolynomial := operationalPolynomialFromSnapshot target.polynomial
            if targetPolynomial.isEmpty then
              throw (.malformedRelation node)
            let rewritten ← targetPolynomial.mapM fun targetTerm => do
              let product ← operationalProductFromFactors
                (accumulated ++ targetTerm.product.factors ++ tail) |>.mapError
                  (fun _ => OperationalError.invalidMatrixParameters node)
              pure {
                coefficient := term.coefficient * targetTerm.coefficient
                product
              }
            pure (some rewritten)
    | _ => pure none
  visit [] term.product.factors

def rewriteOperationalRelationsWithCount
    (node : Nat) (environment : ParamEnvironment)
    (polynomial : OperationalPolynomial) : Except OperationalError (OperationalPolynomial × Nat) := do
  let rec finishTerm : Nat → OperationalTerm →
      Except OperationalError (OperationalPolynomial × Nat)
    | 0, term => do
        match ← rewriteOperationalTermRelation? node environment term with
        | none => pure ([term], 0)
        | some _ => throw (.invalidMatrixParameters node)
    | fuel + 1, term => do
        match ← rewriteOperationalTermRelation? node environment term with
        | none => pure ([term], 0)
        | some rewritten => do
          let mut finished : OperationalPolynomial := []
          let mut count := 1
          for generated in rewritten do
              let (next, nextCount) ← finishTerm fuel generated
              finished := finished ++ next
              count := count + nextCount
            pure (finished, count)
  let mut finished : OperationalPolynomial := []
  let mut count := 0
  for term in polynomial do
    let (next, nextCount) ← finishTerm 64 term
    finished := finished ++ next
    count := count + nextCount
  pure (normalizeOperationalTerms finished, count)

def rewriteOperationalRelations
    (node : Nat) (environment : ParamEnvironment)
    (polynomial : OperationalPolynomial) : Except OperationalError OperationalPolynomial :=
  return (← rewriteOperationalRelationsWithCount node environment polynomial).1

def sameConcreteMatrixShape (left right : Mxx.SamplerParams) : Bool :=
  left.modulus == right.modulus &&
    left.ringDimension == right.ringDimension &&
    left.rows == right.rows &&
    left.columns == right.columns

/-- Decide matrix-product compatibility from evaluated dimensions rather than the syntax of the
dimension expressions. This accepts equivalent forms such as `2` and `1 * 2`, while remaining
fail-closed when any type expression is not closed under the current parameter environment. -/
def concreteMatrixProductMatches
    (leftType rightType outputType : MatrixTypeExpr)
    (environment : ParamEnvironment) : Bool :=
  match leftType.evaluate environment (.constant 0),
      rightType.evaluate environment (.constant 0),
      outputType.evaluate environment (.constant 0) with
  | some left, some right, some output =>
      let sameRing :=
        left.modulus == right.modulus && left.modulus == output.modulus &&
          left.ringDimension == right.ringDimension &&
          left.ringDimension == output.ringDimension
      let outputShapeMatches (rows columns : Nat) :=
        output.rows == rows && output.columns == columns
      sameRing &&
        if left.columns == right.rows then
          outputShapeMatches left.rows right.columns
        else if left.rows == 1 && left.columns == 1 then
          outputShapeMatches right.rows right.columns
        else if right.rows == 1 && right.columns == 1 then
          outputShapeMatches left.rows left.columns
        else if left.rows == 1 && right.rows == 1 && left.columns == right.columns then
          outputShapeMatches 1 left.columns
        else false
  | _, _, _ => false

def equivalentRetypeOperationalFactor
    (outputType : MatrixTypeExpr)
    (standalone : Bool)
    (factor : OperationalFactorKey) : OperationalFactorKey :=
  let updateSummary (summary : OperationalBoundedFactorSummary) := {
    summary with matrixType := outputType
  }
  let leaf := match factor.leaf with
    | .boundedSummary origin summary => .boundedSummary origin (updateSummary summary)
    | .exactTransform tokens _ => .exactTransform tokens outputType
    | primitive => primitive
  {
    factor with
    leaf
    inputType := if standalone then outputType else factor.inputType
    outputType
    boundedSummary := factor.boundedSummary.map updateSummary
  }

/-- Canonicalize a matrix-type expression after proving that its evaluated shape is unchanged.
This changes no value and records no transform, so relation-owner identities remain bare. -/
def equivalentRetypeOperationalPolynomial
    (outputType : MatrixTypeExpr)
    (input : OperationalPolynomial) : Except OperationalFlatError OperationalPolynomial := do
  input.mapM fun term => do
    match term.product.factors.reverse with
    | [] => throw .malformedProduct
    | last :: reversePrefix =>
        let replacement := equivalentRetypeOperationalFactor outputType reversePrefix.isEmpty last
        let factors := (replacement :: reversePrefix).reverse
        pure { term with product := { term.product with factors, outputType } }

def retypeMatrixFact
    (node : Nat)
    (expected : MatrixTypeExpr)
    (fact : OperationalMatrixFact)
    (environment : ParamEnvironment) : Except OperationalError OperationalMatrixFact := do
  if fact.matrixType == expected then return fact
  let expectedParams ← match expected.evaluate environment
      (.constant fact.matrixParams.maxCoefficientBound) with
    | some params => pure params
    | none => throw (.invalidMatrixParameters node)
  if !sameConcreteMatrixShape fact.matrixParams expectedParams then
    throw (.outputTypeMismatch node)
  let polynomial ← equivalentRetypeOperationalPolynomial expected fact.polynomial
    |>.mapError fun error => OperationalError.flat node error
  pure { fact with matrixType := expected, matrixParams := expectedParams, polynomial }

def valueOriginAt
    (scope : ScopeTemplateKey)
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef) : Except OperationalError OperationalValueOrigin := do
  match ← lookupFact node facts wire with
  | { payload := .directValue root, .. } =>
      /- Direct inputs may carry the enclosing parallel coordinate, but a hash query still needs
      the exact semantic source of its key.  Follow only context transport maps to a singleton
      fixed leaf; pointwise and table-backed values have no one source and fail closed. -/
      let rec directOriginAt : Nat → Nat → Except OperationalError OperationalValueOrigin
        | _, 0 => throw (.unsupportedOperationalExpr root)
        | directRoot, fuel + 1 => do
            let value ← match facts.arena.direct.valueAt? directRoot with
              | some value => pure value
              | none => throw (.invalidOperationalExprRef directRoot)
            match value.payload with
            | .shared (.matrix _) (.matrix reference) =>
                match facts.arena.direct.fixed.matrices[reference]? with
                | some { origin := .value originScope originWire, .. } =>
                    pure (.local originScope originWire)
                | some { origin := .protocolInput input, .. } => pure (.protocolInput input)
                | some _ => pure (.local scope wire)
                | none => throw (.invalidOperationalExprRef reference)
            | .shared (.scalar _) (.scalar reference) =>
                match facts.arena.direct.fixed.scalars[reference]? with
                | some (.integer fact) => pure fact.origin
                | some (.bytes fact) => pure fact.origin
                | _ => throw (.unsupportedOperationalExpr directRoot)
            | .mapped _ source map => do
                let sourceValue ← match facts.arena.direct.valueAt? source with
                  | some sourceValue => pure sourceValue
                  | none => throw (.invalidOperationalExprRef source)
                if !map.transportValid || map.source != sourceValue.context ||
                    map.destination != value.context then
                  throw (.unsupportedOperationalExpr directRoot)
                let origin ← directOriginAt source fuel
                let dependencies := operationalValueOriginFreeVariables origin
                let owned := dependencies.filter map.source.binders.contains
                if owned.isEmpty then pure origin
                else if owned.length == dependencies.length then
                  match reindexOperationalValueOrigin map origin with
                  | some origin => pure origin
                  | none => throw (.unsupportedOperationalExpr directRoot)
                else throw (.unsupportedOperationalExpr directRoot)
            /- A rebound changes the graph-wire subject checked at fixed-leaf materialization,
            not the semantic source used as a deterministic hash key. -/
            | .rebound _ source _ => directOriginAt source fuel
            | _ => throw (.unsupportedOperationalExpr directRoot)
      pure (← directOriginAt root (facts.arena.direct.values.size + 1))

def operationalPolynomialNoiseSummary
    (polynomial : OperationalPolynomial) :
    Except OperationalFlatError (Option OperationalBoundedFactorSummary) := do
  let noise := sortOperationalTerms
    (normalizeOperationalTerms (polynomial.filter operationalTermIsNoise))
  if noise.isEmpty then return none
  let summaries ← noise.mapM boundedNoiseTermSummary
  let firstTerm ← match noise.head? with
    | some term => pure term
    | none => throw .malformedProduct
  let firstTermSummary ← match summaries.head? with
    | some summary => pure summary
    | none => throw .malformedProduct
  let hardBound := (noise.zip summaries).foldl (fun current pair =>
    .add current (.multiply
      (.closedInt (.constant (operationalAbsoluteCoefficient pair.1.coefficient)))
      pair.2.hardBound)) (.closedInt (.constant 0))
  let tokens := [.sumStart] ++ ((noise.zip summaries).flatMap fun (term, summary) =>
    boundedNoiseTermTokens term summary) ++ [.summaryBound hardBound, .sumEnd]
  pure (some {
    matrixType := firstTerm.product.outputType
    rowCount := firstTermSummary.rowCount
    hardBound
    metadata := {
      isConstantPolynomial := summaries.all (·.metadata.isConstantPolynomial)
      knownZeroRows := none
    }
    provenance := tokens
  })

def OperationalMatrixFact.noiseHardBound
    (fact : OperationalMatrixFact) : Except OperationalFlatError OperationalBoundExpr := do
  match ← operationalPolynomialNoiseSummary fact.polynomial with
  | some summary => pure summary.hardBound
  | none => pure (.closedInt (.constant 0))

def OperationalMatrixFact.evaluateNoiseHardBound
    (fact : OperationalMatrixFact)
    (environment : ParamEnvironment)
    (states : List OperationalNumericState := []) : Except OperationalError Int := do
  let expression ← fact.noiseHardBound |>.mapError fun _ =>
    OperationalError.invalidMatrixParameters fact.subject.node
  expression.evaluateWithStates environment states

/-- A decoder residual is eligible for numeric noise evaluation only once normalization and
relation consumption have removed every signal (`Large`) term. -/
def OperationalMatrixFact.rejectResidualLargeTerms
    (fact : OperationalMatrixFact) : Except OperationalError Unit :=
  if fact.polynomial.any operationalTermIsSignal then
    throw (.residualContainsLargeTerm fact.subject.node)
  else
    pure ()

def flatErrorAt (node : Nat) : OperationalFlatError → OperationalError
  | error => .flat node error

def polynomialMatrixFact
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (polynomial : OperationalPolynomial)
    (canonicalRange : CanonicalRange := .unknown) : Except OperationalError OperationalMatrixFact := do
  let polynomial ← normalizeOperationalPolynomial polynomial |>.mapError (flatErrorAt nodeIndex)
  let cap ← match matrixCap matrixType environment with
    | some value => pure value
    | none => throw (.invalidMatrixParameters nodeIndex)
  let noiseSummary ← operationalPolynomialNoiseSummary polynomial |>.mapError (flatErrorAt nodeIndex)
  let totalHardBound :=
    if polynomial.any operationalTermIsSignal then
      .closedInt (.constant cap)
    else match noiseSummary with
      | some summary => .minimum (.closedInt (.constant cap)) summary.hardBound
      | none => .closedInt (.constant 0)
  let parameterBound :=
    if totalHardBound.usesPrevious then cap
    else match totalHardBound.evaluate environment #[] with
      | .ok value => min cap value
      | .error _ => cap
  let params ← match matrixType.evaluate environment (.constant parameterBound) with
    | some params => pure params
    | none => throw (.invalidMatrixParameters nodeIndex)
  let metadata := match noiseSummary with
    | some summary => summary.metadata
    | none => {}
  pure {
    subject := { node := nodeIndex, port := outputPort }
    origin := .value temporaryScope { node := nodeIndex, port := outputPort }
    matrixType
    matrixParams := params
    totalHardBound
    polynomial
    metadata
    canonicalRange
  }

/-- Preserve a strict canonical coefficient range through ordinary matrix multiplication when
both inputs are known constant-polynomial matrices.  In that case no negacyclic convolution term
can wrap a negative coefficient to a residue near the modulus: every output coefficient is a sum
of `left.columns` nonnegative scalar products.  For general polynomial inputs the quotient-ring
signs make any sub-modulus range unsafe, so the result remains unknown. -/
def constantPolynomialProductCanonicalRange
    (left right : OperationalMatrixFact) : CanonicalRange :=
  match left.canonicalRange, right.canonicalRange with
  | .below leftUpper, .below rightUpper =>
      if left.metadata.isConstantPolynomial && right.metadata.isConstantPolynomial &&
          left.matrixParams.modulus > 0 &&
          left.matrixParams.modulus == right.matrixParams.modulus &&
          left.matrixParams.columns == right.matrixParams.rows then
        let unreducedUpper :=
          left.matrixParams.columns * (leftUpper - 1) * (rightUpper - 1) + 1
        .below (min left.matrixParams.modulus.toNat unreducedUpper)
      else .unknown
  | _, _ => .unknown

/-- Contract only the matching block boundaries of a column-concatenated left operand and a
row-concatenated right operand.  The ordered concat snapshots, rather than the visible embedded
terms, are authoritative: an all-zero block therefore remains a required physical lane. -/
def contractComplementaryBlocks
    (node : Nat)
    (expectedOutput : MatrixTypeExpr)
    (left right : OperationalMatrixFact)
    (raw : OperationalPolynomial) : Except OperationalError OperationalPolynomial := do
  let layoutMatchesOwner (axis : ConcatAxis) (owner : MatrixTypeExpr)
      (partitions : Array OperationalBlockPartition) : Bool :=
    if partitions.isEmpty then false else
    match partitions[0]? with
    | none => false
    | some first =>
        let sameRing := partitions.all fun partition => operationalSameRing partition.matrixType owner
        match axis with
        | .columns => sameRing &&
            partitions.all (fun partition => operationalDimensionEqual partition.matrixType.rows first.matrixType.rows) &&
            operationalDimensionEqual first.matrixType.rows owner.rows &&
            operationalDimensionEqual
              (partitions.foldl (fun total partition => .add total partition.matrixType.columns)
                (.constant 0)) owner.columns
        | .rows => sameRing &&
            partitions.all (fun partition => operationalDimensionEqual partition.matrixType.columns
              first.matrixType.columns) &&
            operationalDimensionEqual first.matrixType.columns owner.columns &&
            operationalDimensionEqual
              (partitions.foldl (fun total partition => .add total partition.matrixType.rows)
                (.constant 0)) owner.rows
        | .diagonal => false
  match left.blockLayout, right.blockLayout with
  | some leftLayout, some rightLayout =>
      if leftLayout.axis != .columns || rightLayout.axis != .rows then pure raw
      else if !layoutMatchesOwner .columns left.matrixType leftLayout.partitions ||
          !layoutMatchesOwner .rows right.matrixType rightLayout.partitions ||
          !operationalSameRing left.matrixType right.matrixType ||
          !operationalSameRing left.matrixType expectedOutput ||
          !operationalDimensionEqual left.matrixType.rows expectedOutput.rows ||
          !operationalDimensionEqual right.matrixType.columns expectedOutput.columns then
        throw (.malformedRelation node)
      else if leftLayout.partitions.size != rightLayout.partitions.size then
        throw (.malformedRelation node)
      else do
        let mut contracted : OperationalPolynomial := []
        for index in [:leftLayout.partitions.size] do
          let leftPart ← match leftLayout.partitions[index]? with
            | some partition => pure partition
            | none => throw (.malformedRelation node)
          let rightPart ← match rightLayout.partitions[index]? with
            | some partition => pure partition
            | none => throw (.malformedRelation node)
          if !operationalSameRing leftPart.matrixType rightPart.matrixType ||
              !operationalDimensionEqual leftPart.matrixType.columns rightPart.matrixType.rows then
            throw (.malformedRelation node)
          let product ← multiplyOperationalPolynomials leftPart.polynomial rightPart.polynomial
            |>.mapError (flatErrorAt node)
          contracted := contracted ++ product
        normalizeOperationalPolynomial contracted |>.mapError (flatErrorAt node)
  | _, _ => pure raw

def multiplyConcreteMatrixFacts
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (rule : DerivationRule)
    (rightWire : WireRef)
    (environment : ParamEnvironment)
    (left right : OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  let raw ← multiplyOperationalPolynomials left.polynomial right.polynomial
    |>.mapError (flatErrorAt nodeIndex)
  let contracted ← contractComplementaryBlocks nodeIndex matrixType left right raw
  let rewritten ← rewriteOperationalRelations nodeIndex environment contracted
  let polynomial ← match rule with
    | .matrixMultiplyRelation declaredRight => do
        if declaredRight != rightWire then throw (.missingRelation nodeIndex declaredRight)
        if rewritten == contracted then throw (.missingRelation nodeIndex rightWire)
        pure rewritten
    | _ => pure rewritten
  polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
    (constantPolynomialProductCanonicalRange left right)

/-- Concrete direct reduction records exact relation applications without changing the produced
fact.  Ordinary consumers deliberately project the fact alone through `multiplyConcreteMatrixFacts`. -/
def multiplyConcreteMatrixFactsWithRelationRewriteCount
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (rule : DerivationRule)
    (rightWire : WireRef)
    (environment : ParamEnvironment)
    (left right : OperationalMatrixFact) : Except OperationalError (OperationalMatrixFact × Nat) := do
  let raw ← multiplyOperationalPolynomials left.polynomial right.polynomial
    |>.mapError (flatErrorAt nodeIndex)
  let contracted ← contractComplementaryBlocks nodeIndex matrixType left right raw
  let (rewritten, rewriteCount) ← rewriteOperationalRelationsWithCount nodeIndex environment contracted
  let polynomial ← match rule with
    | .matrixMultiplyRelation declaredRight => do
        if declaredRight != rightWire then throw (.missingRelation nodeIndex declaredRight)
        if rewritten == contracted then throw (.missingRelation nodeIndex rightWire)
        pure rewritten
    | _ => pure rewritten
  let fact ← polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
    (constantPolynomialProductCanonicalRange left right)
  pure (fact, rewriteCount)

def addConcreteMatrixFacts
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (subtract : Bool)
    (environment : ParamEnvironment)
    (left right : OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  let left ← retypeMatrixFact nodeIndex matrixType left environment
  let right ← retypeMatrixFact nodeIndex matrixType right environment
  let polynomial := if subtract then
    subtractOperationalPolynomials left.polynomial right.polynomial
  else
    addOperationalPolynomials left.polynomial right.polynomial
  polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial

/-- A direct value is evaluable only when every binder in its declared context has an assigned,
in-range lane.  Checking the whole context at each recursive root prevents an unrelated leaf from
silently accepting a partial assignment; mapped roots construct and validate their own source
assignment before their recursive evaluation. -/
def completeDirectIndexAssignment
    (parameters : ParamEnvironment)
    (context : IndexContext)
    (indices : IndexValueEnvironment) : Bool :=
  validateContext context && context.binders.all fun binder =>
    (evaluateIndexExpr parameters context indices (.variable binder)).isSome

/-- Build one delayed direct matrix operation. -/
def OperationalExprArena.pushDirectMatrixPointwiseN
    (arena : OperationalExprArena)
    (operation : PrimitiveOperation)
    (inputs : Array OperationalFact) : Except OperationalError (OperationalExprArena × OperationalFact) := do
  let inputIds ← inputs.mapM fun input => match input.payload with
    | .directValue id => pure id
  let inputSchemas ← inputIds.mapM fun id => match arena.direct.valueAt? id with
    | some value => pure value.payload.schema
    | none => throw (.invalidOperationalExprRef id)
  if !matrixOperationSchemasValid operation inputSchemas operation.outputSchema then
    throw (.outputTypeMismatch operation.ownerNode)
  let (direct, result) ← match arena.direct.pushPointwise (.matrix operation) inputIds with
    | some result => pure result
    | none => throw (.unsupportedOperationalExpr arena.direct.values.size)
  let value ← match direct.valueAt? result with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef result)
  pure ({ arena with direct }, {
    context := value.context
    payload := .directValue result
    storage := value.storage
  })

def OperationalExprArena.pushDirectMatrixPointwise
    (arena : OperationalExprArena)
    (operation : PrimitiveOperation)
    (left right : OperationalFact) : Except OperationalError (OperationalExprArena × OperationalFact) :=
  arena.pushDirectMatrixPointwiseN operation #[left, right]

/-- Direct scalar primitives use the same context merger and correlated physical-lane reduction
as matrix primitives. -/
def OperationalExprArena.pushDirectScalarPointwiseN
    (arena : OperationalExprArena)
    (operation : DirectScalarOperation)
    (inputs : Array OperationalFact) : Except OperationalError (OperationalExprArena × OperationalFact) := do
  let inputIds ← inputs.mapM fun input => match input.payload with
    | .directValue id => pure id
  let inputSchemas ← inputIds.mapM fun id => match arena.direct.valueAt? id with
    | some value => pure value.payload.schema
    | none => throw (.invalidOperationalExprRef id)
  let (direct, result) ← match arena.direct.pushPointwise (.scalar operation) inputIds with
    | some result => pure result
    | none => throw (.outputTypeMismatch operation.ownerNode)
  let value ← match direct.valueAt? result with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef result)
  if !scalarOperationSchemasValid operation.kind inputSchemas value.payload.schema then
    throw (.outputTypeMismatch operation.ownerNode)
  pure ({ arena with direct }, {
    context := value.context, payload := .directValue result, storage := value.storage })

/-- Structural admission for an owner-aware descriptor.  Evaluation is deliberately deferred:
the same descriptor may be stored under a family, selected dynamically, and gathered before one
complete carrier assignment supplies the exact values of its indexed leaves. -/
def indexedParameterDescriptorValid (context : IndexContext) : IndexedParameterExpr → Bool
  /- `.ir` is the closed Graph-IR expression form; owner-bearing coordinates introduced at an
  indexed carrier boundary use `.index`.  The latter is checked against the exact context here
  and resolved only at the complete fixed-assignment boundary. -/
  | .ir _ => true
  | .index value => indexExpressionInBounds context value
  | .add left right | .subtract left right | .multiply left right | .divide left right |
      .roundDivide left right =>
      indexedParameterDescriptorValid context left && indexedParameterDescriptorValid context right
  | .log2Ceil value => indexedParameterDescriptorValid context value

def indexedMatrixTypeDescriptorValid
    (context : IndexContext) (value : IndexedMatrixTypeExpr) : Bool :=
  indexedParameterDescriptorValid context value.modulus &&
    indexedParameterDescriptorValid context value.ringDimension &&
    indexedParameterDescriptorValid context value.rows &&
    indexedParameterDescriptorValid context value.columns

def indexedOperationalParameterDomainsValid
    (context : IndexContext) : List IndexedOperationalParameterDomain → Bool
  | [] => true
  | .loopIndex binder :: tail => context.binders.contains binder &&
      indexedOperationalParameterDomainsValid context tail
  | .parameter _ _ domains expression :: tail =>
      indexedOperationalParameterDomainsValid context domains &&
        indexedParameterDescriptorValid context expression &&
        indexedOperationalParameterDomainsValid context tail

/-- Store a relation producer over its complete Graph-IR operand list.  Its delayed descriptor
must be well-scoped, but may still contain a loop, selection, or gather coordinate. -/
def validateDirectRelationDescriptor
    (context : IndexContext) (operation : DirectRelationOperation) : Except OperationalError Unit := do
  if !validateContext context || !indexedMatrixTypeDescriptorValid context operation.outputType then
    throw (.unsupportedOperationalExpr operation.ownerNode)
  match operation.kind with
  | .preimage maximum domains =>
      if !indexedParameterDescriptorValid context maximum ||
          !indexedOperationalParameterDomainsValid context domains then
        throw (.unsupportedOperationalExpr operation.ownerNode)
      match maximum, domains with
      | .ir maximum, [] =>
          let _ ← validateContextualCutoffNonnegative operation.ownerNode
            operation.parameterEnvironment [] maximum
          pure ()
      | _, _ => pure ()
  | .decomposition declaredType base small digitCount domains layouts =>
      if !indexedMatrixTypeDescriptorValid context declaredType ||
          !indexedParameterDescriptorValid context base ||
          !indexedParameterDescriptorValid context digitCount ||
          !indexedOperationalParameterDomainsValid context domains then
        throw (.unsupportedOperationalExpr operation.ownerNode)
      /- Fully closed descriptors remain checked at construction, preserving immediate rejection
      of a malformed or absent gadget layout.  Indexed descriptors defer this same check until
      their complete carrier assignment has materialized every owner-bearing field. -/
      match declaredType.closedIr?, base, digitCount, domains with
      | some declaredType, .ir base, .ir digitCount, [] =>
          let bound ← evaluateIntInvariant operation.parameterEnvironment [] base
          let count ← evaluateIntInvariant operation.parameterEnvironment [] digitCount
          if bound <= 1 || count <= 0 then throw (.gadgetLayoutMismatch operation.ownerNode)
          let params ← match declaredType.evaluate operation.parameterEnvironment (.constant 0) with
            | some value => pure value | none => throw (.invalidMatrixParameters operation.ownerNode)
          let outputParams ← match operation.outputSchema.evaluate operation.parameterEnvironment (.constant 0) with
            | some value => pure value | none => throw (.invalidMatrixParameters operation.ownerNode)
          if !sameConcreteMatrixShape params outputParams then
            throw (.outputTypeMismatch operation.ownerNode)
          let descriptor ← resolveGadgetLayout operation.ownerNode layouts params
          let expected := if small then descriptor.smallDigitCount else descriptor.regularDigitCount
          if bound != descriptor.base || count.toNat != expected then
            throw (.gadgetLayoutMismatch operation.ownerNode)
      | _, _, _, _ => pure ()

def OperationalExprArena.pushDirectRelationPointwise
    (arena : OperationalExprArena)
    (operation : DirectRelationOperation)
    (inputs : Array OperationalFact) : Except OperationalError (OperationalExprArena × OperationalFact) := do
  let context ← match mergeIndexContextsN (inputs.toList.map (·.context)) with
    | some value => pure value
    | none => throw (.unsupportedOperationalExpr operation.ownerNode)
  validateDirectRelationDescriptor context operation
  let inputIds ← inputs.mapM fun input => match input.payload with
    | .directValue id => pure id
  let inputSchemas ← inputIds.mapM fun id => match arena.direct.valueAt? id with
    | some value => pure value.payload.schema
    | none => throw (.invalidOperationalExprRef id)
  if !relationOperationSchemasValid operation inputSchemas operation.outputSchema then
    throw (.outputTypeMismatch operation.ownerNode)
  let (direct, result) ← match arena.direct.pushPointwise (.relation operation) inputIds with
    | some result => pure result
    | none => throw (.unsupportedOperationalExpr arena.direct.values.size)
  let value ← match direct.valueAt? result with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef result)
  pure ({ arena with direct }, {
    context := value.context, payload := .directValue result, storage := value.storage })

def OperationalExprArena.pushDirectValueScalarPointwise
    (arena : OperationalExprArena)
    (operation : DirectValueScalarOperation)
    (input : OperationalFact) : Except OperationalError (OperationalExprArena × OperationalFact) := do
  let inputId ← match input.payload with
    | .directValue id => pure id
  let inputSchema ← match arena.direct.valueAt? inputId with
    | some value => pure value.payload.schema
    | none => throw (.invalidOperationalExprRef inputId)
  let (direct, result) ← match arena.direct.pushPointwise (.matrixToScalar operation) #[inputId] with
    | some result => pure result
    | none => throw (.unsupportedOperationalExpr inputId)
  let value ← match direct.valueAt? result with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef result)
  if !pointwiseSchemasValid (.matrixToScalar operation) #[inputSchema] value.payload.schema then
    throw (.outputTypeMismatch operation.ownerNode)
  pure ({ arena with direct }, {
    context := value.context
    payload := .directValue result
    storage := value.storage
  })

def OperationalExprArena.pushDirectIntegerLiftPointwise
    (arena : OperationalExprArena)
    (operation : DirectValueMatrixOperation)
    (input : OperationalFact) : Except OperationalError (OperationalExprArena × OperationalFact) := do
  let inputId ← match input.payload with
    | .directValue id => pure id
  let inputSchema ← match arena.direct.valueAt? inputId with
    | some value => pure value.payload.schema
    | none => throw (.invalidOperationalExprRef inputId)
  let (direct, result) ← match arena.direct.pushPointwise (.matrixFromScalar operation) #[inputId] with
    | some result => pure result
    | none => throw (.unsupportedOperationalExpr inputId)
  let value ← match direct.valueAt? result with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef result)
  if !pointwiseSchemasValid (.matrixFromScalar operation) #[inputSchema] value.payload.schema then
    throw (.outputTypeMismatch operation.ownerNode)
  pure ({ arena with direct }, {
    context := value.context
    payload := .directValue result
    storage := value.storage
  })

def OperationalExprArena.pushDirectBggGrouping
    (arena : OperationalExprArena)
    (vector publicKey plaintext : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let vectorId ← match vector.payload with
    | .directValue id => pure id
  let matrixType ← match arena.direct.valueAt? vectorId with
    | some { payload := payload, .. } => match payload.schema with
      | .matrix matrixType => pure matrixType
      | .scalar _ => throw (.unsupportedOperationalExpr vectorId)
    | none => throw (.invalidOperationalExprRef vectorId)
  let operation : PrimitiveOperation := {
    kind := .bggGrouping, outputType := .fromIr matrixType, outputSchema := matrixType,
    ownerScope := arena.activeScope,
    ownerNode := arena.activeNode.getD 0, outputPort := 0, parameterEnvironment := [] }
  arena.pushDirectMatrixPointwiseN operation #[vector, publicKey, plaintext]

/-- A fixed-assignment direct kernel constructs a fresh executable output.  Its owner descriptor,
not the identities carried by its operands, determines the output wire and scope.  In particular
this changes neither polynomial factors nor relation snapshots transported from the inputs. -/
def directPointwiseMatrixOutput
    (ownerScope : Option ScopeTemplateKey)
    (ownerNode outputPort : Nat)
    (fact : OperationalMatrixFact) : OperationalMatrixFact :=
  let scope := ownerScope.getD temporaryScope
  { fact with
    subject := { node := ownerNode, port := outputPort }
    origin := .value scope { node := ownerNode, port := outputPort }
  }

/-- Matrix-to-scalar kernels likewise construct a fresh scalar output without rebinding any
matrix operand identity. -/
def directPointwiseScalarOutput
    (ownerScope : Option ScopeTemplateKey)
    (ownerNode outputPort : Nat)
    (fact : OperationalScalarFact) : OperationalScalarFact :=
  let scope := ownerScope.getD temporaryScope
  match fact with
  | .integer value => .integer {
      value with
      subject := { node := ownerNode, port := outputPort }
      origin := .local scope { node := ownerNode, port := outputPort }
    }
  | value => value

/-- Concrete operands accepted by the relation-producing direct kernel.  This deliberately has
no arena reference: all identities, target snapshots, and trapdoor ownership come from the
already-correlated physical lane facts. -/
inductive DirectRelationArgument where
  | matrix (fact : OperationalMatrixFact)
  | trapdoor (fact : OperationalTrapdoorFact)

def rebindDirectRelationProducer
    (output : OperationalMatrixFact) : OperationalMatrixFact :=
  ({ output with relations := output.relations.map fun relation => match relation with
    | OperationalMatrixRelation.preimage value =>
        OperationalMatrixRelation.preimage { value with producer := output.origin }
    | OperationalMatrixRelation.decomposition value =>
        OperationalMatrixRelation.decomposition { value with producer := output.origin }
  }).refreshPrimitivePolynomial

/-- Apply one fully-aligned preimage/decomposition lane.  All graph operands are explicit in
`arguments`; this never selects a representative from another family or consults an arena. -/
def applyDirectRelationProducer
    (operation : DirectRelationOperation)
    (matrixType : MatrixTypeExpr)
    (arguments : Array DirectRelationArgument) : Except OperationalError OperationalMatrixFact := do
  let output ← match operation.kind with
  | .preimage maximum loopDomains => do
      let maximum ← match maximum with
        | .ir value => pure value
        | _ => throw (.unsupportedOperationalExpr operation.ownerNode)
      let loopDomains : List OperationalParameterDomain ← match loopDomains with
        | [] => pure []
        | _ => throw (.unsupportedOperationalExpr operation.ownerNode)
      let publicFact ← match arguments[0]? with
        | some (DirectRelationArgument.matrix fact) => pure fact
        | _ => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let trapdoor ← match arguments[1]? with
        | some (DirectRelationArgument.trapdoor fact) => pure fact
        | _ => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let target ← match arguments[2]? with
        | some (DirectRelationArgument.matrix fact) => pure fact
        | _ => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      if arguments.size != 3 then throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      if !sameConcreteMatrixShape publicFact.matrixParams trapdoor.matrixParams ||
          !concreteMatrixProductMatches publicFact.matrixType matrixType target.matrixType
            operation.parameterEnvironment then
        throw (.outputTypeMismatch operation.ownerNode)
      let publicIdentity ← match publicFact.identity with
        | some identity => pure identity
        | none => throw (.missingPublicIdentity operation.ownerNode { node := 0, port := 0 })
      if publicIdentity != trapdoor.publicIdentity then throw (.publicIdentityMismatch operation.ownerNode)
      let trapdoorCutoff ← trapdoor.preimageCutoff.mapM
        (requireMaterializedScalarBound operation.ownerNode)
      let _ ← validatePreimageCutoffAgreement operation.ownerNode operation.parameterEnvironment loopDomains
        maximum trapdoor.publicIdentity trapdoorCutoff
      let bound := OperationalBoundExpr.contextual .maximum operation.parameterEnvironment loopDomains maximum
      let result ← cappedMatrixFactExpr operation.ownerNode operation.outputPort matrixType
        operation.parameterEnvironment bound
      let relation : PreimageRelation := {
        producer := result.origin, publicIdentity, targetOrigin := target.origin,
        targetSummary := matrixTargetSummary target }
      pure ({ result with relations := [.preimage relation] }).refreshPrimitivePolynomial
  | .decomposition declaredType base small digitCount loopDomains layouts => do
      let declaredType ← match declaredType.closedIr? with
        | some value => pure value
        | none => throw (.unsupportedOperationalExpr operation.ownerNode)
      let base ← match base with
        | .ir value => pure value
        | _ => throw (.unsupportedOperationalExpr operation.ownerNode)
      let digitCount ← match digitCount with
        | .ir value => pure value
        | _ => throw (.unsupportedOperationalExpr operation.ownerNode)
      let loopDomains : List OperationalParameterDomain ← match loopDomains with
        | [] => pure []
        | _ => throw (.unsupportedOperationalExpr operation.ownerNode)
      let input ← match arguments with
        | #[DirectRelationArgument.matrix fact] => pure fact
        | _ => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let bound ← evaluateIntInvariant operation.parameterEnvironment loopDomains base
      let count ← evaluateIntInvariant operation.parameterEnvironment loopDomains digitCount
      if bound <= 1 || count <= 0 then throw (.gadgetLayoutMismatch operation.ownerNode)
      let params ← match declaredType.evaluate operation.parameterEnvironment (.constant 0) with
        | some value => pure value | none => throw (.invalidMatrixParameters operation.ownerNode)
      let descriptor ← resolveGadgetLayout operation.ownerNode layouts params
      let expectedCount := if small then descriptor.smallDigitCount else descriptor.regularDigitCount
      if count.toNat != expectedCount || bound != descriptor.base then
        throw (.gadgetLayoutMismatch operation.ownerNode)
      let outputParams ← match matrixType.evaluate operation.parameterEnvironment (.constant 0) with
        | some value => pure value | none => throw (.invalidMatrixParameters operation.ownerNode)
      if !sameConcreteMatrixShape params outputParams || outputParams.modulus != input.matrixParams.modulus ||
          outputParams.ringDimension != input.matrixParams.ringDimension ||
          outputParams.rows != input.matrixParams.rows * count.toNat ||
          outputParams.columns != input.matrixParams.columns then
        throw (.outputTypeMismatch operation.ownerNode)
      let publicIdentity := PublicMatrixIdentity.gadget descriptor.paramsId params
        input.matrixParams.rows bound small count.toNat
      let result ← cappedMatrixFact operation.ownerNode operation.outputPort matrixType
        operation.parameterEnvironment (Int.ofNat (Mxx.gadgetDecompositionBound bound small))
      let status := if !small then ReconstructionStatus.available else
        match input.canonicalRange with
        | .below upper => if upper <= descriptor.smallestCrtModulus then .available
            else .smallRangeMissing descriptor.smallestCrtModulus
        | .unknown => .smallRangeMissing descriptor.smallestCrtModulus
      let relation : DecompositionRelation := {
        producer := result.origin, publicIdentity, inputOrigin := input.origin,
        inputSummary := matrixTargetSummary input, base := bound, small := small,
        digitCount := count.toNat, status }
      let canonicalRange : CanonicalRange :=
        if small then .below bound.natAbs else .unknown
      let output : OperationalMatrixFact := { result with
        canonicalRange := canonicalRange
        relations := [OperationalMatrixRelation.decomposition relation]
      }
      pure output.refreshPrimitivePolynomial
  pure (rebindDirectRelationProducer
    (directPointwiseMatrixOutput operation.ownerScope operation.ownerNode operation.outputPort output))
def applyDirectMatrixFromScalarOperation
    (operation : DirectValueMatrixOperation)
    (matrixType : MatrixTypeExpr)
    (input : OperationalScalarFact) : Except OperationalError OperationalMatrixFact := do
  let output ← match operation.kind with
  | .liftIntegerToConstantPolynomial declaredType => do
      let integer ← match input with
        | .integer value => pure value
        | _ => throw (.operandNotInteger operation.ownerNode { node := 0, port := 0 })
      if !operationalMatrixTypeEqual declaredType matrixType then
        throw (.outputTypeMismatch operation.ownerNode)
      let params ← match matrixType.evaluate operation.parameterEnvironment (.constant 0) with
        | some params => pure params
        | none => throw (.invalidMatrixParameters operation.ownerNode)
      if params.rows != 1 || params.columns != 1 || params.modulus <= 0 ||
          params.ringDimension == 0 then
        throw (.invalidMatrixParameters operation.ownerNode)
      let lower ← requireMaterializedScalarBound operation.ownerNode integer.lowerExpression
      let upper ← requireMaterializedScalarBound operation.ownerNode integer.upperExpression
      let bound := OperationalBoundExpr.maximum (.negate lower) upper
      classifiedMatrixFactExpr operation.ownerNode operation.outputPort matrixType
        operation.parameterEnvironment bound false (.below params.modulus.toNat)
        { isConstantPolynomial := true }
  | .trapdoorPublic declaredType => do
      let trapdoor ← match input with
        | .trapdoor value => pure value
        | _ => throw (.missingPublicIdentity operation.ownerNode { node := 0, port := 0 })
      if !operationalMatrixTypeEqual declaredType matrixType then
        throw (.outputTypeMismatch operation.ownerNode)
      let cap ← match matrixCap matrixType operation.parameterEnvironment with
        | some value => pure value
        | none => throw (.invalidMatrixParameters operation.ownerNode)
      let rec maximum : PublicMatrixIdentity → Int
        | .sampledTrapdoor .. | .gadget .. => cap
        | .indexed _ _ source | .loopInstance _ _ source => maximum source
      let bound := maximum trapdoor.publicIdentity
      let result ← classifiedMatrixFact operation.ownerNode operation.outputPort matrixType
        operation.parameterEnvironment bound true
      pure { result with identity := some trapdoor.publicIdentity }.refreshPrimitivePolynomial
  pure (directPointwiseMatrixOutput operation.ownerScope operation.ownerNode operation.outputPort output)

def applyDirectMatrixToScalarOperation
    (operation : DirectValueScalarOperation)
    (matrix : OperationalMatrixFact) : Except OperationalError OperationalScalarFact := do
  let output ← match operation.kind with
  | .extractCoefficient position => do
      let position ← evaluateIntInvariant operation.parameterEnvironment [] position
      if position < 0 || position >= Int.ofNat matrix.matrixParams.ringDimension then
        throw (.invalidCount operation.ownerNode position)
      let upper := match matrix.canonicalRange with
        | .below upper => Int.ofNat upper
        | .unknown => matrix.matrixParams.modulus
      if upper <= 0 then throw (.invalidMatrixParameters operation.ownerNode)
      integerFact operation.ownerNode operation.outputPort 0 (upper - 1)
  | .thresholdDecodeBool ciphertextModulus plaintextModulus length => do
      let ciphertext ← evaluateIntInvariant operation.parameterEnvironment [] ciphertextModulus
      let plaintext ← evaluateIntInvariant operation.parameterEnvironment [] plaintextModulus
      let count ← evaluateIntInvariant operation.parameterEnvironment [] length
      if matrix.matrixParams.rows != 1 || matrix.matrixParams.columns != 1 ||
          ciphertext != matrix.matrixParams.modulus || plaintext <= 1 || count <= 0 ||
          count > Int.ofNat matrix.matrixParams.ringDimension then
        throw (.invalidMatrixParameters operation.ownerNode)
      pure .boolean
  | .thresholdDecodeInt ciphertextModulus plaintextModulus length => do
      let ciphertext ← evaluateIntInvariant operation.parameterEnvironment [] ciphertextModulus
      let plaintext ← evaluateIntInvariant operation.parameterEnvironment [] plaintextModulus
      let count ← evaluateIntInvariant operation.parameterEnvironment [] length
      if matrix.matrixParams.rows != 1 || matrix.matrixParams.columns != 1 ||
          ciphertext != matrix.matrixParams.modulus || plaintext <= 1 || count <= 0 ||
          count > Int.ofNat matrix.matrixParams.ringDimension then
        throw (.invalidMatrixParameters operation.ownerNode)
      integerFact operation.ownerNode operation.outputPort 0 (plaintext - 1)
  pure (directPointwiseScalarOutput operation.ownerScope operation.ownerNode operation.outputPort output)

def applyDirectScalarPointwiseOperation
    (operation : DirectScalarOperation)
    (arguments : Array OperationalScalarFact) : Except OperationalError OperationalScalarFact := do
  let output ← match operation.kind, arguments with
  | .boolToInt, #[.boolean] => integerFact operation.ownerNode operation.outputPort 0 1
  | .intBinary kind, #[.integer left, .integer right] => do
      let interval ← integerBinaryInterval operation.ownerNode kind left right
      integerFactWithExpressions operation.ownerNode operation.outputPort interval.lower interval.upper
        interval.lowerExpression interval.upperExpression
  | .intCompare _, #[.integer _, .integer _] => pure .boolean
  | .bitExtract position, #[.integer _] =>
      if position < 0 then throw (.invalidCount operation.ownerNode position) else pure .boolean
  | .intToReal, #[.integer _] => pure .real
  | .realBinary _, #[.real, .real] => pure .real
  | .realSqrt, #[.real] => pure .real
  | _, _ => throw (.unsupportedOperationalExpr operation.ownerNode)
  pure (directPointwiseScalarOutput operation.ownerScope operation.ownerNode operation.outputPort output)

def operationalProductTokens
    (term : OperationalTerm) : List OperationalCompressionToken :=
  [.productStart, .termStart term.coefficient] ++
    term.product.factors.flatMap (fun factor => match factor.leaf with
      | .primitive identity => [.primitive identity] ++
          factor.transforms.map OperationalCompressionToken.transform
      | .boundedSummary origin _ => origin.tokens
      | .exactTransform tokens _ => tokens) ++
    term.product.modes.map OperationalCompressionToken.productMode ++
    [.intermediateType term.product.outputType, .termEnd, .productEnd]

def tensorOperationalPolynomials
    (outputType : MatrixTypeExpr)
    (left right : OperationalPolynomial) : Except OperationalFlatError OperationalPolynomial := do
  let rows ← left.mapM fun leftTerm => right.mapM fun rightTerm => do
    let tokens := [.groupStart] ++ operationalProductTokens leftTerm ++ [.groupEnd,
      .groupStart] ++ operationalProductTokens rightTerm ++ [.groupEnd,
      .intermediateType outputType]
    let role := if operationalTermIsSignal leftTerm || operationalTermIsSignal rightTerm then
      OperationalFactorRole.large else .bounded
    let summary ← if role == OperationalFactorRole.bounded then do
      let leftSummary ← boundedNoiseTermSummary leftTerm
      let rightSummary ← boundedNoiseTermSummary rightTerm
      let ringFactor := if leftSummary.metadata.isConstantPolynomial ||
          rightSummary.metadata.isConstantPolynomial then .closedInt (.constant 1)
        else .closedInt outputType.ringDimension
      pure (some {
        matrixType := outputType
        rowCount := leftSummary.rowCount * rightSummary.rowCount
        hardBound := .multiply ringFactor
          (.multiply leftSummary.hardBound rightSummary.hardBound)
        metadata := {
          isConstantPolynomial := leftSummary.metadata.isConstantPolynomial &&
            rightSummary.metadata.isConstantPolynomial
          knownZeroRows := none
        }
        provenance := tokens
      })
    else pure none
    let leaf := match summary with
      | some bounded =>
          let origin : OperationalCompressionOrigin := { kind := .boundedRun, tokens }
          OperationalFactorLeaf.boundedSummary origin bounded
      | none => .exactTransform tokens outputType
    let factor : OperationalFactorKey := {
      leaf
      inputType := outputType
      outputType
      role
      boundedSummary := summary
    }
    pure {
      coefficient := leftTerm.coefficient * rightTerm.coefficient
      product := { factors := [factor], modes := [], outputType }
    }
  pure (normalizeOperationalTerms rows.flatten)

def tensorConcreteMatrixFacts
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (left right : OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  let polynomial ← tensorOperationalPolynomials matrixType left.polynomial right.polynomial
    |>.mapError (flatErrorAt nodeIndex)
  polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial


def transposeOperationalMatrixType (type : MatrixTypeExpr) : MatrixTypeExpr := {
  type with rows := type.columns, columns := type.rows
}

/-- A factor-level transform has only its declared output type, not the enclosing evaluated matrix
parameters.  Its sparse-row certificate is always cleared; when the output row expression is not
locally concrete, retain no usable row witness rather than carrying the old dimension across a
shape-changing transform. -/
def transformedOperationalRowCount (outputType : MatrixTypeExpr) : Int :=
  match normalizeOperationalDimension outputType.rows with
  | .constant rows => rows
  | _ => 0

def transformOperationalFactor
    (transform : OperationalFactorTransform)
    (outputType : MatrixTypeExpr)
    (factor : OperationalFactorKey) : OperationalFactorKey :=
  let transformSummary (summary : OperationalBoundedFactorSummary) := {
    summary with
    matrixType := outputType
    rowCount := transformedOperationalRowCount outputType
    metadata := { summary.metadata with knownZeroRows := none }
    provenance := summary.provenance ++ [.transform transform, .intermediateType outputType]
  }
  let leaf := match factor.leaf with
    | .boundedSummary origin summary =>
        let tokens := origin.tokens ++ [.transform transform, .intermediateType outputType]
        .boundedSummary { origin with tokens } (transformSummary summary)
    | .exactTransform tokens _ =>
        .exactTransform (tokens ++ [.transform transform, .intermediateType outputType]) outputType
    | primitive => primitive
  {
    factor with
    leaf
    transforms := factor.transforms ++ [transform]
    inputType := outputType
    outputType
    boundedSummary := factor.boundedSummary.map transformSummary
    protections := factor.protections.filter fun protection => match protection with
      | .relationOwner | .decompositionOwner => false
      | _ => true
    relations := []
  }

def replaceOperationalFactorAt
    (index : Nat)
    (replacement : OperationalFactorKey)
    (factors : List OperationalFactorKey) : List OperationalFactorKey :=
  let rec visit : Nat → List OperationalFactorKey → List OperationalFactorKey
    | _, [] => []
    | 0, _ :: tail => replacement :: tail
    | remaining + 1, head :: tail => head :: visit remaining tail
  visit index factors

def rowBoundaryIndex (product : OperationalProductKey) : Nat :=
  let rec visit : Nat → List OperationalProductMode → Nat
    | index, .leftPolynomialScalarBroadcast :: tail => visit (index + 1) tail
    | index, _ => index
  visit 0 product.modes

def columnBoundaryIndex (product : OperationalProductKey) : Nat :=
  let rec skipRightScalars : Nat → List OperationalProductMode → Nat
    | index, [] => index
    | index, .rightPolynomialScalarBroadcast :: tail => skipRightScalars (index - 1) tail
    | index, _ => index
  skipRightScalars (product.factors.length - 1) product.modes.reverse

def transformOperationalBoundary
    (axis : ConcatAxis)
    (part : Nat)
    (outputType : MatrixTypeExpr)
    (term : OperationalTerm) : Except OperationalFlatError OperationalTerm := do
  let applyAt (index : Nat) (transform : OperationalFactorTransform)
      (changeType : MatrixTypeExpr → MatrixTypeExpr) :
      Except OperationalFlatError OperationalProductKey := do
    let factor ← match term.product.factors[index]? with
      | some factor => pure factor
      | none => throw OperationalFlatError.malformedProduct
    let replacement := transformOperationalFactor transform (changeType factor.outputType) factor
    operationalProductFromFactors
      (replaceOperationalFactorAt index replacement term.product.factors)
  let product ← match axis with
    | .rows => applyAt (rowBoundaryIndex term.product) (.rowEmbed .rows part) (fun matrixType =>
        { matrixType with rows := outputType.rows })
    | .columns => applyAt (columnBoundaryIndex term.product) (.columnEmbed .columns part) (fun matrixType =>
        { matrixType with columns := outputType.columns })
    | .diagonal => do
        let rowProduct ← applyAt (rowBoundaryIndex term.product)
          (.rowEmbed .diagonal part)
          (fun matrixType => { matrixType with rows := outputType.rows })
        let index := columnBoundaryIndex rowProduct
        let factor ← match rowProduct.factors[index]? with
          | some factor => pure factor
          | none => throw .malformedProduct
        let replacement := transformOperationalFactor (.columnEmbed .diagonal part)
          { factor.outputType with columns := outputType.columns } factor
        operationalProductFromFactors (replaceOperationalFactorAt index replacement rowProduct.factors)
  pure { term with product }

def concatOperationalPolynomials
    (axis : ConcatAxis)
    (outputType : MatrixTypeExpr)
    (inputs : List OperationalPolynomial) : Except OperationalFlatError OperationalPolynomial := do
  let rows ← inputs.zipIdx.mapM fun (terms, part) =>
    terms.mapM (transformOperationalBoundary axis part outputType)
  pure (normalizeOperationalTerms rows.flatten)

def concatCanonicalRange (inputs : Array OperationalMatrixFact) : CanonicalRange :=
  if inputs.all (fun input => match input.canonicalRange with
      | .below _ => true
      | .unknown => false) then
    .below (inputs.foldl (fun result input => match input.canonicalRange with
      | .below value => max result value
      | .unknown => result) 0)
  else .unknown

/-- Negation preserves a canonical coefficient interval only for the exact-zero interval.
Ordinary canonical representatives are not closed under additive inversion: for example, the
negation of a coefficient in `[0, 2)` modulo `17` can be `16`. -/
def negateCanonicalRange : CanonicalRange → CanonicalRange
  | .below upper => if upper <= 1 then .below upper else .unknown
  | .unknown => .unknown

/-- Scaling preserves a canonical coefficient interval only for a provably identity scalar, or
for the exact-zero interval.  Other scalars require a separately proved modular range transfer. -/
def scaleCanonicalRange (scalarValues : List Int) : CanonicalRange → CanonicalRange
  | .below upper =>
      if upper <= 1 || (!scalarValues.isEmpty && scalarValues.all (· == 1)) then .below upper
      else .unknown
  | .unknown => .unknown

def concatConcreteMatrixFacts
    (nodeIndex outputPort : Nat)
    (axis : ConcatAxis)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (inputs : Array OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  let polynomial ← concatOperationalPolynomials axis matrixType
    (inputs.toList.map (·.polynomial)) |>.mapError (flatErrorAt nodeIndex)
  let result ← polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
    (concatCanonicalRange inputs)
  pure { result with blockLayout := some { axis, partitions := inputs.map fun input => {
    matrixType := input.matrixType, polynomial := input.polynomial } } }

def transposeOperationalPolynomial
    (terms : OperationalPolynomial) : Except OperationalFlatError OperationalPolynomial := do
  let terms ← terms.mapM fun term => do
    let factors := term.product.factors.reverse.map fun factor =>
      transformOperationalFactor .transpose (transposeOperationalMatrixType factor.outputType) factor
    let product ← operationalProductFromFactors factors
    pure { term with product }
  pure (normalizeOperationalTerms terms)

def sliceOperationalPolynomial
    (rows columns : Option (IntExpr × IntExpr))
    (outputType : MatrixTypeExpr)
    (terms : OperationalPolynomial) : Except OperationalFlatError OperationalPolynomial := do
  let terms ← terms.mapM fun term => do
    let term ← match rows with
      | none => pure term
      | some (start, stop) => do
          let index := rowBoundaryIndex term.product
          let factor ← match term.product.factors[index]? with
            | some factor => pure factor
            | none => throw .malformedProduct
          let replacement := transformOperationalFactor (.rowSlice start stop)
            { factor.outputType with rows := outputType.rows } factor
          let product ← operationalProductFromFactors
            (replaceOperationalFactorAt index replacement term.product.factors)
          pure { term with product }
    match columns with
    | none => pure term
    | some (start, stop) => do
        let index := columnBoundaryIndex term.product
        let factor ← match term.product.factors[index]? with
          | some factor => pure factor
          | none => throw .malformedProduct
        let replacement := transformOperationalFactor (.columnSlice start stop)
          { factor.outputType with columns := outputType.columns } factor
        let product ← operationalProductFromFactors
          (replaceOperationalFactorAt index replacement term.product.factors)
        pure { term with product }
  pure (normalizeOperationalTerms terms)

def boundedStructuralTransformPolynomial
    (transform : OperationalFactorTransform)
    (outputType : MatrixTypeExpr)
    (input : OperationalPolynomial) : Except OperationalFlatError OperationalPolynomial := do
  if input.any operationalTermIsSignal then throw .cannotPreserveNoiseSeparation
  let compressed ← compressBoundedNoiseSum input
  compressed.mapM fun term => do
    match term.product.factors with
    | [factor] =>
        let transformed := transformOperationalFactor transform outputType factor
        pure { term with product := {
          factors := [transformed]
          modes := []
          outputType
        }}
    | _ => throw .malformedProduct

def transformConcreteMatrixFact
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (operation : OperationalFactorTransform)
    (environment : ParamEnvironment)
    (input : OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  let polynomial ← match operation with
    | .negate =>
        let input ← retypeMatrixFact nodeIndex matrixType input environment
        pure (scaleOperationalPolynomial (-1) input.polynomial)
    | .transpose =>
        transposeOperationalPolynomial input.polynomial |>.mapError (flatErrorAt nodeIndex)
    | .rowSlice start stop =>
        sliceOperationalPolynomial (some (start, stop)) none matrixType input.polynomial
          |>.mapError (flatErrorAt nodeIndex)
    | .columnSlice start stop =>
        sliceOperationalPolynomial none (some (start, stop)) matrixType input.polynomial
          |>.mapError (flatErrorAt nodeIndex)
    | .rowEmbed axis part | .columnEmbed axis part =>
        input.polynomial.mapM (transformOperationalBoundary axis part matrixType)
          |>.mapError (flatErrorAt nodeIndex)
  let canonicalRange := match operation with
    | .negate => negateCanonicalRange input.canonicalRange
    | .transpose | .rowSlice _ _ | .columnSlice _ _ | .rowEmbed _ _ | .columnEmbed _ _ =>
        input.canonicalRange
  polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial canonicalRange


def parameterScalarPolynomial
    (environment : ParamEnvironment)
    (domains : List OperationalParameterDomain)
    (value : IntExpr)
    (matrixType : MatrixTypeExpr) : OperationalPolynomial :=
  let scalarType : MatrixTypeExpr := {
    modulus := matrixType.modulus
    ringDimension := matrixType.ringDimension
    rows := .constant 1
    columns := .constant 1
  }
  let identity := OperationalPrimitiveIdentity.parameterScalar environment domains value
  let metadata : OperationalMatrixMetadata := { isConstantPolynomial := true }
  let summary : OperationalBoundedFactorSummary := {
    matrixType := scalarType
    rowCount := 1
    hardBound := .contextual .maximumAbsolute environment domains value
    metadata
    provenance := [.primitive identity]
  }
  [{
    coefficient := 1
    product := {
      factors := [{
        leaf := .primitive identity
        inputType := scalarType
        outputType := scalarType
        role := .bounded
        boundedSummary := some summary
      }]
      modes := []
      outputType := scalarType
    }
  }]



def scaleConcreteMatrixFact
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (scalar : IntExpr)
    (scalarValues : List Int)
    (environment : ParamEnvironment)
    (loopDomains : List OperationalParameterDomain)
    (input : OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  let input ← retypeMatrixFact nodeIndex matrixType input environment
  let first ← match scalarValues with
    | first :: _ => pure first
    | [] => throw (.invalidMatrixParameters nodeIndex)
  let polynomial ←
    if scalarValues.all (· == first) then
      pure (scaleOperationalPolynomial first input.polynomial)
    else
      multiplyOperationalPolynomials
        (parameterScalarPolynomial environment loopDomains scalar matrixType)
        input.polynomial |>.mapError (flatErrorAt nodeIndex)
  polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
    (scaleCanonicalRange scalarValues input.canonicalRange)

def directBggProductTokens
    (term : OperationalTerm) : List OperationalCompressionToken :=
  [.productStart, .termStart term.coefficient] ++
    term.product.factors.flatMap (fun factor => match factor.leaf with
      | .primitive identity => [.primitive identity] ++
          factor.transforms.map OperationalCompressionToken.transform
      | .boundedSummary origin _ => origin.tokens
      | .exactTransform tokens _ => tokens) ++
    term.product.modes.map OperationalCompressionToken.productMode ++
    [.intermediateType term.product.outputType, .termEnd, .productEnd]

def directGroupBggEncodingSignal
    (vector publicKey plaintext : OperationalMatrixFact) :
    Except OperationalFlatError OperationalMatrixFact := do
  let signal := sortOperationalTerms (vector.polynomial.filter operationalTermIsSignal)
  let noise := vector.polynomial.filter operationalTermIsNoise
  if signal.isEmpty then return { vector with polynomial := (← compressBoundedNoiseSum noise) }
  let tokens := [.groupStart, .primitive (.matrix publicKey.origin),
    .primitive (.matrix plaintext.origin), .sumStart] ++
    signal.flatMap directBggProductTokens ++
    [.sumEnd, .intermediateType vector.matrixType, .groupEnd]
  let factor : OperationalFactorKey := {
    leaf := .exactTransform tokens vector.matrixType
    inputType := vector.matrixType
    outputType := vector.matrixType
    role := .large
  }
  let groupedSignal : OperationalTerm := {
    coefficient := 1
    product := { factors := [factor], modes := [], outputType := vector.matrixType }
  }
  let compressedNoise ← compressBoundedNoiseSum noise
  pure { vector with polynomial := groupedSignal :: compressedNoise }

def applyDirectMatrixPointwiseOperation
    (operation : PrimitiveOperation)
    (matrixType : MatrixTypeExpr)
    (arguments : Array OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  if operation.outputSchema != matrixType then throw (.unsupportedOperationalExpr operation.ownerNode)
  let output ← match operation.kind with
  | .add subtract => do
      let left ← match arguments[0]? with
        | some fact => pure fact | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let right ← match arguments[1]? with
        | some fact => pure fact | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      if arguments.size != 2 then throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      addConcreteMatrixFacts operation.ownerNode operation.outputPort matrixType subtract
        operation.parameterEnvironment left right
  | .multiply rule rightWire => do
      let left ← match arguments[0]? with
        | some fact => pure fact | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let right ← match arguments[1]? with
        | some fact => pure fact | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      if arguments.size != 2 then throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      multiplyConcreteMatrixFacts operation.ownerNode operation.outputPort matrixType rule rightWire
        operation.parameterEnvironment left right
  | .tensor => do
      let left ← match arguments[0]? with
        | some fact => pure fact | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let right ← match arguments[1]? with
        | some fact => pure fact | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      if arguments.size != 2 then throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      tensorConcreteMatrixFacts operation.ownerNode operation.outputPort matrixType
        operation.parameterEnvironment left right
  | .concat axis =>
      concatConcreteMatrixFacts operation.ownerNode operation.outputPort axis matrixType
        operation.parameterEnvironment arguments
  | .transform transform => do
      let input ← match arguments with
        | #[input] => pure input
        | _ => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      transformConcreteMatrixFact operation.ownerNode operation.outputPort matrixType transform
        operation.parameterEnvironment input
  | .slice rows columns => do
      let input ← match arguments with
        | #[input] => pure input
        | _ => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let polynomial ← sliceOperationalPolynomial rows columns matrixType input.polynomial
        |>.mapError (flatErrorAt operation.ownerNode)
      polynomialMatrixFact operation.ownerNode operation.outputPort matrixType
        operation.parameterEnvironment polynomial input.canonicalRange
  | .scale scalar loopDomains => do
      let input ← match arguments with
        | #[input] => pure input
        | _ => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let scalar ← match scalar with
        | .ir value => pure value | _ => throw (.unsupportedOperationalExpr operation.ownerNode)
      let loopDomains : List OperationalParameterDomain ← match loopDomains with
        | [] => pure [] | _ => throw (.unsupportedOperationalExpr operation.ownerNode)
      let value ← evaluateIntInvariant operation.parameterEnvironment loopDomains scalar
      scaleConcreteMatrixFact operation.ownerNode operation.outputPort matrixType scalar [value]
        operation.parameterEnvironment loopDomains input
  | .bggGrouping => do
      let vector ← match arguments[0]? with
        | some fact => pure fact | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let publicKey ← match arguments[1]? with
        | some fact => pure fact | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let plaintext ← match arguments[2]? with
        | some fact => pure fact | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      if arguments.size != 3 then throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      directGroupBggEncodingSignal vector publicKey plaintext |>.mapError (.flat operation.ownerNode)
  pure (directPointwiseMatrixOutput operation.ownerScope operation.ownerNode operation.outputPort output)

/-- The structural direct reducer uses this instrumented twin.  It is intentionally private to
reduction telemetry: fixed-assignment queries and acceptance retain the fact-only dispatcher. -/
private def applyDirectMatrixPointwiseOperationWithRelationRewriteCount
    (operation : PrimitiveOperation)
    (matrixType : MatrixTypeExpr)
    (arguments : Array OperationalMatrixFact) : Except OperationalError (OperationalMatrixFact × Nat) := do
  if operation.outputSchema != matrixType then throw (.unsupportedOperationalExpr operation.ownerNode)
  match operation.kind with
  | .multiply rule rightWire => do
      let left ← match arguments[0]? with
        | some fact => pure fact | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let right ← match arguments[1]? with
        | some fact => pure fact | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      if arguments.size != 2 then throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let (output, rewriteCount) ← multiplyConcreteMatrixFactsWithRelationRewriteCount
        operation.ownerNode operation.outputPort matrixType rule rightWire operation.parameterEnvironment left right
      pure (directPointwiseMatrixOutput operation.ownerScope operation.ownerNode operation.outputPort output,
        rewriteCount)
  | _ => pure (← applyDirectMatrixPointwiseOperation operation matrixType arguments, 0)

/-- Recover the capture-free maps beneath a delayed direct annotation.  Fixed-assignment lookup
needs the same transport as structural reduction when a result-bound annotation overwrites a
source that is still represented by lazy mapped views. -/
private def directPendingMaps
    (arena : DirectOperationalIndexedArena) (id : OperationalIndexedValueId) : Nat →
    Except OperationalError (List IndexMap)
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => do
      let value ← match arena.valueAt? id with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef id)
      match value.payload with
      | .mapped _ source map =>
          if !map.transportValid || map.destination != value.context then
            throw (.unsupportedOperationalExpr id)
          return (← directPendingMaps arena source fuel) ++ [map]
      | .rebound _ source _ | .matrixResultBound _ source _ =>
          directPendingMaps arena source fuel
      | _ => pure []

/-- Resolve a gather descriptor only through an exact fixed integer table.  This deliberately
does not invoke the raw carrier evaluator, so numeric consumers can materialize gathered bounds
inside the fixed-assignment mutual evaluator without creating a recursive evaluator cycle. -/
private def exactGatherSelections
    (arena : DirectOperationalIndexedArena) (root : OperationalIndexedValueId) : Nat →
    Option (List (IndexExpr × Nat × Int))
  | 0 => none
  | fuel + 1 => do
      let value ← arena.valueAt? root
      match value.payload with
      | .shared (.scalar .integer) (.scalar reference) => do
          match arena.fixed.scalars[reference]? with
          | some (.integer fact) => if fact.lower == fact.upper then some [(.constant 0, 0, fact.lower)] else none
          | _ => none
      | .explicit (.scalar .integer) binder references => references.toList.mapIdxM fun lane reference => do
          match reference with
          | .scalar reference => match arena.fixed.scalars[reference]? with
              | some (.integer fact) =>
                  if fact.lower == fact.upper then some (.variable binder, lane, fact.lower) else none
              | _ => none
          | .matrix _ => none
      | .explicitValues (.scalar .integer) binder values => values.toList.mapIdxM fun lane child => do
          match ← exactGatherSelections arena child fuel with
          | [(_, _, selected)] => some (.variable binder, lane, selected)
          | _ => none
      | .mapped (.scalar .integer) source map => do
          if !map.transportValid || map.destination != value.context then none else do
            let entries ← exactGatherSelections arena source fuel
            entries.mapM fun (key, lane, selected) => return (← reindex map key, lane, selected)
      | .rebound (.scalar .integer) source _ => exactGatherSelections arena source fuel
      | _ => none

private def exactGatherIndex
    (arena : DirectOperationalIndexedArena) (parameters : ParamEnvironment) (context : IndexContext)
    (indices : IndexValueEnvironment) : IndexExpr → Nat → Except OperationalError Int
  | _, 0 => throw (.unsupportedOperationalExpr arena.values.size)
  | .constant value, _ => pure value
  | .variable binder, _ => match evaluateIndexExpr parameters context indices (.variable binder) with
      | some value => pure value | none => throw (.unsupportedOperationalExpr arena.values.size)
  | .offset base amount, fuel + 1 => return (← exactGatherIndex arena parameters context indices base fuel) + amount
  | .gather owner count position, fuel + 1 => do
      let count ← match count.evaluate parameters with | some count => pure count | none => throw .nonClosedExpression
      if count <= 0 then throw (.invalidCount owner.indices.node count)
      let position ← exactGatherIndex arena parameters context indices position fuel
      if position < 0 then throw (.invalidCount owner.indices.node position)
      let registered ← match arena.gatherIntegerRoot? owner with
        | some registered => pure registered | none => throw (.unsupportedOperationalExpr owner.indices.node)
      let root := registered.root
      let rootValue ← match arena.valueAt? root with
        | some value => pure value | none => throw (.invalidOperationalExprRef root)
      let binder := registered.position
      let entries ← match exactGatherSelections arena root fuel with
        | some entries => pure entries | none => throw (.unsupportedOperationalExpr root)
      /- A mapped executable integer table retains its physical source lane separately from the
      transported selection key.  At lookup position `p`, the root's sole destination binder is
      assigned `p`; exactly one recovered key must then select its recorded physical lane.  Do
      not equate `p` with that lane directly: an offset or nested map changes the key while the
      physical table remains unchanged. -/
      let matchedEntries ← entries.filterMapM fun (key, physicalLane, selected) => do
        let lane ← exactGatherIndex arena parameters rootValue.context [(.variable binder, position)] key fuel
        if lane == physicalLane then pure (some selected) else pure none
      let selected ← match matchedEntries with
        | [selected] => pure selected
        | [] => throw (.invalidCount owner.indices.node position)
        | _ => throw (.unsupportedOperationalExpr owner.indices.node)
      if selected < 0 || selected >= count then throw (.invalidCount owner.indices.node selected)
      pure selected

private def materializeDirectScalarFact
    (arena : DirectOperationalIndexedArena) (parameters : ParamEnvironment) (context : IndexContext)
    (indices : IndexValueEnvironment) (fact : OperationalScalarFact) : Except OperationalError OperationalScalarFact := do
  let rec evaluate : IndexedParameterExpr → Except OperationalError Int
    | .ir value => match value.evaluate parameters with | some value => pure value | none => throw .nonClosedExpression
    | .index value => exactGatherIndex arena parameters context indices value (arena.values.size + 1)
    | .add left right => return (← evaluate left) + (← evaluate right)
    | .subtract left right => return (← evaluate left) - (← evaluate right)
    | .multiply left right => return (← evaluate left) * (← evaluate right)
    | .divide left right => do let right ← evaluate right; if right = 0 then throw .nonClosedExpression else return (← evaluate left) / right
    | .roundDivide left right => do let right ← evaluate right; if right = 0 then throw .nonClosedExpression else return Mxx.Ir.roundDiv (← evaluate left) right
    | .log2Ceil value => return Mxx.Ir.log2Ceil (← evaluate value)
  let close (bound : IndexedOperationalBoundExpr) := bound.materializeWith (fun _ _ _ value => evaluate value) parameters context indices
  match fact with
  | .integer fact => pure (.integer { fact with lowerExpression := .closed (← close fact.lowerExpression), upperExpression := .closed (← close fact.upperExpression) })
  | .trapdoor fact => pure (.trapdoor { fact with maximum := .closed (← close fact.maximum), preimageCutoff := (← fact.preimageCutoff.mapM (fun bound => return .closed (← close bound))) })
  | fact => pure fact

/-- Close one relation descriptor at the same fixed-assignment boundary used by scalar bounds.
The evaluator is arena-aware, so `.index` retains gathered owner semantics until this point. -/
private def materializeDirectRelationOperation
    (arena : DirectOperationalIndexedArena)
    (parameters : ParamEnvironment)
    (context : IndexContext)
    (indices : IndexValueEnvironment)
    (operation : DirectRelationOperation) : Except OperationalError DirectRelationOperation := do
  let rec evaluateAt (environment : ParamEnvironment) : IndexedParameterExpr → Except OperationalError Int
    | .ir value => match value.evaluate environment with
        | some value => pure value | none => throw .nonClosedExpression
    | .index value => exactGatherIndex arena environment context indices value (arena.values.size + 1)
    | .add left right => return (← evaluateAt environment left) + (← evaluateAt environment right)
    | .subtract left right => return (← evaluateAt environment left) - (← evaluateAt environment right)
    | .multiply left right => return (← evaluateAt environment left) * (← evaluateAt environment right)
    | .divide left right => do
        let right ← evaluateAt environment right
        if right = 0 then throw .nonClosedExpression else return (← evaluateAt environment left) / right
    | .roundDivide left right => do
        let right ← evaluateAt environment right
        if right = 0 then throw .nonClosedExpression else return Mxx.Ir.roundDiv (← evaluateAt environment left) right
    | .log2Ceil value => return Mxx.Ir.log2Ceil (← evaluateAt environment value)
  let materializeType (value : IndexedMatrixTypeExpr) : Except OperationalError MatrixTypeExpr := do
    let modulusValue ← evaluateAt parameters value.modulus
    let ringDimensionValue ← evaluateAt parameters value.ringDimension
    let rowCount ← evaluateAt parameters value.rows
    let columnCount ← evaluateAt parameters value.columns
    let result : MatrixTypeExpr := {
      modulus := .constant modulusValue
      ringDimension := .constant ringDimensionValue
      rows := .constant rowCount
      columns := .constant columnCount
    }
    pure result
  let rec materializeDomains (environment : ParamEnvironment) :
      List IndexedOperationalParameterDomain → Except OperationalError Unit
    | [] => pure ()
    | .loopIndex binder :: remaining => do
        if !context.binders.contains binder then throw (.unsupportedOperationalExpr operation.ownerNode)
        let count ← match binder.count.evaluate environment with
          | some value => pure value | none => throw .nonClosedExpression
        let lane ← exactGatherIndex arena environment context indices (.variable binder)
          (arena.values.size + 1)
        if count <= 0 || lane < 0 || lane >= count then throw (.invalidCount operation.ownerNode lane)
        materializeDomains environment remaining
    | .parameter _ environment domains expression :: remaining => do
        materializeDomains environment domains
        let _ ← evaluateAt environment expression
        materializeDomains environment remaining
  let outputType ← materializeType operation.outputType
  let kind ← match operation.kind with
  | .preimage maximum domains => do
      materializeDomains parameters domains
      let maximum ← evaluateAt parameters maximum
      if maximum < 0 then throw (.invalidBound operation.ownerNode maximum)
      pure (.preimage (.ir (.constant maximum)) [])
  | .decomposition declaredType base small digitCount domains layouts => do
      materializeDomains parameters domains
      let declaredType ← materializeType declaredType
      let base ← evaluateAt parameters base
      let digitCount ← evaluateAt parameters digitCount
      pure (.decomposition (.fromIr declaredType) (.ir (.constant base)) small
        (.ir (.constant digitCount)) [] layouts)
  let materialized : DirectRelationOperation := {
    kind := kind
    outputType := .fromIr outputType
    outputSchema := outputType
    ownerScope := operation.ownerScope
    ownerNode := operation.ownerNode
    outputPort := operation.outputPort
    parameterEnvironment := operation.parameterEnvironment
  }
  pure materialized

mutual

/-- Evaluate an owner-bearing direct index expression.  Unlike the generic indexed-facts helper,
this resolves a gather through the arena's unique executable integer producer at the correlated
position.  The producer's physical lane is selected by the position expression, while its
integer result is the exact source-family ordinal. -/
def DirectOperationalIndexedArena.indexExprAt
    (arena : DirectOperationalIndexedArena)
    (parameters : ParamEnvironment)
    (context : IndexContext)
    (indices : IndexValueEnvironment)
    (expression : IndexExpr) : Nat → Except OperationalError Int
  | 0 => throw (.unsupportedOperationalExpr arena.values.size)
  | fuel + 1 => do
      if !validateContext context || !indexExpressionInBounds context expression then
        throw (.unsupportedOperationalExpr arena.values.size)
      match expression with
      | .constant lane => pure lane
      | .variable binder => match evaluateIndexExpr parameters context indices (.variable binder) with
          | some lane => pure lane
          | none => throw (.unsupportedOperationalExpr arena.values.size)
      | .offset base amount => return (← arena.indexExprAt parameters context indices base fuel) + amount
      | .gather owner sourceCount position => do
          let sourceBound ← match sourceCount.evaluate parameters with
            | some bound => pure bound
            | none => throw .nonClosedExpression
          if sourceBound <= 0 then throw (.invalidCount owner.indices.node sourceBound)
          let position ← arena.indexExprAt parameters context indices position fuel
          let registered ← match arena.gatherIntegerRoot? owner with
            | some registered => pure registered
            | none => throw (.unsupportedOperationalExpr owner.indices.node)
          let root := registered.root
          let rootValue ← match arena.valueAt? root with
            | some value => pure value
            | none => throw (.invalidOperationalExprRef root)
          let positionBinder := registered.position
          let selected ← arena.scalarFactAt parameters [(.variable positionBinder, position)] root fuel
          let ordinal ← match selected with
            | .integer fact => pure fact
            | _ =>
                let wire : WireRef := { node := owner.indices.node, port := owner.indices.port }
                throw (.operandNotInteger owner.indices.node wire)
          if ordinal.lower != ordinal.upper || ordinal.lower < 0 || ordinal.lower >= sourceBound then
            throw (.invalidCount owner.indices.node ordinal.upper)
          pure ordinal.lower

/-- Evaluate a direct indexed parameter at one complete fixed assignment.  All `.index` leaves
use `indexExprAt`, so a gathered descriptor is resolved through the unique registered scalar
producer rather than being projected to a slot-only Graph-IR expression. -/
def DirectOperationalIndexedArena.indexedParameterAt
    (arena : DirectOperationalIndexedArena)
    (parameters : ParamEnvironment)
    (context : IndexContext)
    (indices : IndexValueEnvironment)
    (value : IndexedParameterExpr) : Except OperationalError Int :=
  let rec visit : IndexedParameterExpr → Except OperationalError Int
    | .ir value => match value.evaluate parameters with
        | some value => pure value
        | none => throw .nonClosedExpression
    | .index value => arena.indexExprAt parameters context indices value (arena.values.size + 1)
    | .add left right => return (← visit left) + (← visit right)
    | .subtract left right => return (← visit left) - (← visit right)
    | .multiply left right => return (← visit left) * (← visit right)
    | .divide left right => do
        let denominator ← visit right
        if denominator = 0 then throw .nonClosedExpression else return (← visit left) / denominator
    | .roundDivide left right => do
        let denominator ← visit right
        if denominator = 0 then throw .nonClosedExpression
        else return Mxx.Ir.roundDiv (← visit left) denominator
    | .log2Ceil value => return Mxx.Ir.log2Ceil (← visit value)
  visit value

def DirectOperationalIndexedArena.indexedMatrixTypeAt
    (arena : DirectOperationalIndexedArena)
    (parameters : ParamEnvironment)
    (context : IndexContext)
    (indices : IndexValueEnvironment)
    (value : IndexedMatrixTypeExpr) : Except OperationalError MatrixTypeExpr := do
  pure {
    modulus := .constant (← arena.indexedParameterAt parameters context indices value.modulus)
    ringDimension := .constant (← arena.indexedParameterAt parameters context indices value.ringDimension)
    rows := .constant (← arena.indexedParameterAt parameters context indices value.rows)
    columns := .constant (← arena.indexedParameterAt parameters context indices value.columns)
  }

/-- Indexing a closed uniform loop output happens while the direct evaluator is already reducing
its source.  These local transport helpers deliberately mirror the later family-selection
transport, but live before the recursive evaluator so the overlay stays executable. -/
private def overlayMapPrimitiveIdentity
    (mapOrigin : MatrixOriginIdentity → MatrixOriginIdentity)
    (mapPublic : PublicMatrixIdentity → PublicMatrixIdentity)
    (mapValue : OperationalValueOrigin → OperationalValueOrigin) :
    OperationalPrimitiveIdentity → OperationalPrimitiveIdentity
  | .matrix identity => .matrix (mapOrigin identity)
  | .publicMatrix identity => .publicMatrix (mapPublic identity)
  | .value identity => .value (mapValue identity)
  | .parameterScalar environment domains value => .parameterScalar environment domains value
  | .identityMatrix type => .identityMatrix type
  | .indexedArtifact input index => .indexedArtifact input index
  | .recurrenceResult scope node path => .recurrenceResult scope node path
  | .carriedInput path => .carriedInput path

private def overlayMapCompressionToken
    (mapOrigin : MatrixOriginIdentity → MatrixOriginIdentity)
    (mapPublic : PublicMatrixIdentity → PublicMatrixIdentity)
    (mapValue : OperationalValueOrigin → OperationalValueOrigin)
    (mapBound : OperationalBoundExpr → OperationalBoundExpr) :
    OperationalCompressionToken → OperationalCompressionToken
  | .primitive identity => .primitive (overlayMapPrimitiveIdentity mapOrigin mapPublic mapValue identity)
  | .summaryBound bound => .summaryBound (mapBound bound)
  | token => token

private def overlayMapBoundedSummary
    (mapOrigin : MatrixOriginIdentity → MatrixOriginIdentity)
    (mapPublic : PublicMatrixIdentity → PublicMatrixIdentity)
    (mapValue : OperationalValueOrigin → OperationalValueOrigin)
    (mapBound : OperationalBoundExpr → OperationalBoundExpr)
    (summary : OperationalBoundedFactorSummary) : OperationalBoundedFactorSummary := {
  summary with
  hardBound := mapBound summary.hardBound
  provenance := summary.provenance.map (overlayMapCompressionToken mapOrigin mapPublic mapValue mapBound)
}

private def overlayMapRelationSnapshotPolynomial
    (mapOrigin : MatrixOriginIdentity → MatrixOriginIdentity)
    (mapPublic : PublicMatrixIdentity → PublicMatrixIdentity)
    (mapValue : OperationalValueOrigin → OperationalValueOrigin)
    (mapBound : OperationalBoundExpr → OperationalBoundExpr)
    (polynomial : RelationSnapshotPolynomial) : RelationSnapshotPolynomial :=
  polynomial.map fun term => {
    term with
    product := {
      term.product with
      factors := term.product.factors.map fun factor =>
      let boundedSummary := factor.boundedSummary.map
        (overlayMapBoundedSummary mapOrigin mapPublic mapValue mapBound)
      let leaf := match factor.leaf with
        | .primitive identity => .primitive (overlayMapPrimitiveIdentity mapOrigin mapPublic mapValue identity)
        | .boundedSummary origin summary =>
            let tokens := origin.tokens.map
              (overlayMapCompressionToken mapOrigin mapPublic mapValue mapBound)
            .boundedSummary { origin with tokens }
              (overlayMapBoundedSummary mapOrigin mapPublic mapValue mapBound summary)
        | .exactTransform tokens type =>
            let tokens := tokens.map (overlayMapCompressionToken mapOrigin mapPublic mapValue mapBound)
            .exactTransform tokens type
        { factor with leaf, boundedSummary }
    }
  }

private def overlayMapOperationalPolynomial
    (mapOrigin : MatrixOriginIdentity → MatrixOriginIdentity)
    (mapPublic : PublicMatrixIdentity → PublicMatrixIdentity)
    (mapValue : OperationalValueOrigin → OperationalValueOrigin)
    (mapBound : OperationalBoundExpr → OperationalBoundExpr)
    (mapRelation : OperationalMatrixRelation → OperationalMatrixRelation)
    (polynomial : OperationalPolynomial) : OperationalPolynomial :=
  polynomial.map fun term => {
    term with
    product := {
      term.product with
      factors := term.product.factors.map fun factor =>
      let boundedSummary := factor.boundedSummary.map
        (overlayMapBoundedSummary mapOrigin mapPublic mapValue mapBound)
      let leaf := match factor.leaf with
        | .primitive identity => .primitive (overlayMapPrimitiveIdentity mapOrigin mapPublic mapValue identity)
        | .boundedSummary origin summary =>
            let tokens := origin.tokens.map
              (overlayMapCompressionToken mapOrigin mapPublic mapValue mapBound)
            .boundedSummary { origin with tokens }
              (overlayMapBoundedSummary mapOrigin mapPublic mapValue mapBound summary)
        | .exactTransform tokens type =>
            let tokens := tokens.map (overlayMapCompressionToken mapOrigin mapPublic mapValue mapBound)
            .exactTransform tokens type
        { factor with leaf, boundedSummary, relations := factor.relations.map mapRelation }
    }
  }

private def overlayIndexMatrixFact
    (binder : FamilyTemplateBinder) (selection : DynamicSelectionIdentity) (subject : WireRef)
    (fact : OperationalMatrixFact) : OperationalMatrixFact :=
  let mapOrigin (origin : MatrixOriginIdentity) := .indexed binder selection.expression origin
  let mapPublic (identity : PublicMatrixIdentity) := .indexed binder selection.expression identity
  let mapValue (origin : OperationalValueOrigin) := .indexed binder selection.expression origin
  let mapTarget (target : RelationTargetSummary) : RelationTargetSummary := {
    target with
    origin := mapOrigin target.origin
    polynomial := overlayMapRelationSnapshotPolynomial mapOrigin mapPublic mapValue id target.polynomial
  }
  let mapRelation : OperationalMatrixRelation → OperationalMatrixRelation
    | .decomposition relation => .decomposition {
        relation with
        producer := mapOrigin relation.producer
        publicIdentity := mapPublic relation.publicIdentity
        inputOrigin := mapOrigin relation.inputOrigin
        inputSummary := mapTarget relation.inputSummary
      }
    | .preimage relation => .preimage {
        relation with
        producer := mapOrigin relation.producer
        publicIdentity := mapPublic relation.publicIdentity
        targetOrigin := mapOrigin relation.targetOrigin
        targetSummary := mapTarget relation.targetSummary
      }
  { fact with
    subject
    origin := mapOrigin fact.origin
    identity := fact.identity.map mapPublic
    relations := fact.relations.map mapRelation
    polynomial := overlayMapOperationalPolynomial mapOrigin mapPublic mapValue id mapRelation fact.polynomial
  }

private def overlayIndexScalarFact
    (binder : FamilyTemplateBinder) (selection : DynamicSelectionIdentity) (subject : WireRef) :
    OperationalScalarFact → OperationalScalarFact
  | .integer fact => .integer {
      fact with
      subject
      origin := .indexed binder selection.expression fact.origin
    }
  | .trapdoor fact => .trapdoor {
      fact with
      subject
      publicIdentity := .indexed binder selection.expression fact.publicIdentity
    }
  | .bytes fact => .bytes {
      fact with
      subject
      origin := .indexed binder selection.expression fact.origin
    }
  | fact => fact

/-- Evaluate one direct indexed matrix value at a complete index assignment.  This is the only
place direct delayed nodes invoke the fixed-assignment matrix kernels. -/
def rebindOperationalScalarFact
    (subject : WireRef) : OperationalScalarFact → OperationalScalarFact
  | .integer fact => .integer { fact with subject }
  | .trapdoor fact => .trapdoor { fact with subject }
  | .bytes fact => .bytes { fact with subject }
  | .boolean => .boolean
  | .real => .real
  | .typedBlob typeName schemaHash => .typedBlob typeName schemaHash
  | .unknown wireType => .unknown wireType

def rebindMatrixSubject
    (subject : WireRef) (fact : OperationalMatrixFact) :
    Except OperationalError OperationalMatrixFact :=
  if fact.relations.all fun relation => match relation with
      | .decomposition relation => relation.producer == fact.origin
      | .preimage relation => relation.producer == fact.origin then
    pure { fact with subject }
  else throw (.malformedRelation subject.node)

def DirectOperationalIndexedArena.matrixFactAt
    (arena : DirectOperationalIndexedArena)
    (parameters : ParamEnvironment)
    (indices : IndexValueEnvironment)
    (id : OperationalIndexedValueId) : Nat → Except OperationalError OperationalMatrixFact
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => do
      let value ← match arena.valueAt? id with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef id)
      if !completeDirectIndexAssignment parameters value.context indices then
        throw (.unsupportedOperationalExpr id)
      match value.payload with
      | .shared (.matrix _) (.matrix reference) =>
          match arena.fixed.matrices[reference]? with
          | some fact => pure fact
          | none => throw (.invalidOperationalExprRef reference)
      | .explicit (.matrix _) binder references => do
          let lane ← arena.indexExprAt parameters value.context indices (.variable binder) fuel
          let reference ← match references[lane.toNat]? with
            | some (.matrix reference) => pure reference
            | some _ => throw (.unsupportedOperationalExpr id)
            | none => throw (.invalidOperationalExprRef lane.toNat)
          match arena.fixed.matrices[reference]? with
          | some fact => pure fact
          | none => throw (.invalidOperationalExprRef reference)
      | .explicitValues (.matrix _) binder values => do
          let lane ← arena.indexExprAt parameters value.context indices (.variable binder) fuel
          let branch ← match values[lane.toNat]? with
            | some branch => pure branch
            | none => throw (.invalidOperationalExprRef lane.toNat)
          arena.matrixFactAt parameters indices branch fuel
      | .mapped (.matrix _) source map => do
          if !map.transportValid || map.destination != value.context then
            throw (.unsupportedOperationalExpr id)
          let sourceIndices ← map.source.binders.toList.mapM fun binder => do
            let expression ← match map.assignmentFor binder with
              | some expression => pure expression
              | none => throw (.unsupportedOperationalExpr id)
            let lane ← arena.indexExprAt parameters value.context indices expression fuel
            pure (.variable binder, lane)
          let sourceValue ← match arena.valueAt? source with
            | some value => pure value
            | none => throw (.invalidOperationalExprRef source)
          match sourceValue.payload with
          /- `pushRebound` normalizes a selected output to `mapped (rebound source subject) map`.
          Apply the pending map to the fixed fact before checking the output subject, matching
          reduced evaluation and keeping the subject overlay lazy. -/
          | .rebound (.matrix _) reboundSource subject => do
              let fact ← arena.matrixFactAt parameters sourceIndices reboundSource fuel
              let fact ← match reindexOperationalMatrixFact parameters map fact with
                | some fact => pure fact
                | none => throw (.unsupportedOperationalExpr id)
              rebindMatrixSubject subject fact
          | _ =>
              let fact ← arena.matrixFactAt parameters sourceIndices source fuel
              match reindexOperationalMatrixFact parameters map fact with
              | some fact => pure fact
              | none => throw (.unsupportedOperationalExpr id)
      | .rebound (.matrix _) source subject => do
          rebindMatrixSubject subject (← arena.matrixFactAt parameters indices source fuel)
      | .indexedOutput (.matrix _) source binder selection subject => do
          let fact ← arena.matrixFactAt parameters indices source fuel
          pure (overlayIndexMatrixFact binder selection subject fact)
      | .matrixResultBound (.matrix _) source totalHardBound => do
          let maps ← directPendingMaps arena source (arena.values.size + 1)
          let source ← arena.matrixFactAt parameters indices source fuel
          let totalHardBound ← maps.foldlM (fun bound map =>
            match reindexOperationalBoundExpr parameters map bound with
            | some bound => pure bound
            | none => throw (.unsupportedOperationalExpr id)) totalHardBound
          pure { source with totalHardBound }
      | .pointwise (.matrix matrixType) (.matrix operation) inputs => do
          let arguments ← inputs.mapM fun input => arena.matrixFactAt parameters indices input fuel
          applyDirectMatrixPointwiseOperation operation matrixType arguments
      | .pointwise (.matrix _) (.relation operation) inputs => do
          let arguments ← inputs.mapM fun input => do
            let value ← match arena.valueAt? input with
              | some value => pure value
              | none => throw (.invalidOperationalExprRef input)
            match value.payload.schema with
            | .matrix _ => return .matrix (← arena.matrixFactAt parameters indices input fuel)
            | .scalar (.trapdoor _ _ _ _ _) =>
                let scalar ← arena.scalarFactAt parameters indices input fuel
                let scalar ← materializeDirectScalarFact arena parameters value.context indices scalar
                match scalar with
                | .trapdoor fact => return .trapdoor fact
                | _ => throw (.unsupportedOperationalExpr input)
            | .scalar _ => throw (.unsupportedOperationalExpr input)
          let operation ← materializeDirectRelationOperation arena parameters value.context indices operation
          applyDirectRelationProducer operation operation.outputSchema arguments
      | .pointwise (.matrix matrixType) (.matrixFromScalar operation) inputs => do
          let input ← match inputs with
            | #[input] => pure input
            | _ => throw (.unsupportedOutputArity operation.ownerNode inputs.size)
          let value ← match arena.valueAt? input with
            | some value => pure value | none => throw (.invalidOperationalExprRef input)
          let scalar ← arena.scalarFactAt parameters indices input fuel
          applyDirectMatrixFromScalarOperation operation matrixType
            (← materializeDirectScalarFact arena parameters value.context indices scalar)
      | _ => throw (.unsupportedOperationalExpr id)

/-- Evaluate a direct indexed scalar at a complete assignment. Matrix-to-scalar kernels evaluate
their matrix input at that identical assignment, preserving shared-selector correlation. -/
def materializeOperationalScalarFact
    (parameters : ParamEnvironment) (context : IndexContext) (indices : IndexValueEnvironment) :
    OperationalScalarFact → Except OperationalError OperationalScalarFact
  | .integer fact => do
      let lowerExpression ← match fact.lowerExpression.materialize parameters context indices with
        | some value => pure value | none => throw (.unsupportedOperationalExpr fact.subject.node)
      let upperExpression ← match fact.upperExpression.materialize parameters context indices with
        | some value => pure value | none => throw (.unsupportedOperationalExpr fact.subject.node)
      pure (.integer { fact with
        lowerExpression := .closed lowerExpression
        upperExpression := .closed upperExpression })
  | .trapdoor fact => do
      let maximum ← match fact.maximum.materialize parameters context indices with
        | some value => pure value | none => throw (.unsupportedOperationalExpr fact.subject.node)
      let preimageCutoff ← fact.preimageCutoff.mapM fun cutoff =>
        match cutoff.materialize parameters context indices with
        | some value => pure value | none => throw (.unsupportedOperationalExpr fact.subject.node)
      pure (.trapdoor { fact with
        maximum := .closed maximum
        preimageCutoff := preimageCutoff.map .closed })
  | fact => pure fact

def DirectOperationalIndexedArena.scalarFactAt
    (arena : DirectOperationalIndexedArena)
    (parameters : ParamEnvironment)
    (indices : IndexValueEnvironment)
    (id : OperationalIndexedValueId) : Nat → Except OperationalError OperationalScalarFact
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => do
      let value ← match arena.valueAt? id with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef id)
      if !completeDirectIndexAssignment parameters value.context indices then
        throw (.unsupportedOperationalExpr id)
      let fact ← match value.payload with
      | .shared (.scalar _) (.scalar reference) =>
          match arena.fixed.scalars[reference]? with
          | some fact => pure fact
          | none => throw (.invalidOperationalExprRef reference)
      | .explicit (.scalar _) binder references => do
          let lane ← arena.indexExprAt parameters value.context indices (.variable binder) fuel
          let reference ← match references[lane.toNat]? with
            | some (.scalar reference) => pure reference
            | some _ => throw (.unsupportedOperationalExpr id)
            | none => throw (.invalidOperationalExprRef lane.toNat)
          match arena.fixed.scalars[reference]? with
          | some fact => pure fact
          | none => throw (.invalidOperationalExprRef reference)
      | .explicitValues (.scalar _) binder values => do
          let lane ← arena.indexExprAt parameters value.context indices (.variable binder) fuel
          let branch ← match values[lane.toNat]? with
            | some branch => pure branch
            | none => throw (.invalidOperationalExprRef lane.toNat)
          arena.scalarFactAt parameters indices branch fuel
      | .mapped (.scalar _) source map => do
          if !map.transportValid || map.destination != value.context then
            throw (.unsupportedOperationalExpr id)
          let sourceIndices ← map.source.binders.toList.mapM fun binder => do
            let expression ← match map.assignmentFor binder with
              | some expression => pure expression
              | none => throw (.unsupportedOperationalExpr id)
            let lane ← arena.indexExprAt parameters value.context indices expression fuel
            pure (.variable binder, lane)
          let sourceValue ← match arena.valueAt? source with
            | some value => pure value
            | none => throw (.invalidOperationalExprRef source)
          match sourceValue.payload with
          | .rebound (.scalar _) reboundSource subject => do
              let fact ← arena.scalarFactAt parameters sourceIndices reboundSource fuel
              let fact ← match reindexOperationalScalarFact parameters map fact with
                | some fact => pure fact
                | none => throw (.unsupportedOperationalExpr id)
              pure (rebindOperationalScalarFact subject fact)
          | _ =>
              let fact ← arena.scalarFactAt parameters sourceIndices source fuel
              match reindexOperationalScalarFact parameters map fact with
              | some fact => pure fact
              | none => throw (.unsupportedOperationalExpr id)
      | .rebound (.scalar _) source subject =>
          pure (rebindOperationalScalarFact subject (← arena.scalarFactAt parameters indices source fuel))
      | .indexedOutput (.scalar _) source binder selection subject =>
          pure (overlayIndexScalarFact binder selection subject
            (← arena.scalarFactAt parameters indices source fuel))
      | .pointwise (.scalar _) (.matrixToScalar operation) inputs => do
          let input ← match inputs with
            | #[input] => pure input
            | _ => throw (.unsupportedOutputArity operation.ownerNode inputs.size)
          applyDirectMatrixToScalarOperation operation
            (← arena.matrixFactAt parameters indices input fuel)
      | .pointwise (.scalar _) (.scalar operation) inputs => do
        let arguments ← inputs.mapM fun input => do
          let value ← match arena.valueAt? input with
            | some value => pure value | none => throw (.invalidOperationalExprRef input)
          materializeDirectScalarFact arena parameters value.context indices
            (← arena.scalarFactAt parameters indices input fuel)
        applyDirectScalarPointwiseOperation operation arguments
      | _ => throw (.unsupportedOperationalExpr id)
      pure fact

end

/-- Materialize one fully selected scalar after raw carrier traversal has finished.  Keeping this
outside the recursive carrier evaluator preserves its structural termination proof while giving
cutoff and interval descriptors the arena-aware gather resolver. -/
def DirectOperationalIndexedArena.materializeScalarFact
    (arena : DirectOperationalIndexedArena)
    (parameters : ParamEnvironment)
    (context : IndexContext)
    (indices : IndexValueEnvironment)
    (fact : OperationalScalarFact) : Except OperationalError OperationalScalarFact :=
  materializeDirectScalarFact arena parameters context indices fact

def DirectOperationalIndexedArena.materializedScalarFactAt
    (arena : DirectOperationalIndexedArena)
    (parameters : ParamEnvironment)
    (indices : IndexValueEnvironment)
    (id : OperationalIndexedValueId) : Nat → Except OperationalError OperationalScalarFact
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => do
      let value ← match arena.valueAt? id with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef id)
      /- Materialization is a post-traversal boundary, not another carrier edge.  Preserve the
      caller's remaining traversal budget for the scalar carrier itself; otherwise one mapped
      view followed by its fixed table reaches `indexExprAt` with zero fuel despite a valid
      complete assignment. -/
      let fact ← arena.scalarFactAt parameters indices id (fuel + 1)
      arena.materializeScalarFact parameters value.context indices fact

/-- Read a direct matrix value only after its indexed context has been fully assigned.  Empty
contexts are complete assignments, so ordinary direct wires never fall back to the removed
selection evaluator. -/
def OperationalExprArena.directValueFactAt
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (expression : IndexedOperationalFact) : Except OperationalError OperationalMatrixFact := do
  let root ← match expression.payload with
    | .directValue root => pure root
  if !expression.context.binders.isEmpty then throw (.unsupportedOperationalExpr root)
  arena.direct.matrixFactAt environment [] root (arena.direct.values.size + 1)

def OperationalExprArena.directValueScalarFactAt
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (expression : IndexedOperationalFact) : Except OperationalError OperationalScalarFact := do
  let root ← match expression.payload with
    | .directValue root => pure root
  if !expression.context.binders.isEmpty then throw (.unsupportedOperationalExpr root)
  arena.direct.materializedScalarFactAt environment [] root (arena.direct.values.size + 1)

/-- Production scalar boundaries admit only fully assigned direct values. -/
def directScalarFactAt
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef) : Except OperationalError OperationalScalarFact := do
  let expression ← lookupFact node facts wire
  facts.arena.directValueScalarFactAt [] expression

def requireBooleanFact
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef) : Except OperationalError Unit := do
  match ← directScalarFactAt node facts wire with
  | .boolean => pure ()
  | _ => throw (.operandNotBoolean node wire)

def requireRealFact
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef) : Except OperationalError Unit := do
  match ← directScalarFactAt node facts wire with
  | .real => pure ()
  | _ => throw (.operandNotReal node wire)

def trapdoorFactAt
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef) : Except OperationalError OperationalTrapdoorFact := do
  match ← directScalarFactAt node facts wire with
  | .trapdoor fact => pure fact
  | _ => throw (.missingPublicIdentity node wire)

def integerFactAt
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef) : Except OperationalError OperationalIntegerFact := do
  match ← directScalarFactAt node facts wire with
  | OperationalScalarFact.integer fact => pure fact
  | _ => throw (.operandNotInteger node wire)

/-- Read the complete interval of an ordered direct integer family without choosing one lane.
Only fixed scalar tables and mapped views thereof are accepted here; arithmetic/opaque scalar
nodes remain fail-closed until their indexed interval transfer is implemented. -/
private partial def DirectOperationalIndexedArena.fixedIntegerInterval
    (arena : DirectOperationalIndexedArena) (owner : Nat) (wire : WireRef)
    (id : OperationalIndexedValueId) : Nat → Except OperationalError (Int × Int)
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => do
      let value ← match arena.valueAt? id with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef id)
      let intervals ← match value.payload with
        | .shared (.scalar .integer) (.scalar reference) =>
            match arena.fixed.scalars[reference]? with
            | some (.integer fact) => pure [(fact.lower, fact.upper)]
            | _ => throw (.operandNotInteger owner wire)
        | .explicit (.scalar .integer) _ references => references.toList.mapM fun reference =>
            match reference with
            | .scalar scalar => match arena.fixed.scalars[scalar]? with
              | some (.integer fact) => pure (fact.lower, fact.upper)
              | _ => throw (.operandNotInteger owner wire)
            | .matrix _ => throw (.operandNotInteger owner wire)
        | .explicitValues (.scalar .integer) _ values =>
            values.toList.mapM fun child => arena.fixedIntegerInterval owner wire child fuel
        | .mapped (.scalar .integer) source _ => return ← arena.fixedIntegerInterval owner wire source fuel
        | _ => throw (.operandNotInteger owner wire)
      match intervals with
      | [] => throw (.invalidCount owner 0)
      | first :: rest => pure <| rest.foldl (fun (lower, upper) (nextLower, nextUpper) =>
          (min lower nextLower, max upper nextUpper)) first

def directSingleIndexBinder
    (node : Nat)
    (expression : IndexedOperationalFact) : Except OperationalError IndexVariable :=
  match expression.context.binders.toList with
  | [binder] => pure binder
  | _ => throw (.loopInputModeMismatch node 0)

/-- Materialize the exact prepared-scope path and integer-family producer wire for a gather
owner.  Source-family identity is retained by the gathered matrix provenance instead. -/
def operationalGatherIndicesWire (scope : ScopeTemplateKey) (wire : WireRef) : GatherLookupWire := {
  scope := scope.toGatherScopeTemplateKey
  node := wire.node
  port := wire.port
}

/-- Read a deterministic representative of a direct indexed matrix value for structural bound
queries.  Each free binder is assigned its first valid lane; the direct carrier retains the
complete context for identity-sensitive rewrites, while the fixed transfer supplies its
construction-uniform hard bound. -/
def OperationalExprArena.directValueRepresentativeFactAt
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (expression : IndexedOperationalFact) : Except OperationalError OperationalMatrixFact := do
  let root ← match expression.payload with
    | .directValue root => pure root
  if !validateContext expression.context then throw (.unsupportedOperationalExpr root)
  let indices := expression.context.binders.toList.map fun binder => (.variable binder, 0)
  arena.direct.matrixFactAt environment indices root (arena.direct.values.size + 1)

/-- Enumerate every declared direct-index assignment before reducing a family to one numeric
bound.  This is deliberately structural: an unresolved cardinality or an unevaluable mapped
selector remains an error instead of silently choosing lane zero. -/
def directIndexAssignments
    (environment : ParamEnvironment)
    (context : IndexContext) : Except OperationalError (List IndexValueEnvironment) := do
  if !validateContext context then throw (.unsupportedOperationalExpr context.binders.size)
  context.binders.foldrM (fun binder tails => do
    let count ← match binder.count.evaluate environment with
      | some count => pure count
      | none => throw .nonClosedExpression
    if count <= 0 then throw (.invalidCount binder.slot count)
    pure <| (List.range count.toNat).flatMap fun lane =>
      tails.map fun tail => (.variable binder, Int.ofNat lane) :: tail) [[]]

/-- Materialize every fixed assignment of one direct indexed matrix value.  The carrier itself
continues to store a compact table or mapped template; only bound consumers enumerate the finite
context. -/
def OperationalExprArena.directValueFactsAt
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (expression : IndexedOperationalFact) : Except OperationalError (List OperationalMatrixFact) := do
  let root ← match expression.payload with
    | .directValue root => pure root
  let assignments ← directIndexAssignments environment expression.context
  assignments.mapM fun indices =>
    arena.direct.matrixFactAt environment indices root (arena.direct.values.size + 1)

/-- The canonical structural driver of one correlated direct-family lane.  This deliberately
retains the complete owner-bearing index expression; its ordinal records the selected physical
lane without reconstructing a root assignment environment. -/
abbrev DirectCorrelationKey := IndexExpr

/-- The selector assignments that identify one physical direct lane.  Nested direct tables add
independent assignments to this environment; repeating a selector must agree with its existing
ordinal.  The list preserves the first structural occurrence, while all comparisons below are
set-like, so mapped/nested paths cannot depend on traversal order. -/
abbrev DirectCorrelationEnvironment := List (DirectCorrelationKey × Nat)

/-- Canonicalize only identity-preserving selector syntax before comparing correlations.  In
particular a closed map may express a loop lane as `i + 0`; retaining it as a distinct key would
allow the same owner to receive inconsistent ordinals.  Nonzero offsets remain distinct because
they denote a different physical lane. -/
private def canonicalDirectCorrelationKey : DirectCorrelationKey → DirectCorrelationKey
  | .offset base 0 => canonicalDirectCorrelationKey base
  | expression => expression

private def normalizeDirectCorrelation
    (environment : DirectCorrelationEnvironment) : Option DirectCorrelationEnvironment :=
  environment.foldlM (fun normalized assignment =>
    let assignment := (canonicalDirectCorrelationKey assignment.1, assignment.2)
    match normalized.find? fun retained => retained.1 == assignment.1 with
    | none => pure (normalized ++ [assignment])
    | some retained => if retained.2 == assignment.2 then pure normalized else none) []

/-- Close a physical-lane correlation under the explicit owner-aware selector equalities carried
by the direct arena.  Equality propagates an existing ordinal to its peer; conflicting explicit
ordinals reject that alternative.  It never relates owners without a declared equality, so
independent selectors retain their full Cartesian set of feasible alternatives. -/
private def mergeDirectCorrelation
    (left right : DirectCorrelationEnvironment) : Option DirectCorrelationEnvironment :=
  normalizeDirectCorrelation (left ++ right)

private def directCorrelationContained
    (required available : DirectCorrelationEnvironment) : Bool :=
  required.all fun assignment => available.contains assignment

private def sameDirectCorrelation
    (left right : DirectCorrelationEnvironment) : Bool :=
  directCorrelationContained left right && directCorrelationContained right left

/-- Recover the executable position carried by one gather alternative.  The gathered result
identity and its selected result ordinal remain in the complete correlation environment; this
helper merely exposes the independently recorded runtime position used to align two gathers. -/
private def directGatherExecutablePosition?
    (correlation : DirectCorrelationEnvironment) : Option (IndexExpr × Nat) := do
  let gathers := correlation.filterMap fun assignment => match assignment.1 with
    | .gather _ _ position => some position
    | _ => none
  let position ← match gathers with
    | [position] => some position
    | _ => none
  let assignments := correlation.filter fun assignment =>
    canonicalDirectCorrelationKey assignment.1 == canonicalDirectCorrelationKey position
  match assignments with
  | [assignment] => some (canonicalDirectCorrelationKey assignment.1, assignment.2)
  | _ => none

/-- One gather alternative's operand-local proof that its runtime position crossed a real
capture-free map after the gather was materialized.  This is not arena state and cannot be
borrowed by another operand: it is reconstructed only from that leaf's pending map stack. -/
structure DirectGatherPositionPath where
  gather : IndexExpr
  destination : IndexExpr
  ordinal : Nat
  deriving BEq, Repr

private def sameGatherOwnerAndDomain (left right : IndexExpr) : Bool :=
  match left, right with
  | .gather leftOwner leftCount _, .gather rightOwner rightCount _ =>
      leftOwner == rightOwner && leftCount == rightCount
  | _, _ => false

/-- Locate the operand's own map that carried this gather.  A compacted map may already have
rewritten the gather's position, so matching the whole pre-map expression would lose precisely
the evidence needed after composition.  The gather owner and source domain remain immutable. -/
private def gatherMapDestination?
    (maps : List IndexMap) (gathered : IndexExpr) : Option (IndexExpr × List IndexMap) :=
  match maps with
  | [] => none
  | map :: remaining =>
      match map.assignments.toList.filter (sameGatherOwnerAndDomain gathered) with
      | [destination@(.gather _ _ _)] => some (destination, remaining)
      | _ => gatherMapDestination? remaining gathered

private def transportGatherPositionPath
    (parameters : ParamEnvironment) (maps : List IndexMap) (position : IndexExpr) (ordinal : Nat) : Option (IndexExpr × Nat) := do
  let mut position := position
  let mut ordinal := ordinal
  let mut transported := false
  for map in maps do
    if position.freeVariables.any map.source.binders.contains then
      let translated ← reindex map position
      match translated with
      | .constant lane =>
          /- A closed transport is an assertion about the *existing* executable coordinate.  It
          must not manufacture a matching lane by replacing the incoming ordinal. -/
          if ordinal != lane then failure else pure ()
          position := .constant lane
          transported := true
      | .variable binder =>
          let bound ← binder.count.evaluate parameters
          if bound <= 0 || Int.ofNat ordinal >= bound then failure else pure ()
          position := .variable binder
          transported := true
      | .offset (.variable binder) amount =>
          let shifted := Int.ofNat ordinal - amount
          if shifted < 0 then failure else pure ()
          let bound ← binder.count.evaluate parameters
          if bound <= 0 || shifted >= bound then failure else pure ()
          position := .variable binder
          ordinal := shifted.toNat
          transported := true
      | .offset (.constant lane) amount =>
          let destination := Int.ofNat lane + amount
          if destination < 0 then failure else pure ()
          if ordinal != destination.toNat then failure else pure ()
          position := .constant destination.toNat
          transported := true
      | .gather _ _ _ => failure
      | .offset _ _ => failure
  if transported then some (canonicalDirectCorrelationKey position, ordinal) else none

private def directGatherPositionPaths
    (parameters : ParamEnvironment) (maps : List IndexMap)
    (correlation : DirectCorrelationEnvironment) : List DirectGatherPositionPath :=
  correlation.filterMap fun assignment => match assignment.1 with
    | gathered@(.gather _ _ position) => do
        let (_, positionOrdinal) ← directGatherExecutablePosition? correlation
        let (mappedGather, suffix) ← gatherMapDestination? maps gathered
        let mappedPosition ← match mappedGather with
          | .gather _ _ mappedPosition => some mappedPosition
          | _ => none
        let (destination, ordinal) ←
          transportGatherPositionPath parameters suffix mappedPosition positionOrdinal
        some { gather := gathered, destination, ordinal }
    | _ => none

/-- Merge inherited gather provenance with records introduced at the current carrier boundary.
Identical records are harmless aliases of the same transport and are retained once.  A single
immutable gather identity cannot claim two destination coordinates or ordinals in one reduced
alternative: that would make later correlation alignment traversal-order dependent. -/
private def mergeDirectGatherPositionPaths
    (inherited introduced : List DirectGatherPositionPath) : Option (List DirectGatherPositionPath) :=
  (inherited ++ introduced).foldlM (fun retained path =>
    match retained.find? fun candidate => candidate.gather == path.gather with
    | none => pure (retained ++ [path])
    | some candidate =>
        if candidate.destination == path.destination && candidate.ordinal == path.ordinal then
          pure retained
        else none) []

structure ReducedDirectMatrixFact where
  correlation : DirectCorrelationEnvironment
  fact : OperationalMatrixFact
  positionPaths : List DirectGatherPositionPath := []

def ReducedDirectMatrixFact.key (entry : ReducedDirectMatrixFact) : Option DirectCorrelationKey :=
  entry.correlation.head?.map (·.1)

def ReducedDirectMatrixFact.ordinal (entry : ReducedDirectMatrixFact) : Nat :=
  entry.correlation.head?.map (·.2) |>.getD 0

structure ReducedDirectScalarFact where
  correlation : DirectCorrelationEnvironment
  fact : OperationalScalarFact
  positionPaths : List DirectGatherPositionPath := []

def ReducedDirectScalarFact.key (entry : ReducedDirectScalarFact) : Option DirectCorrelationKey :=
  entry.correlation.head?.map (·.1)

def ReducedDirectScalarFact.ordinal (entry : ReducedDirectScalarFact) : Nat :=
  entry.correlation.head?.map (·.2) |>.getD 0

structure ReducedDirectRelationArgument where
  correlation : DirectCorrelationEnvironment
  payload : DirectRelationArgument
  positionPaths : List DirectGatherPositionPath := []

/- The reporting identity of a map deliberately omits `closedIndex`: that field records how a
transport was admitted, whereas a rewrite event is identified by its fully transported source
assignment.  Thus `A → 0` and `A → B → 0` normalize to the same event path. -/
private structure DirectRelationRewriteMapKey where
  source : IndexContext
  destination : IndexContext
  assignments : Array IndexExpr
  deriving BEq, DecidableEq, Repr

private def directRelationRewriteMapKey (map : IndexMap) : DirectRelationRewriteMapKey := {
  source := map.source, destination := map.destination, assignments := map.assignments }

private def normalizeDirectRelationRewriteMaps (maps : List IndexMap) : List DirectRelationRewriteMapKey :=
  let rec normalize (current : IndexMap) : List IndexMap → Option DirectRelationRewriteMapKey
    | [] => some (directRelationRewriteMapKey current)
    | next :: remaining => do
        if current.destination != next.source then none else pure ()
        let assignments ← current.assignments.toList.mapM (reindex next)
        let composed : IndexMap := {
          source := current.source, destination := next.destination, assignments := assignments.toArray }
        normalize composed remaining
  /- Nested `.mapped` descent prepends the innermost map after its outer caller has already
  supplied the earlier transport, so this stack is source-to-destination order.  Compose it
  solely for stable rewrite-event identity; semantic reduction keeps the individual maps. -/
  match maps with
  | [] => []
  | first :: remaining => match normalize first remaining with
    | some key => [key]
    | none => maps.map directRelationRewriteMapKey

/-- Reindexing clones a delayed carrier node, so its allocation ID is not an event identity.
The executable owner identifies the one pointwise graph operation across those clones. -/
private structure DirectRelationRewritePointwiseKey where
  ownerScope : Option ScopeTemplateKey
  ownerNode : Nat
  outputPort : Nat
  deriving BEq, DecidableEq, Repr

/-- One successful exact relation application during structural direct reduction.  The pointwise
node, normalized fully transported map path, correlated physical lane, and local rewrite ordinal
make events stable across shared DAG traversal and repeated bound queries. -/
structure DirectRelationRewriteEventKey where
  pointwise : DirectRelationRewritePointwiseKey
  maps : List DirectRelationRewriteMapKey
  correlation : DirectCorrelationEnvironment
  localOrdinal : Nat
  deriving BEq, DecidableEq, Repr

private def sameDirectRelationRewriteEvent
    (left right : DirectRelationRewriteEventKey) : Bool :=
  left.pointwise == right.pointwise && left.maps == right.maps &&
    sameDirectCorrelation left.correlation right.correlation && left.localOrdinal == right.localOrdinal

private def deduplicateDirectRelationRewriteEvents
    (events : List DirectRelationRewriteEventKey) : List DirectRelationRewriteEventKey :=
  events.foldl (fun retained event =>
    if retained.any (sameDirectRelationRewriteEvent event) then retained else retained ++ [event]) []

/-- Work performed by the exact gather-position zipper.  These counters are reporting-only;
they are incremented at the index insertion and lookup sites, never inferred from a fixture. -/
structure DirectGatherAlignmentWork where
  indexBuildEntries : Nat := 0
  driverLookupVisits : Nat := 0
  deriving BEq, DecidableEq, Repr

private def DirectGatherAlignmentWork.add
    (left right : DirectGatherAlignmentWork) : DirectGatherAlignmentWork := {
  indexBuildEntries := left.indexBuildEntries + right.indexBuildEntries
  driverLookupVisits := left.driverLookupVisits + right.driverLookupVisits }

private structure ReducedDirectMatrixEvaluation where
  entries : List ReducedDirectMatrixFact
  rewriteEvents : List DirectRelationRewriteEventKey := []
  /-- Maximum normalized polynomial width encountered while reducing this direct sub-DAG. -/
  maximumPolynomialTerms : Nat := 0

private structure ReducedDirectScalarEvaluation where
  entries : List ReducedDirectScalarFact
  rewriteEvents : List DirectRelationRewriteEventKey := []
  /-- Scalar reduction can invoke matrix-to-scalar nodes, so preserve their observed width. -/
  maximumPolynomialTerms : Nat := 0

private structure DirectCorrelationEntry (α : Type) where
  correlation : DirectCorrelationEnvironment
  payload : α
  positionPaths : List DirectGatherPositionPath

/-- A reduced operand may carry several provenance records only when they name the same
executable coordinate.  Selecting with `any` would let an unrelated path in the same operand
manufacture a join, so ambiguity is rejected before correlation alignment. -/
private def agreedDirectGatherPositionPath?
    (paths : List DirectGatherPositionPath) : Option DirectGatherPositionPath :=
  match paths with
  | [] => none
  | first :: remaining =>
      if remaining.all fun candidate =>
          candidate.destination == first.destination && candidate.ordinal == first.ordinal then
        some first
      else none

/-- One canonical executable gather coordinate.  The custom hash uses the derived structural
representation, while `Std.HashMap` still compares this complete key with `BEq`; hash collisions
therefore remain distinct buckets and can never manufacture or reject a selector match. -/
private structure DirectGatherPositionKey where
  destination : IndexExpr
  ordinal : Nat
  deriving BEq

private instance : Hashable DirectGatherPositionKey where
  hash key := hash (reprStr key.destination) ^^^ UInt64.ofNat key.ordinal

private abbrev DirectGatherPositionIndex (α : Type) :=
  Std.HashMap DirectGatherPositionKey (DirectCorrelationEntry α)

private def directGatherPositionIndex {α : Type}
    (id : Nat) (entries : List (DirectCorrelationEntry α)) :
    Except OperationalError (DirectGatherPositionIndex α × Nat) := do
  let index ← entries.foldlM (fun index entry => do
    let path ← match agreedDirectGatherPositionPath? entry.positionPaths with
      | some path => pure path
      | none => throw (.unsupportedOperationalExpr id)
    let key : DirectGatherPositionKey := {
      destination := canonicalDirectCorrelationKey path.destination, ordinal := path.ordinal }
    match index[key]? with
    | none => pure (index.insert key entry)
    | some _ =>
        /- Repeated executable coordinates in one operand are ambiguous.  Hash collisions reach
        independent `Std.HashMap` keys by the full `BEq` instance and never get here. -/
        throw (.unsupportedOperationalExpr id)) {}
  pure (index, entries.length)

private def directGatherEntryAt? {α : Type}
    (index : DirectGatherPositionIndex α) (key : DirectGatherPositionKey) :
    Option (DirectCorrelationEntry α) := do
  index[key]?

private def directGatherIdentities
    (correlation : DirectCorrelationEnvironment) : List IndexExpr :=
  correlation.filterMap fun assignment => match assignment.1 with
    | gathered@(.gather _ _ _) => some gathered
    | _ => none

/-- Relation factors may carry more than one gather path after generic composition.  Compare the
complete post-reindex inventory, preserving gather owner/domain and the canonical executable
coordinate.  This accepts syntactically different maps that reach the same selector, but never
selects an arbitrary singleton path from a mixed operand. -/
private def directGatherOwner? : IndexExpr → Option GatherLookupOwner
  | .gather owner _ _ => some owner
  | _ => none

private structure DirectGatherRelationPathKey where
  selector : GatherLookupOwner
  destination : IndexExpr
  ordinal : Nat
  deriving BEq

private instance : Hashable DirectGatherRelationPathKey where
  hash key := hash key.selector ^^^ hash (reprStr key.destination) ^^^ UInt64.ofNat key.ordinal

private def directGatherRelationPathKey?
    (path : DirectGatherPositionPath) : Option DirectGatherRelationPathKey := do
  let selector ← directGatherOwner? path.gather
  some { selector, destination := canonicalDirectCorrelationKey path.destination, ordinal := path.ordinal }

private def directGatherRelationInventory
    (id : Nat) (paths : List DirectGatherPositionPath) : Except OperationalError
    (Std.HashMap DirectGatherRelationPathKey Unit) :=
  paths.foldlM (fun inventory path => do
    let key ← match directGatherRelationPathKey? path with
      | some key => pure key
      | none => throw (.unsupportedOperationalExpr id)
    if inventory.contains key then throw (.unsupportedOperationalExpr id)
    else pure (inventory.insert key ())) {}

private def sameDirectGatherRelationInventory
    (id : Nat) (left right : List DirectGatherPositionPath) : Except OperationalError Bool := do
  let left ← directGatherRelationInventory id left
  let right ← directGatherRelationInventory id right
  pure (left.size == right.size && left.toList.all fun (key, _) => right.contains key)

private def directRelationGatherPathsCompatible
    (id : Nat) (paths : List (List DirectGatherPositionPath)) : Except OperationalError Bool :=
  match paths.filter (fun paths => !paths.isEmpty) with
  | [] | [_] => pure true
  | first :: remaining => remaining.foldlM (fun compatible next => do
      if !compatible then pure false else sameDirectGatherRelationInventory id first next) true

private def directGatherSelectorInventory
    (id : Nat) (correlation : DirectCorrelationEnvironment) : Except OperationalError
    (Std.HashMap GatherLookupOwner Unit) :=
  (directGatherIdentities correlation).foldlM (fun inventory gathered => do
    let owner ← match directGatherOwner? gathered with
      | some owner => pure owner
      | none => throw (.unsupportedOperationalExpr id)
    pure (inventory.insert owner ())) {}

private def directRelationSelectedInputsCompatible {α : Type}
    (id : Nat) (entries : List (DirectCorrelationEntry α)) : Except OperationalError Bool := do
  let paths := entries.map (·.positionPaths)
  if paths.any (fun paths => !paths.isEmpty) then
    directRelationGatherPathsCompatible id paths
  else
    match entries with
    | [] | [_] => pure true
    | first :: remaining => do
        let first ← directGatherSelectorInventory id first.correlation
        remaining.allM fun entry => do
          let current ← directGatherSelectorInventory id entry.correlation
          pure (first.size == current.size && first.toList.all fun (owner, _) => current.contains owner)

private def alignDirectCorrelationEntriesWithWork {α : Type}
    (owner id : Nat)
    (inputs : List (List (DirectCorrelationEntry α))) :
    Except OperationalError ((List (Array α × DirectCorrelationEnvironment × List DirectGatherPositionPath ×
      List (DirectCorrelationEntry α))) × DirectGatherAlignmentWork) := do
  let inputs := inputs.map fun entries => entries.filterMap fun entry => do
    let correlation ← normalizeDirectCorrelation entry.correlation
    pure { entry with correlation }
  if inputs.isEmpty || inputs.any List.isEmpty then throw (.invalidCount owner 0)
  /- A gather result ordinal is an ordinal in its source family, not an executable zip position.
  When all gather alternatives expose one exact runtime position, zip them by that position even
  if their source families have different sizes.  Each operand must contribute exactly one
  alternative at each position; duplicates, absent evidence, and ambiguous position proofs fail
  closed.  Non-gather operands still use the ordinary subset rule below. -/
  let gatherIdentities := inputs.foldl (fun retained entries => entries.foldl (fun retained entry =>
      (directGatherIdentities entry.correlation).foldl (fun retained gathered =>
        if retained.contains gathered then retained else retained ++ [gathered]) retained) retained) []
  let gatheredInputs := inputs.filter fun entries =>
    entries.any fun entry => !entry.positionPaths.isEmpty
  /- Each reduced operand transports its own gather position through its own pending map stack
  before it reaches this zipper.  Thus equal normalized position expressions here are exactly a
  path-scoped proof that both operands arrived at the same executable destination coordinate.
  No arena-wide equality registry is consulted: unrelated maps, branching views, and equal
  numeric slots cannot create a match. -/
  if gatherIdentities.length > 1 then
    if gatheredInputs.length != inputs.length ||
        !gatheredInputs.all (fun entries => entries.all fun entry =>
          (agreedDirectGatherPositionPath? entry.positionPaths).isSome) then
      throw (.unsupportedOperationalExpr id)
    let indexed ← gatheredInputs.mapM (directGatherPositionIndex id)
    let indexes := indexed.map (·.1)
    let indexBuildEntries := indexed.foldl (fun total (_, count) => total + count) 0
    let driver? := gatheredInputs.foldl (fun selected? entries =>
      match selected? with
      | none => some entries
      | some selected => if entries.length > selected.length then some entries else selected) none
    match driver? with
    | some driver =>
        let results ← driver.mapM fun driverEntry => do
          let path ← match agreedDirectGatherPositionPath? driverEntry.positionPaths with
            | some path => pure path
            | none => throw (.unsupportedOperationalExpr id)
          let position : DirectGatherPositionKey := {
            destination := canonicalDirectCorrelationKey path.destination, ordinal := path.ordinal }
          let selected ← indexes.mapM fun index =>
            match directGatherEntryAt? index position with
            | some entry => pure entry
            | none => throw (.unsupportedOperationalExpr id)
          let correlation ← selected.foldlM (fun current entry =>
            match mergeDirectCorrelation current entry.correlation with
            | some merged => pure merged
            | none => throw (.unsupportedOperationalExpr id)) []
          pure (selected.map (·.payload) |>.toArray, correlation,
            selected.flatMap (·.positionPaths), selected)
        return (results, { indexBuildEntries, driverLookupVisits := driver.length * indexes.length })
    | none => throw (.invalidCount owner 0)
  let driver? := inputs.foldl (fun selected? entries =>
    match selected? with
    | none => some entries
    | some selected =>
        let selectedDepth := selected.foldl
          (fun maximum (entry : DirectCorrelationEntry α) => max maximum entry.correlation.length) 0
        let entryDepth := entries.foldl
          (fun maximum (entry : DirectCorrelationEntry α) => max maximum entry.correlation.length) 0
        if entryDepth > selectedDepth then some entries else selected?) none
  match driver? with
  | some driver =>
      let driverDepth := driver.foldl
        (fun maximum (entry : DirectCorrelationEntry α) => max maximum entry.correlation.length) 0
      let competing := inputs.filter fun entries =>
        entries.foldl (fun maximum (entry : DirectCorrelationEntry α) =>
          max maximum entry.correlation.length) 0 == driverDepth
      if !competing.all (fun entries => entries.all fun entry =>
          driver.any fun driverEntry =>
            sameDirectCorrelation entry.correlation driverEntry.correlation) then
        throw (.unsupportedOperationalExpr id)
      if driver.length == 1 && inputs.all (fun entries => entries.length == 1) then do
        let entries ← inputs.mapM fun entries => match entries with
          | [entry] => pure entry | _ => throw (.unsupportedOperationalExpr id)
        let correlation ← entries.foldlM (fun current entry =>
          match mergeDirectCorrelation current entry.correlation with
          | some merged => pure merged | none => throw (.unsupportedOperationalExpr id)) []
        pure ([(entries.map (·.payload) |>.toArray, correlation,
          entries.flatMap (·.positionPaths), entries)], {})
      else do
        let results ← driver.mapM fun driverEntry => do
          let selected ← inputs.mapM fun entries => match entries with
            | [entry] =>
                if directCorrelationContained entry.correlation driverEntry.correlation then pure entry
                else throw (.unsupportedOperationalExpr id)
            | _ => match entries.filter fun entry =>
                directCorrelationContained entry.correlation driverEntry.correlation with
              | [entry] => pure entry
              | _ => throw (.unsupportedOperationalExpr id)
          pure (selected.map (·.payload) |>.toArray, driverEntry.correlation,
            driverEntry.positionPaths, selected)
        pure (results, {})
  | none => throw (.invalidCount owner 0)

private def alignDirectCorrelationEntries {α : Type}
    (owner id : Nat)
    (inputs : List (List (DirectCorrelationEntry α))) :
    Except OperationalError (List (Array α × DirectCorrelationEnvironment × List DirectGatherPositionPath ×
      List (DirectCorrelationEntry α))) := do
  let (results, _) ← alignDirectCorrelationEntriesWithWork owner id inputs
  pure results

/-- Reporting-only work counter for the same gather-position zipper used by direct pointwise
reduction.  It accepts already reduced operand lanes so fixtures can assert the exact linear
index-build and lookup work without a synthetic estimator or acceptance dependency. -/
def directGatherAlignmentWorkForMatrixInputs
    (owner id : Nat) (inputs : List (List ReducedDirectMatrixFact)) :
    Except OperationalError DirectGatherAlignmentWork := do
  let inputs := inputs.map fun entries => entries.map fun entry => {
    correlation := entry.correlation, payload := entry.fact, positionPaths := entry.positionPaths }
  let (_, work) ← alignDirectCorrelationEntriesWithWork owner id inputs
  pure work

/-- Force failure-only evidence at the single correlation zipper.  This shows whether a mapped
zip produced incompatible owner-bearing environments before any operation-specific transfer is
attempted; it does not enumerate or merge missing Cartesian combinations. -/
private def directCorrelationAlignmentFailureDiagnostic {α : Type}
    (owner id : Nat) (inputs : List (List (DirectCorrelationEntry α)))
    (error : OperationalError) : Bool :=
  let correlations := inputs.map fun entries => entries.map (·.correlation)
  let positionPaths := inputs.map fun entries => entries.map (·.positionPaths)
  operationalProgress "direct_correlation_alignment" "failure" "" id inputs.length
    ("owner=" ++ toString owner ++ "; correlations=" ++ reprStr correlations ++
      "; position_paths=" ++ reprStr positionPaths ++
      "; error=" ++ reprStr error)

/-- The sole correlation zipper for mixed direct relation operands.  It selects a pre-existing
most-specific lane and permits only operands whose assignments are a compatible subset; it never
creates a Cartesian product of independent tables. -/
def alignDirectRelationArguments
    (arena : DirectOperationalIndexedArena) (owner id : Nat)
    (inputs : List (List ReducedDirectRelationArgument)) :
    Except OperationalError (List (Array DirectRelationArgument × DirectCorrelationEnvironment ×
      List DirectGatherPositionPath × List (DirectCorrelationEntry DirectRelationArgument))) :=
  (alignDirectCorrelationEntries owner id
    (inputs.map (fun entries => entries.map fun entry => {
      correlation := entry.correlation, payload := entry.payload, positionPaths := entry.positionPaths }))).map
    (List.map fun (arguments, correlation, paths, inputEntries) =>
      (arguments, correlation, paths, inputEntries))

/-- Recover exact integer table lanes from the compact direct carrier.  This is deliberately
limited to fixed integer leaves and capture-free maps: a gathered family selection may use those
lanes to correlate its runtime position with the owning loop coordinate, but an interval-only or
computed scalar producer remains an unresolved gather and is handled conservatively below. -/
partial def directFixedIntegerSelections
    (arena : DirectOperationalIndexedArena)
    (root : OperationalIndexedValueId) : Nat → Option (List (IndexExpr × Nat × Int))
  | 0 => none
  | fuel + 1 => do
      let value ← arena.valueAt? root
      match value.payload with
      | .shared (.scalar .integer) (.scalar reference) => do
          let fact ← arena.fixed.scalars[reference]?
          match fact with
          | .integer integer => if integer.lower == integer.upper then
              some [(.constant 0, 0, integer.lower)] else none
          | _ => none
      | .explicit (.scalar .integer) binder references =>
          references.toList.mapIdxM fun ordinal reference => do
            match reference with
            | .scalar reference => do
                let fact ← arena.fixed.scalars[reference]?
                match fact with
                | .integer integer => if integer.lower == integer.upper then
                    some (.variable binder, ordinal, integer.lower) else none
                | _ => none
            | .matrix _ => none
      | .mapped (.scalar .integer) source map => do
          let selections ← directFixedIntegerSelections arena source fuel
          selections.mapM fun (key, ordinal, selected) => do
            let key ← reindex map key
            some (key, ordinal, selected)
      | .rebound (.scalar .integer) source _ =>
          directFixedIntegerSelections arena source fuel
      | _ => none

/-- The lookup position of a gather is an executable index expression, not the integer
producer's table binder.  Recover its evaluated half-open domain before attaching an exact
producer lane to that position, so a malformed producer table cannot introduce an impossible
position assignment. -/
private def gatherPositionDomain
    (parameters : ParamEnvironment) : IndexExpr → Except OperationalError (Int × Int)
  | .constant value => pure (value, value + 1)
  | .variable binder => do
      let count ← match binder.count.evaluate parameters with
        | some count => pure count
        | none => throw .nonClosedExpression
      if count <= 0 then throw .nonClosedExpression
      pure (0, count)
  | .offset base amount => do
      let (lower, upper) ← gatherPositionDomain parameters base
      pure (lower + amount, upper + amount)
  | .gather _ sourceCount _ => do
      let count ← match sourceCount.evaluate parameters with
        | some count => pure count
        | none => throw .nonClosedExpression
      if count <= 0 then throw .nonClosedExpression
      pure (0, count)

private def gatherPositionAssignments
    (arena : DirectOperationalIndexedArena)
    (parameters : ParamEnvironment)
    (id : OperationalIndexedValueId)
    (gathered : IndexExpr)
    (owner : GatherLookupOwner)
    (sourceCount : IntExpr)
    (position : IndexExpr)
    (ordinal : Nat) : Except OperationalError (List DirectCorrelationEnvironment) := do
  let bound ← match sourceCount.evaluate parameters with
    | some bound => pure bound | none => throw .nonClosedExpression
  if bound <= 0 || ordinal >= bound.toNat then return []
  let registered ← match arena.gatherIntegerRoot? owner with
    | some registered => pure registered | none => throw (.unsupportedOperationalExpr id)
  let root := registered.root
  let rootValue ← match arena.valueAt? root with
    | some value => pure value | none => throw (.invalidOperationalExprRef root)
  /- For an exact integer table, recover `(transported selection key, physical lane, selected
  value)`.  Enumerate the executable lookup position separately: a mapped table can select its
  physical lane through `position + offset`, so the two are equal only when the transported key
  proves it.  The constraints retain all three facts without conflating their owners.  If the
  producer is only interval-known, retain the conservative unresolved gather alternative rather
  than inventing a position-to-ordinal function. -/
  match rootValue.payload.schema with
  | .scalar .integer =>
      match directFixedIntegerSelections arena root (arena.values.size + 1) with
      | some selections => do
          let (lower, upper) ← gatherPositionDomain parameters position
          let positions := List.range (upper - lower).toNat |>.map fun offset => lower + offset
          let binder := registered.position
          let candidates ← selections.foldlM (fun retained (key, physicalLane, selected) => do
            if selected != Int.ofNat ordinal then pure retained else do
            let matchingPositions ← positions.filterMapM fun executablePosition => do
              if executablePosition < 0 then pure none else do
                let selectedLane ← exactGatherIndex arena parameters rootValue.context
                  [(.variable binder, executablePosition)] key (arena.values.size + 1)
                if selectedLane == physicalLane then
                  pure (some [(gathered, selected.toNat),
                    (canonicalDirectCorrelationKey position, executablePosition.toNat),
                    (canonicalDirectCorrelationKey key, physicalLane)])
                else pure none
            pure (retained ++ matchingPositions)) []
          pure candidates
      | none => pure [[(gathered, ordinal)]]
  | _ => throw (.unsupportedOperationalExpr root)

private def transportDirectCorrelationComponent
    (arena : DirectOperationalIndexedArena)
    (parameters : ParamEnvironment)
    (id : OperationalIndexedValueId)
    (maps : List IndexMap)
    (key : DirectCorrelationKey)
    (ordinal : Nat) : Except OperationalError (List DirectCorrelationEnvironment) := do
  let mut key := key
  let mut ordinal := ordinal
  for map in maps do
    let source := key
    /- A gather's codomain is an `IntExpr` domain witness, so its free index atoms contain only
    the runtime position.  It nevertheless substitutes the mapped source family lane; always
    transport the complete gather key through that map. -/
    let transports := match source with
      | .gather _ _ _ => true
      | _ => source.freeVariables.any map.source.binders.contains
    if transports then
      let translated ← match reindex map source with
        | some translated => pure translated
        | none => throw (.unsupportedOperationalExpr id)
      match translated with
      | .constant lane =>
          if ordinal != lane then return []
          return [[]]
      | .offset (.constant lane) amount =>
          let lane := Int.ofNat lane + amount
          if lane < 0 || ordinal != lane.toNat then return []
          return [[]]
      | .variable destination => key := .variable destination
      | .offset (.variable destination) amount =>
          let destinationOrdinal := Int.ofNat ordinal - amount
          let count ← match destination.count.evaluate parameters with
            | some count => pure count
            | none => throw .nonClosedExpression
          if destinationOrdinal < 0 || destinationOrdinal >= count then return []
          key := .variable destination
          ordinal := destinationOrdinal.toNat
      /- A gather is a dependent function application.  `ordinal` remains the physical source
      table lane, while the complete owner-bearing gather expression names the runtime lookup. -/
      | gathered@(.gather owner sourceCount position) =>
          return ← gatherPositionAssignments arena parameters id gathered owner sourceCount position ordinal
      | .offset _ _ => throw (.unsupportedOperationalExpr id)
  pure [[(key, ordinal)]]

private def transportDirectCorrelation
    (arena : DirectOperationalIndexedArena)
    (parameters : ParamEnvironment)
    (id : OperationalIndexedValueId)
    (maps : List IndexMap)
    (correlation : DirectCorrelationEnvironment) : Except OperationalError (List DirectCorrelationEnvironment) := do
  let mut alternatives : List DirectCorrelationEnvironment := [[]]
  for assignment in correlation do
    let components ← transportDirectCorrelationComponent arena parameters id maps assignment.1 assignment.2
    alternatives := alternatives.flatMap fun accumulated =>
      components.filterMap fun component =>
        normalizeDirectCorrelation (accumulated ++ component)
  pure alternatives

/-- Materialize the pending mapped-view stack only at a fixed leaf.  Recursive reduction prepends
each enclosing map while descending, so the resulting list is inner-to-outer and is applied in
its stored order. -/
private def reindexReducedMatrixFact
    (parameters : ParamEnvironment) (maps : List IndexMap) (fact : OperationalMatrixFact) :
    Except OperationalError OperationalMatrixFact :=
  maps.foldlM (fun fact map => match reindexOperationalMatrixFact parameters map fact with
    | some fact => pure fact
    | none =>
        if operationalDiagnostic "reindex" ("payload=matrix_fact reason=" ++
            operationalMatrixFactReindexFailureField parameters map fact ++ " source=" ++
            reprStr map.source ++ " destination=" ++ reprStr map.destination ++
            " assignments=" ++ reprStr map.assignments) then
          throw (.unsupportedOperationalExpr 0)
        else throw (.unsupportedOperationalExpr 0)) fact

private def reindexReducedScalarFact
    (parameters : ParamEnvironment) (maps : List IndexMap) (fact : OperationalScalarFact) :
    Except OperationalError OperationalScalarFact :=
  maps.foldlM (fun fact map => match reindexOperationalScalarFact parameters map fact with
    | some fact => pure fact
    | none => throw (.unsupportedOperationalExpr 0)) fact

private def reindexReducedPointwiseOperation
    (parameters : ParamEnvironment) (maps : List IndexMap)
    (operation : OperationalIndexedPointwiseOperation) :
    Except OperationalError OperationalIndexedPointwiseOperation :=
  maps.foldlM (fun operation map => match reindexOperationalIndexedPointwiseOperation parameters map operation with
    | some operation => pure operation
    | none =>
        if operationalDiagnostic "reindex" ("payload=pointwise_operation descriptor_kind=" ++
            operationalPointwiseOperationDescriptorKind operation ++ " reason=" ++
            operationalPointwiseOperationReindexFailureField map operation ++ " source=" ++
            reprStr map.source ++ " destination=" ++ reprStr map.destination ++
            " assignments=" ++ reprStr map.assignments) then
          throw (.unsupportedOperationalExpr 0)
        else throw (.unsupportedOperationalExpr 0)) operation

/-- A closed parallel output carries an executable selection identity in addition to the delayed
payload below it.  When an enclosing zip reindexes that output, reduction must transport this
identity through the same owner-aware maps as the fixed leaves and pointwise descriptors.  A
direct-carrier context lift has no source coordinate to substitute, so it deliberately leaves the
already-installed lexical selection unchanged. -/
private def reindexReducedDynamicSelection
    (maps : List IndexMap) (selection : DynamicSelectionIdentity) :
    Except OperationalError DynamicSelectionIdentity :=
  maps.foldlM (fun selection map =>
    if map.isDirectCarrierContextLift then pure selection else
    match reindexDynamicSelectionIdentity map selection with
    | some selection => pure selection
    | none => throw (.unsupportedOperationalExpr 0)) selection

/-- Matrix and scalar reduction cross the same ordered map stack.  A relation descriptor must
use that final context when closing its indexed fields, rather than the source context stored at
the delayed pointwise root. -/
private def reducedDirectMatrixContext
    (source : IndexContext) (maps : List IndexMap) : IndexContext :=
  maps.foldl (fun _ map => map.destination) source

private def reducedDirectPayloadSchemaDiagnostic : OperationalIndexedPayloadSchema → String
  | .matrix matrixType => "matrix:" ++ reprStr matrixType
  | .scalar .integer => "scalar:integer"
  | .scalar .boolean => "scalar:boolean"
  | .scalar .real => "scalar:real"
  | .scalar (.trapdoor _ _ _ _ _) => "scalar:trapdoor"
  | .scalar (.bytes _) => "scalar:bytes"
  | .scalar (.typedBlob _ _) => "scalar:typed_blob"
  | .scalar (.unknown _) => "scalar:unknown"

private def reducedDirectPrimitiveOperationKindDiagnostic : PrimitiveOperationKind → String
  | .add false => "matrix:add"
  | .add true => "matrix:subtract"
  | .multiply _ _ => "matrix:multiply"
  | .tensor => "matrix:tensor"
  | .concat .rows => "matrix:concat_rows"
  | .concat .columns => "matrix:concat_columns"
  | .concat .diagonal => "matrix:concat_diagonal"
  | .transform .negate => "matrix:transform_negate"
  | .transform .transpose => "matrix:transform_transpose"
  | .transform (.rowSlice _ _) => "matrix:transform_row_slice"
  | .transform (.columnSlice _ _) => "matrix:transform_column_slice"
  | .transform (.rowEmbed _ _) => "matrix:transform_row_embed"
  | .transform (.columnEmbed _ _) => "matrix:transform_column_embed"
  | .slice _ _ => "matrix:slice"
  | .scale _ _ => "matrix:scale"
  | .bggGrouping => "matrix:bgg_grouping"

private def reducedDirectPointwiseOperationDiagnostic : OperationalIndexedPointwiseOperation → String
  | .matrix operation => reducedDirectPrimitiveOperationKindDiagnostic operation.kind ++
      "; owner=" ++ toString operation.ownerNode ++ "; output_port=" ++ toString operation.outputPort ++
      "; output_schema=" ++ reprStr operation.outputSchema
  | .relation operation =>
      let kind := match operation.kind with
        | .preimage _ _ => "relation:preimage"
        | .decomposition _ _ _ _ _ _ => "relation:decomposition"
      kind ++ "; owner=" ++ toString operation.ownerNode ++ "; output_port=" ++
        toString operation.outputPort ++ "; output_schema=" ++ reprStr operation.outputSchema
  | .scalar operation =>
      let kind := match operation.kind with
        | .boolToInt => "scalar:bool_to_int"
        | .intBinary .add => "scalar:int_add"
        | .intBinary .subtract => "scalar:int_subtract"
        | .intBinary .multiply => "scalar:int_multiply"
        | .intBinary .divide => "scalar:int_divide"
        | .intBinary .remainder => "scalar:int_remainder"
        | .intCompare .equal => "scalar:int_equal"
        | .intCompare .less => "scalar:int_less"
        | .intCompare .lessEqual => "scalar:int_less_equal"
        | .bitExtract _ => "scalar:bit_extract"
        | .intToReal => "scalar:int_to_real"
        | .realBinary .add => "scalar:real_add"
        | .realBinary .subtract => "scalar:real_subtract"
        | .realBinary .multiply => "scalar:real_multiply"
        | .realBinary .divide => "scalar:real_divide"
        | .realSqrt => "scalar:real_sqrt"
      kind ++ "; owner=" ++ toString operation.ownerNode ++ "; output_port=" ++
        toString operation.outputPort
  | .matrixToScalar operation =>
      let kind := match operation.kind with
        | .extractCoefficient _ => "matrix_to_scalar:extract_coefficient"
        | .thresholdDecodeBool _ _ _ => "matrix_to_scalar:threshold_decode_bool"
        | .thresholdDecodeInt _ _ _ => "matrix_to_scalar:threshold_decode_int"
      kind ++ "; owner=" ++ toString operation.ownerNode ++ "; output_port=" ++
        toString operation.outputPort
  | .matrixFromScalar operation =>
      let kind := match operation.kind with
        | .liftIntegerToConstantPolynomial _ => "matrix_from_scalar:lift_integer"
        | .trapdoorPublic _ => "matrix_from_scalar:trapdoor_public"
      kind ++ "; owner=" ++ toString operation.ownerNode ++ "; output_port=" ++
        toString operation.outputPort

/-- Failure-only structural telemetry for delayed direct reduction.  This is intentionally kept
at the reduction boundary: it records the concrete payload and complete pending-map stack that
produced a fail-closed result, without adding a success-path traversal or a protocol-specific
exception. -/
private def reducedDirectPayloadDiagnostic : OperationalIndexedPayload → String
  | .shared schema reference => "shared; schema=" ++ reducedDirectPayloadSchemaDiagnostic schema ++
      "; reference=" ++ reprStr reference
  | .explicit schema binder references =>
      "explicit; schema=" ++ reducedDirectPayloadSchemaDiagnostic schema ++ "; binder=" ++ reprStr binder ++
        "; entries=" ++ toString references.size
  | .explicitValues schema binder values =>
      "explicit_values; schema=" ++ reducedDirectPayloadSchemaDiagnostic schema ++ "; binder=" ++
        reprStr binder ++ "; entries=" ++ toString values.size
  | .mapped schema source map =>
      "mapped; schema=" ++ reducedDirectPayloadSchemaDiagnostic schema ++ "; source=" ++ toString source ++
        "; map=" ++ reprStr map
  | .rebound schema source subject =>
      "rebound; schema=" ++ reducedDirectPayloadSchemaDiagnostic schema ++ "; source=" ++ toString source ++
        "; subject=" ++ reprStr subject
  | .indexedOutput schema source binder selection subject =>
      "indexed_output; source=" ++ toString source ++ "; binder=" ++ reprStr binder ++
        "; schema=" ++ reducedDirectPayloadSchemaDiagnostic schema ++ "; selection=" ++ reprStr selection ++
        "; subject=" ++ reprStr subject
  | .matrixResultBound schema source _ =>
      "matrix_result_bound; schema=" ++ reducedDirectPayloadSchemaDiagnostic schema ++
        "; source=" ++ toString source
  | .pointwise schema operation inputs =>
      "pointwise; schema=" ++ reducedDirectPayloadSchemaDiagnostic schema ++ "; operation=" ++
        reducedDirectPointwiseOperationDiagnostic operation ++ "; inputs=" ++ reprStr inputs

private def reducedDirectPointwiseFailureDiagnostic
    (phase : String) (id : OperationalIndexedValueId) (operation : OperationalIndexedPointwiseOperation)
    (maps : List IndexMap) (detail : String) (error : OperationalError) : Bool :=
  operationalProgress "reduced_direct_pointwise" phase "" id maps.length
    ("operation=" ++ reducedDirectPointwiseOperationDiagnostic operation ++ "; maps=" ++ reprStr maps ++
      "; " ++ detail ++ "; error=" ++ reprStr error)

private def reducedDirectReductionFailureDiagnostic
    (kind : String) (id : OperationalIndexedValueId) (value : OperationalIndexedValue)
    (maps : List IndexMap) (error : OperationalError) : Bool :=
  operationalProgress "reduced_direct_reduction" (kind ++ "_failure") "" id maps.length
    ("context=" ++ reprStr value.context ++ "; payload=" ++ reducedDirectPayloadDiagnostic value.payload ++
      "; maps=" ++ reprStr maps ++ "; error=" ++ reprStr error)

mutual

private def reducedDirectMatrixFactAt
    (arena : DirectOperationalIndexedArena)
    (parameters : ParamEnvironment)
    (maps : List IndexMap)
    (id : OperationalIndexedValueId) : Nat → Except OperationalError ReducedDirectMatrixEvaluation
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => do
      let value ← match arena.valueAt? id with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef id)
      let evaluation ← try
        match value.payload with
      | .shared (.matrix _) (.matrix reference) => do
          let fact ← match arena.fixed.matrices[reference]? with
            | some fact => pure fact
            | none => throw (.invalidOperationalExprRef reference)
          let fact ← reindexReducedMatrixFact parameters maps fact
          pure { entries := [{ correlation := [], fact }] }
      | .explicit (.matrix _) binder references => do
          let mapped ← references.toList.mapIdxM fun ordinal reference => do
            let fact ← match reference with
              | .matrix reference => match arena.fixed.matrices[reference]? with
                | some fact => pure fact
                | none => throw (.invalidOperationalExprRef reference)
              | .scalar _ => throw (.unsupportedOperationalExpr id)
            let fact ← reindexReducedMatrixFact parameters maps fact
            let correlation := if references.size == 1 then [] else [(.variable binder, ordinal)]
            let correlations ← transportDirectCorrelation arena parameters id maps correlation
            pure (correlations.map fun correlation => {
              correlation, fact, positionPaths := directGatherPositionPaths parameters maps correlation })
          let entries := mapped.flatten
          pure { entries := entries }
      | .explicitValues (.matrix _) binder values => do
          let lanes ← values.toList.mapIdxM fun ordinal child => do
            let outer := if values.size == 1 then [] else [(.variable binder, ordinal)]
            let outer ← transportDirectCorrelation arena parameters id maps outer
            outer.mapM fun outer => do
                let introduced := directGatherPositionPaths parameters maps outer
                let evaluation ← reducedDirectMatrixFactAt arena parameters maps child fuel
                /- A direct-value table adds one owner-bearing table-lane assignment to every
                child alternative.  A conflicting assignment rejects only that physical
                alternative: another child lane may still be compatible with the enclosing
                selection.  `mergeDirectCorrelation` compares the complete `IndexExpr`, hence
                equal slot/count pairs belonging to distinct owners remain independent. -/
                let candidates ← evaluation.entries.mapM fun entry => do
                  match mergeDirectCorrelation outer entry.correlation with
                  | none => pure none
                  | some correlation => do
                      let positionPaths ← match
                          mergeDirectGatherPositionPaths entry.positionPaths introduced with
                        | some paths => pure paths
                        | none => throw (.unsupportedOperationalExpr id)
                      pure (some { entry with correlation, positionPaths })
                let entries := candidates.filterMap fun candidate => candidate
                pure { evaluation with entries }
          let lanes := lanes.flatten
          let entries := lanes.flatMap (fun lane => lane.entries)
          let rewriteEvents := deduplicateDirectRelationRewriteEvents
            (lanes.flatMap (fun lane => lane.rewriteEvents))
          let maximumPolynomialTerms := lanes.foldl (fun maximum lane =>
            max maximum lane.maximumPolynomialTerms) 0
          pure { entries := entries, rewriteEvents := rewriteEvents, maximumPolynomialTerms }
      | .mapped (.matrix _) source map => do
          if !map.transportValid || map.destination != value.context then
            throw (.unsupportedOperationalExpr id)
          reducedDirectMatrixFactAt arena parameters (map :: maps) source fuel
      | .rebound (.matrix _) source subject => do
          let evaluation ← reducedDirectMatrixFactAt arena parameters maps source fuel
          let entries ← evaluation.entries.mapM fun entry => do
            let fact ← rebindMatrixSubject subject entry.fact
            pure { entry with fact }
          pure { evaluation with entries }
      | .indexedOutput (.matrix _) source binder selection subject => do
          let evaluation ← reducedDirectMatrixFactAt arena parameters maps source fuel
          let selection ← reindexReducedDynamicSelection maps selection
          let entries := evaluation.entries.map fun entry =>
            { entry with fact := overlayIndexMatrixFact binder selection subject entry.fact }
          pure { evaluation with entries }
      | .matrixResultBound (.matrix _) source totalHardBound => do
          let evaluation ← reducedDirectMatrixFactAt arena parameters maps source fuel
          /- The pending maps are normally applied at fixed leaves.  This annotation is installed
          after its source has been reduced, so its replacement bound must receive that same
          transport before it overwrites the source result. -/
          let sourceMaps ← directPendingMaps arena source fuel
          let totalHardBound ← (sourceMaps ++ maps).foldlM (fun bound map =>
            match reindexOperationalBoundExpr parameters map bound with
            | some bound => pure bound
            | none => throw (.unsupportedOperationalExpr id)) totalHardBound
          let entries := evaluation.entries.map fun entry =>
            { entry with fact := { entry.fact with totalHardBound } }
          let result : ReducedDirectMatrixEvaluation := {
            entries := entries, rewriteEvents := evaluation.rewriteEvents }
          pure { result with maximumPolynomialTerms := evaluation.maximumPolynomialTerms }
      | .pointwise (.matrix matrixType) (.matrix operation) inputs => do
          let descriptor : OperationalIndexedPointwiseOperation := .matrix operation
          let operation ← try reindexReducedPointwiseOperation parameters maps descriptor catch error =>
            if reducedDirectPointwiseFailureDiagnostic "descriptor_transport" id descriptor maps
                ("payload_schema=" ++ reducedDirectPayloadSchemaDiagnostic (.matrix matrixType)) error then
              throw error
            else throw (.unsupportedOperationalExpr id)
          let operation ← match operation with
            | .matrix operation => pure operation
            | _ => throw (.unsupportedOperationalExpr id)
          let inputEvaluations ← inputs.toList.mapIdxM fun childIndex childId =>
            try reducedDirectMatrixFactAt arena parameters maps childId fuel catch error =>
              if reducedDirectPointwiseFailureDiagnostic "child_reduction" id descriptor maps
                  ("child_index=" ++ toString childIndex ++ "; child_id=" ++ toString childId) error then
                throw error
              else throw (.unsupportedOperationalExpr id)
          let inputEntries := inputEvaluations.map (·.entries)
          let correlationInputs := inputEntries.map fun entries =>
            entries.map fun entry => {
              correlation := entry.correlation, payload := entry.fact, positionPaths := entry.positionPaths }
          let aligned ← try
            alignDirectCorrelationEntries operation.ownerNode id correlationInputs
          catch error =>
            if directCorrelationAlignmentFailureDiagnostic operation.ownerNode id correlationInputs error then
              throw error
            else throw (.unsupportedOperationalExpr id)
          let entriesAndEvents ← aligned.mapIdxM fun alignmentIndex
              (arguments, correlation, positionPaths, inputEntries) => do
            let (fact, rewriteCount) ← try
              applyDirectMatrixPointwiseOperationWithRelationRewriteCount operation matrixType arguments
              catch error =>
                if reducedDirectPointwiseFailureDiagnostic "primitive_application" id descriptor maps
                    ("alignment_index=" ++ toString alignmentIndex ++ "; correlation=" ++
                      reprStr correlation ++ "; argument_count=" ++ toString arguments.size) error then
                  throw error
                else throw (.unsupportedOperationalExpr id)
            match operation.kind with
            | .multiply (.matrixMultiplyRelation _) _ =>
                if !(← directRelationSelectedInputsCompatible id inputEntries) then
                  throw (.unsupportedOperationalExpr id)
                if rewriteCount == 0 then throw (.unsupportedOperationalExpr id)
            | _ => pure ()
            let pointwise : DirectRelationRewritePointwiseKey := {
              ownerScope := operation.ownerScope, ownerNode := operation.ownerNode,
              outputPort := operation.outputPort }
            let events := (List.range rewriteCount).map fun localOrdinal => {
              pointwise := pointwise,
              maps := normalizeDirectRelationRewriteMaps maps, correlation := correlation,
              localOrdinal := localOrdinal }
            pure ({ correlation, fact, positionPaths }, events)
          let entries := entriesAndEvents.map (fun value => value.1)
          let rewriteEvents := deduplicateDirectRelationRewriteEvents
            (inputEvaluations.flatMap (fun value => value.rewriteEvents) ++
              entriesAndEvents.flatMap (fun value => value.2))
          let maximumPolynomialTerms := inputEvaluations.foldl (fun maximum evaluation =>
            max maximum evaluation.maximumPolynomialTerms) 0
          pure { entries := entries, rewriteEvents := rewriteEvents, maximumPolynomialTerms }
      | .pointwise (.matrix _) (.relation operation) inputs => do
          let operation ← reindexReducedPointwiseOperation parameters maps (.relation operation)
          let operation ← match operation with
            | .relation operation => pure operation
            | _ => throw (.unsupportedOperationalExpr id)
          let inputEvaluations : List (List ReducedDirectRelationArgument ×
              List DirectRelationRewriteEventKey × Nat) ← inputs.toList.mapM fun inputId => do
            let input ← match arena.valueAt? inputId with
              | some value => pure value
              | none => throw (.invalidOperationalExprRef inputId)
            match input.payload.schema with
            | .matrix _ =>
                let evaluation ← reducedDirectMatrixFactAt arena parameters maps inputId fuel
                pure (evaluation.entries.map fun entry =>
                  { correlation := entry.correlation,
                    payload := DirectRelationArgument.matrix entry.fact, positionPaths := entry.positionPaths },
                  evaluation.rewriteEvents,
                  evaluation.maximumPolynomialTerms)
            | .scalar (.trapdoor _ _ _ _ _) =>
                let evaluation ← reducedDirectScalarFactAt arena parameters maps inputId fuel
                let entries ← evaluation.entries.mapM fun entry => do
                  match entry.fact with
                  | OperationalScalarFact.trapdoor fact =>
                      pure (ReducedDirectRelationArgument.mk entry.correlation
                        (DirectRelationArgument.trapdoor fact) entry.positionPaths)
                  | _ => throw (.unsupportedOperationalExpr id)
                pure (entries, evaluation.rewriteEvents, evaluation.maximumPolynomialTerms)
            | .scalar _ => throw (.unsupportedOperationalExpr id)
          let inputEntries := inputEvaluations.map (·.1)
          let aligned ← alignDirectRelationArguments arena operation.ownerNode id inputEntries
          let entries ← aligned.mapM fun
              (arguments, correlation, positionPaths, inputEntries) => do
            if !(← directRelationSelectedInputsCompatible id inputEntries) then
              throw (.unsupportedOperationalExpr id)
            let context := reducedDirectMatrixContext value.context maps
            let indices := correlation.map fun (key, ordinal) => (key, Int.ofNat ordinal)
            let operation ← materializeDirectRelationOperation arena parameters context indices operation
            let fact ← applyDirectRelationProducer operation operation.outputSchema arguments
            pure { correlation, fact, positionPaths }
          let rewriteEvents := deduplicateDirectRelationRewriteEvents
            (inputEvaluations.flatMap (fun value => value.2.1))
          let maximumPolynomialTerms := inputEvaluations.foldl (fun maximum value =>
            max maximum value.2.2) 0
          pure { entries := entries, rewriteEvents := rewriteEvents, maximumPolynomialTerms }
      | .pointwise (.matrix matrixType) (.matrixFromScalar operation) inputs => do
          let operation ← reindexReducedPointwiseOperation parameters maps (.matrixFromScalar operation)
          let operation ← match operation with
            | .matrixFromScalar operation => pure operation
            | _ => throw (.unsupportedOperationalExpr id)
          let input ← match inputs with
            | #[input] => pure input
            | _ => throw (.unsupportedOutputArity operation.ownerNode inputs.size)
          let evaluation ← reducedDirectScalarFactAt arena parameters maps input fuel
          let entries : List ReducedDirectMatrixFact ← evaluation.entries.mapM fun entry => do
            let fact ← applyDirectMatrixFromScalarOperation operation matrixType entry.fact
            pure { correlation := entry.correlation, fact, positionPaths := entry.positionPaths }
          let result : ReducedDirectMatrixEvaluation := {
            entries := entries, rewriteEvents := evaluation.rewriteEvents }
          pure { result with maximumPolynomialTerms := evaluation.maximumPolynomialTerms }
        | _ => throw (.unsupportedOperationalExpr id)
      catch error =>
        if reducedDirectReductionFailureDiagnostic "matrix" id value maps error then throw error
        else throw (.unsupportedOperationalExpr id)
      let observed := evaluation.entries.foldl (fun maximum entry =>
        max maximum entry.fact.polynomial.length) 0
      pure { evaluation with maximumPolynomialTerms :=
        if observed > evaluation.maximumPolynomialTerms then observed
        else evaluation.maximumPolynomialTerms }

private def reducedDirectScalarContext
    (source : IndexContext) (maps : List IndexMap) : IndexContext :=
  maps.foldl (fun _ map => map.destination) source

/-- Matrix and scalar reduction cross the same ordered map stack.  A relation descriptor must
use that final context when closing its indexed fields, rather than the source context stored at
the delayed pointwise root. -/
private def reducedDirectScalarFactAt
    (arena : DirectOperationalIndexedArena)
    (parameters : ParamEnvironment)
    (maps : List IndexMap)
    (id : OperationalIndexedValueId) : Nat → Except OperationalError ReducedDirectScalarEvaluation
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => do
      let value ← match arena.valueAt? id with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef id)
      let evaluation ← try
        match value.payload with
      | .shared (.scalar _) (.scalar reference) => do
          let fact ← match arena.fixed.scalars[reference]? with
            | some fact => pure fact
            | none => throw (.invalidOperationalExprRef reference)
          let fact ← reindexReducedScalarFact parameters maps fact
          let fact ← arena.materializeScalarFact parameters
            (reducedDirectScalarContext value.context maps) [] fact
          pure { entries := [{ correlation := [], fact }] }
      | .explicit (.scalar _) binder references => do
          let mapped ← references.toList.mapIdxM fun ordinal reference => do
            let fact ← match reference with
              | .scalar reference => match arena.fixed.scalars[reference]? with
                | some fact => pure fact
                | none => throw (.invalidOperationalExprRef reference)
              | .matrix _ => throw (.unsupportedOperationalExpr id)
            let fact ← reindexReducedScalarFact parameters maps fact
            let correlation := if references.size == 1 then [] else [(.variable binder, ordinal)]
            let correlations ← transportDirectCorrelation arena parameters id maps correlation
            correlations.mapM fun correlation => do
              let materialized ← arena.materializeScalarFact parameters
                (reducedDirectScalarContext value.context maps)
                (correlation.map fun (key, ordinal) => (key, Int.ofNat ordinal)) fact
              let paths := directGatherPositionPaths parameters maps correlation
              pure ⟨correlation, materialized, paths⟩
          let entries := mapped.flatten
          pure { entries := entries }
      | .explicitValues (.scalar _) binder values => do
          let lanes ← values.toList.mapIdxM fun ordinal child => do
            let outer := if values.size == 1 then [] else [(.variable binder, ordinal)]
            let outer ← transportDirectCorrelation arena parameters id maps outer
            outer.mapM fun outer => do
                let introduced := directGatherPositionPaths parameters maps outer
                let evaluation ← reducedDirectScalarFactAt arena parameters maps child fuel
                let candidates ← evaluation.entries.mapM fun entry => do
                  match mergeDirectCorrelation outer entry.correlation with
                  | none => pure none
                  | some correlation => do
                      let positionPaths ← match
                          mergeDirectGatherPositionPaths entry.positionPaths introduced with
                        | some paths => pure paths
                        | none => throw (.unsupportedOperationalExpr id)
                      pure (some { entry with correlation, positionPaths })
                let entries := candidates.filterMap fun candidate => candidate
                pure { evaluation with entries }
          let lanes := lanes.flatten
          let entries := lanes.flatMap (fun lane => lane.entries)
          let rewriteEvents := deduplicateDirectRelationRewriteEvents
            (lanes.flatMap (fun lane => lane.rewriteEvents))
          let maximumPolynomialTerms := lanes.foldl (fun maximum lane =>
            max maximum lane.maximumPolynomialTerms) 0
          pure { entries := entries, rewriteEvents := rewriteEvents, maximumPolynomialTerms }
      | .mapped (.scalar _) source map => do
          if !map.transportValid || map.destination != value.context then
            throw (.unsupportedOperationalExpr id)
          reducedDirectScalarFactAt arena parameters (map :: maps) source fuel
      | .rebound (.scalar _) source subject => do
          let evaluation ← reducedDirectScalarFactAt arena parameters maps source fuel
          let entries := evaluation.entries.map fun entry =>
            { entry with fact := rebindOperationalScalarFact subject entry.fact }
          pure { evaluation with entries }
      | .indexedOutput (.scalar _) source binder selection subject => do
          let evaluation ← reducedDirectScalarFactAt arena parameters maps source fuel
          let selection ← reindexReducedDynamicSelection maps selection
          let entries := evaluation.entries.map fun entry =>
            { entry with fact := overlayIndexScalarFact binder selection subject entry.fact }
          pure { evaluation with entries }
      | .pointwise (.scalar _) (.matrixToScalar operation) inputs => do
          let operation ← reindexReducedPointwiseOperation parameters maps (.matrixToScalar operation)
          let operation ← match operation with
            | .matrixToScalar operation => pure operation
            | _ => throw (.unsupportedOperationalExpr id)
          let input ← match inputs with
            | #[input] => pure input
            | _ => throw (.unsupportedOutputArity operation.ownerNode inputs.size)
          let evaluation ← reducedDirectMatrixFactAt arena parameters maps input fuel
          let entries ← evaluation.entries.mapM fun entry => do
            let fact ← applyDirectMatrixToScalarOperation operation entry.fact
            pure { correlation := entry.correlation, fact, positionPaths := entry.positionPaths }
          let result : ReducedDirectScalarEvaluation := {
            entries := entries, rewriteEvents := evaluation.rewriteEvents }
          pure { result with maximumPolynomialTerms := evaluation.maximumPolynomialTerms }
      | .pointwise (.scalar _) (.scalar operation) inputs => do
          let operation ← reindexReducedPointwiseOperation parameters maps (.scalar operation)
          let operation ← match operation with
            | .scalar operation => pure operation
            | _ => throw (.unsupportedOperationalExpr id)
          let inputEvaluations ← inputs.toList.mapM fun input =>
            reducedDirectScalarFactAt arena parameters maps input fuel
          let inputEntries := inputEvaluations.map (·.entries)
          let correlationInputs := inputEntries.map fun entries =>
            entries.map fun entry => {
              correlation := entry.correlation, payload := entry.fact, positionPaths := entry.positionPaths }
          let aligned ← try
            alignDirectCorrelationEntries operation.ownerNode id correlationInputs
          catch error =>
            if directCorrelationAlignmentFailureDiagnostic operation.ownerNode id correlationInputs error then
              throw error
            else throw (.unsupportedOperationalExpr id)
          let entries ← aligned.mapM fun (arguments, correlation, positionPaths, _) => do
            let fact ← applyDirectScalarPointwiseOperation operation arguments
            pure { correlation, fact, positionPaths }
          let rewriteEvents := deduplicateDirectRelationRewriteEvents
            (inputEvaluations.flatMap (fun value => value.rewriteEvents))
          let maximumPolynomialTerms := inputEvaluations.foldl (fun maximum evaluation =>
            max maximum evaluation.maximumPolynomialTerms) 0
          pure { entries := entries, rewriteEvents := rewriteEvents, maximumPolynomialTerms }
        | _ => throw (.unsupportedOperationalExpr id)
      catch error =>
        if reducedDirectReductionFailureDiagnostic "scalar" id value maps error then throw error
        else throw (.unsupportedOperationalExpr id)
      pure evaluation

/-- Hull the authoritative direct scalar reduction.  This is the only interval endpoint used by
dynamic family selection, so scalar pointwise operations (notably Tall's coefficient remainder)
share exactly the same semantics as ordinary direct reduction. -/
def DirectOperationalIndexedArena.integerInterval
    (arena : DirectOperationalIndexedArena) (owner : Nat) (wire : WireRef)
    (id : OperationalIndexedValueId) : Nat → Except OperationalError (Int × Int)
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => do
      let evaluation ← reducedDirectScalarFactAt arena [] [] id fuel
      let intervals ← evaluation.entries.mapM fun entry => match entry.fact with
        | .integer fact => pure (fact.lower, fact.upper)
        | _ => throw (.operandNotInteger owner wire)
      if operationalProgress "direct_integer_interval" "entries" "" owner intervals.length
          ("wire=" ++ reprStr wire ++ "; root=" ++ toString id ++ "; intervals=" ++
            reprStr intervals ++ "; correlations=" ++
            reprStr (evaluation.entries.map (·.correlation))) then pure () else
        throw (.unsupportedOperationalExpr id)
      match intervals with
      | [] => throw (.invalidCount owner 0)
      | first :: rest => pure <| rest.foldl (fun (lower, upper) (nextLower, nextUpper) =>
          (min lower nextLower, max upper nextUpper)) first

end

/-- Reduce storage directly to concrete physical lanes while preserving only proven shared
correlation.  Unlike `directValueFactsAt`, this never enumerates a root `IndexValueEnvironment`.
Independent driver keys are rejected before any matrix operation can form a Cartesian product. -/
def OperationalExprArena.reducedDirectValueFactsAt
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (expression : IndexedOperationalFact) : Except OperationalError (List ReducedDirectMatrixFact) := do
  let root ← match expression.payload with
    | .directValue root => pure root
  let entries ← (← reducedDirectMatrixFactAt arena.direct environment [] root
    (arena.direct.values.size + 1)).entries.mapM fun entry => do
      pure entry
  pure entries

/-- Scalar companion to `reducedDirectValueFactsAt`.  Sequential recurrences use this to retain
integer lower and upper components for every direct physical lane instead of reducing a scalar
family to an arbitrary representative. -/
def OperationalExprArena.reducedDirectScalarValueFactsAt
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (expression : IndexedOperationalFact) : Except OperationalError (List ReducedDirectScalarFact) := do
  let root ← match expression.payload with
    | .directValue root => pure root
  let evaluation ← reducedDirectScalarFactAt arena.direct environment [] root
    (arena.direct.values.size + 1)
  let entries ← evaluation.entries.mapM fun entry => do
      pure entry
  pure entries

/-- Structural direct reduction plus the deduplicated exact relation applications it performed.
This is reporting-only: acceptance consumes the public fact projection above and fixed-assignment
queries never invoke it. -/
def OperationalExprArena.reducedDirectValueFactsAtWithRelationRewriteEvents
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (expression : IndexedOperationalFact) :
    Except OperationalError (List ReducedDirectMatrixFact × List DirectRelationRewriteEventKey) := do
  let root ← match expression.payload with
    | .directValue root => pure root
  let evaluation ← reducedDirectMatrixFactAt arena.direct environment [] root
    (arena.direct.values.size + 1)
  pure (evaluation.entries, deduplicateDirectRelationRewriteEvents evaluation.rewriteEvents)

/-- Reporting-only structural reduction telemetry.  This exposes the maximum normalized
polynomial width reached by the actual direct evaluation path without participating in
acceptance. -/
def OperationalExprArena.reducedDirectValueFactsAtWithDiagnostics
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (expression : IndexedOperationalFact) :
    Except OperationalError (List ReducedDirectMatrixFact × List DirectRelationRewriteEventKey × Nat) := do
  let root ← match expression.payload with
    | .directValue root => pure root
  let evaluation ← reducedDirectMatrixFactAt arena.direct environment [] root
    (arena.direct.values.size + 1)
  pure (evaluation.entries, deduplicateDirectRelationRewriteEvents evaluation.rewriteEvents,
    evaluation.maximumPolynomialTerms)

/-- Sequential recurrences consume a direct carrier through its fixed assignments.  This keeps
the recurrence's numeric state independent of storage while rejecting a non-uniform carried
schema instead of summarizing an arbitrary representative. -/
def sequentialFactNumericExpressions
    (arena : OperationalExprArena)
    (slot : Nat)
    (fact : OperationalFact) : Except OperationalError
  (List (OperationalBoundPath × OperationalBoundExpr)) :=
  match fact with
  | expression@{ payload := .directValue _, .. } => do
      let value ← match arena.direct.valueAt? expression.payload.root with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef expression.payload.root)
      match value.payload.schema with
      | .matrix _ =>
          let entries ← arena.reducedDirectValueFactsAt [] expression
          let maximum ← match entries with
            | [] => throw (.invalidCount slot 0)
            | first :: remaining => pure <| remaining.foldl (fun bound entry =>
                .maximum bound entry.fact.totalHardBound) first.fact.totalHardBound
          pure [(.matrixMaximum 0 slot, maximum)]
      | .scalar _ =>
          let entries ← arena.reducedDirectScalarValueFactsAt [] expression
          let mergeComponent : List (OperationalBoundPath × OperationalBoundExpr) →
              OperationalBoundPath × OperationalBoundExpr → List (OperationalBoundPath × OperationalBoundExpr) :=
            fun accumulated component =>
              if accumulated.any (fun existing => existing.1 == component.1) then
                accumulated.map fun existing =>
                  if existing.1 != component.1 then existing else
                    (existing.1, match existing.1 with
                      | .integerLower .. => .minimum existing.2 component.2
                      | .matrixMaximum .. | .integerUpper .. => .maximum existing.2 component.2)
              else accumulated ++ [component]
          pure <| entries.foldl (fun accumulated entry =>
            (scalarFactNumericExpressions slot entry.fact).foldl mergeComponent accumulated) []

def sameSequentialCarriedSchema
    (arena : OperationalExprArena)
    (left right : OperationalFact) : Bool :=
  match left, right with
  | left@{ payload := .directValue _, .. }, right@{ payload := .directValue _, .. } =>
      left.context.binders.toList.all (right.context.binders.contains) &&
        right.context.binders.toList.all (left.context.binders.contains) &&
      match arena.direct.valueAt? left.payload.root, arena.direct.valueAt? right.payload.root with
      | some leftValue, some rightValue => match leftValue.payload.schema, rightValue.payload.schema with
        | .matrix _, .matrix _ =>
            match arena.reducedDirectValueFactsAt [] left, arena.reducedDirectValueFactsAt [] right with
            | .ok leftFacts, .ok rightFacts => leftFacts.length == rightFacts.length &&
                (leftFacts.zip rightFacts).all fun (left, right) =>
                  left.correlation == right.correlation &&
                    sameCarriedMatrixFactSchema left.fact right.fact
            | _, _ => false
        | .scalar _, .scalar _ =>
            match arena.reducedDirectScalarValueFactsAt [] left,
                arena.reducedDirectScalarValueFactsAt [] right with
            | .ok leftFacts, .ok rightFacts => leftFacts.length == rightFacts.length &&
                (leftFacts.zip rightFacts).all fun (left, right) =>
                  left.correlation == right.correlation &&
                    scalarSchemaTag left.fact == scalarSchemaTag right.fact
            | _, _ => false
        | _, _ => false
      | _, _ => false

def sequentialCarriedLargeFactorCounts
    (arena : OperationalExprArena)
    (fact : OperationalFact) : Except OperationalError (List Nat) :=
  match fact with
  | expression@{ payload := .directValue _, .. } => do
      let entries ← arena.reducedDirectValueFactsAt [] expression
      pure <| entries.flatMap fun entry => entry.fact.polynomial.map operationalLargeFactorCount


/-- Group the already-derived exact signal part of a BGG encoding while retaining its bounded
noise as a separate top-level term.  The complete pre-grouping signal polynomial is embedded in a
flat token sequence, so this cannot create a false cancellation or hide bounded noise.  The paired
public-key/plaintext origins identify the one executable BGG value selected at runtime. -/
def groupExactSignal
    (identityTokens : List OperationalCompressionToken)
    (vector : OperationalMatrixFact) :
    Except OperationalFlatError OperationalMatrixFact := do
  let signal := sortOperationalTerms (vector.polynomial.filter operationalTermIsSignal)
  let noise := vector.polynomial.filter operationalTermIsNoise
  if signal.isEmpty then
    return { vector with polynomial := (← compressBoundedNoiseSum noise) }
  let tokens := [.groupStart] ++ identityTokens ++ [.sumStart] ++
    signal.flatMap operationalProductTokens ++
    [.sumEnd, .intermediateType vector.matrixType, .groupEnd]
  let factor : OperationalFactorKey := {
    leaf := .exactTransform tokens vector.matrixType
    inputType := vector.matrixType
    outputType := vector.matrixType
    role := .large
  }
  let groupedSignal : OperationalTerm := {
    coefficient := 1
    product := { factors := [factor], modes := [], outputType := vector.matrixType }
  }
  let compressedNoise ← compressBoundedNoiseSum noise
  pure { vector with polynomial := groupedSignal :: compressedNoise }

def groupBggEncodingSignal
    (vector publicKey plaintext : OperationalMatrixFact) :
    Except OperationalFlatError OperationalMatrixFact :=
  groupExactSignal
    [.primitive (.matrix publicKey.origin), .primitive (.matrix plaintext.origin)] vector

def groupPublicKeySignal
    (fact : OperationalMatrixFact) : Except OperationalError OperationalMatrixFact :=
  groupExactSignal [] fact |>.mapError (.flat 0)

/-- Promote the output of a separately validated exact Boolean carrier selection to one Large
signal factor. The validator below proves that this value is exactly `select(bit, zero, carrier)`
with a deterministic constant carrier, so this grouping cannot hide sampler noise. -/
def groupProtocolBooleanSignal
    (fact : OperationalMatrixFact) : OperationalMatrixFact :=
  let tokens := [
    .groupStart,
    .primitive (.matrix fact.origin),
    .intermediateType fact.matrixType,
    .groupEnd
  ]
  let factor : OperationalFactorKey := {
    leaf := .exactTransform tokens fact.matrixType
    inputType := fact.matrixType
    outputType := fact.matrixType
    role := .large
  }
  let term : OperationalTerm := {
    coefficient := 1
    product := { factors := [factor], modes := [], outputType := fact.matrixType }
  }
  { fact with polynomial := [term], metadata := {} }


def derivationAttachmentRole
    (attachment : DerivationAttachment)
    (role : String) : Except OperationalError WireRef :=
  match attachment.roles.filter (fun candidate => candidate.1 == role) with
  | [(_, wire)] => pure wire
  | _ => throw (.missingDerivationAttachmentRole attachment.ownerNamespace attachment.ruleName role)

def validateDerivationAttachment
    (scope : Scope) (attachment : DerivationAttachment) : Except OperationalError Unit := do
  let isBggEncoding := attachment.ownerNamespace == "mxx-bgg" &&
    attachment.ruleName == "encoding-family-pairing"
  let isBggPublicKey := attachment.ownerNamespace == "mxx-bgg" &&
    attachment.ruleName == "public-key-signal-grouping"
  let isProtocolBoolean := attachment.ownerNamespace == "mxx-correctness" &&
    attachment.ruleName == "protocol-boolean-signal-grouping"
  if !(isBggEncoding || isBggPublicKey || isProtocolBoolean) then
    throw (.unknownDerivationAttachment attachment.ownerNamespace attachment.ruleName)
  let required := if isBggEncoding then ["vector", "public-key", "plaintext"]
    else if isProtocolBoolean then ["value", "selector", "zero", "carrier"]
    else ["value"]
  for role in required do
    let wire ← derivationAttachmentRole attachment role
    let outputCount ← match scope.nodes[wire.node]? with
      | some node => pure node.outputCount
      | none => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    if wire.port >= outputCount then
      throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
  if attachment.roles.length != required.length then
    throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
  if isProtocolBoolean then
    let valueWire ← derivationAttachmentRole attachment "value"
    let selectorWire ← derivationAttachmentRole attachment "selector"
    let zeroWire ← derivationAttachmentRole attachment "zero"
    let carrierWire ← derivationAttachmentRole attachment "carrier"
    let valueNode ← match scope.nodes[valueWire.node]? with
      | some value => pure value
      | none => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    let indexWire ← match valueNode.kind, valueNode.arguments with
      | .select, [index, zero, carrier] =>
          if valueWire.port != 0 || zero != zeroWire || carrier != carrierWire then
            throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
          pure index
      | _, _ => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    let indexNode ← match scope.nodes[indexWire.node]? with
      | some value => pure value
      | none => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    match indexNode.kind, indexNode.arguments with
    | .boolToInt, [source] =>
        if indexWire.port != 0 || source != selectorWire then
          throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    | _, _ => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    let selectorNode ← match scope.nodes[selectorWire.node]? with
      | some value => pure value
      | none => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    if selectorWire.port != 0 || selectorNode.outputTypes != [.boolean] then
      throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    match selectorNode.kind with
    | .input _ => pure ()
    | _ => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    let zeroNode ← match scope.nodes[zeroWire.node]? with
      | some value => pure value
      | none => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    let carrierNode ← match scope.nodes[carrierWire.node]? with
      | some value => pure value
      | none => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    match zeroNode.kind, carrierNode.kind with
    | .zeroMatrix zeroType, .constantMatrix carrierType coefficients =>
        if zeroWire.port != 0 || carrierWire.port != 0 || zeroType != carrierType ||
            coefficients.isEmpty || coefficients.all (· == .constant 0) ||
            valueNode.outputTypes != [.matrix zeroType] then
          throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    | _, _ => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)

def replaceOperationalFact
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef)
    (fact : OperationalFact) : Except OperationalError OperationalScopeFacts := do
  let outputs ← match facts.values[wire.node]? with
    | some outputs => pure outputs
    | none => throw (.missingOperand node wire)
  if wire.port >= outputs.size then throw (.missingOperand node wire)
  pure { facts with values := facts.values.set! wire.node (outputs.set! wire.port fact) }

def applyDerivationAttachment
    (node : Nat)
    (attachment : DerivationAttachment)
    (facts : OperationalScopeFacts) : Except OperationalError OperationalScopeFacts := do
  if attachment.ownerNamespace == "mxx-bgg" &&
      attachment.ruleName == "encoding-family-pairing" then
    let vectorWire ← derivationAttachmentRole attachment "vector"
    let publicKeyWire ← derivationAttachmentRole attachment "public-key"
    let plaintextWire ← derivationAttachmentRole attachment "plaintext"
    let vector ← lookupFact node facts vectorWire
    let publicKey ← lookupFact node facts publicKeyWire
    let plaintext ← lookupFact node facts plaintextWire
    let (arena, grouped) ← facts.arena.pushDirectBggGrouping vector publicKey plaintext
    replaceOperationalFact node { facts with arena } vectorWire grouped
  else if attachment.ownerNamespace == "mxx-correctness" &&
      attachment.ruleName == "protocol-boolean-signal-grouping" then
    let valueWire ← derivationAttachmentRole attachment "value"
    let value ← lookupFact node facts valueWire
    match value with
    | expression@{ payload := .directValue root, .. } => do
        let value ← match facts.arena.direct.valueAt? root with
          | some value => pure value
          | none => throw (.invalidOperationalExprRef root)
        if value.context != expression.context then throw (.unsupportedOperationalExpr root)
        let (direct, grouped) ← facts.arena.direct.mapMatrixValue root
          (fun fact => pure (groupProtocolBooleanSignal fact))
        let value ← match direct.valueAt? grouped with
          | some value => pure value
          | none => throw (.invalidOperationalExprRef grouped)
        replaceOperationalFact node { facts with arena := { facts.arena with direct } } valueWire {
          context := value.context, payload := .directValue grouped, storage := value.storage }
  else
    let valueWire ← derivationAttachmentRole attachment "value"
    let value ← lookupFact node facts valueWire
    match value with
    | expression@{ payload := .directValue root, .. } => do
        let value ← match facts.arena.direct.valueAt? root with
          | some value => pure value
          | none => throw (.invalidOperationalExprRef root)
        if value.context != expression.context then throw (.unsupportedOperationalExpr root)
        let (direct, grouped) ← facts.arena.direct.mapMatrixValue root groupPublicKeySignal
        let value ← match direct.valueAt? grouped with
          | some value => pure value
          | none => throw (.invalidOperationalExprRef grouped)
        replaceOperationalFact node { facts with arena := { facts.arena with direct } } valueWire {
          context := value.context, payload := .directValue grouped, storage := value.storage }

def applyPreparedDerivationAttachments
    (node : Nat)
    (attachments : Array DerivationAttachment)
    (facts : OperationalScopeFacts) : Except OperationalError OperationalScopeFacts :=
  attachments.foldlM (init := facts) fun current attachment =>
    applyDerivationAttachment node attachment current

def availableRelation
    (node : Nat)
    (wire : WireRef)
    (fact : OperationalMatrixFact) : Except OperationalError OperationalMatrixRelation := do
  match fact.relations with
  | [] => throw (.missingRelation node wire)
  | [relation] =>
      let available := match relation with
        | .decomposition relation => relation.status == ReconstructionStatus.available
        | .preimage _ => true
      if available then pure relation else throw (OperationalError.unavailableRelation node wire)
  | _ => throw (.ambiguousRelation node wire)

def relationPublicIdentity : OperationalMatrixRelation → PublicMatrixIdentity
  | .decomposition relation => relation.publicIdentity
  | .preimage relation => relation.publicIdentity

def relationTarget : OperationalMatrixRelation → RelationTargetSummary
  | .decomposition relation => relation.inputSummary
  | .preimage relation => relation.targetSummary

def publicIdentityIsLarge : PublicMatrixIdentity → Bool
  | .sampledTrapdoor .. | .gadget .. => true
  | .indexed _ _ source => publicIdentityIsLarge source
  | .loopInstance _ _ source => publicIdentityIsLarge source

def publicIdentityMaximum
    (residueCap : Int) : PublicMatrixIdentity → Int
  | .sampledTrapdoor .. => residueCap
  | .gadget .. => residueCap
  | .indexed _ _ source => publicIdentityMaximum residueCap source
  | .loopInstance _ _ source => publicIdentityMaximum residueCap source

/-- Rebind a direct wire-level fact without cloning its carrier DAG.  The subject overlay is
validated and applied when reduction reaches the fixed matrix/scalar leaf. -/
def rebindOperationalFact
    (subject : WireRef) : OperationalExprArena → OperationalFact →
    ParamEnvironment →
    Except OperationalError (OperationalExprArena × OperationalFact)
  | arena, expression@{ payload := .directValue root, .. }, environment => do
      let value ← match arena.direct.valueAt? root with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef root)
      if value.context != expression.context then throw (.unsupportedOperationalExpr root)
      let (direct, rebound) ← match arena.direct.pushRebound root subject with
        | some result => pure result
        | none => throw (.unsupportedOperationalExpr root)
      let value ← match direct.valueAt? rebound with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef rebound)
      pure ({ arena with direct }, {
        context := value.context
        payload := .directValue rebound
        storage := value.storage
      })

def canonicalSelectionExpression
    (origin : OperationalValueOrigin) (count : IntExpr) : IndexExpr :=
  .variable {
    owner := {
      stage := ⟨s!"operational-selection:{reprStr origin}"⟩
      scope := ⟨[]⟩
      node := ⟨0⟩
    }
    slot := 0
    count
  }

def selectionExpressionForOrigin
    (origin : OperationalValueOrigin) : IndexExpr → IndexExpr
  | .constant value => .constant value
  | .variable binder => canonicalSelectionExpression origin binder.count
  | .offset base amount => .offset (selectionExpressionForOrigin origin base) amount
  | .gather owner sourceCount position =>
      .gather owner sourceCount (selectionExpressionForOrigin origin position)

def DynamicSelectionIdentity.withOrigin
    (selection : DynamicSelectionIdentity)
    (origin : OperationalValueOrigin) : DynamicSelectionIdentity := {
  index := origin
  expression := selectionExpressionForOrigin origin selection.expression
}

partial def namespaceFreshValueOrigin
    (scope : ScopeTemplateKey)
    (wire : WireRef) : OperationalValueOrigin → OperationalValueOrigin
  | .local originScope originWire =>
      if originScope == temporaryScope && originWire == wire then .local scope originWire
      else .local originScope originWire
  | origin@(.protocolInput _) => origin
  | origin@(.protocolFamilyElement _ _) => origin
  | .loopInstance slot index source =>
      .loopInstance slot index (namespaceFreshValueOrigin scope wire source)
  | .indexed binder expression source =>
      .indexed binder expression (namespaceFreshValueOrigin scope wire source)

def namespaceFreshOrigin
    (scope : ScopeTemplateKey)
    (wire : WireRef) : MatrixOriginIdentity → MatrixOriginIdentity
  | .value originScope originWire =>
      if originScope == temporaryScope && originWire == wire then .value scope originWire
      else .value originScope originWire
  | origin@(.protocolInput _) => origin
  | origin@(.protocolFamilyElement _ _) => origin
  | origin@(.deterministicHash _) => origin
  | .loopInstance slot index source =>
      .loopInstance slot index (namespaceFreshOrigin scope wire source)
  | .indexed binder expression source =>
      .indexed binder expression (namespaceFreshOrigin scope wire source)

def namespaceFreshPublicIdentity
    (scope : ScopeTemplateKey)
    (wire : WireRef) : PublicMatrixIdentity → PublicMatrixIdentity
  | .sampledTrapdoor originScope originWire =>
      if originScope == temporaryScope && originWire.node == wire.node then
        .sampledTrapdoor scope originWire
      else .sampledTrapdoor originScope originWire
  | identity@(.gadget ..) => identity
  | .indexed binder expression source =>
      .indexed binder expression (namespaceFreshPublicIdentity scope wire source)
  | .loopInstance slot index source =>
      .loopInstance slot index (namespaceFreshPublicIdentity scope wire source)

def mapOperationalPrimitiveIdentity
    (mapOrigin : MatrixOriginIdentity → MatrixOriginIdentity)
    (mapPublic : PublicMatrixIdentity → PublicMatrixIdentity)
    (mapValue : OperationalValueOrigin → OperationalValueOrigin) :
    OperationalPrimitiveIdentity → OperationalPrimitiveIdentity
  | .matrix identity => .matrix (mapOrigin identity)
  | .publicMatrix identity => .publicMatrix (mapPublic identity)
  | .value identity => .value (mapValue identity)
  | .parameterScalar environment domains value => .parameterScalar environment domains value
  | .identityMatrix type => .identityMatrix type
  | .indexedArtifact input index => .indexedArtifact input index
  | .recurrenceResult scope node path => .recurrenceResult scope node path
  | .carriedInput path => .carriedInput path

/-- Recursively rebuild a direct carrier under one capture-free index substitution.  Unlike the
ordinary subject-rebinding walkers, this visits both sides of matrix/scalar conversion nodes and
revalidates each reconstructed pointwise schema, so delayed descriptors cannot retain stale
loop arithmetic after a static, dynamic, offset, or gather selection. -/
partial def DirectOperationalIndexedArena.reindexValue
    (environment : ParamEnvironment) (arena : DirectOperationalIndexedArena) (map : IndexMap)
    (root : OperationalIndexedValueId) : Except OperationalError
      (DirectOperationalIndexedArena × OperationalIndexedValueId) := do
  let rec visit : Nat → DirectOperationalIndexedArena →
      Std.HashMap OperationalIndexedValueId OperationalIndexedValueId → OperationalIndexedValueId →
      Except OperationalError (DirectOperationalIndexedArena ×
        Std.HashMap OperationalIndexedValueId OperationalIndexedValueId × OperationalIndexedValueId)
    | 0, _, _, id => throw (.unsupportedOperationalExpr id)
    | fuel + 1, arena, memo, id => match memo[id]? with
      | some mapped => pure (arena, memo, mapped)
      | none => do
          let value ← match arena.valueAt? id with
            | some value => pure value | none => throw (.invalidOperationalExprRef id)
          if !validateContext value.context then throw (.unsupportedOperationalExpr id)
          let (arena, memo, mapped) ← match value.payload with
            | .shared schema reference => do
                let schema ← match reindexOperationalIndexedPayloadSchema map schema with
                  | some schema => pure schema | none => throw (.unsupportedOperationalExpr id)
                let (fixed, reference) ← match reference with
                  | .matrix reference => do
                      let fact ← match arena.fixed.matrices[reference]? with
                        | some fact => pure fact | none => throw (.invalidOperationalExprRef reference)
                      let fact ← match reindexOperationalMatrixFact environment map fact with
                        | some fact => pure fact | none => throw (.unsupportedOperationalExpr id)
                      pure (arena.fixed.pushMatrix fact)
                  | .scalar reference => do
                      let fact ← match arena.fixed.scalars[reference]? with
                        | some fact => pure fact | none => throw (.invalidOperationalExprRef reference)
                      let fact ← match reindexOperationalScalarFact environment map fact with
                        | some fact => pure fact | none => throw (.unsupportedOperationalExpr id)
                      pure (arena.fixed.pushScalar fact)
                let arena := { arena with fixed }
                let (arena, mapped) ← match arena.pushShared value.context schema reference with
                  | some result => pure result | none => throw (.unsupportedOperationalExpr id)
                pure (arena, memo, mapped)
            | .explicit schema binder references => do
                if value.context != { binders := #[binder] } ||
                    !explicitCountValid environment binder references then throw (.unsupportedOperationalExpr id)
                let schema ← match reindexOperationalIndexedPayloadSchema map schema with
                  | some schema => pure schema | none => throw (.unsupportedOperationalExpr id)
                let (arena, references) ← references.foldlM (fun (arena, mapped) reference => do
                  let (fixed, replacement) ← match reference with
                    | .matrix reference => do
                        let fact ← match arena.fixed.matrices[reference]? with
                          | some fact => pure fact | none => throw (.invalidOperationalExprRef reference)
                        let fact ← match reindexOperationalMatrixFact environment map fact with
                          | some fact => pure fact | none => throw (.unsupportedOperationalExpr id)
                        pure (arena.fixed.pushMatrix fact)
                    | .scalar reference => do
                        let fact ← match arena.fixed.scalars[reference]? with
                          | some fact => pure fact | none => throw (.invalidOperationalExprRef reference)
                        let fact ← match reindexOperationalScalarFact environment map fact with
                          | some fact => pure fact | none => throw (.unsupportedOperationalExpr id)
                        pure (arena.fixed.pushScalar fact)
                  pure ({ arena with fixed }, mapped.push replacement)) (arena, #[])
                let (arena, mapped) ← match arena.pushExplicit environment value.context binder schema references with
                  | some result => pure result | none => throw (.unsupportedOperationalExpr id)
                pure (arena, memo, mapped)
            | .explicitValues schema binder values => do
                let (arena, memo, values) ← values.foldlM (fun (arena, memo, mapped) child => do
                  let (arena, memo, child) ← visit fuel arena memo child
                  pure (arena, memo, mapped.push child)) (arena, memo, #[])
                let schema ← match reindexOperationalIndexedPayloadSchema map schema with
                  | some schema => pure schema | none => throw (.unsupportedOperationalExpr id)
                let (arena, mapped) ← match arena.pushExplicitValues environment binder values with
                  | some result => pure result | none => throw (.unsupportedOperationalExpr id)
                let mappedValue ← match arena.valueAt? mapped with
                  | some value => pure value | none => throw (.invalidOperationalExprRef mapped)
                if mappedValue.context != value.context || mappedValue.payload.schema != schema then
                  throw (.unsupportedOperationalExpr id)
                pure (arena, memo, mapped)
            | .mapped schema source innerMap => do
                let sourceValue ← match arena.valueAt? source with
                  | some value => pure value | none => throw (.invalidOperationalExprRef source)
                if !innerMap.transportValid || innerMap.source != sourceValue.context ||
                    innerMap.destination != value.context || sourceValue.payload.schema != schema then
                  throw (.unsupportedOperationalExpr id)
                let (arena, memo, source) ← visit fuel arena memo source
                let (arena, mapped) ← match arena.pushMapped source innerMap with
                  | some result => pure result | none => throw (.unsupportedOperationalExpr id)
                pure (arena, memo, mapped)
            | .rebound schema source subject => do
                let (arena, memo, source) ← visit fuel arena memo source
                let (arena, mapped) ← match arena.pushRebound source subject with
                  | some result => pure result | none => throw (.unsupportedOperationalExpr id)
                let mappedValue ← match arena.valueAt? mapped with
                  | some value => pure value | none => throw (.invalidOperationalExprRef mapped)
                if mappedValue.context != value.context || mappedValue.payload.schema != schema then
                  throw (.unsupportedOperationalExpr id)
                pure (arena, memo, mapped)
            | .indexedOutput schema source binder selection subject => do
                let (arena, memo, source) ← visit fuel arena memo source
                let (arena, mapped) ← match arena.pushIndexedOutput source binder selection subject with
                  | some result => pure result | none => throw (.unsupportedOperationalExpr id)
                let mappedValue ← match arena.valueAt? mapped with
                  | some value => pure value | none => throw (.invalidOperationalExprRef mapped)
                if mappedValue.context != value.context || mappedValue.payload.schema != schema then
                  throw (.unsupportedOperationalExpr id)
                pure (arena, memo, mapped)
            | .matrixResultBound schema source totalHardBound => do
                let (arena, memo, source) ← visit fuel arena memo source
                let bound ← match reindexOperationalBoundExpr environment map totalHardBound with
                  | some bound => pure bound | none => throw (.unsupportedOperationalExpr id)
                let (arena, mapped) ← match arena.pushMatrixResultBound source bound with
                  | some result => pure result | none => throw (.unsupportedOperationalExpr id)
                let mappedValue ← match arena.valueAt? mapped with
                  | some value => pure value | none => throw (.invalidOperationalExprRef mapped)
                let schema ← match reindexOperationalIndexedPayloadSchema map schema with
                  | some schema => pure schema | none => throw (.unsupportedOperationalExpr id)
                if mappedValue.context != value.context || mappedValue.payload.schema != schema then
                  throw (.unsupportedOperationalExpr id)
                pure (arena, memo, mapped)
            | .pointwise schema operation inputs => do
                let originalInputs ← match inputs.toList.mapM arena.valueAt? with
                  | some values => pure values | none => throw (.invalidOperationalExprRef id)
                let (originalContext, _) ← match mergeIndexedFactShapeN originalInputs with
                  | some shape => pure shape | none => throw (.unsupportedOperationalExpr id)
                if originalContext != value.context ||
                    !pointwiseSchemasValid operation (originalInputs.toArray.map (·.payload.schema)) schema then
                  throw (.unsupportedOperationalExpr id)
                let (arena, memo, inputs) ← inputs.foldlM (fun (arena, memo, mapped) child => do
                  let (arena, memo, child) ← visit fuel arena memo child
                  pure (arena, memo, mapped.push child)) (arena, memo, #[])
                let operation ← match reindexOperationalIndexedPointwiseOperation environment map operation with
                  | some operation => pure operation | none => throw (.unsupportedOperationalExpr id)
                let (arena, mapped) ← match arena.pushPointwise operation inputs with
                  | some result => pure result | none => throw (.unsupportedOperationalExpr id)
                let mappedValue ← match arena.valueAt? mapped with
                  | some value => pure value | none => throw (.invalidOperationalExprRef mapped)
                let schema ← match reindexOperationalIndexedPayloadSchema map schema with
                  | some schema => pure schema | none => throw (.unsupportedOperationalExpr id)
                if mappedValue.context != value.context || mappedValue.payload.schema != schema then
                  throw (.unsupportedOperationalExpr id)
                pure (arena, memo, mapped)
          pure (arena, memo.insert id mapped, mapped)
  let (arena, _, mapped) ← visit (arena.values.size + 1) arena {} root
  pure (arena, mapped)

/-- Reindex one direct carrier value, matrix or scalar.  The storage map retains the indexed
shape and composes without materializing a family, while every reachable fixed leaf receives the
same capture-free substitution.  Thus scalar selector schemas as well as matrix identities cannot
disagree after static, dynamic, offset, or gather reindexing. -/
def OperationalExprArena.reindexDirectFact
    (arena : OperationalExprArena)
    (map : IndexMap)
    (expression : IndexedOperationalFact)
    (environment : ParamEnvironment := []) : Except OperationalError
      (OperationalExprArena × IndexedOperationalFact) := do
  let root ← match expression.payload with
    | .directValue root => pure root
  if !map.transportValid || map.source != expression.context then
    if operationalProgress "reindex_direct_fact" "admission_failed" ""
        root arena.direct.values.size ("transport_valid=" ++ toString map.transportValid ++
          "; map_source=" ++ reprStr map.source ++
          "; fact_context=" ++ reprStr expression.context ++
          "; destination_context=" ++ reprStr map.destination ++
          "; assignments=" ++ reprStr map.assignments) then
      throw (.unsupportedOperationalExpr root)
    else throw (.unsupportedOperationalExpr root)
  /- `pushMapped` composes adjacent capture-free maps.  No fixed table, relation inventory, or
  delayed pointwise DAG is copied here: final reduction transports the one selected fixed result
  through that composed map before any enclosing `.rebound` subject overlay is validated. -/
  let source ← match arena.direct.valueAt? root with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef root)
  let schema ← match reindexOperationalIndexedPayloadSchema map source.payload.schema with
    | some value => pure value
    | none =>
        if operationalProgress "reindex_direct_fact" "schema_transport_failed" ""
            root arena.direct.values.size ("source_context=" ++ reprStr expression.context ++
              "; destination_context=" ++ reprStr map.destination ++
              "; assignments=" ++ reprStr map.assignments ++
              "; schema_kind=" ++ match source.payload.schema with
                | .matrix _ => "matrix"
                | .scalar _ => "scalar") then
          throw (.unsupportedOperationalExpr root)
        else throw (.unsupportedOperationalExpr root)
  let (direct, mapped) ← match arena.direct.pushMappedWithSchema root map schema with
    | some result => pure result
    | none =>
        if operationalProgress "reindex_direct_fact" "mapped_construction_failed" ""
            root arena.direct.values.size ("source_context=" ++ reprStr expression.context ++
              "; destination_context=" ++ reprStr map.destination ++
              "; assignments=" ++ reprStr map.assignments) then
          throw (.unsupportedOperationalExpr root)
        else throw (.unsupportedOperationalExpr root)
  let value ← match direct.valueAt? mapped with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef mapped)
  pure ({ arena with direct }, {
    context := value.context
    payload := .directValue mapped
    storage := value.storage
  })


def mapOperationalCompressionToken
    (mapOrigin : MatrixOriginIdentity → MatrixOriginIdentity)
    (mapPublic : PublicMatrixIdentity → PublicMatrixIdentity)
    (mapValue : OperationalValueOrigin → OperationalValueOrigin)
    (mapBound : OperationalBoundExpr → OperationalBoundExpr) :
    OperationalCompressionToken → OperationalCompressionToken
  | .primitive identity =>
      .primitive (mapOperationalPrimitiveIdentity mapOrigin mapPublic mapValue identity)
  | .summaryBound bound => .summaryBound (mapBound bound)
  | token => token

def mapOperationalBoundedSummary
    (mapOrigin : MatrixOriginIdentity → MatrixOriginIdentity)
    (mapPublic : PublicMatrixIdentity → PublicMatrixIdentity)
    (mapValue : OperationalValueOrigin → OperationalValueOrigin)
    (mapBound : OperationalBoundExpr → OperationalBoundExpr)
    (summary : OperationalBoundedFactorSummary) : OperationalBoundedFactorSummary := {
  summary with
  hardBound := mapBound summary.hardBound
  provenance := summary.provenance.map
    (mapOperationalCompressionToken mapOrigin mapPublic mapValue mapBound)
}

def mapOperationalPolynomial
    (mapOrigin : MatrixOriginIdentity → MatrixOriginIdentity)
    (mapPublic : PublicMatrixIdentity → PublicMatrixIdentity)
    (mapValue : OperationalValueOrigin → OperationalValueOrigin)
    (mapBound : OperationalBoundExpr → OperationalBoundExpr)
    (mapRelation : OperationalMatrixRelation → OperationalMatrixRelation)
    (polynomial : OperationalPolynomial) : OperationalPolynomial :=
  polynomial.map fun term => { term with product := {
    term.product with factors := term.product.factors.map fun factor =>
      let boundedSummary := factor.boundedSummary.map
        (mapOperationalBoundedSummary mapOrigin mapPublic mapValue mapBound)
      let leaf := match factor.leaf with
        | .primitive identity => .primitive
            (mapOperationalPrimitiveIdentity mapOrigin mapPublic mapValue identity)
        | .boundedSummary origin summary =>
            let tokens := origin.tokens.map
              (mapOperationalCompressionToken mapOrigin mapPublic mapValue mapBound)
            .boundedSummary { origin with tokens }
              (mapOperationalBoundedSummary mapOrigin mapPublic mapValue mapBound summary)
        | .exactTransform tokens type =>
            let tokens := tokens.map
              (mapOperationalCompressionToken mapOrigin mapPublic mapValue mapBound)
            .exactTransform tokens type
      { factor with
        leaf
        boundedSummary
        relations := factor.relations.map mapRelation
      }
  }}

def mapRelationSnapshotPolynomial
    (mapOrigin : MatrixOriginIdentity → MatrixOriginIdentity)
    (mapPublic : PublicMatrixIdentity → PublicMatrixIdentity)
    (mapValue : OperationalValueOrigin → OperationalValueOrigin)
    (mapBound : OperationalBoundExpr → OperationalBoundExpr)
    (polynomial : RelationSnapshotPolynomial) : RelationSnapshotPolynomial :=
  polynomial.map fun term => { term with product := {
    term.product with factors := term.product.factors.map fun factor =>
      let boundedSummary := factor.boundedSummary.map
        (mapOperationalBoundedSummary mapOrigin mapPublic mapValue mapBound)
      let leaf := match factor.leaf with
        | .primitive identity => .primitive
            (mapOperationalPrimitiveIdentity mapOrigin mapPublic mapValue identity)
        | .boundedSummary origin summary =>
            let tokens := origin.tokens.map
              (mapOperationalCompressionToken mapOrigin mapPublic mapValue mapBound)
            .boundedSummary { origin with tokens }
              (mapOperationalBoundedSummary mapOrigin mapPublic mapValue mapBound summary)
        | .exactTransform tokens type =>
            let tokens := tokens.map
              (mapOperationalCompressionToken mapOrigin mapPublic mapValue mapBound)
            .exactTransform tokens type
      { factor with leaf, boundedSummary }
  }}

def namespaceFreshSummary
    (scope : ScopeTemplateKey)
    (wire : WireRef)
    (summary : RelationTargetSummary) : RelationTargetSummary := {
  summary with
  origin := namespaceFreshOrigin scope wire summary.origin
  polynomial := mapRelationSnapshotPolynomial
    (namespaceFreshOrigin scope wire)
    (namespaceFreshPublicIdentity scope wire)
    id id summary.polynomial
}

def namespaceFreshRelation
    (scope : ScopeTemplateKey)
    (wire : WireRef) : OperationalMatrixRelation → OperationalMatrixRelation
  | .decomposition relation => .decomposition {
      relation with
      producer := namespaceFreshOrigin scope wire relation.producer
      inputOrigin := namespaceFreshOrigin scope wire relation.inputOrigin
      inputSummary := namespaceFreshSummary scope wire relation.inputSummary
      publicIdentity := namespaceFreshPublicIdentity scope wire relation.publicIdentity
    }
  | .preimage relation => .preimage {
      relation with
      producer := namespaceFreshOrigin scope wire relation.producer
      targetOrigin := namespaceFreshOrigin scope wire relation.targetOrigin
      targetSummary := namespaceFreshSummary scope wire relation.targetSummary
      publicIdentity := namespaceFreshPublicIdentity scope wire relation.publicIdentity
    }

def shiftTargetPreviousDepth
    (target : RelationTargetSummary) : RelationTargetSummary := {
  target with
  totalHardBound := shiftPreviousDepth target.totalHardBound
  polynomial := mapRelationSnapshotPolynomial id id id shiftPreviousDepth target.polynomial
}

def shiftRelationPreviousDepth :
    OperationalMatrixRelation → OperationalMatrixRelation
  | .decomposition relation => .decomposition {
      relation with inputSummary := shiftTargetPreviousDepth relation.inputSummary }
  | .preimage relation => .preimage {
      relation with targetSummary := shiftTargetPreviousDepth relation.targetSummary }

def shiftMatrixFactPreviousDepth
    (fact : OperationalMatrixFact) : OperationalMatrixFact := {
  fact with
  totalHardBound := shiftPreviousDepth fact.totalHardBound
  relations := fact.relations.map shiftRelationPreviousDepth
  polynomial := mapOperationalPolynomial id id id shiftPreviousDepth
    shiftRelationPreviousDepth fact.polynomial
}

/-- Insert a new innermost recurrence state. References already present in invariant facts then
refer to the enclosing state, while the new carried placeholders continue to use depth zero. -/
partial def shiftFactPreviousDepth
    (environment : ParamEnvironment) (arena : OperationalExprArena) : OperationalFact →
    Except OperationalError (OperationalExprArena × OperationalFact)
  | expression@{ payload := .directValue root, .. } => do
      let value ← match arena.direct.valueAt? root with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef root)
      let (direct, mapped) ← match value.payload.schema with
        | .matrix _ => do
            arena.direct.mapMatrixValue root (fun fact => pure (shiftMatrixFactPreviousDepth fact))
        | .scalar _ => do
            arena.direct.mapScalarValue environment root fun
              | .trapdoor fact => do
                  let maximum ← requireMaterializedScalarBound root fact.maximum
                  pure (.trapdoor { fact with maximum := .closed (shiftPreviousDepth maximum) })
              | .integer fact => pure (.integer { fact with
                  lowerExpression := .closed (shiftPreviousDepth
                    (fact.lowerExpression.closedOperational?.getD (.closedInt (.constant fact.lower))))
                  upperExpression := .closed (shiftPreviousDepth
                    (fact.upperExpression.closedOperational?.getD (.closedInt (.constant fact.upper)))) })
              | fact => pure fact
      let value ← match direct.valueAt? mapped with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef mapped)
      if value.context != expression.context then throw (.unsupportedOperationalExpr mapped)
      pure ({ arena with direct }, {
        context := value.context
        payload := .directValue mapped
        storage := value.storage
      })

/-- Namespace only identities created by this exact output.  Caller origins transported through
an input are deliberately left unchanged. -/
def namespaceFreshMatrixFact
    (scope : ScopeTemplateKey)
    (wire : WireRef)
    (fact : OperationalMatrixFact) : OperationalMatrixFact := {
  fact with
  origin := namespaceFreshOrigin scope wire fact.origin
  identity := fact.identity.map (namespaceFreshPublicIdentity scope wire)
  relations := fact.relations.map (namespaceFreshRelation scope wire)
  polynomial := mapOperationalPolynomial
    (namespaceFreshOrigin scope wire)
    (namespaceFreshPublicIdentity scope wire)
    (namespaceFreshValueOrigin scope wire)
    id
    (namespaceFreshRelation scope wire)
    fact.polynomial
}

def namespaceFreshScalarFact
    (scope : ScopeTemplateKey)
    (wire : WireRef) : OperationalScalarFact → OperationalScalarFact
  | .trapdoor fact => .trapdoor {
      fact with publicIdentity := namespaceFreshPublicIdentity scope wire fact.publicIdentity }
  | .integer fact => .integer {
      fact with origin := namespaceFreshValueOrigin scope wire fact.origin }
  | .bytes fact => .bytes {
      fact with origin := namespaceFreshValueOrigin scope wire fact.origin }
  | fact => fact

/-- Namespace newly materialized direct leaves.  Mapped and delayed roots retain their
producer-installed namespaces: they transport existing values and do not introduce a fresh fixed
leaf at this boundary. -/
def namespaceFreshDirectOutput
    (scope : ScopeTemplateKey)
    (wire : WireRef)
    (arena : OperationalExprArena)
    (fact : OperationalFact) : Except OperationalError (OperationalExprArena × OperationalFact) := do
  let id ← match fact.payload with
    | .directValue id => pure id
  match arena.direct.valueAt? id with
  | some { context, payload := .shared (.matrix matrixType) (.matrix reference), .. } =>
      let leaf ← match arena.direct.fixed.matrices[reference]? with
        | some leaf => pure leaf
        | none => throw (.invalidOperationalExprRef reference)
      let (fixed, replacement) := arena.direct.fixed.pushMatrix (namespaceFreshMatrixFact scope wire leaf)
      let direct := { arena.direct with fixed }
      let (direct, replacement) ← match direct.pushShared context (.matrix matrixType) replacement with
        | some replacement => pure replacement
        | none => throw (.unsupportedOperationalExpr id)
      pure ({ arena with direct }, { fact with payload := .directValue replacement })
  | some { context, payload := .shared (.scalar scalarType) (.scalar reference), .. } =>
      let leaf ← match arena.direct.fixed.scalars[reference]? with
        | some leaf => pure leaf
        | none => throw (.invalidOperationalExprRef reference)
      let (fixed, replacement) := arena.direct.fixed.pushScalar
        (namespaceFreshScalarFact scope wire leaf)
      let direct := { arena.direct with fixed }
      let (direct, replacement) ← match direct.pushShared context (.scalar scalarType) replacement with
        | some replacement => pure replacement
        | none => throw (.unsupportedOperationalExpr id)
      pure ({ arena with direct }, { fact with payload := .directValue replacement })
  | some { payload := .explicit .., .. }
  | some { payload := .explicitValues .., .. }
  | some { payload := .mapped .., .. }
  | some { payload := .rebound .., .. }
  | some { payload := .indexedOutput .., .. }
  | some { payload := .matrixResultBound .., .. }
  | some { payload := .pointwise .., .. } => pure (arena, fact)
  | some _ => throw (.unsupportedOperationalExpr id)
  | none => throw (.invalidOperationalExprRef id)

def namespaceFreshOutput
    (scope : ScopeTemplateKey)
    (wire : WireRef)
    (arena : OperationalExprArena)
    (fact : OperationalFact) : Except OperationalError (OperationalExprArena × OperationalFact) := do
  match fact with
  | expression@{ payload := .directValue _, .. } =>
      namespaceFreshDirectOutput scope wire arena expression


def joinCanonicalRanges : List CanonicalRange → CanonicalRange
  | [] => .unknown
  | ranges =>
      if ranges.all (fun range => match range with | .below _ => true | .unknown => false) then
        .below (ranges.foldl (fun result range => match range with
          | .below value => max result value
          | .unknown => result) 0)
      else .unknown

def isLoopTemplateSelection
    (binder : FamilyTemplateBinder)
    (origin : OperationalValueOrigin) : Bool :=
  let base : OperationalValueOrigin :=
    .local temporaryScope { node := binder.producerNode, port := 0 }
  let namespacedBase : OperationalValueOrigin :=
    .local binder.owner { node := binder.producerNode, port := 0 }
  let rec containsBase : OperationalValueOrigin → Bool
    | candidate@(.local ..) => candidate == base || candidate == namespacedBase
    | .loopInstance _ _ source => containsBase source
    | .indexed _ _ source => containsBase source
    | _ => false
  containsBase origin


/-- Substitute the symbolic index of a previously constructed uniform loop family with the
current consumer-loop index. This preserves correlation when a uniform family is zipped into a
later loop instead of treating the producer's template index as an independent selection. -/
partial def substituteLoopTemplateValueOrigin
    (binder : FamilyTemplateBinder)
    (replacement : OperationalValueOrigin) : OperationalValueOrigin → OperationalValueOrigin
  | origin@(.local ..) => if isLoopTemplateSelection binder origin then replacement else origin
  | origin@(.protocolInput _) => origin
  | origin@(.protocolFamilyElement _ _) => origin
  | origin@(.loopInstance slot index source) =>
      if isLoopTemplateSelection binder origin then replacement
      else .loopInstance slot index (substituteLoopTemplateValueOrigin binder replacement source)
  | .indexed indexedBinder expression source =>
      .indexed indexedBinder expression
        (substituteLoopTemplateValueOrigin binder replacement source)

def substituteLoopTemplateHashIdentity
    (binder : FamilyTemplateBinder)
    (replacement : OperationalValueOrigin)
    (identity : DeterministicHashIdentity) : DeterministicHashIdentity := {
  identity with
  keyOrigin := substituteLoopTemplateValueOrigin binder replacement identity.keyOrigin
  trailingIntegerOrigins := identity.trailingIntegerOrigins.map
    (substituteLoopTemplateValueOrigin binder replacement)
}

partial def substituteLoopTemplateMatrixOrigin
    (binder : FamilyTemplateBinder)
    (replacement : OperationalValueOrigin) : MatrixOriginIdentity → MatrixOriginIdentity
  | origin@(.value ..) => origin
  | origin@(.protocolInput _) => origin
  | origin@(.protocolFamilyElement _ _) => origin
  | .deterministicHash identity =>
      .deterministicHash (substituteLoopTemplateHashIdentity binder replacement identity)
  | .loopInstance slot index source =>
      .loopInstance slot index (substituteLoopTemplateMatrixOrigin binder replacement source)
  | .indexed selectedBinder expression source =>
      .indexed selectedBinder expression
        (substituteLoopTemplateMatrixOrigin binder replacement source)

partial def substituteLoopTemplatePublicIdentity
    (binder : FamilyTemplateBinder)
    (replacement : OperationalValueOrigin) : PublicMatrixIdentity → PublicMatrixIdentity
  | identity@(.sampledTrapdoor ..) => identity
  | identity@(.gadget ..) => identity
  | .loopInstance slot index source =>
      .loopInstance slot index (substituteLoopTemplatePublicIdentity binder replacement source)
  | .indexed selectedBinder expression source =>
      .indexed selectedBinder expression
        (substituteLoopTemplatePublicIdentity binder replacement source)

def substituteLoopTemplateTarget
    (binder : FamilyTemplateBinder)
    (replacement : OperationalValueOrigin)
    (target : RelationTargetSummary) : RelationTargetSummary := {
  target with
  origin := substituteLoopTemplateMatrixOrigin binder replacement target.origin
  polynomial := mapRelationSnapshotPolynomial
    (substituteLoopTemplateMatrixOrigin binder replacement)
    (substituteLoopTemplatePublicIdentity binder replacement)
    (substituteLoopTemplateValueOrigin binder replacement)
    id target.polynomial
}

def substituteLoopTemplateRelation
    (binder : FamilyTemplateBinder)
    (replacement : OperationalValueOrigin) :
    OperationalMatrixRelation → OperationalMatrixRelation
  | .decomposition relation => .decomposition {
      relation with
      producer := substituteLoopTemplateMatrixOrigin binder replacement relation.producer
      publicIdentity := substituteLoopTemplatePublicIdentity binder replacement
        relation.publicIdentity
      inputOrigin := substituteLoopTemplateMatrixOrigin binder replacement relation.inputOrigin
      inputSummary := substituteLoopTemplateTarget binder replacement relation.inputSummary
    }
  | .preimage relation => .preimage {
      relation with
      producer := substituteLoopTemplateMatrixOrigin binder replacement relation.producer
      publicIdentity := substituteLoopTemplatePublicIdentity binder replacement
        relation.publicIdentity
      targetOrigin := substituteLoopTemplateMatrixOrigin binder replacement relation.targetOrigin
      targetSummary := substituteLoopTemplateTarget binder replacement relation.targetSummary
    }

def substituteLoopTemplateMatrixFact
    (binder : FamilyTemplateBinder)
    (replacement : OperationalValueOrigin)
    (fact : OperationalMatrixFact) : OperationalMatrixFact := {
  fact with
  origin := substituteLoopTemplateMatrixOrigin binder replacement fact.origin
  identity := fact.identity.map (substituteLoopTemplatePublicIdentity binder replacement)
  relations := fact.relations.map (substituteLoopTemplateRelation binder replacement)
  polynomial := mapOperationalPolynomial
    (substituteLoopTemplateMatrixOrigin binder replacement)
    (substituteLoopTemplatePublicIdentity binder replacement)
    (substituteLoopTemplateValueOrigin binder replacement)
    id
    (substituteLoopTemplateRelation binder replacement)
    fact.polynomial
}

/-- Select one element of a uniform family by the exact executable index wire.  The wrapper is
structural: two selections compare equal only when the family binder and index-wire instance are
identical.  Arithmetic equivalence of two index computations is never inferred. -/
def dynamicSelectionScope : OperationalValueOrigin → ScopeTemplateKey
  | .local scope _ => scope
  | .protocolInput _ | .protocolFamilyElement _ _ => temporaryScope
  | .loopInstance _ _ source => dynamicSelectionScope source
  | .indexed _ _ source => dynamicSelectionScope source

def indexValueOrigin
    (binder : FamilyTemplateBinder)
    (selection : DynamicSelectionIdentity)
    (source : OperationalValueOrigin) : OperationalValueOrigin :=
  .indexed binder selection.expression source

def indexMatrixFact
    (binder : FamilyTemplateBinder)
    (selection : DynamicSelectionIdentity)
    (subject : WireRef)
    (fact : OperationalMatrixFact) : OperationalMatrixFact :=
  let mapOrigin (origin : MatrixOriginIdentity) := .indexed binder selection.expression origin
  let mapPublic (identity : PublicMatrixIdentity) := .indexed binder selection.expression identity
  let mapValue := indexValueOrigin binder selection
  let mapTarget (target : RelationTargetSummary) : RelationTargetSummary := {
    target with
    origin := mapOrigin target.origin
    polynomial := mapRelationSnapshotPolynomial mapOrigin mapPublic mapValue id target.polynomial
  }
  let mapRelation : OperationalMatrixRelation → OperationalMatrixRelation
    | .decomposition relation => .decomposition {
        relation with
        producer := mapOrigin relation.producer
        publicIdentity := mapPublic relation.publicIdentity
        inputOrigin := mapOrigin relation.inputOrigin
        inputSummary := mapTarget relation.inputSummary
      }
    | .preimage relation => .preimage {
        relation with
        producer := mapOrigin relation.producer
        publicIdentity := mapPublic relation.publicIdentity
        targetOrigin := mapOrigin relation.targetOrigin
        targetSummary := mapTarget relation.targetSummary
      }
  { fact with
    subject
    origin := mapOrigin fact.origin
    identity := fact.identity.map mapPublic
    relations := fact.relations.map mapRelation
    polynomial := mapOperationalPolynomial mapOrigin mapPublic mapValue id mapRelation fact.polynomial
  }

def indexScalarFact
    (binder : FamilyTemplateBinder)
    (selection : DynamicSelectionIdentity)
    (subject : WireRef) : OperationalScalarFact → OperationalScalarFact
  | .integer fact => .integer {
      fact with subject, origin := indexValueOrigin binder selection fact.origin }
  | .trapdoor fact => .trapdoor {
      fact with subject, publicIdentity := .indexed binder selection.expression fact.publicIdentity }
  | .bytes fact => .bytes {
      fact with subject, origin := indexValueOrigin binder selection fact.origin }
  | fact => fact

/-- Represent a dynamic choice from a construction-uniform matrix family by one checked schema
envelope.  The selected representative carries the unresolved index in every matrix and relation
identity, so the envelope is not an equal-value collapse. -/

def packedDirectFamilyBinder
    (scope : ScopeTemplateKey)
    (node : Nat)
    (count : IntExpr) : IndexVariable := {
  owner := {
    stage := ⟨s!"operational-family-pack:{reprStr scope}"⟩
    scope := ⟨[]⟩
    node := ⟨node⟩
  }
  slot := 0
  count
}

def directFamilyLaneBinder
    (scope : ScopeTemplateKey) (producerNode : Nat) (producer : Node) (familyWire : WireRef)
    (countExpression : IntExpr) (count : Nat) : Except OperationalError IndexVariable := do
  if familyWire.node != producerNode || count == 0 then
    throw (.loopInputModeMismatch producerNode familyWire.port)
  match producer.kind with
  | .familyPack => pure (packedDirectFamilyBinder scope producerNode countExpression)
  | .parallelLoop _ _ indexSlot _ _ =>
      parallelLoopLaneBinder scope producerNode indexSlot countExpression
  | .select =>
      let selection := DynamicSelectionIdentity.fromDeclaredCount (.local scope familyWire) countExpression
      match selection.expression with
      | .variable binder => pure binder
      | _ => throw (.loopInputModeMismatch producerNode familyWire.port)
  | _ => throw (.loopInputModeMismatch producerNode familyWire.port)
def packDirectMatrixFamily
    (scope : ScopeTemplateKey)
    (node : Nat)
    (environment : ParamEnvironment)
    (count : IntExpr)
    (arena : OperationalExprArena)
    (elements : Array OperationalFact) : Except OperationalError
      (OperationalExprArena × OperationalFact) := do
  let binder := packedDirectFamilyBinder scope node count
  let ids ← elements.mapM fun element => match element.payload with
    | .directValue id => pure id
  let values ← ids.mapM fun id => match arena.direct.valueAt? id with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef id)
  let schema ← match values[0]? with
    | some value => match value.payload.schema with
      | .matrix matrixType => pure (.matrix matrixType)
      | .scalar _ => throw (.operandNotMatrix node { node, port := 0 })
    | none => throw (.invalidCount node 0)
  if values.any (fun value => value.payload.schema != schema) then
    throw (.outputTypeMismatch node)
  let explicitReferences := values.mapM fun value => match value with
    | { context, payload := .shared (.matrix _) reference, .. } =>
        if context.binders.isEmpty then some reference else none
    | _ => none
  let (direct, result) ← match explicitReferences with
    | some references => do
      match arena.direct.pushExplicit environment { binders := #[binder] } binder schema references with
      | some result => pure result
      | none => throw (.unsupportedOperationalExpr arena.direct.values.size)
    | none => do
      match arena.direct.pushExplicitValues environment binder ids with
      | some result => pure result
      | none => throw (.unsupportedOperationalExpr arena.direct.values.size)
  let value ← match direct.valueAt? result with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef result)
  pure ({ arena with direct }, {
    context := value.context
    payload := .directValue result
    storage := value.storage
  })


/-- Select one matrix family as application of an ordered direct family table.  Each branch family
is first reindexed onto the output lane selector; the branch-table binder is then substituted by
the executable selector, preserving both dimensions. -/
def selectUniformMatrixFamiliesWithLaneBinders
    (scopeKey : ScopeTemplateKey)
    (node : Nat)
    (selection : OperationalIntegerFact)
    (selectionExpression : Option IndexExpr)
    (matrixType : MatrixTypeExpr)
    (declaredCount : IntExpr)
    (expectedCount : Nat)
    (branches : List OperationalFact)
    (branchLaneBinders : List IndexVariable)
    (environment : ParamEnvironment)
    (arena : OperationalExprArena) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  if expectedCount = 0 || branches.isEmpty || branches.length != branchLaneBinders.length then
    throw (.invalidCount node expectedCount)
  let outputLane : OperationalValueOrigin := .local scopeKey { node, port := 0 }
  let outputSelection := DynamicSelectionIdentity.fromDeclaredCount outputLane declaredCount
  let mut arena := arena
  let mut normalizedBranches : Array OperationalFact := #[]
  for (branch, sourceBinder) in branches.zip branchLaneBinders do
    let expression ← match branch with
      | expression@{ payload := .directValue _, .. } => pure expression
    let value ← match arena.direct.valueAt? expression.payload.root with
      | some value => pure value
      | none => throw (.invalidOperationalExprRef expression.payload.root)
    if value.context != expression.context || value.payload.schema != .matrix matrixType then
      throw (.outputTypeMismatch node)
    if !expression.context.binders.contains sourceBinder then
      throw (.loopInputModeMismatch node 1)
    let map ← match dynamicIndexMap expression.context sourceBinder outputSelection.expression with
      | some map => pure map
      | none => throw (.loopInputModeMismatch node 1)
    let (nextArena, normalized) ← arena.reindexDirectFact map expression environment
    arena := nextArena
    normalizedBranches := normalizedBranches.push normalized
  let choiceCount := normalizedBranches.size
  let choiceCountExpression := IntExpr.constant (Int.ofNat choiceCount)
  let choiceBinder := packedDirectFamilyBinder scopeKey node choiceCountExpression
  let (nextArena, table) ← packDirectMatrixFamily scopeKey node environment choiceCountExpression arena
    normalizedBranches
  arena := nextArena
  if !table.context.binders.contains choiceBinder then
    throw (.unsupportedOperationalExpr node)
  let choiceMap ←
    if selection.lower == selection.upper then
      match closedStaticIndexMap environment table.context choiceBinder selection.lower.toNat with
      | some map => pure map
      | none => throw (.unsupportedOperationalExpr node)
    else
      let choiceSelection := selectionExpression.getD
        (DynamicSelectionIdentity.fromOrigin selection.origin choiceCount).expression
      match dynamicIndexMap table.context choiceBinder choiceSelection with
      | some map => pure map
      | none => throw (.unsupportedOperationalExpr node)
  let (finalArena, selected) ← arena.reindexDirectFact choiceMap table environment
  rebindOperationalFact { node, port := 0 } finalArena selected environment

def selectionIndexedContext
    (selection : DynamicSelectionIdentity)
    (root : Nat) : Except OperationalError IndexContext := do
  match selection.expression with
  | .variable binder => pure { binders := #[binder] }
  | _ => throw (.unsupportedOperationalExpr root)

def exactlyOneIndexedBinder
    (context : IndexContext)
    (root : Nat) : Except OperationalError IndexVariable := do
  if context.binders.size != 1 then throw (.unsupportedOperationalExpr root)
  match context.binders[0]? with
  | some binder => pure binder
  | none => throw (.unsupportedOperationalExpr root)

/-- A parallel-loop result may retain independent selector dimensions (for example a select
followed by a dynamic family get).  Close only the loop's own lexical binder and preserve those
other dimensions; requiring a singleton context here would incorrectly reject that supported
composition. -/
def parallelLoopOutputBinder
    (selection : DynamicSelectionIdentity) (context : IndexContext) (root : Nat) :
    Except OperationalError IndexVariable :=
  match selection.expression with
  | .variable expected => match context.binders.toList.filter (· == expected) with
      | [binder] => pure binder
      | _ => throw (.unsupportedOperationalExpr root)
  | _ => throw (.unsupportedOperationalExpr root)

/-- Emit complete owner-aware context evidence when closing a parallel body cannot find its
lexical lane.  The diagnostic is failure-only; the carrier remains the authoritative source and
the result still fails closed. -/
private def parallelLoopOutputBinderFailureDiagnostic
    (selection : DynamicSelectionIdentity)
    (output : OperationalFact)
  (arena : OperationalExprArena) : Bool :=
  let root := output.payload.root
  let payload := match arena.direct.valueAt? root with
    | some { payload := .shared .., .. } => "shared"
    | some { payload := .explicit _ binder references, .. } =>
        "explicit; binder=" ++ reprStr binder ++ "; entries=" ++ toString references.size
    | some { payload := .explicitValues _ binder values, .. } =>
        "explicit_values; binder=" ++ reprStr binder ++ "; entries=" ++ toString values.size
    | some { payload := .mapped _ source map, .. } =>
        "mapped; source=" ++ toString source ++ "; map=" ++ reprStr map
    | some { payload := .rebound _ source subject, .. } =>
        "rebound; source=" ++ toString source ++ "; subject=" ++ reprStr subject
    | some { payload := .indexedOutput _ source binder selection subject, .. } =>
        "indexed_output; source=" ++ toString source ++ "; binder=" ++ reprStr binder ++
          "; selection=" ++ reprStr selection ++ "; subject=" ++ reprStr subject
    | some { payload := .matrixResultBound _ source _, .. } =>
        "matrix_result_bound; source=" ++ toString source
    | some { payload := .pointwise _ _ inputs, .. } =>
        "pointwise; inputs=" ++ reprStr inputs
    | none => "missing_direct_root"
  operationalProgress "parallel_loop_output_binder" "unresolved" ""
    root arena.direct.values.size
    ("expected=" ++ reprStr selection.expression ++ "; output_context=" ++ reprStr output.context ++
      "; root=" ++ toString root ++ "; payload=" ++ payload)

/-- Compact direct-carrier path for failures after a parallel-output map has been admitted. -/
private partial def parallelLoopOutputCarrierTrace
    (arena : DirectOperationalIndexedArena) (root : OperationalIndexedValueId) : Nat → String
  | 0 => "fuel_exhausted(root=" ++ toString root ++ ")"
  | fuel + 1 =>
      match arena.valueAt? root with
      | none => "missing(root=" ++ toString root ++ ")"
      | some { payload := .shared .., .. } => "shared(root=" ++ toString root ++ ")"
      | some { payload := .explicit _ binder references, .. } =>
          "explicit(root=" ++ toString root ++ "; binder=" ++ reprStr binder ++
            "; entries=" ++ toString references.size ++ ")"
      | some { payload := .explicitValues _ binder values, .. } =>
          "explicit_values(root=" ++ toString root ++ "; binder=" ++ reprStr binder ++
            "; entries=" ++ toString values.size ++ ")"
      | some { payload := .mapped _ source map, .. } =>
          "mapped(root=" ++ toString root ++ "; map=" ++ reprStr map ++ ") -> " ++
            parallelLoopOutputCarrierTrace arena source fuel
      | some { payload := .rebound _ source subject, .. } =>
          "rebound(root=" ++ toString root ++ "; subject=" ++ reprStr subject ++ ") -> " ++
            parallelLoopOutputCarrierTrace arena source fuel
      | some { payload := .indexedOutput _ source binder selection subject, .. } =>
          "indexed_output(root=" ++ toString root ++ "; binder=" ++ reprStr binder ++
            "; selection=" ++ reprStr selection ++ "; subject=" ++ reprStr subject ++ ") -> " ++
            parallelLoopOutputCarrierTrace arena source fuel
      | some { payload := .matrixResultBound _ source _, .. } =>
          "matrix_result_bound(root=" ++ toString root ++ ") -> " ++
            parallelLoopOutputCarrierTrace arena source fuel
      | some { payload := .pointwise _ _ inputs, .. } =>
          "pointwise(root=" ++ toString root ++ "; inputs=" ++ reprStr inputs ++ ")"

private def parallelLoopOutputMapFailureDiagnostic
    (event : String) (sourceBinder : IndexVariable) (selection : DynamicSelectionIdentity)
    (output : OperationalFact) (arena : OperationalExprArena) : Bool :=
  let root := output.payload.root
  let candidate := dynamicIndexMap output.context sourceBinder selection.expression
  let selectorVariables := selection.expression.freeVariables
  operationalProgress "parallel_loop_output_map" event "" root arena.direct.values.size
    ("source_context=" ++ reprStr output.context ++ "; source_binder=" ++ reprStr sourceBinder ++
      "; selection=" ++ reprStr selection.expression ++ "; selection_free_variables=" ++
      reprStr selectorVariables ++ "; context_valid=" ++ toString (validateContext output.context) ++
      "; source_present=" ++ toString (output.context.binders.contains sourceBinder) ++
      "; map_constructed=" ++ toString candidate.isSome ++ "; carrier=" ++
      parallelLoopOutputCarrierTrace arena.direct root (arena.direct.values.size + 1))

def packDirectScalarFamily
    (scope : ScopeTemplateKey)
    (node : Nat)
    (environment : ParamEnvironment)
    (count : IntExpr)
    (arena : OperationalExprArena)
    (elements : Array OperationalFact) : Except OperationalError
      (OperationalExprArena × OperationalFact) := do
  let binder := packedDirectFamilyBinder scope node count
  let ids := elements.map fun element => element.payload.root
  let values ← ids.mapM fun id => match arena.direct.valueAt? id with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef id)
  let first ← match values[0]? with
    | some value => pure value
    | none => throw (.invalidCount node 0)
  let schema ← match first.payload.schema with
    | .scalar schema => pure (.scalar schema)
    | .matrix _ => throw (.operandNotInteger node { node, port := 0 })
  if values.any (fun value => value.payload.schema != schema) then throw (.outputTypeMismatch node)
  let explicitReferences := values.mapM fun value => match value with
    | { context, payload := .shared (.scalar _) reference, .. } =>
        if context.binders.isEmpty then some reference else none
    | _ => none
  let (direct, result) ← match explicitReferences with
    | some references => match arena.direct.pushExplicit environment { binders := #[binder] } binder schema references with
      | some result => pure result
      | none => throw (.unsupportedOperationalExpr arena.direct.values.size)
    | none => match arena.direct.pushExplicitValues environment binder ids with
      | some result => pure result
      | none => throw (.unsupportedOperationalExpr arena.direct.values.size)
  let value ← match direct.valueAt? result with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef result)
  pure ({ arena with direct }, {
    context := value.context, payload := .directValue result, storage := value.storage })

def selectDirectMatrixBranches
    (scope : ScopeTemplateKey) (node : Nat) (selection : OperationalIntegerFact) (subject : WireRef)
    (matrixType : MatrixTypeExpr) (environment : ParamEnvironment) (arena : OperationalExprArena)
    (branches : Array OperationalFact) (selectionExpression : Option IndexExpr) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  if branches.isEmpty then throw (.invalidCount node 0)
  let count := branches.size
  let values ← branches.mapM fun branch => match arena.direct.valueAt? branch.payload.root with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef branch.payload.root)
  if values.any fun value => value.payload.schema != .matrix matrixType then throw (.outputTypeMismatch node)
  let familyCount := IntExpr.constant (Int.ofNat count)
  let binder := packedDirectFamilyBinder scope node familyCount
  let (arena, family) ← packDirectMatrixFamily scope node environment familyCount arena branches
  if !family.context.binders.contains binder then throw (.unsupportedOperationalExpr node)
  let map ← if selection.lower == selection.upper then
      match closedStaticIndexMap environment family.context binder selection.lower.toNat with
      | some map => pure map | none => throw (.unsupportedOperationalExpr node)
    else
      let executableSelection ← match selectionExpression with
        | some expression => pure expression
        | none => throw (.unsupportedOperationalExpr node)
      match dynamicIndexMap family.context binder executableSelection with
      | some map => pure map | none => throw (.unsupportedOperationalExpr node)
  let (arena, selected) ← arena.reindexDirectFact map family environment
  rebindOperationalFact subject arena selected environment

def selectDirectScalarBranches
    (scope : ScopeTemplateKey) (node : Nat) (selection : OperationalIntegerFact) (subject : WireRef)
    (schema : OperationalFixedScalarSchema) (environment : ParamEnvironment)
    (arena : OperationalExprArena) (branches : Array OperationalFact) (selectionExpression : Option IndexExpr) : Except OperationalError
      (OperationalExprArena × OperationalFact) := do
  if branches.isEmpty then throw (.invalidCount node 0)
  let count := branches.size
  let values ← branches.mapM fun branch => match arena.direct.valueAt? branch.payload.root with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef branch.payload.root)
  if values.any fun value => value.payload.schema != .scalar schema then throw (.outputTypeMismatch node)
  let familyCount := IntExpr.constant (Int.ofNat count)
  let binder := packedDirectFamilyBinder scope node familyCount
  let (arena, family) ← packDirectScalarFamily scope node environment familyCount arena branches
  if !family.context.binders.contains binder then throw (.unsupportedOperationalExpr node)
  let map ← if selection.lower == selection.upper then
      match closedStaticIndexMap environment family.context binder selection.lower.toNat with
      | some map => pure map | none => throw (.unsupportedOperationalExpr node)
    else
      let executableSelection ← match selectionExpression with
        | some expression => pure expression
        | none => throw (.unsupportedOperationalExpr node)
      match dynamicIndexMap family.context binder executableSelection with
      | some map => pure map | none => throw (.unsupportedOperationalExpr node)
  let (arena, selected) ← arena.reindexDirectFact map family environment
  rebindOperationalFact subject arena selected environment

def closeParallelDirectMatrixOutput
    (scope : ScopeTemplateKey) (node indexSlot port : Nat) (declaredCount : IntExpr)
    (environment : ParamEnvironment) (arena : OperationalExprArena) (output : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let root := output.payload.root
  let subject : WireRef := { node, port }
  let binder := parallelLoopFamilyBinder scope node indexSlot
  let selection := parallelLoopLaneSelection scope node indexSlot declaredCount
  if output.context.binders.isEmpty then
    let (direct, indexed) ← arena.direct.mapMatrixValue root
      (fun fact => pure (indexMatrixFact binder selection subject fact))
    let indexedValue ← match direct.valueAt? indexed with
      | some value => pure value | none => throw (.invalidOperationalExprRef indexed)
    if indexedValue.context != emptyContext then throw (.unsupportedOperationalExpr indexed)
    let destination ← selectionIndexedContext selection indexed
    let map : IndexMap := { source := emptyContext, destination, assignments := #[] }
    let expression : OperationalFact := {
      context := emptyContext, payload := .directValue indexed, storage := indexedValue.storage }
    ({ arena with direct }).reindexDirectFact map expression environment
  else
    let expected ← match selection.expression with
      | .variable binder => pure binder
      | _ => throw (.unsupportedOperationalExpr root)
    if output.context.binders.contains expected then
      let sourceBinder ← match parallelLoopOutputBinder selection output.context root with
        | .ok binder => pure binder
        | .error error =>
            if parallelLoopOutputBinderFailureDiagnostic selection output arena then throw error
            else throw error
      let map ← match dynamicIndexMap output.context sourceBinder selection.expression with
        | some map => pure map
        | none =>
            if parallelLoopOutputMapFailureDiagnostic "construct_failed" sourceBinder selection output arena then
              throw (.unsupportedOperationalExpr root)
            else throw (.unsupportedOperationalExpr root)
      match arena.reindexDirectFact map output environment with
      | .ok result => pure result
      | .error error =>
          if parallelLoopOutputMapFailureDiagnostic "reindex_failed" sourceBinder selection output arena then
            throw error
          else throw error
    else
      let (direct, indexed) ← match arena.direct.pushIndexedOutput root binder selection subject with
        | some result => pure result
        | none => throw (.unsupportedOperationalExpr root)
      let value ← match direct.valueAt? indexed with
        | some value => pure value | none => throw (.invalidOperationalExprRef indexed)
      pure ({ arena with direct }, {
        context := value.context, payload := .directValue indexed, storage := value.storage })

def closeParallelDirectScalarOutput
    (scope : ScopeTemplateKey) (node indexSlot port : Nat) (declaredCount : IntExpr)
    (environment : ParamEnvironment) (arena : OperationalExprArena) (output : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let root := output.payload.root
  let subject : WireRef := { node, port }
  let binder := parallelLoopFamilyBinder scope node indexSlot
  let selection := parallelLoopLaneSelection scope node indexSlot declaredCount
  let value ← match arena.direct.valueAt? root with
    | some value => pure value | none => throw (.invalidOperationalExprRef root)
  match value.payload.schema with
  | .matrix _ => throw (.operandNotInteger node subject)
  | .scalar _ =>
      if output.context.binders.isEmpty then
        let (direct, indexed) ← arena.direct.mapScalarValue environment root
          (fun fact => pure (indexScalarFact binder selection subject fact))
        let indexedValue ← match direct.valueAt? indexed with
          | some value => pure value | none => throw (.invalidOperationalExprRef indexed)
        if indexedValue.context != emptyContext then throw (.unsupportedOperationalExpr indexed)
        let destination ← selectionIndexedContext selection indexed
        let map : IndexMap := { source := emptyContext, destination, assignments := #[] }
        let expression : OperationalFact := {
          context := emptyContext, payload := .directValue indexed, storage := indexedValue.storage }
        ({ arena with direct }).reindexDirectFact map expression environment
      else
        let expected ← match selection.expression with
          | .variable binder => pure binder
          | _ => throw (.unsupportedOperationalExpr root)
        if output.context.binders.contains expected then
          let sourceBinder ← match parallelLoopOutputBinder selection output.context root with
            | .ok binder => pure binder
            | .error error =>
                if parallelLoopOutputBinderFailureDiagnostic selection output arena then throw error
                else throw error
          let map ← match dynamicIndexMap output.context sourceBinder selection.expression with
            | some map => pure map
            | none =>
                if parallelLoopOutputMapFailureDiagnostic "construct_failed" sourceBinder selection output arena then
                  throw (.unsupportedOperationalExpr root)
                else throw (.unsupportedOperationalExpr root)
          match arena.reindexDirectFact map output environment with
          | .ok result => pure result
          | .error error =>
              if parallelLoopOutputMapFailureDiagnostic "reindex_failed" sourceBinder selection output arena then
                throw error
              else throw error
        else
          let (direct, indexed) ← match arena.direct.pushIndexedOutput root binder selection subject with
            | some result => pure result
            | none => throw (.unsupportedOperationalExpr root)
          let value ← match direct.valueAt? indexed with
            | some value => pure value | none => throw (.invalidOperationalExprRef indexed)
          pure ({ arena with direct }, {
            context := value.context, payload := .directValue indexed, storage := value.storage })

def parallelLoopIndexedMatrixOutput
    (scope : ScopeTemplateKey) (node indexSlot port : Nat) (declaredCount : IntExpr) (count : Nat)
    (environment : ParamEnvironment) (arena : OperationalExprArena) (output : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  if count = 0 then throw (.invalidCount node 0)
  closeParallelDirectMatrixOutput scope node indexSlot port declaredCount environment arena output

def loopTemplateArgumentExprWithDirectLaneBinder
    (arena : OperationalExprArena) (scope : ScopeTemplateKey) (node indexSlot argument : Nat)
    (declaredCount : IntExpr) (count : Nat) (mode : LoopInputMode)
    (directLaneBinder : Option IndexVariable) (environment : ParamEnvironment) (fact : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  match mode with
  | .broadcast =>
      if fact.context.binders.isEmpty then
        let consumer := parallelLoopLaneSelection scope node indexSlot declaredCount
        let destination ← selectionIndexedContext consumer fact.payload.root
        let map : IndexMap := { source := emptyContext, destination, assignments := #[] }
        arena.reindexDirectFact map fact environment
      else pure (arena, fact)
  | .zip | .zipOffset _ =>
      let sourceBinder ← match directLaneBinder with
        | some binder => pure binder | none => throw (.loopInputModeMismatch node argument)
      if !fact.context.binders.contains sourceBinder then throw (.loopInputModeMismatch node argument)
      let offset := match mode with | .zipOffset value => value | _ => 0
      let sourceCount ← match sourceBinder.count.evaluate environment with
        | some value => if value > 0 then pure value.toNat else throw (.loopInputModeMismatch node argument)
        | none => throw (.loopInputModeMismatch node argument)
      if count + offset > sourceCount then throw (.loopInputModeMismatch node argument)
      let consumer := parallelLoopLaneSelection scope node indexSlot declaredCount
      let assignment := .offset consumer.expression (Int.ofNat offset)
      let map ← match dynamicIndexMap fact.context sourceBinder assignment with
        | some map => pure map | none => throw (.loopInputModeMismatch node argument)
      arena.reindexDirectFact map fact environment

/-- Carry one sequential-loop input into the lexical body coordinate.  Sequential state remains
recurrence-owned, but body descriptors that refer to `loopIndex indexSlot` must resolve that
coordinate through an owner-bearing context rather than a slot-only fallback. -/
def sequentialLoopTemplateArgumentExpr
    (arena : OperationalExprArena) (scope : ScopeTemplateKey) (node indexSlot : Nat)
    (declaredCount : IntExpr) (environment : ParamEnvironment) (fact : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let selection := sequentialLoopLaneSelection scope node indexSlot declaredCount
  let destination ← selectionIndexedContext selection fact.payload.root
  if fact.context.binders.isEmpty then
    let map : IndexMap := { source := emptyContext, destination, assignments := #[] }
    arena.reindexDirectFact map fact environment
  else if fact.context == destination then
    pure (arena, fact)
  else
    throw (.loopInputModeMismatch node indexSlot)

/-- A relation-free, identity-free summary of a compact direct matrix carrier.  This is an
analysis result, never a replacement `OperationalMatrixFact`: callers which need provenance,
relations, or an exact polynomial must continue through `reducedDirectValueFactsAt`.  It bounds
ordinary delayed pointwise addition without materializing a Cartesian product of gather domains. -/
structure DirectMatrixEnvelope where
  matrixType : MatrixTypeExpr
  hardBound : OperationalBoundExpr
  hasLarge : Bool
  canonicalRange : CanonicalRange
  relationFree : Bool
  storedAlternativeVisits : Nat
  maximumPolynomialTerms : Nat
  deriving Repr

private def directMatrixEnvelopeForFact
    (fact : OperationalMatrixFact) : Except OperationalError DirectMatrixEnvelope := do
  let canonicalRange ← match fact.canonicalRange with
    | .below 0 => throw (.invalidMatrixParameters fact.subject.node)
    | range => pure range
  pure {
    matrixType := fact.matrixType
    hardBound := fact.totalHardBound
    hasLarge := fact.polynomial.any operationalTermIsSignal
    canonicalRange
    relationFree := !matrixFactHasRelation fact
    storedAlternativeVisits := 1
    maximumPolynomialTerms := fact.polynomial.length
  }

/-- Check the final direct-carrier invariant before a bound-only consumer traverses a delayed
carrier.  Constructors enforce the same facts on the normal path, but endpoint analysis must
also reject a manually assembled malformed arena rather than treating one surviving branch as a
valid envelope.  This is storage validation only: it neither reduces a carrier nor changes the
exact reduction semantics. -/
private partial def directCarrierPayloadValidAt
    (arena : DirectOperationalIndexedArena)
    (environment : ParamEnvironment)
    (id : OperationalIndexedValueId) : Nat → Bool
  | 0 => false
  | fuel + 1 => match arena.valueAt? id with
    | none => false
    | some value =>
        validateContext value.context && match value.payload with
        | .shared schema reference => arena.fixed.refHasSchema schema reference
        | .explicit schema binder references =>
            value.context == { binders := #[binder] } && explicitCountValid environment binder references &&
              references.all (arena.fixed.refHasSchema schema)
        | .explicitValues schema binder values =>
            match values.toList.mapM arena.valueAt? with
            | none => false
            | some children =>
                !values.isEmpty &&
                  explicitCountValid environment binder
                    (Array.replicate values.size (.matrix 0)) &&
                  children.all (fun child => child.payload.schema == schema) &&
                  values.toList.all (fun child =>
                    directCarrierPayloadValidAt arena environment child fuel) &&
                  match mergeIndexContextsN (children.map (·.context)) with
                  | some context =>
                      match extendContext context binder with
                      | some context => value.context == context
                      | none => false
                  | none => false
        | .mapped schema source map =>
            match arena.valueAt? source, reindexOperationalIndexedPayloadSchema map schema with
            | some sourceValue, some _ =>
                directCarrierPayloadValidAt arena environment source fuel && map.transportValid &&
                  map.source == sourceValue.context && map.destination == value.context &&
                  reindexOperationalIndexedPayloadSchema map sourceValue.payload.schema == some schema
            | _, _ => false
        | .rebound schema source _ =>
            match arena.valueAt? source with
            | some sourceValue =>
                directCarrierPayloadValidAt arena environment source fuel &&
                  sourceValue.context == value.context && sourceValue.payload.schema == schema
            | none => false
        | .indexedOutput schema source _ selection _ =>
            match arena.valueAt? source, selection.expression.identityVariable? with
            | some sourceValue, some selector =>
                directCarrierPayloadValidAt arena environment source fuel &&
                  sourceValue.payload.schema == schema &&
                  match extendContext sourceValue.context selector with
                  | some context => value.context == context &&
                      indexExpressionInBounds value.context selection.expression
                  | none => false
            | _, _ => false
        | .matrixResultBound schema source _ =>
            match arena.valueAt? source with
            | some sourceValue =>
                directCarrierPayloadValidAt arena environment source fuel &&
                  sourceValue.context == value.context && sourceValue.payload.schema == schema &&
                  match schema with | .matrix _ => true | .scalar _ => false
            | none => false
        | .pointwise schema operation inputs =>
            match inputs.toList.mapM arena.valueAt? with
            | none => false
            | some children =>
                inputs.toList.all (fun child =>
                  directCarrierPayloadValidAt arena environment child fuel) &&
                  match mergeIndexedFactShapeN children with
                  | some (context, _) =>
                      value.context == context &&
                        pointwiseSchemasValid operation (children.toArray.map (·.payload.schema)) schema
                  | none => false

/-- `PrimitiveOperation.outputType` is an owner-aware descriptor and may not silently disagree
with the Graph-IR output schema retained beside it.  Envelope evaluation has no fixed index
assignment, so only descriptors closed in the request environment are admitted here. -/
private def closedPrimitiveOutputTypeMatchesSchema
    (environment : ParamEnvironment)
    (operation : PrimitiveOperation) : Bool :=
  match operation.outputType.closedIr?, operation.outputSchema.evaluate environment (.constant 0) with
  | some outputType, some schema =>
      match outputType.evaluate environment (.constant 0) with
      | some outputType => sameConcreteMatrixShape outputType schema
      | none => false
  | _, _ => false

/- Fold alternatives of one carrier domain.  The domain denotes a choice, hence its bound is a
maximum rather than an addition. -/
private def joinDirectMatrixEnvelopeAlternatives
    (left right : DirectMatrixEnvelope) : Except OperationalError DirectMatrixEnvelope := do
  if left.matrixType != right.matrixType then throw (.outputTypeMismatch 0)
  pure {
    matrixType := left.matrixType
    hardBound := .maximum left.hardBound right.hardBound
    hasLarge := left.hasLarge || right.hasLarge
    canonicalRange := joinCanonicalRanges [left.canonicalRange, right.canonicalRange]
    relationFree := left.relationFree && right.relationFree
    storedAlternativeVisits := left.storedAlternativeVisits + right.storedAlternativeVisits
    maximumPolynomialTerms := max left.maximumPolynomialTerms right.maximumPolynomialTerms
  }

/- Transfer a relation-free ordinary pointwise addition/subtraction.  Unlike domain folding,
both operands occur in every result, so their hard bounds add. -/
private def transferDirectMatrixEnvelopeAdd
    (environment : ParamEnvironment)
    (subtract : Bool)
    (left right : DirectMatrixEnvelope) : Except OperationalError DirectMatrixEnvelope := do
  if left.matrixType != right.matrixType then throw (.outputTypeMismatch 0)
  let modulus ← match left.matrixType.modulus.evaluate environment with
    | some modulus => pure modulus
    | none => throw .nonClosedExpression
  if modulus <= 0 then throw (.invalidMatrixParameters 0)
  let canonicalRange := match left.canonicalRange, right.canonicalRange with
    | .below leftUpper, .below rightUpper =>
        if leftUpper == 0 || rightUpper == 0 then .unknown else
        if subtract then
          /- Residue subtraction wraps unless the right operand is exactly zero. -/
          if rightUpper <= 1 then .below leftUpper else .unknown
        else
          /- For `0 ≤ a < A`, `0 ≤ b < B`, the representative of `a+b mod q` is
          below `min q (A+B-1)`.  This is the same canonical-residue arithmetic used by the
          concrete matrix path, but applied to a branch hull rather than a representative. -/
          .below (min modulus.toNat (leftUpper + rightUpper - 1))
    | _, _ => .unknown
  pure {
    matrixType := left.matrixType
    hardBound := .add left.hardBound right.hardBound
    hasLarge := left.hasLarge || right.hasLarge
    canonicalRange
    relationFree := left.relationFree && right.relationFree
    storedAlternativeVisits := left.storedAlternativeVisits + right.storedAlternativeVisits
    maximumPolynomialTerms := left.maximumPolynomialTerms + right.maximumPolynomialTerms
  }

/-- Fold a direct matrix carrier into an envelope without assigning unresolved gather
coordinates.  Mapped views preserve the source bound but must still be structurally valid.
Only relation-free add/sub is admitted; every other delayed primitive continues to require the
exact correlation zipper. -/
private def directMatrixEnvelopeAt
    (arena : DirectOperationalIndexedArena)
    (environment : ParamEnvironment)
    (maps : List IndexMap)
    (id : OperationalIndexedValueId) : Nat → Except OperationalError DirectMatrixEnvelope
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => do
      let value ← match arena.valueAt? id with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef id)
      match value.payload with
      | .shared (.matrix _) (.matrix reference) =>
          match arena.fixed.matrices[reference]? with
          | some fact => directMatrixEnvelopeForFact (← reindexReducedMatrixFact environment maps fact)
          | none => throw (.invalidOperationalExprRef reference)
      | .explicit (.matrix _) _ references => do
          let summaries ← references.toList.mapM fun reference => match reference with
            | .matrix reference => match arena.fixed.matrices[reference]? with
              | some fact => do
                  directMatrixEnvelopeForFact (← reindexReducedMatrixFact environment maps fact)
              | none => throw (.invalidOperationalExprRef reference)
            | .scalar _ => throw (.unsupportedOperationalExpr id)
          match summaries with
          | [] => throw (.invalidCount id 0)
          | first :: remaining => remaining.foldlM joinDirectMatrixEnvelopeAlternatives first
      | .explicitValues (.matrix _) _ values => do
          let summaries ← values.toList.mapM fun child =>
            directMatrixEnvelopeAt arena environment maps child fuel
          match summaries with
          | [] => throw (.invalidCount id 0)
          | first :: remaining => remaining.foldlM joinDirectMatrixEnvelopeAlternatives first
      | .mapped (.matrix _) source map =>
          if !map.transportValid || map.destination != value.context then
            throw (.unsupportedOperationalExpr id)
          else directMatrixEnvelopeAt arena environment (map :: maps) source fuel
      | .rebound (.matrix _) source _ | .indexedOutput (.matrix _) source _ _ _ =>
          directMatrixEnvelopeAt arena environment maps source fuel
      /- A result-bound annotation can carry an indexed expression which needs the exact mapped
      assignment.  Do not silently drop that transport in the compact envelope path. -/
      | .matrixResultBound .. => throw (.unsupportedOperationalExpr id)
      | .pointwise (.matrix matrixType) (.matrix operation) inputs => do
          let descriptor ← reindexReducedPointwiseOperation environment maps (.matrix operation)
          let operation ← match descriptor with
            | .matrix operation => pure operation
            | _ => throw (.unsupportedOperationalExpr id)
          if operation.outputSchema != matrixType ||
              !closedPrimitiveOutputTypeMatchesSchema environment operation then
            throw (.outputTypeMismatch operation.ownerNode)
          match operation.kind, inputs.toList with
          | .add subtract, [left, right] => do
              let left ← directMatrixEnvelopeAt arena environment maps left fuel
              let right ← directMatrixEnvelopeAt arena environment maps right fuel
              if !matrixOperationSchemasValid operation #[.matrix left.matrixType, .matrix right.matrixType]
                  matrixType then throw (.outputTypeMismatch operation.ownerNode)
              let result ← transferDirectMatrixEnvelopeAdd environment subtract left right
              if result.matrixType != matrixType || !result.relationFree then
                throw (.unsupportedOperationalExpr id)
              let cap ← match matrixCap matrixType environment with
                | some cap => pure cap
                | none => throw (.invalidMatrixParameters operation.ownerNode)
              let hardBound := OperationalBoundExpr.minimum
                (OperationalBoundExpr.closedInt (IntExpr.constant cap)) result.hardBound
              pure { result with matrixType := matrixType, hardBound := hardBound }
          | _, _ => throw (.unsupportedOperationalExpr id)
      | _ => throw (.unsupportedOperationalExpr id)

/-- Conservative bound-only analysis for a delayed direct carrier.  Exact reductions stay the
authority for all structural consumers; this API deliberately exposes no representative fact,
identity, or relation inventory. -/
def OperationalExprArena.directMatrixEnvelope
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (expression : OperationalFact) : Except OperationalError DirectMatrixEnvelope := do
  let root := expression.payload.root
  if !directCarrierPayloadValidAt arena.direct environment root (arena.direct.values.size + 1) then
    throw (.unsupportedOperationalExpr root)
  let envelope ← directMatrixEnvelopeAt arena.direct environment [] root (arena.direct.values.size + 1)
  if !envelope.relationFree then throw (.unsupportedOperationalExpr root)
  pure envelope

/-- Cheap structural prefilter for a delayed matrix envelope.  It recognizes only an ordinary
binary add/sub through carrier views; it does not inspect contexts, maps, schemas, relations, or
expression bounds.  Those are admission invariants of `directMatrixEnvelope`, so a shared outer
loop binder cannot make this prefilter reject independently gathered nested operands. -/
private def directMatrixEnvelopeShapePrefilterAt
    (arena : DirectOperationalIndexedArena) (id : OperationalIndexedValueId) : Nat → Bool
  | 0 => false
  | fuel + 1 => match arena.valueAt? id with
      | some { payload := .mapped (.matrix _) source _, .. }
      | some { payload := .rebound (.matrix _) source _, .. }
      | some { payload := .indexedOutput (.matrix _) source _ _ _, .. } =>
          directMatrixEnvelopeShapePrefilterAt arena source fuel
      | some { payload := .pointwise (.matrix _) (.matrix { kind := .add _, .. }) #[_, _], .. } => true
      | _ => false

/-- Cheap structural prefilter only.  Every accepted candidate must subsequently pass the
authoritative recursive validation in `directMatrixEnvelope`. -/
def OperationalExprArena.directMatrixEnvelopeShapePrefilter
    (arena : OperationalExprArena) (expression : OperationalFact) : Bool :=
  directMatrixEnvelopeShapePrefilterAt arena.direct expression.payload.root
    (arena.direct.values.size + 1)

/-- Conservative interval transfer for coefficient extraction from a compact matrix envelope.
This is deliberately an interval-only API: it cannot manufacture the scalar
identity needed by a relation or exact scalar pointwise operation.  A strict canonical range is
retained; otherwise every canonical residue modulo `q` remains possible. -/
private def directScalarExactIntervalAt
    (arena : DirectOperationalIndexedArena)
    (environment : ParamEnvironment)
    (maps : List IndexMap)
    (id : OperationalIndexedValueId) : Nat → Except OperationalError (Int × Int)
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => do
      let evaluation ← reducedDirectScalarFactAt arena environment maps id fuel
      let intervals ← evaluation.entries.mapM fun entry => match entry.fact with
        | .integer fact => pure (fact.lower, fact.upper)
        | _ => throw (.operandNotInteger 0 { node := 0, port := 0 })
      match intervals with
      | [] => throw (.invalidCount id 0)
      | first :: remaining => pure <| remaining.foldl (fun (lower, upper) (nextLower, nextUpper) =>
          (min lower nextLower, max upper nextUpper)) first

private def directScalarEnvelopeIntervalAt
    (arena : DirectOperationalIndexedArena)
    (environment : ParamEnvironment)
    (maps : List IndexMap)
    (id : OperationalIndexedValueId) : Nat → Except OperationalError (Int × Int)
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => do
      let value ← match arena.valueAt? id with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef id)
      match value.payload with
      | .mapped (.scalar _) source map =>
          if !map.transportValid || map.destination != value.context then
            throw (.unsupportedOperationalExpr id)
          else directScalarEnvelopeIntervalAt arena environment (map :: maps) source fuel
      | .rebound (.scalar _) source _ | .indexedOutput (.scalar _) source _ _ _ =>
          directScalarEnvelopeIntervalAt arena environment maps source fuel
      | .pointwise (.scalar .integer) (.matrixToScalar operation) #[matrix] =>
          let descriptor ← reindexReducedPointwiseOperation environment maps (.matrixToScalar operation)
          let operation ← match descriptor with
            | .matrixToScalar operation => pure operation
            | _ => throw (.unsupportedOperationalExpr id)
          if !directMatrixEnvelopeShapePrefilterAt arena matrix fuel then
            directScalarExactIntervalAt arena environment maps id fuel
          else match operation.kind with
          | .extractCoefficient position => do
              let envelope ← directMatrixEnvelopeAt arena environment maps matrix fuel
              let modulus ← match envelope.matrixType.modulus.evaluate environment with
                | some modulus => pure modulus
                | none => throw .nonClosedExpression
              if modulus <= 0 then throw (.invalidMatrixParameters operation.ownerNode)
              let ringDimension ← match envelope.matrixType.ringDimension.evaluate environment with
                | some value => pure value
                | none => throw .nonClosedExpression
              let position ← evaluateIntInvariant operation.parameterEnvironment [] position
              if position < 0 || ringDimension <= 0 || position >= ringDimension then
                throw (.invalidCount operation.ownerNode position)
              match envelope.canonicalRange with
              | .below 0 => throw (.invalidMatrixParameters operation.ownerNode)
              | .below upper => pure (0, upper - 1)
              | .unknown => pure (0, modulus - 1)
          | _ => throw (.unsupportedOperationalExpr id)
      | .pointwise (.scalar .integer) (.scalar operation) #[left, right] => do
          let descriptor ← reindexReducedPointwiseOperation environment maps (.scalar operation)
          let operation ← match descriptor with
            | .scalar operation => pure operation
            | _ => throw (.unsupportedOperationalExpr id)
          match operation.kind with
          | .intBinary kind => do
              let (leftLower, leftUpper) ← directScalarEnvelopeIntervalAt arena environment maps left fuel
              let (rightLower, rightUpper) ← directScalarEnvelopeIntervalAt arena environment maps right fuel
              let left ← match ← integerFact operation.ownerNode operation.outputPort leftLower leftUpper with
                | .integer fact => pure fact | _ => throw (.operandNotInteger operation.ownerNode
                    { node := operation.ownerNode, port := operation.outputPort })
              let right ← match ← integerFact operation.ownerNode operation.outputPort rightLower rightUpper with
                | .integer fact => pure fact | _ => throw (.operandNotInteger operation.ownerNode
                    { node := operation.ownerNode, port := operation.outputPort })
              let interval ← integerBinaryInterval operation.ownerNode kind left right
              pure (interval.lower, interval.upper)
          | _ => directScalarExactIntervalAt arena environment maps id fuel
      | _ => directScalarExactIntervalAt arena environment maps id fuel

def OperationalExprArena.directIntegerEnvelopeInterval
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (owner : Nat)
    (wire : WireRef)
    (expression : OperationalFact) : Except OperationalError (Int × Int) := do
  let _ := owner
  let _ := wire
  if !directCarrierPayloadValidAt arena.direct environment expression.payload.root
      (arena.direct.values.size + 1) then
    throw (.unsupportedOperationalExpr expression.payload.root)
  directScalarEnvelopeIntervalAt arena.direct environment [] expression.payload.root
    (arena.direct.values.size + 1)

private def directScalarEnvelopeShapePrefilterAt
    (arena : DirectOperationalIndexedArena) (id : OperationalIndexedValueId) : Nat → Bool
  | 0 => false
  | fuel + 1 => match arena.valueAt? id with
      | some { payload := .mapped (.scalar _) source _, .. }
      | some { payload := .rebound (.scalar _) source _, .. }
      | some { payload := .indexedOutput (.scalar _) source _ _ _, .. } =>
          directScalarEnvelopeShapePrefilterAt arena source fuel
      | some { payload := .pointwise (.scalar .integer) descriptor inputs, .. } =>
          match descriptor, inputs with
          | .matrixToScalar { kind := .extractCoefficient _, .. }, #[matrix] =>
              directMatrixEnvelopeShapePrefilterAt arena matrix fuel
          | .scalar { kind := .intBinary _, .. }, #[left, right] =>
              directScalarEnvelopeShapePrefilterAt arena left fuel ||
                directScalarEnvelopeShapePrefilterAt arena right fuel
          | _, _ => false
      | _ => false

/-- Cheap scalar-extraction structural prefilter.  This never admits a carrier by itself;
`directIntegerEnvelopeInterval` performs the authoritative recursive validation. -/
def OperationalExprArena.directScalarEnvelopeShapePrefilter
    (arena : OperationalExprArena) (expression : OperationalFact) : Bool :=
  directScalarEnvelopeShapePrefilterAt arena.direct expression.payload.root
    (arena.direct.values.size + 1)

end Mxx.Certificate
