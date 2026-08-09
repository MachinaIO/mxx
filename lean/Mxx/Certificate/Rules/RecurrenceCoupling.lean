import Mxx.Certificate.Typing
import Mxx.Certificate.Workflow
import Mxx.Certificate.SymbolicRecurrenceConstruction
import Mxx.Certificate.FrozenDependencySlice
import Mxx.Certificate.Rules.RequirementAcceptance

namespace Mxx.Certificate

/-! # Closed recurrence-coupling matcher foundation

This file contains analyzer-owned syntax and structural checks for relating two distinct
sequential recurrences.  None of these types is reachable from `SparseCertificate` or from the
serialized protocol declaration.  Semantic initial/step evidence is intentionally not modeled
until the symbolic-form transfer exists; structural acceptance alone is not a soundness proof.
-/

def frozenScopeAtPath?
    (program : Mxx.Ir.Prog)
    (scope : StaticScopeId) : Option Mxx.Ir.Scope :=
  match scope.path.reverse with
  | [] => some program.root
  | definition :: _ => Mxx.Ir.lookupDefinition definition program.definitions

/-- Exact frozen `gadgetDecompose` occurrence retained by a dependency slice.  This replaces the
obsolete model that searched for a literal gadget matrix expression: the executable BGG bodies
decompose a selected public key and expose base and digit count on this node. -/
structure CheckedGadgetDecompositionNode (program : Mxx.Ir.Prog) where
  site : CoreNodeRef
  scope : Mxx.Ir.Scope
  scopeFound : frozenScopeAtPath? program site.scope = some scope
  node : Mxx.Ir.Node
  nodeFound : scope.nodes[site.node.value]? = some node
  decompositionType : MatrixTypeExpr
  base : IntExpr
  digitCount : IntExpr
  input : Mxx.Ir.WireRef
  outputPort : Nat
  kindMatches : node.kind = .gadgetDecompose decompositionType base digitCount
  argumentsMatch : node.arguments = [input]
  outputInBounds : outputPort < node.outputCount

private def checkGadgetDecompositionAt
    (program : Mxx.Ir.Prog)
    (site : CoreNodeRef)
    (outputPort : Nat := 0) : Option (CheckedGadgetDecompositionNode program) :=
  match scopeFound : frozenScopeAtPath? program site.scope with
  | none => none
  | some scope =>
      match nodeFound : scope.nodes[site.node.value]? with
      | none => none
      | some node =>
          match kindMatches : node.kind with
          | .gadgetDecompose decompositionType base digitCount =>
              match argumentsMatch : node.arguments with
              | [input] =>
                  if outputInBounds : outputPort < node.outputCount then
                    some {
                      site
                      scope
                      scopeFound
                      node
                      nodeFound
                      decompositionType
                      base
                      digitCount
                      input
                      outputPort
                      kindMatches
                      argumentsMatch
                      outputInBounds
                    }
                  else none
              | _ => none
          | _ => none

/-- All actual decomposition nodes reachable from one frozen output slice. -/
def findGadgetDecompositions
    (program : Mxx.Ir.Prog)
    (slice : FrozenDependencySlice) : List (CheckedGadgetDecompositionNode program) :=
  slice.sites.eraseDups.filterMap (checkGadgetDecompositionAt program)

private def appendDecompositionIfMissing
    {program : Mxx.Ir.Prog}
    (nodes : List (CheckedGadgetDecompositionNode program))
    (node : CheckedGadgetDecompositionNode program) :
    List (CheckedGadgetDecompositionNode program) :=
  if nodes.any fun candidate => candidate.site = node.site then nodes else nodes ++ [node]

/-- The frozen executable interface of one analyzer-produced recurrence transfer. The body
slices are rooted at the exact declared outputs and retain nested call sites. -/
structure FrozenSequentialRecurrenceInterface where
  transfer : SymbolicRecurrenceTransfer
  program : Mxx.Ir.Prog
  loopScope : Mxx.Ir.Scope
  loopScopeFound : frozenScopeAtPath? program transfer.source.loop.site.scope = some loopScope
  loopNode : Mxx.Ir.Node
  loopNodeFound : loopScope.nodes[transfer.source.loop.site.node.value]? = some loopNode
  definition : String
  count : IntExpr
  indexSlot : Nat
  bindings : List (String × IntExpr)
  carriedCount : Nat
  loopKindMatches : loopNode.kind =
    .sequentialLoop definition count indexSlot bindings carriedCount
  body : Mxx.Ir.Scope
  bodyFound : Mxx.Ir.lookupDefinition definition program.definitions = some body
  outputSlices : List FrozenDependencySlice

/-- Distinct executable decomposition nodes reachable from every carried output. -/
def FrozenSequentialRecurrenceInterface.gadgetDecompositions
    (interface : FrozenSequentialRecurrenceInterface) :
    List (CheckedGadgetDecompositionNode interface.program) :=
  (interface.outputSlices.flatMap (findGadgetDecompositions interface.program)).foldl
    appendDecompositionIfMissing []

/-- A candidate that has exactly one reachable decomposition node. Full BGG matching applies
gate-shape and role-indexed origin checks after this cheap prefilter. -/
def FrozenSequentialRecurrenceInterface.uniqueGadgetDecomposition?
    (interface : FrozenSequentialRecurrenceInterface) :
    Option (CheckedGadgetDecompositionNode interface.program) :=
  match interface.gadgetDecompositions with
  | [node] => some node
  | _ => none

private def matrixFamilyType? : CarriedValueSchema → Option (IntExpr × MatrixTypeExpr)
  | .family count (.matrix matrixType _) => some (count, matrixType)
  | _ => none

private def isBooleanFamily : CarriedValueSchema → Bool
  | .family _ .boolean => true
  | _ => false

private def isOneByOneMatrixFamily : CarriedValueSchema → Bool
  | .family _ (.matrix matrixType _) =>
      matrixType.rows == .constant 1 && matrixType.columns == .constant 1
  | _ => false

private def isEncryptionBggPrefilter
    (interface : FrozenSequentialRecurrenceInterface) : Bool :=
  match interface.transfer.carriedSchemas with
  | [schema] => (matrixFamilyType? schema).isSome &&
      interface.uniqueGadgetDecomposition?.isSome
  | _ => false

private def isDecryptionBggPrefilter
    (interface : FrozenSequentialRecurrenceInterface) : Bool :=
  match interface.transfer.carriedSchemas with
  | [left, middle, right] =>
      [left, middle, right].all (fun schema => (matrixFamilyType? schema).isSome) &&
        ([left, middle, right].filter isOneByOneMatrixFamily).length == 1 &&
        interface.uniqueGadgetDecomposition?.isSome
  | _ => false

private def isBooleanInterpreterPrefilter
    (interface : FrozenSequentialRecurrenceInterface) : Bool :=
  match interface.transfer.carriedSchemas with
  | [schema] => isBooleanFamily schema
  | _ => false

inductive FrozenRecurrenceInterfaceError where
  | missingLoopScope
  | missingLoopNode
  | notSequentialLoop
  | countMismatch
  | carriedArityMismatch
  | iterationSlotMismatch
  | missingBody
  | bodyInputMismatch
  | bodyOutputArityMismatch
  | invalidBodyOutput (slot : Nat)
  deriving BEq, DecidableEq, Repr

/-- Recover the exact frozen loop and all carried-output dependency slices. This accepts no node
number or definition name other than the identity already present in the analyzer-produced
transfer. -/
def checkFrozenRecurrenceInterface
    (program : Mxx.Ir.Prog)
    (transfer : SymbolicRecurrenceTransfer) :
    Except FrozenRecurrenceInterfaceError FrozenSequentialRecurrenceInterface :=
  match loopScopeFound : frozenScopeAtPath? program transfer.source.loop.site.scope with
  | none => throw .missingLoopScope
  | some loopScope =>
      match loopNodeFound : loopScope.nodes[transfer.source.loop.site.node.value]? with
      | none => throw .missingLoopNode
      | some loopNode =>
          match loopKindMatches : loopNode.kind with
          | .sequentialLoop definition count indexSlot bindings carriedCount => do
              unless count = transfer.source.count do throw .countMismatch
              unless carriedCount = transfer.source.carriedArity do
                throw .carriedArityMismatch
              unless indexSlot = transfer.source.iterationVariable.slot do
                throw .iterationSlotMismatch
              match bodyFound : Mxx.Ir.lookupDefinition definition program.definitions with
              | none => throw .missingBody
              | some body => do
                  unless transfer.source.bodyInputs.toList.all fun input =>
                      input.definition = {
                        stage := transfer.source.loop.site.stage
                        name := definition
                      } do
                    throw .bodyInputMismatch
                  unless body.outputs.length = carriedCount do
                    throw .bodyOutputArityMismatch
                  let mut outputSlices : List FrozenDependencySlice := []
                  for output in body.outputs, slot in [0:body.outputs.length] do
                    let slice ← buildFrozenDependencySlice program
                      transfer.source.loop.site.stage
                      ⟨transfer.source.loop.site.scope.path ++ [definition]⟩ body output.2
                      (program.definitions.length + 1)
                      |>.elim (throw (.invalidBodyOutput slot)) pure
                    outputSlices := outputSlices ++ [slice]
                  return {
                    transfer
                    program
                    loopScope
                    loopScopeFound
                    loopNode
                    loopNodeFound
                    definition
                    count
                    indexSlot
                    bindings
                    carriedCount
                    loopKindMatches
                    body
                    bodyFound
                    outputSlices
                  }
          | _ => throw .notSequentialLoop

/-- The closed relation universe for pairs of recurrence results. -/
inductive RecurrenceRelationKind where
  | quotientEqual
  | bggEncodingOf
  deriving BEq, DecidableEq, Repr

/-- A coarse typed path to a matrix-valued element of a carried family.  Unlike
`MatrixFactPath`, it cannot address an affine term. `nestedFamilyDepth = 0` denotes a matrix
element of the outer family, one denotes a matrix element of a nested family, and so on. -/
structure FamilyMatrixPath where
  carriedSlot : Nat
  nestedFamilyDepth : Nat := 0
  deriving BEq, DecidableEq, Repr

private def familyMatrixTypeAtDepth : ValueFactSchema → Nat → Option MatrixTypeExpr
  | .family _ (.matrix matrixType ..), 0 => some matrixType
  | .family _ element, depth + 1 => familyMatrixTypeAtDepth element depth
  | _, _ => none

private def familyMatrixTypeOfFact (fact : ValueFact) (depth : Nat) : Option MatrixTypeExpr :=
  match fact with
  | .family family => familyMatrixTypeAtDepth (.family family.count family.elementSchema) depth
  | _ => none

/-- Resolve a family-matrix role only when the initial value and one-step output template expose
the same coarse matrix type.  Exact/affine form and term count are deliberately irrelevant. -/
def SequentialRecurrenceSource.resolveFamilyMatrixType
    (recurrence : SequentialRecurrenceSource)
    (path : FamilyMatrixPath) : Option MatrixTypeExpr := do
  let initial ← recurrence.initial.toList[path.carriedSlot]?
  let output ← recurrence.bodyOutputs.toList[path.carriedSlot]?
  let initialType ← familyMatrixTypeOfFact initial.fact path.nestedFamilyDepth
  let outputType ← familyMatrixTypeAtDepth output.schema path.nestedFamilyDepth
  if initialType == outputType then some initialType else none

/-- Exact analyzer evidence that two count expressions have the same frozen expression origin.
For current IR integer expressions, parameter names and loop slots are stable origin identities;
the matcher does not compare evaluated integers. -/
structure CheckedIntExprOriginEquality (left right : IntExpr) where
  origin : IntExpr
  leftEq : left = origin
  rightEq : right = origin

private def intExprHasStableGlobalOrigin : IntExpr → Bool
  | .constant _ | .parameter _ => true
  | .loopIndex _ => false
  | .add left right | .subtract left right | .multiply left right |
      .divide left right | .roundDivide left right =>
      intExprHasStableGlobalOrigin left && intExprHasStableGlobalOrigin right
  | .log2Ceil value => intExprHasStableGlobalOrigin value

def checkIntExprOriginEquality
    (left right : IntExpr) : Option (CheckedIntExprOriginEquality left right) :=
  if intExprHasStableGlobalOrigin left then
    if equal : left = right then
      some { origin := left, leftEq := rfl, rightEq := equal.symm }
    else none
  else none

/-- Exact frozen-value origin equality.  Numerically equal values with different producer
identities are rejected. -/
structure CheckedValueOriginEquality where
  origin : ValueInstanceRef
  left : ValueInstanceRef
  right : ValueInstanceRef
  leftEq : left = origin
  rightEq : right = origin

def checkValueOriginEquality
    (left right : ValueInstanceRef) : Option CheckedValueOriginEquality :=
  if equal : left = right then
    some { origin := left, left, right, leftEq := rfl, rightEq := equal.symm }
  else none

/-- Two actual decomposition nodes use exactly the same output type, base, and digit count. -/
structure CheckedCanonicalGadgetDecomposition where
  leftProgram : Mxx.Ir.Prog
  rightProgram : Mxx.Ir.Prog
  left : CheckedGadgetDecompositionNode leftProgram
  right : CheckedGadgetDecompositionNode rightProgram
  decompositionType : MatrixTypeExpr
  base : IntExpr
  digitCount : IntExpr
  leftTypeMatches : left.decompositionType = decompositionType
  rightTypeMatches : right.decompositionType = decompositionType
  leftBaseMatches : left.base = base
  rightBaseMatches : right.base = base
  leftDigitCountMatches : left.digitCount = digitCount
  rightDigitCountMatches : right.digitCount = digitCount

inductive CanonicalGadgetMatchError where
  | matrixTypeMismatch
  | baseMismatch
  | digitCountMismatch
  deriving BEq, DecidableEq, Repr

def matchCanonicalGadgetDecompositions
    {leftProgram rightProgram : Mxx.Ir.Prog}
    (left : CheckedGadgetDecompositionNode leftProgram)
    (right : CheckedGadgetDecompositionNode rightProgram) :
    Except CanonicalGadgetMatchError CheckedCanonicalGadgetDecomposition :=
  if typeEqual : left.decompositionType = right.decompositionType then
    if baseEqual : left.base = right.base then
      if digitEqual : left.digitCount = right.digitCount then
        .ok {
          leftProgram
          rightProgram
          left
          right
          decompositionType := left.decompositionType
          base := left.base
          digitCount := left.digitCount
          leftTypeMatches := rfl
          rightTypeMatches := typeEqual.symm
          leftBaseMatches := rfl
          rightBaseMatches := baseEqual.symm
          leftDigitCountMatches := rfl
          rightDigitCountMatches := digitEqual.symm
        }
      else .error .digitCountMismatch
    else .error .baseMismatch
  else .error .matrixTypeMismatch

/-- Unique structural candidates for the current BGG/Boolean three-trace relation. This is only
the cheap prefilter; full acceptance additionally checks gate lanes, transported controls,
carried roles, initial relations, and one-step preservation. -/
structure CheckedBggRecurrencePrefilter where
  encryption : FrozenSequentialRecurrenceInterface
  decryption : FrozenSequentialRecurrenceInterface
  booleanInterpreter : FrozenSequentialRecurrenceInterface
  encryptionDecomposition : CheckedGadgetDecompositionNode encryption.program
  decryptionDecomposition : CheckedGadgetDecompositionNode decryption.program
  matchingDecomposition : CheckedCanonicalGadgetDecomposition
  encryptionDecryptionCount : CheckedIntExprOriginEquality
    encryption.transfer.source.count decryption.transfer.source.count
  encryptionBooleanCount : CheckedIntExprOriginEquality
    encryption.transfer.source.count booleanInterpreter.transfer.source.count
  requirementAcceptance : CheckedRequirementAcceptance

inductive BggRecurrencePrefilterError where
  | missingOrAmbiguousEncryption
  | missingOrAmbiguousDecryption
  | missingOrAmbiguousBooleanInterpreter
  | missingEncryptionDecomposition
  | missingDecryptionDecomposition
  | decompositionMismatch (error : CanonicalGadgetMatchError)
  | countOriginMismatch
  | missingOrAmbiguousRequirementAcceptance
  deriving Repr

/-- Run the fail-closed unique-candidate prefilter using only analyzer-owned interfaces and
acceptance evidence. -/
def checkBggRecurrencePrefilter
    (interfaces : List FrozenSequentialRecurrenceInterface)
    (acceptances : List CheckedRequirementAcceptance) :
    Except BggRecurrencePrefilterError CheckedBggRecurrencePrefilter := do
  let encryption ← match interfaces.filter isEncryptionBggPrefilter with
    | [candidate] => pure candidate
    | _ => throw .missingOrAmbiguousEncryption
  let decryption ← match interfaces.filter isDecryptionBggPrefilter with
    | [candidate] => pure candidate
    | _ => throw .missingOrAmbiguousDecryption
  let booleanInterpreter ← match interfaces.filter isBooleanInterpreterPrefilter with
    | [candidate] => pure candidate
    | _ => throw .missingOrAmbiguousBooleanInterpreter
  let encryptionDecomposition ← encryption.uniqueGadgetDecomposition?.elim
    (throw .missingEncryptionDecomposition) pure
  let decryptionDecomposition ← decryption.uniqueGadgetDecomposition?.elim
    (throw .missingDecryptionDecomposition) pure
  let matchingDecomposition ←
    matchCanonicalGadgetDecompositions encryptionDecomposition decryptionDecomposition
      |>.mapError .decompositionMismatch
  let encryptionDecryptionCount ← checkIntExprOriginEquality
    encryption.transfer.source.count decryption.transfer.source.count
    |>.elim (throw .countOriginMismatch) pure
  let encryptionBooleanCount ← checkIntExprOriginEquality
    encryption.transfer.source.count booleanInterpreter.transfer.source.count
    |>.elim (throw .countOriginMismatch) pure
  let requirementAcceptance ← match acceptances.filter fun acceptance =>
      acceptance.wrapper.selectedRecurrence = booleanInterpreter.transfer.identity with
    | [acceptance] => pure acceptance
    | _ => throw .missingOrAmbiguousRequirementAcceptance
  return {
    encryption
    decryption
    booleanInterpreter
    encryptionDecomposition
    decryptionDecomposition
    matchingDecomposition
    encryptionDecryptionCount
    encryptionBooleanCount
    requirementAcceptance
  }

/-- The four BGG family roles and deterministic gadget constant found by the frozen interface
matcher.  This is analyzer output, not protocol or sparse-certificate input. -/
structure CheckedBggEncodingSlots where
  encryptionPublicKeys : FamilyMatrixPath
  encodingVectors : FamilyMatrixPath
  encodingPublicKeys : FamilyMatrixPath
  plaintextMatrices : FamilyMatrixPath
  gadgetDecomposition : CheckedCanonicalGadgetDecomposition
  encryptionPublicKeyType : MatrixTypeExpr
  encodingVectorType : MatrixTypeExpr
  plaintextMatrixType : MatrixTypeExpr

inductive BggEncodingSlotMatchError where
  | invalidEncryptionPublicKeys
  | invalidEncodingVectors
  | invalidEncodingPublicKeys
  | invalidPlaintextMatrices
  | duplicateDecryptionRole
  | publicKeyTypeMismatch
  | plaintextNotOneByOne
  | plaintextRingMismatch
  | plaintextGadgetProductTyping
  | plaintextGadgetProductMismatch
  | encodingVectorRingMismatch
  deriving BEq, DecidableEq, Repr

/-- Validate roles already located by the frozen BGG interface matcher.  Kept private so no
protocol-facing API can choose a role.  Future interface discovery in this module will be the
only caller outside the regression fixtures below. -/
private def checkBggEncodingSlots
    (encryption decryption : SequentialRecurrenceSource)
    (encryptionPublicKeys encodingVectors encodingPublicKeys plaintextMatrices :
      FamilyMatrixPath)
    (gadgetDecomposition : CheckedCanonicalGadgetDecomposition) :
    Except BggEncodingSlotMatchError CheckedBggEncodingSlots := do
  let encryptionPublicKeyType ← match encryption.resolveFamilyMatrixType encryptionPublicKeys with
    | some matrixType => pure matrixType
    | none => throw .invalidEncryptionPublicKeys
  let encodingVectorType ← match decryption.resolveFamilyMatrixType encodingVectors with
    | some matrixType => pure matrixType
    | none => throw .invalidEncodingVectors
  let encodingPublicKeyType ← match decryption.resolveFamilyMatrixType encodingPublicKeys with
    | some matrixType => pure matrixType
    | none => throw .invalidEncodingPublicKeys
  let plaintextMatrixType ← match decryption.resolveFamilyMatrixType plaintextMatrices with
    | some matrixType => pure matrixType
    | none => throw .invalidPlaintextMatrices
  if encodingVectors = encodingPublicKeys || encodingVectors = plaintextMatrices ||
      encodingPublicKeys = plaintextMatrices then
    throw .duplicateDecryptionRole
  unless encryptionPublicKeyType = encodingPublicKeyType do
    throw .publicKeyTypeMismatch
  unless gadgetDecomposition.decompositionType.modulus = encodingPublicKeyType.modulus &&
      gadgetDecomposition.decompositionType.ringDimension =
        encodingPublicKeyType.ringDimension do
    throw .plaintextRingMismatch
  unless plaintextMatrixType.rows = .constant 1 &&
      plaintextMatrixType.columns = .constant 1 do
    throw .plaintextNotOneByOne
  unless plaintextMatrixType.modulus = encryptionPublicKeyType.modulus &&
      plaintextMatrixType.ringDimension = encryptionPublicKeyType.ringDimension do
    throw .plaintextRingMismatch
  /- The `gadgetDecompose` node's declared type is the decomposition/preimage matrix type.  The
  corresponding gadget matrix has the selected public-key type; for a 1×1 plaintext `m`, the
  signal term is `m * G` with that public-key type. -/
  let plaintextGadgetProduct ← inferMatrixProductType plaintextMatrixType
    encryptionPublicKeyType
    |>.mapError fun _ => .plaintextGadgetProductTyping
  unless plaintextGadgetProduct.output = encryptionPublicKeyType do
    throw .plaintextGadgetProductMismatch
  unless encodingVectorType.modulus = encryptionPublicKeyType.modulus &&
      encodingVectorType.ringDimension = encryptionPublicKeyType.ringDimension do
    throw .encodingVectorRingMismatch
  return {
    encryptionPublicKeys
    encodingVectors
    encodingPublicKeys
    plaintextMatrices
    gadgetDecomposition
    encryptionPublicKeyType
    encodingVectorType
    plaintextMatrixType
  }

private def decompositionOuterInputSlots
    (interface : FrozenSequentialRecurrenceInterface)
    (decomposition : CheckedGadgetDecompositionNode interface.program) : List Nat :=
  (interface.outputSlices.filterMap fun slice =>
      if slice.containsSite decomposition.site then
        slice.projectInputToOuterScope? interface.program decomposition.site decomposition.input
      else none).flatten.eraseDups

inductive BggCarriedRoleInferenceError where
  | decompositionInputNotUnique (slots : List Nat)
  | invalidPublicKeySlot
  | plaintextSlotNotUnique
  | encodingVectorSlotNotUnique
  | familyCountMismatch
  | publicKeyTypeMismatch
  | invalidSlots (error : BggEncodingSlotMatchError)
  deriving Repr

/-- Full carried-role result after the public-key slot has been identified by actual
decomposition reachability. -/
structure CheckedBggRecurrenceCandidate where
  prefilter : CheckedBggRecurrencePrefilter
  publicKeySlot : Nat
  encodingVectorSlot : Nat
  plaintextSlot : Nat
  slots : CheckedBggEncodingSlots

/-- One loop argument whose frozen root producer is a protocol input bound to the exact stage or
requirement containing the recurrence.  This records cross-program origin through the bundle's
input binding rather than equating distinct template-wire identities. -/
structure CheckedDirectLoopControl
    (bundle : ClosedProtocolBundle)
    (interface : FrozenSequentialRecurrenceInterface)
    (destinationFor : String → ProtocolInputDestination) where
  argumentIndex : Fin interface.loopNode.arguments.length
  inputNode : Mxx.Ir.Node
  inputName : String
  inputNodeFound : interface.loopScope.nodes[
    interface.loopNode.arguments[argumentIndex].node]? = some inputNode
  inputKind : inputNode.kind = .input inputName
  binding : ProtocolInputBinding
  bindingFound : bundle.inputBindings.find? (fun candidate =>
    candidate.destinations.contains (destinationFor inputName)) = some binding

/-- Protocol-input identities occur in loop-argument order.  Later gate-shape checking assigns
the four positions their active/kind/left/right roles; this foundation merely proves that all
three executable loops receive the same four frozen protocol inputs in the same order. -/
def CheckedDirectLoopControl.protocolInput
    {bundle : ClosedProtocolBundle}
    {interface : FrozenSequentialRecurrenceInterface}
    {destinationFor : String → ProtocolInputDestination}
    (control : CheckedDirectLoopControl bundle interface destinationFor) : ProtocolInputId :=
  control.binding.input

private def checkDirectLoopControlAt
    (bundle : ClosedProtocolBundle)
    (interface : FrozenSequentialRecurrenceInterface)
    (destinationFor : String → ProtocolInputDestination)
    (index : Nat) : Option (CheckedDirectLoopControl bundle interface destinationFor) :=
  if inBounds : index < interface.loopNode.arguments.length then
    let argumentIndex : Fin interface.loopNode.arguments.length := ⟨index, inBounds⟩
    let argument := interface.loopNode.arguments[argumentIndex]
    match inputNodeFound : interface.loopScope.nodes[argument.node]? with
    | none => none
    | some inputNode =>
        match inputKind : inputNode.kind with
        | .input inputName =>
            match bindingFound : bundle.inputBindings.find? (fun candidate =>
                candidate.destinations.contains (destinationFor inputName)) with
            | none => none
            | some binding => some {
                argumentIndex
                inputNode
                inputName
                inputNodeFound
                inputKind
                binding
                bindingFound
              }
        | _ => none
  else none

private def checkedDirectLoopControls
    (bundle : ClosedProtocolBundle)
    (interface : FrozenSequentialRecurrenceInterface)
    (destinationFor : String → ProtocolInputDestination) :
    List (CheckedDirectLoopControl bundle interface destinationFor) :=
  (List.range interface.loopNode.arguments.length).filterMap fun index =>
    if index < interface.transfer.source.carriedArity then none
    else checkDirectLoopControlAt bundle interface destinationFor index

/-- Frozen active/inactive selector at the final body of one carried-output lane. -/
structure CheckedActiveSelection (scope : Mxx.Ir.Scope) where
  outputName : String
  output : Mxx.Ir.WireRef
  outputFound : scope.outputs[0]? = some (outputName, output)
  outputPortZero : output.port = 0
  activeNode : Mxx.Ir.Node
  activeNodeFound : scope.nodes[output.node]? = some activeNode
  activeSelector : Mxx.Ir.WireRef
  inactiveValue : Mxx.Ir.WireRef
  gateResult : Mxx.Ir.WireRef
  activeKind : activeNode.kind = .select
  activeArguments : activeNode.arguments = [activeSelector, inactiveValue, gateResult]

private def checkActiveSelection (scope : Mxx.Ir.Scope) : Option (CheckedActiveSelection scope) :=
  match outputFound : scope.outputs[0]? with
  | none => none
  | some (outputName, output) =>
      if outputPortZero : output.port = 0 then
        match activeNodeFound : scope.nodes[output.node]? with
        | none => none
        | some activeNode =>
            match activeKind : activeNode.kind with
            | .select =>
                match activeArguments : activeNode.arguments with
                | [activeSelector, inactiveValue, gateResult] =>
                    some {
                      outputName
                      output
                      outputFound
                      outputPortZero
                      activeNode
                      activeNodeFound
                      activeSelector
                      inactiveValue
                      gateResult
                      activeKind
                      activeArguments
                    }
                | _ => none
            | _ => none
      else none

/-- Exact six-way gate selector found anywhere in the frozen dependency slice of one carried
output.  This permits graph factoring: encryption and Boolean currently combine gate and active
selection in one body, while decryption uses two consecutive parallel loops. -/
structure CheckedSixWayGateSelection (program : Mxx.Ir.Prog) where
  site : CoreNodeRef
  scope : Mxx.Ir.Scope
  scopeFound : frozenScopeAtPath? program site.scope = some scope
  node : Mxx.Ir.Node
  nodeFound : scope.nodes[site.node.value]? = some node
  gateSelector : Mxx.Ir.WireRef
  candidates : List Mxx.Ir.WireRef
  gateKind : node.kind = .select
  gateArguments : node.arguments = gateSelector :: candidates
  sixCandidates : candidates.length = 6

private def checkSixWayGateSelectionAt
    (program : Mxx.Ir.Prog)
    (site : CoreNodeRef) : Option (CheckedSixWayGateSelection program) :=
  match scopeFound : frozenScopeAtPath? program site.scope with
  | none => none
  | some scope =>
      match nodeFound : scope.nodes[site.node.value]? with
      | none => none
      | some node =>
          match gateKind : node.kind with
          | .select =>
              match gateArguments : node.arguments with
              | gateSelector :: candidates =>
                  if sixCandidates : candidates.length = 6 then
                    some {
                      site
                      scope
                      scopeFound
                      node
                      nodeFound
                      gateSelector
                      candidates
                      gateKind
                      gateArguments
                      sixCandidates
                    }
                  else none
              | _ => none
          | _ => none

private def findSixWayGateSelections
    (program : Mxx.Ir.Prog)
    (slice : FrozenDependencySlice) : List (CheckedSixWayGateSelection program) :=
  slice.sites.eraseDups.filterMap (checkSixWayGateSelectionAt program)

private def isTwoScalarMatrix : FrozenPointwiseMatrixFormula → Bool
  | .constant matrixType [.constant 2] =>
      matrixType.rows == .constant 1 && matrixType.columns == .constant 1
  | _ => false

private def isTwiceFormula
    (twice input : FrozenPointwiseMatrixFormula) : Bool :=
  match twice with
  | .scale (.constant 2) candidate => candidate == input
  | .multiply left right =>
      (left == input && isTwoScalarMatrix right) ||
        (isTwoScalarMatrix left && right == input)
  | _ => false

/-- Exact common six-candidate matrix skeleton.  The `andFormula` remains explicit because its
closed BGG shape differs between public-key, encoding-vector, and plaintext lanes; the subsequent
role matcher checks that shape. -/
structure CheckedSixWayMatrixSkeleton where
  formulas : List FrozenPointwiseMatrixFormula
  zeroFormula : FrozenPointwiseMatrixFormula
  oneFormula : FrozenPointwiseMatrixFormula
  leftFormula : FrozenPointwiseMatrixFormula
  notFormula : FrozenPointwiseMatrixFormula
  andFormula : FrozenPointwiseMatrixFormula
  xorFormula : FrozenPointwiseMatrixFormula
  rightFormula : FrozenPointwiseMatrixFormula
  twiceAndFormula : FrozenPointwiseMatrixFormula
  formulasMatch : formulas =
    [zeroFormula, oneFormula, leftFormula, notFormula, andFormula, xorFormula]
  zeroMatches : zeroFormula = .subtract oneFormula oneFormula
  notMatches : notFormula = .subtract oneFormula leftFormula
  xorMatches : xorFormula =
    .subtract (.add leftFormula rightFormula) twiceAndFormula
  twiceAndMatches : isTwiceFormula twiceAndFormula andFormula = true

private def checkSixWayMatrixSkeleton
    (formulas : List FrozenPointwiseMatrixFormula) :
    Option CheckedSixWayMatrixSkeleton :=
  match formulasMatch : formulas with
  | [zeroFormula, oneFormula, leftFormula, notFormula, andFormula,
      .subtract (.add xorLeft rightFormula) twiceAndFormula] =>
      if zeroMatches : zeroFormula = .subtract oneFormula oneFormula then
        if notMatches : notFormula = .subtract oneFormula leftFormula then
          if xorLeftMatches : xorLeft = leftFormula then
            if twiceAndMatches : isTwiceFormula twiceAndFormula andFormula = true then
              some {
                formulas
                zeroFormula
                oneFormula
                leftFormula
                notFormula
                andFormula
                xorFormula := .subtract (.add xorLeft rightFormula) twiceAndFormula
                rightFormula
                twiceAndFormula
                formulasMatch
                zeroMatches
                notMatches
                xorMatches := by rw [xorLeftMatches]
                twiceAndMatches
              }
            else none
          else none
        else none
      else none
  | _ => none

/-- A successful six-way check retains exactly the input formula list.  This public elimination
lemma lets later semantic modules use the checker result without unfolding its private matcher or
pretending that formula erasure is injective. -/
theorem CheckedSixWayMatrixSkeleton.formulas_eq_input
    {formulas : List FrozenPointwiseMatrixFormula}
    {skeleton : CheckedSixWayMatrixSkeleton}
    (found : checkSixWayMatrixSkeleton formulas = some skeleton) :
    skeleton.formulas = formulas := by
  unfold checkSixWayMatrixSkeleton at found
  split at found <;> simp_all
  all_goals aesop

/-- A six-way lane whose `AND` candidate is exactly public-key multiplication by the gadget
decomposition of the right input.  The decomposition parameters are retained from the frozen
executable node rather than supplied by a certificate. -/
structure CheckedPublicKeyGateFormula
    (formulas : List FrozenPointwiseMatrixFormula) where
  skeleton : CheckedSixWayMatrixSkeleton
  skeletonFound : checkSixWayMatrixSkeleton formulas = some skeleton
  decompositionType : MatrixTypeExpr
  base : IntExpr
  digitCount : IntExpr
  andMatches : skeleton.andFormula = .multiply skeleton.leftFormula
    (.decompose decompositionType base digitCount skeleton.rightFormula)

private def checkPublicKeyGateFormula
    (formulas : List FrozenPointwiseMatrixFormula) :
    Option (CheckedPublicKeyGateFormula formulas) :=
  match skeletonFound : checkSixWayMatrixSkeleton formulas with
  | none => none
  | some skeleton =>
      match andMatches : skeleton.andFormula with
      | .multiply left (.decompose decompositionType base digitCount right) =>
          if leftMatches : left = skeleton.leftFormula then
            if rightMatches : right = skeleton.rightFormula then
              some {
                skeleton
                skeletonFound
                decompositionType
                base
                digitCount
                andMatches := by simpa [leftMatches, rightMatches] using andMatches
              }
            else none
          else none
      | _ => none

/-- A six-way lane whose `AND` candidate is ordinary matrix multiplication. -/
structure CheckedPlaintextGateFormula
    (formulas : List FrozenPointwiseMatrixFormula) where
  skeleton : CheckedSixWayMatrixSkeleton
  skeletonFound : checkSixWayMatrixSkeleton formulas = some skeleton
  andMatches : skeleton.andFormula =
    .multiply skeleton.leftFormula skeleton.rightFormula

private def checkPlaintextGateFormula
    (formulas : List FrozenPointwiseMatrixFormula) :
    Option (CheckedPlaintextGateFormula formulas) :=
  match skeletonFound : checkSixWayMatrixSkeleton formulas with
  | none => none
  | some skeleton =>
      if andMatches : skeleton.andFormula =
          .multiply skeleton.leftFormula skeleton.rightFormula then
        some { skeleton, skeletonFound, andMatches }
      else
        none

/-- The exact four-lane BGG gate formula coupling.  In particular, the vector lane must use the
same right public-key decomposition as the public-key lane and the same left plaintext selected
by the plaintext lane.  This is the frozen-DAG counterpart of
`v_L * D(pk_R) + m_L * v_R`; no protocol label or node number is involved. -/
structure CheckedBggGateFormulaCoupling
    (encryptionPublicKeyFormulas encodingVectorFormulas decryptionPublicKeyFormulas
      plaintextFormulas : List FrozenPointwiseMatrixFormula) where
  encryptionPublicKey : CheckedPublicKeyGateFormula encryptionPublicKeyFormulas
  decryptionPublicKey : CheckedPublicKeyGateFormula decryptionPublicKeyFormulas
  plaintext : CheckedPlaintextGateFormula plaintextFormulas
  encodingVector : CheckedSixWayMatrixSkeleton
  encodingVectorFound : checkSixWayMatrixSkeleton encodingVectorFormulas = some encodingVector
  vectorAndMatches : encodingVector.andFormula =
    .add
      (.multiply encodingVector.leftFormula
        (.decompose decryptionPublicKey.decompositionType decryptionPublicKey.base
          decryptionPublicKey.digitCount decryptionPublicKey.skeleton.rightFormula))
      (.multiply encodingVector.rightFormula plaintext.skeleton.leftFormula)

private def checkBggGateFormulaCoupling
    (encryptionPublicKeyFormulas encodingVectorFormulas decryptionPublicKeyFormulas
      plaintextFormulas : List FrozenPointwiseMatrixFormula) :
    Option (CheckedBggGateFormulaCoupling encryptionPublicKeyFormulas encodingVectorFormulas
      decryptionPublicKeyFormulas plaintextFormulas) := do
  let encryptionPublicKey ← checkPublicKeyGateFormula encryptionPublicKeyFormulas
  let decryptionPublicKey ← checkPublicKeyGateFormula decryptionPublicKeyFormulas
  let plaintext ← checkPlaintextGateFormula plaintextFormulas
  match encodingVectorFound : checkSixWayMatrixSkeleton encodingVectorFormulas with
  | none => none
  | some encodingVector =>
      if vectorAndMatches : encodingVector.andFormula =
          .add
            (.multiply encodingVector.leftFormula
              (.decompose decryptionPublicKey.decompositionType decryptionPublicKey.base
                decryptionPublicKey.digitCount decryptionPublicKey.skeleton.rightFormula))
            (.multiply encodingVector.rightFormula plaintext.skeleton.leftFormula) then
        some {
          encryptionPublicKey
          decryptionPublicKey
          plaintext
          encodingVector
          encodingVectorFound
          vectorAndMatches
        }
      else
        none

/-- Analyzer-only scalar syntax for the six Boolean gate candidates in one frozen lane body. -/
inductive FrozenPointwiseScalarFormula where
  | atom (wire : Mxx.Ir.WireRef)
  | integer (value : Int)
  | boolean (value : Bool)
  | boolToInt (input : FrozenPointwiseScalarFormula)
  | intBinary (operation : Mxx.Ir.IntBinaryOp)
      (left right : FrozenPointwiseScalarFormula)
  | compare (operation : Mxx.Ir.IntCompareOp)
      (left right : FrozenPointwiseScalarFormula)
  deriving BEq, DecidableEq

/-- Exact value semantics for the analyzer-only scalar view.  Arithmetic and comparisons delegate
to the executable IR evaluators; this definition does not introduce a second scalar semantics. -/
def FrozenPointwiseScalarFormula.evaluate
    (atoms : Mxx.Ir.WireRef → Option Mxx.Ir.Value) :
    FrozenPointwiseScalarFormula → Option Mxx.Ir.Value
  | .atom wire => atoms wire
  | .integer value => some (.integer value)
  | .boolean value => some (.boolean value)
  | .boolToInt input => do
      let .boolean value ← input.evaluate atoms | none
      return .integer (if value then 1 else 0)
  | .intBinary operation left right => do
      let .integer left ← left.evaluate atoms | none
      let .integer right ← right.evaluate atoms | none
      return .integer (← Mxx.Ir.evaluateIntBinary operation left right)
  | .compare operation left right => do
      let .integer left ← left.evaluate atoms | none
      let .integer right ← right.evaluate atoms | none
      return .boolean (Mxx.Ir.evaluateIntCompare operation left right)

private partial def normalizePointwiseScalarWire
    (scope : Mxx.Ir.Scope)
    (fuel : Nat)
    (wire : Mxx.Ir.WireRef) : Option FrozenPointwiseScalarFormula := do
  guard (wire.port = 0)
  let node ← scope.nodes[wire.node]?
  match fuel with
  | 0 => none
  | fuel + 1 =>
      match node.kind, node.arguments with
      | .constantInt value, [] => some (.integer value)
      | .constantBool value, [] => some (.boolean value)
      | .boolToInt, [input] =>
          return .boolToInt (← normalizePointwiseScalarWire scope fuel input)
      | .intBinary operation, [left, right] =>
          return .intBinary operation
            (← normalizePointwiseScalarWire scope fuel left)
            (← normalizePointwiseScalarWire scope fuel right)
      | .intCompare operation, [left, right] =>
          return .compare operation
            (← normalizePointwiseScalarWire scope fuel left)
            (← normalizePointwiseScalarWire scope fuel right)
      | _, _ => some (.atom wire)

/-- Exact six Boolean candidates emitted by the generic Boolean interpreter. -/
structure CheckedSixWayBooleanSkeleton where
  formulas : List FrozenPointwiseScalarFormula
  leftFormula : FrozenPointwiseScalarFormula
  rightFormula : FrozenPointwiseScalarFormula
  formulasMatch : formulas = [
    .compare .equal (.boolToInt (.boolean false)) (.integer 1),
    .compare .equal (.boolToInt (.boolean true)) (.integer 1),
    leftFormula,
    .compare .equal (.boolToInt leftFormula) (.integer 0),
    .compare .equal
      (.intBinary .multiply (.boolToInt leftFormula) (.boolToInt rightFormula)) (.integer 1),
    .compare .equal
      (.intBinary .add (.boolToInt leftFormula) (.boolToInt rightFormula)) (.integer 1)
  ]

private def checkSixWayBooleanSkeleton
    (scope : Mxx.Ir.Scope)
    (candidates : List Mxx.Ir.WireRef) : Option CheckedSixWayBooleanSkeleton := do
  let formulas ← candidates.mapM (normalizePointwiseScalarWire scope (scope.nodes.length + 1))
  match formulasMatch : formulas with
  | [.compare .equal (.boolToInt (.boolean false)) (.integer 1),
      .compare .equal (.boolToInt (.boolean true)) (.integer 1), leftFormula,
      .compare .equal (.boolToInt notLeft) (.integer 0),
      .compare .equal
        (.intBinary .multiply (.boolToInt andLeft) (.boolToInt andRight)) (.integer 1),
      .compare .equal
        (.intBinary .add (.boolToInt xorLeft) (.boolToInt xorRight)) (.integer 1)] =>
      if leftMatches : notLeft = leftFormula && andLeft = leftFormula &&
          xorLeft = leftFormula then
        if rightMatches : andRight = xorRight then
          some {
            formulas
            leftFormula
            rightFormula := andRight
            formulasMatch := by
              simp only [Bool.and_eq_true, decide_eq_true_eq] at leftMatches
              rcases leftMatches with ⟨⟨notMatches, andMatches⟩, xorMatches⟩
              rw [notMatches, andMatches, xorMatches] at formulasMatch
              rw [← rightMatches] at formulasMatch
              exact formulasMatch
          }
        else none
      else none
  | _ => none

/-- Exact outer parallel-lane binder at one carried body output. -/
structure CheckedRecurrenceLaneOutput
    (interface : FrozenSequentialRecurrenceInterface) where
  outputSlot : Nat
  outputName : String
  output : Mxx.Ir.WireRef
  outputFound : interface.body.outputs[outputSlot]? = some (outputName, output)
  node : Mxx.Ir.Node
  nodeFound : interface.body.nodes[output.node]? = some node
  definition : String
  count : IntExpr
  indexSlot : Nat
  bindings : List (String × IntExpr)
  inputModes : List Mxx.Ir.LoopInputMode
  kindMatches : node.kind = .parallelLoop definition count indexSlot bindings inputModes
  body : Mxx.Ir.Scope
  bodyFound : Mxx.Ir.lookupDefinition definition interface.program.definitions = some body
  nestedFuelPositive : 0 < interface.program.definitions.length - 1
  activeSelection : CheckedActiveSelection body
  gateSelection : CheckedSixWayGateSelection interface.program
  outputSlice : FrozenDependencySlice
  outputSliceFound : interface.outputSlices[outputSlot]? = some outputSlice
  gateCandidateProgramFormulas : List FrozenPointwiseMatrixProgramFormula
  gateCandidateProgramFormulasFound : gateSelection.candidates.mapM (fun candidate =>
    outputSlice.normalizePointwiseMatrixProgramAt? interface.program gateSelection.site candidate) =
      some gateCandidateProgramFormulas
  gateCandidateProgramFormulasValid : gateCandidateProgramFormulas.all
    (FrozenPointwiseMatrixProgramFormula.validIn interface.program) = true
  gateCandidateFormulas : List FrozenPointwiseMatrixFormula
  gateCandidateFormulasMatch : gateCandidateFormulas =
    gateCandidateProgramFormulas.map FrozenPointwiseMatrixProgramFormula.erase
  matrixGateSkeleton : Option CheckedSixWayMatrixSkeleton
  matrixGateSkeletonMatches : matrixGateSkeleton =
    checkSixWayMatrixSkeleton gateCandidateFormulas
  booleanGateSkeleton : Option CheckedSixWayBooleanSkeleton
  booleanGateSkeletonMatches : booleanGateSkeleton =
    checkSixWayBooleanSkeleton gateSelection.scope gateSelection.candidates
  activeSite : CoreNodeRef
  activeSiteMatches : activeSite = {
    stage := interface.transfer.source.loop.site.stage
    scope := ⟨interface.transfer.source.loop.site.scope.path ++
      [interface.definition, definition]⟩
    node := ⟨activeSelection.output.node⟩
  }
  activeControlSlot : Nat
  activeControlProjection : outputSlice.projectInputToOuterScopeAny? interface.program activeSite
    activeSelection.activeSelector = some [activeControlSlot]
  gateControlSlot : Nat
  gateControlProjection : outputSlice.projectInputToOuterScopeAny? interface.program
    gateSelection.site gateSelection.gateSelector = some [gateControlSlot]

/-- The two selectors used by one matched lane resolve to actual direct loop arguments.  This
prevents a structurally similar nested body from being paired through a closure-local selector or
through a different outer input. -/
structure CheckedLaneControlBinding
    {bundle : ClosedProtocolBundle}
    {interface : FrozenSequentialRecurrenceInterface}
    {destinationFor : String → ProtocolInputDestination}
    (controls : List (CheckedDirectLoopControl bundle interface destinationFor)) where
  lane : CheckedRecurrenceLaneOutput interface
  activeControl : CheckedDirectLoopControl bundle interface destinationFor
  activeControlFound : controls.find? (fun control =>
    control.argumentIndex.val = lane.activeControlSlot) = some activeControl
  gateControl : CheckedDirectLoopControl bundle interface destinationFor
  gateControlFound : controls.find? (fun control =>
    control.argumentIndex.val = lane.gateControlSlot) = some gateControl

private def checkLaneControlBinding
    {bundle : ClosedProtocolBundle}
    {interface : FrozenSequentialRecurrenceInterface}
    {destinationFor : String → ProtocolInputDestination}
    (controls : List (CheckedDirectLoopControl bundle interface destinationFor))
    (lane : CheckedRecurrenceLaneOutput interface) :
    Option (CheckedLaneControlBinding controls) :=
  match activeControlFound : controls.find? (fun control =>
      control.argumentIndex.val = lane.activeControlSlot) with
  | none => none
  | some activeControl =>
      match gateControlFound : controls.find? (fun control =>
          control.argumentIndex.val = lane.gateControlSlot) with
      | none => none
      | some gateControl =>
          some { lane, activeControl, activeControlFound, gateControl, gateControlFound }

/-- One decryption carried role resolved from the inferred slot to exactly one matched lane. -/
structure CheckedCarriedRoleLane
    {bundle : ClosedProtocolBundle}
    {interface : FrozenSequentialRecurrenceInterface}
    {destinationFor : String → ProtocolInputDestination}
    {controls : List (CheckedDirectLoopControl bundle interface destinationFor)}
    (lanes : List (CheckedLaneControlBinding controls))
    (slot : Nat) where
  binding : CheckedLaneControlBinding controls
  found : lanes.find? (fun candidate => candidate.lane.outputSlot == slot) = some binding

private def checkCarriedRoleLane
    {bundle : ClosedProtocolBundle}
    {interface : FrozenSequentialRecurrenceInterface}
    {destinationFor : String → ProtocolInputDestination}
    {controls : List (CheckedDirectLoopControl bundle interface destinationFor)}
    (lanes : List (CheckedLaneControlBinding controls))
    (slot : Nat) : Option (CheckedCarriedRoleLane lanes slot) :=
  match found : lanes.find? (fun candidate => candidate.lane.outputSlot == slot) with
  | none => none
  | some binding => some { binding, found }

/-- Bundle-owned workflow program from which a frozen recurrence interface was constructed. -/
structure CheckedWorkflowRecurrenceOrigin
    (bundle : ClosedProtocolBundle)
    (interface : FrozenSequentialRecurrenceInterface) : Type where
  stage : Mxx.Ir.Stage
  stageFound : bundle.workflow.stages.find? (fun candidate =>
    candidate.id = interface.transfer.source.loop.site.stage.name &&
      candidate.program = interface.program) = some stage

theorem CheckedWorkflowRecurrenceOrigin.stageMatches
    {bundle : ClosedProtocolBundle}
    {interface : FrozenSequentialRecurrenceInterface}
    (origin : CheckedWorkflowRecurrenceOrigin bundle interface) :
    origin.stage.id = interface.transfer.source.loop.site.stage.name := by
  have selected := List.find?_some origin.stageFound
  simp only [Bool.and_eq_true, decide_eq_true_eq] at selected
  exact selected.1

theorem CheckedWorkflowRecurrenceOrigin.programMatches
    {bundle : ClosedProtocolBundle}
    {interface : FrozenSequentialRecurrenceInterface}
    (origin : CheckedWorkflowRecurrenceOrigin bundle interface) :
    origin.stage.program = interface.program := by
  have selected := List.find?_some origin.stageFound
  simp only [Bool.and_eq_true, decide_eq_true_eq] at selected
  exact selected.2

theorem CheckedWorkflowRecurrenceOrigin.stageMember
    {bundle : ClosedProtocolBundle}
    {interface : FrozenSequentialRecurrenceInterface}
    (origin : CheckedWorkflowRecurrenceOrigin bundle interface) :
    origin.stage ∈ bundle.workflow.stages :=
  List.mem_of_find?_eq_some origin.stageFound

/-- Bundle-owned requirement program from which the Boolean recurrence was constructed. -/
structure CheckedRequirementRecurrenceOrigin
    (bundle : ClosedProtocolBundle)
    (interface : FrozenSequentialRecurrenceInterface)
    (acceptance : CheckedRequirementAcceptance) : Type where
  programAt : bundle.requirements[acceptance.requirementIndex]? = some interface.program
  stageMatches : interface.transfer.source.loop.site.stage.name =
    s!"$requirement:{acceptance.requirementIndex}"

inductive BggThreeTraceInterfaceError where
  | controlCountMismatch (encryption decryption booleanInterpreter : Nat)
  | controlOriginMismatch
  | missingLaneOutput (role : Nat) (slot : Nat)
  | laneCountMismatch
  | laneIndexMismatch
  | invalidLaneBody (role slot : Nat)
  | missingLaneControl (role slot : Nat)
  | missingCarriedRoleLane (role : Nat)
  | invalidMatrixGateSkeleton (role slot : Nat)
  | invalidBooleanGateSkeleton
  | invalidBggGateFormulaCoupling
  | activeControlOriginMismatch
  | gateControlOriginMismatch
  | missingEncryptionProgramOrigin
  | missingDecryptionProgramOrigin
  | missingBooleanProgramOrigin
  deriving Repr

private def checkWorkflowRecurrenceOrigin
    (bundle : ClosedProtocolBundle)
    (interface : FrozenSequentialRecurrenceInterface) :
    Option (CheckedWorkflowRecurrenceOrigin bundle interface) :=
  match stageFound : bundle.workflow.stages.find? (fun candidate =>
      candidate.id = interface.transfer.source.loop.site.stage.name &&
        candidate.program = interface.program) with
  | none => none
  | some stage => some { stage, stageFound }

private def checkRequirementRecurrenceOrigin
    (bundle : ClosedProtocolBundle)
    (interface : FrozenSequentialRecurrenceInterface)
    (acceptance : CheckedRequirementAcceptance) :
    Option (CheckedRequirementRecurrenceOrigin bundle interface acceptance) :=
  match programFound : bundle.requirements[acceptance.requirementIndex]? with
  | none => none
  | some program =>
      if programMatches : program = interface.program then
        if stageMatches : interface.transfer.source.loop.site.stage.name =
            s!"$requirement:{acceptance.requirementIndex}" then
          some {
            programAt := by simpa [programMatches] using programFound
            stageMatches
          }
        else none
      else none

private def checkRecurrenceLaneOutput
    (interface : FrozenSequentialRecurrenceInterface)
    (role outputSlot : Nat) :
    Except BggThreeTraceInterfaceError (CheckedRecurrenceLaneOutput interface) :=
  match outputFound : interface.body.outputs[outputSlot]? with
  | none => .error (.missingLaneOutput role outputSlot)
  | some (outputName, output) =>
      match nodeFound : interface.body.nodes[output.node]? with
      | none => .error (.missingLaneOutput role outputSlot)
      | some node =>
          match kindMatches : node.kind with
          | .parallelLoop definition count indexSlot bindings inputModes =>
              match bodyFound : Mxx.Ir.lookupDefinition definition
                  interface.program.definitions with
              | none => .error (.missingLaneOutput role outputSlot)
              | some body =>
                  if nestedFuelPositive : 0 < interface.program.definitions.length - 1 then
                    match checkActiveSelection body with
                    | none => .error (.invalidLaneBody role outputSlot)
                    | some activeSelection =>
                        match outputSliceFound : interface.outputSlices[outputSlot]? with
                        | none => .error (.invalidLaneBody role outputSlot)
                        | some outputSlice =>
                            match findSixWayGateSelections interface.program outputSlice with
                            | [gateSelection] =>
                                match gateCandidateProgramFormulasFound :
                                    gateSelection.candidates.mapM (fun candidate =>
                                      outputSlice.normalizePointwiseMatrixProgramAt?
                                        interface.program gateSelection.site candidate) with
                                | none => .error (.invalidLaneBody role outputSlot)
                                | some gateCandidateProgramFormulas =>
                                  if gateCandidateProgramFormulasValid :
                                      gateCandidateProgramFormulas.all
                                        (FrozenPointwiseMatrixProgramFormula.validIn
                                          interface.program) = true then
                                    let gateCandidateFormulas :=
                                      gateCandidateProgramFormulas.map
                                        FrozenPointwiseMatrixProgramFormula.erase
                                    let matrixGateSkeleton :=
                                      checkSixWayMatrixSkeleton gateCandidateFormulas
                                    let booleanGateSkeleton :=
                                      checkSixWayBooleanSkeleton gateSelection.scope
                                        gateSelection.candidates
                                    let activeSite : CoreNodeRef := {
                                      stage := interface.transfer.source.loop.site.stage
                                      scope := ⟨interface.transfer.source.loop.site.scope.path ++
                                        [interface.definition, definition]⟩
                                      node := ⟨activeSelection.output.node⟩
                                    }
                                    match activeControlProjection :
                                        outputSlice.projectInputToOuterScopeAny? interface.program
                                          activeSite activeSelection.activeSelector with
                                    | some [activeControlSlot] =>
                                        match gateControlProjection :
                                            outputSlice.projectInputToOuterScopeAny?
                                              interface.program gateSelection.site
                                              gateSelection.gateSelector with
                                        | some [gateControlSlot] =>
                                            .ok {
                                              outputSlot := outputSlot
                                              outputName := outputName
                                              output := output
                                              outputFound := outputFound
                                              node := node
                                              nodeFound := nodeFound
                                              definition := definition
                                              count := count
                                              indexSlot := indexSlot
                                              bindings := bindings
                                              inputModes := inputModes
                                              kindMatches := kindMatches
                                              body := body
                                              bodyFound := bodyFound
                                              nestedFuelPositive := nestedFuelPositive
                                              activeSelection := activeSelection
                                              gateSelection := gateSelection
                                              outputSlice := outputSlice
                                              outputSliceFound := outputSliceFound
                                              gateCandidateProgramFormulas :=
                                                gateCandidateProgramFormulas
                                              gateCandidateProgramFormulasFound :=
                                                gateCandidateProgramFormulasFound
                                              gateCandidateProgramFormulasValid :=
                                                gateCandidateProgramFormulasValid
                                              gateCandidateFormulas := gateCandidateFormulas
                                              gateCandidateFormulasMatch := rfl
                                              matrixGateSkeleton := matrixGateSkeleton
                                              matrixGateSkeletonMatches := rfl
                                              booleanGateSkeleton := booleanGateSkeleton
                                              booleanGateSkeletonMatches := rfl
                                              activeSite := activeSite
                                              activeSiteMatches := rfl
                                              activeControlSlot := activeControlSlot
                                              activeControlProjection := activeControlProjection
                                              gateControlSlot := gateControlSlot
                                              gateControlProjection := gateControlProjection
                                            }
                                        | _ => .error (.invalidLaneBody role outputSlot)
                                    | _ => .error (.invalidLaneBody role outputSlot)
                                  else .error (.invalidLaneBody role outputSlot)
                            | _ => .error (.invalidLaneBody role outputSlot)
                  else .error (.missingLaneOutput role outputSlot)
          | _ => .error (.missingLaneOutput role outputSlot)

/-- First complete static three-trace foundation.  It proves only frozen interface
correspondence; no BGG semantic relation or endpoint rewrite follows from this structure alone. -/
structure CheckedBggThreeTraceInterface (bundle : ClosedProtocolBundle) where
  candidate : CheckedBggRecurrenceCandidate
  encryptionDestination : String → ProtocolInputDestination
  decryptionDestination : String → ProtocolInputDestination
  booleanDestination : String → ProtocolInputDestination
  encryptionControls : List (CheckedDirectLoopControl bundle
    candidate.prefilter.encryption encryptionDestination)
  decryptionControls : List (CheckedDirectLoopControl bundle
    candidate.prefilter.decryption decryptionDestination)
  booleanControls : List (CheckedDirectLoopControl bundle
    candidate.prefilter.booleanInterpreter booleanDestination)
  controlOrigins : List ProtocolInputId
  fourControls : controlOrigins.length = 4
  fourDecryptionControls : decryptionControls.length = 4
  fourBooleanControls : booleanControls.length = 4
  encryptionOrigins : encryptionControls.map (·.protocolInput) = controlOrigins
  decryptionOrigins : decryptionControls.map (·.protocolInput) = controlOrigins
  booleanOrigins : booleanControls.map (·.protocolInput) = controlOrigins
  encryptionProgramOrigin : CheckedWorkflowRecurrenceOrigin bundle
    candidate.prefilter.encryption
  decryptionProgramOrigin : CheckedWorkflowRecurrenceOrigin bundle
    candidate.prefilter.decryption
  booleanProgramOrigin : CheckedRequirementRecurrenceOrigin bundle
    candidate.prefilter.booleanInterpreter candidate.prefilter.requirementAcceptance
  encryptionLaneControl : CheckedLaneControlBinding encryptionControls
  encryptionMatrixGateSkeletonPresent :
    encryptionLaneControl.lane.matrixGateSkeleton.isSome = true
  decryptionLaneControls : List (CheckedLaneControlBinding decryptionControls)
  decryptionMatrixGateSkeletonsPresent : decryptionLaneControls.all (fun binding =>
    binding.lane.matrixGateSkeleton.isSome) = true
  encodingVectorLane : CheckedCarriedRoleLane decryptionLaneControls candidate.encodingVectorSlot
  decryptionPublicKeyLane : CheckedCarriedRoleLane decryptionLaneControls candidate.publicKeySlot
  plaintextLane : CheckedCarriedRoleLane decryptionLaneControls candidate.plaintextSlot
  gateFormulaCoupling : CheckedBggGateFormulaCoupling
    encryptionLaneControl.lane.gateCandidateFormulas
    encodingVectorLane.binding.lane.gateCandidateFormulas
    decryptionPublicKeyLane.binding.lane.gateCandidateFormulas
    plaintextLane.binding.lane.gateCandidateFormulas
  booleanLaneControl : CheckedLaneControlBinding booleanControls
  booleanGateSkeleton : CheckedSixWayBooleanSkeleton
  booleanGateSkeletonFound : booleanLaneControl.lane.booleanGateSkeleton =
    some booleanGateSkeleton
  activeControlOrigins : List ProtocolInputId
  activeControlOriginsMatch : activeControlOrigins =
    encryptionLaneControl.activeControl.protocolInput ::
      decryptionLaneControls.map (fun binding => binding.activeControl.protocolInput) ++
      [booleanLaneControl.activeControl.protocolInput]
  allActiveControlOrigins : activeControlOrigins.all
    (fun origin => origin = encryptionLaneControl.activeControl.protocolInput) = true
  gateControlOrigins : List ProtocolInputId
  gateControlOriginsMatch : gateControlOrigins =
    encryptionLaneControl.gateControl.protocolInput ::
      decryptionLaneControls.map (fun binding => binding.gateControl.protocolInput) ++
      [booleanLaneControl.gateControl.protocolInput]
  allGateControlOrigins : gateControlOrigins.all
    (fun origin => origin = encryptionLaneControl.gateControl.protocolInput) = true
  laneBinders : List (IntExpr × Nat)
  laneBindersMatch : laneBinders =
    (encryptionLaneControl.lane.count, encryptionLaneControl.lane.indexSlot) ::
      decryptionLaneControls.map (fun binding =>
        (binding.lane.count, binding.lane.indexSlot)) ++
      [(booleanLaneControl.lane.count, booleanLaneControl.lane.indexSlot)]
  allLaneCounts :
    laneBinders.all (fun lane => lane.1 = encryptionLaneControl.lane.count) = true
  allLaneIndices :
    laneBinders.all (fun lane => lane.2 = encryptionLaneControl.lane.indexSlot) = true

/-- Check shared protocol controls and paired outer lane binders without using labels or node
numbers.  Gate-branch shape and the semantic one-step relation are checked by the subsequent
closed matcher. -/
def checkBggThreeTraceInterface
    (bundle : ClosedProtocolBundle)
    (candidate : CheckedBggRecurrenceCandidate) :
    Except BggThreeTraceInterfaceError (CheckedBggThreeTraceInterface bundle) := do
  let encryptionDestination := fun inputName =>
    ProtocolInputDestination.workflowStage
      candidate.prefilter.encryption.transfer.source.loop.site.stage inputName
  let decryptionDestination := fun inputName =>
    ProtocolInputDestination.workflowStage
      candidate.prefilter.decryption.transfer.source.loop.site.stage inputName
  let booleanDestination := fun inputName =>
    ProtocolInputDestination.requirement
      candidate.prefilter.requirementAcceptance.requirementIndex inputName
  let encryptionControls := checkedDirectLoopControls bundle
    candidate.prefilter.encryption encryptionDestination
  let decryptionControls := checkedDirectLoopControls bundle
    candidate.prefilter.decryption decryptionDestination
  let booleanControls := checkedDirectLoopControls bundle
    candidate.prefilter.booleanInterpreter booleanDestination
  let encryptionOrigins := encryptionControls.map (·.protocolInput)
  let decryptionOrigins := decryptionControls.map (·.protocolInput)
  let booleanOrigins := booleanControls.map (·.protocolInput)
  let encryptionProgramOrigin ← checkWorkflowRecurrenceOrigin bundle
    candidate.prefilter.encryption |>.elim (throw .missingEncryptionProgramOrigin) pure
  let decryptionProgramOrigin ← checkWorkflowRecurrenceOrigin bundle
    candidate.prefilter.decryption |>.elim (throw .missingDecryptionProgramOrigin) pure
  let booleanProgramOrigin ← checkRequirementRecurrenceOrigin bundle
    candidate.prefilter.booleanInterpreter candidate.prefilter.requirementAcceptance
    |>.elim (throw .missingBooleanProgramOrigin) pure
  let encryptionLane ← checkRecurrenceLaneOutput candidate.prefilter.encryption 0 0
  let mut decryptionLanes := []
  for slot in [0:candidate.prefilter.decryption.outputSlices.length] do
    decryptionLanes := decryptionLanes ++ [
      ← checkRecurrenceLaneOutput candidate.prefilter.decryption 1 slot]
  let booleanLane ← checkRecurrenceLaneOutput candidate.prefilter.booleanInterpreter 2 0
  let encryptionLaneControl ← checkLaneControlBinding encryptionControls encryptionLane
    |>.elim (throw (.missingLaneControl 0 0)) pure
  let mut decryptionLaneControls := []
  for lane in decryptionLanes, slot in [0:decryptionLanes.length] do
    decryptionLaneControls := decryptionLaneControls ++ [
      ← checkLaneControlBinding decryptionControls lane
        |>.elim (throw (.missingLaneControl 1 slot)) pure]
  let booleanLaneControl ← checkLaneControlBinding booleanControls booleanLane
    |>.elim (throw (.missingLaneControl 2 0)) pure
  let encryptionMatrixGateSkeletonPresent ←
    (if present : encryptionLaneControl.lane.matrixGateSkeleton.isSome = true then
      .ok (PLift.up present)
    else
      .error (.invalidMatrixGateSkeleton 0 0) :
        Except BggThreeTraceInterfaceError
          (PLift (encryptionLaneControl.lane.matrixGateSkeleton.isSome = true)))
  let decryptionMatrixGateSkeletonsPresent ←
    (if present : decryptionLaneControls.all (fun binding =>
        binding.lane.matrixGateSkeleton.isSome) = true then
      .ok (PLift.up present)
    else
      .error (.invalidMatrixGateSkeleton 1 0) :
        Except BggThreeTraceInterfaceError
          (PLift (decryptionLaneControls.all (fun binding =>
            binding.lane.matrixGateSkeleton.isSome) = true)))
  let encodingVectorLane ← checkCarriedRoleLane decryptionLaneControls
    candidate.encodingVectorSlot |>.elim (throw (.missingCarriedRoleLane 0)) pure
  let decryptionPublicKeyLane ← checkCarriedRoleLane decryptionLaneControls
    candidate.publicKeySlot |>.elim (throw (.missingCarriedRoleLane 1)) pure
  let plaintextLane ← checkCarriedRoleLane decryptionLaneControls candidate.plaintextSlot
    |>.elim (throw (.missingCarriedRoleLane 2)) pure
  let gateFormulaCoupling ← checkBggGateFormulaCoupling
    encryptionLaneControl.lane.gateCandidateFormulas
    encodingVectorLane.binding.lane.gateCandidateFormulas
    decryptionPublicKeyLane.binding.lane.gateCandidateFormulas
    plaintextLane.binding.lane.gateCandidateFormulas
    |>.elim (throw .invalidBggGateFormulaCoupling) pure
  let ⟨booleanGateSkeleton, booleanGateSkeletonFound⟩ ←
    match found : booleanLaneControl.lane.booleanGateSkeleton with
    | none => throw .invalidBooleanGateSkeleton
    | some skeleton =>
        pure (Sigma.mk skeleton (PLift.up found))
  let activeControlOrigins := encryptionLaneControl.activeControl.protocolInput ::
    decryptionLaneControls.map (fun binding => binding.activeControl.protocolInput) ++
    [booleanLaneControl.activeControl.protocolInput]
  let gateControlOrigins := encryptionLaneControl.gateControl.protocolInput ::
    decryptionLaneControls.map (fun binding => binding.gateControl.protocolInput) ++
    [booleanLaneControl.gateControl.protocolInput]
  let laneBinders :=
    (encryptionLaneControl.lane.count, encryptionLaneControl.lane.indexSlot) ::
      decryptionLaneControls.map (fun binding =>
        (binding.lane.count, binding.lane.indexSlot)) ++
      [(booleanLaneControl.lane.count, booleanLaneControl.lane.indexSlot)]
  if fourEncryption : encryptionOrigins.length = 4 then
    if fourDecryption : decryptionOrigins.length = 4 then
      if fourBoolean : booleanOrigins.length = 4 then
        if decryptionOriginsMatch : decryptionOrigins = encryptionOrigins then
          if booleanOriginsMatch : booleanOrigins = encryptionOrigins then
            if allLaneCounts : laneBinders.all
                (fun lane => lane.1 = encryptionLaneControl.lane.count) = true then
              if allLaneIndices : laneBinders.all
                  (fun lane => lane.2 = encryptionLaneControl.lane.indexSlot) = true then
                if allActiveControlOrigins : activeControlOrigins.all (fun origin =>
                    origin = encryptionLaneControl.activeControl.protocolInput) = true then
                  if allGateControlOrigins : gateControlOrigins.all (fun origin =>
                      origin = encryptionLaneControl.gateControl.protocolInput) = true then
                    return {
                      candidate := candidate
                      encryptionDestination := encryptionDestination
                      decryptionDestination := decryptionDestination
                      booleanDestination := booleanDestination
                      encryptionControls := encryptionControls
                      decryptionControls := decryptionControls
                      booleanControls := booleanControls
                      controlOrigins := encryptionOrigins
                      fourControls := fourEncryption
                      fourDecryptionControls := by
                        simpa [decryptionOrigins] using fourDecryption
                      fourBooleanControls := by simpa [booleanOrigins] using fourBoolean
                      encryptionOrigins := rfl
                      decryptionOrigins := decryptionOriginsMatch
                      booleanOrigins := booleanOriginsMatch
                      encryptionProgramOrigin := encryptionProgramOrigin
                      decryptionProgramOrigin := decryptionProgramOrigin
                      booleanProgramOrigin := booleanProgramOrigin
                      encryptionLaneControl := encryptionLaneControl
                      encryptionMatrixGateSkeletonPresent :=
                        encryptionMatrixGateSkeletonPresent.down
                      decryptionLaneControls := decryptionLaneControls
                      decryptionMatrixGateSkeletonsPresent :=
                        decryptionMatrixGateSkeletonsPresent.down
                      encodingVectorLane
                      decryptionPublicKeyLane
                      plaintextLane
                      gateFormulaCoupling
                      booleanLaneControl := booleanLaneControl
                      booleanGateSkeleton
                      booleanGateSkeletonFound := booleanGateSkeletonFound.down
                      activeControlOrigins := activeControlOrigins
                      activeControlOriginsMatch := rfl
                      allActiveControlOrigins := allActiveControlOrigins
                      gateControlOrigins := gateControlOrigins
                      gateControlOriginsMatch := rfl
                      allGateControlOrigins := allGateControlOrigins
                      laneBinders := laneBinders
                      laneBindersMatch := rfl
                      allLaneCounts := allLaneCounts
                      allLaneIndices := allLaneIndices
                    }
                  else throw .gateControlOriginMismatch
                else throw .activeControlOriginMismatch
              else throw .laneIndexMismatch
            else throw .laneCountMismatch
          else throw .controlOriginMismatch
        else throw .controlOriginMismatch
      else throw (.controlCountMismatch encryptionOrigins.length decryptionOrigins.length
        booleanOrigins.length)
    else throw (.controlCountMismatch encryptionOrigins.length decryptionOrigins.length
      booleanOrigins.length)
  else throw (.controlCountMismatch encryptionOrigins.length decryptionOrigins.length
    booleanOrigins.length)

/-- Infer decryption carried roles without accepting raw slot numbers. The public-key slot is the
unique outer body input on which the actual decomposition input depends; the plaintext slot is
the unique 1×1 matrix family; the remaining carried slot is the encoding vector. -/
def inferBggCarriedRoles
    (prefilter : CheckedBggRecurrencePrefilter) :
    Except BggCarriedRoleInferenceError CheckedBggRecurrenceCandidate := do
  let decompositionInputs := decompositionOuterInputSlots prefilter.decryption
    prefilter.decryptionDecomposition
  let publicKeySlot ← match decompositionInputs with
    | [slot] => pure slot
    | _ => throw (.decompositionInputNotUnique decompositionInputs)
  unless publicKeySlot < 3 do throw .invalidPublicKeySlot
  let schemas := prefilter.decryption.transfer.carriedSchemas
  let plaintextSlots := (List.range schemas.length).filter fun slot =>
    (schemas[slot]?).any isOneByOneMatrixFamily
  let plaintextSlot ← match plaintextSlots with
    | [slot] => pure slot
    | _ => throw .plaintextSlotNotUnique
  let vectorSlots := (List.range schemas.length).filter fun slot =>
    slot != publicKeySlot && slot != plaintextSlot &&
      (schemas[slot]? >>= matrixFamilyType?).isSome
  let encodingVectorSlot ← match vectorSlots with
    | [slot] => pure slot
    | _ => throw .encodingVectorSlotNotUnique
  let ⟨publicKeyCount, publicKeyType⟩ ←
    (schemas[publicKeySlot]? >>= matrixFamilyType?).elim
      (throw .invalidPublicKeySlot) pure
  let ⟨vectorCount, _⟩ ←
    (schemas[encodingVectorSlot]? >>= matrixFamilyType?).elim
      (throw .encodingVectorSlotNotUnique) pure
  let ⟨plaintextCount, _⟩ ←
    (schemas[plaintextSlot]? >>= matrixFamilyType?).elim
      (throw .plaintextSlotNotUnique) pure
  let ⟨encryptionCount, encryptionType⟩ ← match
      prefilter.encryption.transfer.carriedSchemas with
    | [schema] => (matrixFamilyType? schema).elim (throw .publicKeyTypeMismatch) pure
    | _ => throw .publicKeyTypeMismatch
  unless publicKeyCount = vectorCount && publicKeyCount = plaintextCount &&
      publicKeyCount = encryptionCount do
    throw .familyCountMismatch
  unless publicKeyType = encryptionType do throw .publicKeyTypeMismatch
  let slots ← checkBggEncodingSlots prefilter.encryption.transfer.source
    prefilter.decryption.transfer.source ⟨0, 0⟩ ⟨encodingVectorSlot, 0⟩
    ⟨publicKeySlot, 0⟩ ⟨plaintextSlot, 0⟩ prefilter.matchingDecomposition
    |>.mapError .invalidSlots
  return { prefilter, publicKeySlot, encodingVectorSlot, plaintextSlot, slots }

/-- Unique frozen-table resolution for a recurrence occurrence. -/
structure CheckedRecurrenceResolution
    (analysis : AnalysisResult)
    (reference : SequentialRecurrenceInstanceRef) where
  transfer : SymbolicRecurrenceTransfer
  unique : analysis.symbolicRecurrences.filter (fun entry => entry.identity = reference) =
    [transfer]

def resolveUniqueRecurrence
    (analysis : AnalysisResult)
    (reference : SequentialRecurrenceInstanceRef) :
    Option (CheckedRecurrenceResolution analysis reference) :=
  match resolved : analysis.symbolicRecurrences.filter
      (fun entry => entry.identity = reference) with
  | [transfer] => some { transfer, unique := resolved }
  | _ => none

/-- Closed structural coupling data.  The payload invariant makes an absent BGG role table legal
only for quotient equality.  Initial and step semantic evidence will be added by the symbolic
transfer theorem; this structure alone deliberately has no soundness eliminator. -/
structure CheckedRecurrenceCoupling (analysis : AnalysisResult) where
  kind : RecurrenceRelationKind
  left : SequentialRecurrenceInstanceRef
  right : SequentialRecurrenceInstanceRef
  leftResolution : CheckedRecurrenceResolution analysis left
  rightResolution : CheckedRecurrenceResolution analysis right
  bggSlots : Option CheckedBggEncodingSlots
  payloadMatches : match kind with
    | .quotientEqual => bggSlots = none
    | .bggEncodingOf => bggSlots.isSome = true
  countIdentity : CheckedIntExprOriginEquality leftResolution.transfer.source.count
    rightResolution.transfer.source.count
  sharedControls : List CheckedValueOriginEquality

inductive RecurrenceCouplingMatchError where
  | missingOrAmbiguousLeft
  | missingOrAmbiguousRight
  | countOriginMismatch
  | invalidBggSlots (error : BggEncodingSlotMatchError)
  deriving BEq, DecidableEq, Repr

/-- Closed quotient-equality foundation.  Candidate discovery supplies only frozen recurrence
identities and already matched shared-control origins. -/
def deriveQuotientEqualCoupling
    (analysis : AnalysisResult)
    (left right : SequentialRecurrenceInstanceRef)
    (sharedControls : List CheckedValueOriginEquality) :
    Except RecurrenceCouplingMatchError (CheckedRecurrenceCoupling analysis) := do
  let leftResolution ← resolveUniqueRecurrence analysis left
    |>.elim (throw .missingOrAmbiguousLeft) pure
  let rightResolution ← resolveUniqueRecurrence analysis right
    |>.elim (throw .missingOrAmbiguousRight) pure
  let countIdentity ← checkIntExprOriginEquality leftResolution.transfer.source.count
    rightResolution.transfer.source.count |>.elim (throw .countOriginMismatch) pure
  return {
    kind := .quotientEqual
    left
    right
    leftResolution
    rightResolution
    bggSlots := none
    payloadMatches := rfl
    countIdentity
    sharedControls
  }

/-- Assemble the BGG coupling foundation after the frozen interface matcher has produced the
checked role table.  This function does not accept raw slot numbers, expressions, or protocol
labels. -/
private def deriveBggEncodingCouplingFromCheckedSlots
    (analysis : AnalysisResult)
    (left right : SequentialRecurrenceInstanceRef)
    (slots : CheckedBggEncodingSlots)
    (sharedControls : List CheckedValueOriginEquality) :
    Except RecurrenceCouplingMatchError (CheckedRecurrenceCoupling analysis) := do
  let leftResolution ← resolveUniqueRecurrence analysis left
    |>.elim (throw .missingOrAmbiguousLeft) pure
  let rightResolution ← resolveUniqueRecurrence analysis right
    |>.elim (throw .missingOrAmbiguousRight) pure
  let countIdentity ← checkIntExprOriginEquality leftResolution.transfer.source.count
    rightResolution.transfer.source.count |>.elim (throw .countOriginMismatch) pure
  return {
    kind := .bggEncodingOf
    left
    right
    leftResolution
    rightResolution
    bggSlots := some slots
    payloadMatches := rfl
    countIdentity
    sharedControls
  }

private def fixtureMatrixType (rows columns : Int) : MatrixTypeExpr := {
  modulus := .parameter "q"
  ringDimension := .parameter "n"
  rows := .constant rows
  columns := .constant columns
}

private def fixtureMatrixSchema (matrixType : MatrixTypeExpr) : ValueFactSchema :=
  .matrix matrixType .exact [] .unknown

private def fixtureMatrixTemplate
    (name : String)
    (matrixType : MatrixTypeExpr) : ValueFactTemplate := {
  fact := .matrix {
    subject := .protocolInput ⟨name⟩
    primary := .exact (.zero matrixType)
    relations := []
    totalNormBound := .constant 0
  }
  schema := fixtureMatrixSchema matrixType
}

private def fixtureFamilyFact (matrixType : MatrixTypeExpr) : ValueFact :=
  .family {
    aggregate := .carriedInput 0
    count := .parameter "width"
    elementSchema := fixtureMatrixSchema matrixType
  }

private def fixtureFamilyFactAt (slot : Nat) (matrixType : MatrixTypeExpr) : ValueFact :=
  .family {
    aggregate := .carriedInput slot
    count := .parameter "width"
    elementSchema := fixtureMatrixSchema matrixType
  }

private def fixtureFamilyFactWithSchemaAt
    (slot : Nat) (schema : ValueFactSchema) : ValueFact :=
  .family {
    aggregate := .carriedInput slot
    count := .parameter "width"
    elementSchema := schema
  }

private def fixtureRecurrence (matrixType : MatrixTypeExpr) : SequentialRecurrenceSource where
  loop := { site := { stage := ⟨"fixture"⟩, scope := ⟨[]⟩, node := ⟨0⟩ } }
  count := .parameter "depth"
  carriedArity := 1
  initial := ⟨#[{
    fact := fixtureFamilyFact matrixType
    schema := .family (.parameter "width") (fixtureMatrixSchema matrixType)
  }], rfl⟩
  bodyInputs := ⟨#[{
    definition := { stage := ⟨"fixture"⟩, name := "body" }
    bodyScope := ⟨[]⟩
    node := ⟨0⟩
    port := 0
  }], rfl⟩
  bodyOutputs := ⟨#[{
    fact := fixtureFamilyFact matrixType
    schema := .family (.parameter "width") (fixtureMatrixSchema matrixType)
  }], rfl⟩
  familyElementTemplates := [
    (.carriedInput 0, fixtureMatrixTemplate "fixture-family-element" matrixType)
  ]
  invariantInputs := []
  iterationVariable := ⟨0⟩

private def fixturePublicKeyType := fixtureMatrixType 1 4
private def fixtureOtherPublicKeyType := fixtureMatrixType 1 5
private def fixtureEncodingVectorType := fixtureMatrixType 1 4
private def fixturePlaintextType := fixtureMatrixType 1 1
private def fixtureDecompositionType := fixtureMatrixType 4 4

private def fixtureDecompositionProgram : Mxx.Ir.Prog := {
  root := {
    nodes := #[{
      kind := .gadgetDecompose fixtureDecompositionType (.constant 2) (.constant 4)
      arguments := [{ node := 0, port := 0 }]
      outputCount := 1
      outputTypes := [.preimage fixtureDecompositionType]
    }]
    outputs := [("output", { node := 0, port := 0 })]
    inputNames := []
  }
  definitions := []
}

private def fixtureDecompositionNode :
    CheckedGadgetDecompositionNode fixtureDecompositionProgram :=
  ((checkGadgetDecompositionAt fixtureDecompositionProgram {
    stage := ⟨"fixture"⟩, scope := ⟨[]⟩, node := ⟨0⟩
  }).get (by native_decide))

private def fixtureCheckedGadget : CheckedCanonicalGadgetDecomposition :=
  ((matchCanonicalGadgetDecompositions fixtureDecompositionNode fixtureDecompositionNode).toOption
    |>.get (by native_decide))

private def fixtureBggDecryptionRecurrenceWithPlaintext
    (plaintextFact : ValueFact)
    (plaintextSchema : ValueFactSchema) : SequentialRecurrenceSource where
  loop := { site := { stage := ⟨"fixture"⟩, scope := ⟨[]⟩, node := ⟨1⟩ } }
  count := .parameter "depth"
  carriedArity := 3
  initial := ⟨#[
    { fact := fixtureFamilyFactAt 0 fixtureEncodingVectorType,
      schema := .family (.parameter "width") (fixtureMatrixSchema fixtureEncodingVectorType) },
    { fact := fixtureFamilyFactAt 1 fixturePublicKeyType,
      schema := .family (.parameter "width") (fixtureMatrixSchema fixturePublicKeyType) },
    { fact := plaintextFact,
      schema := .family (.parameter "width") plaintextSchema }
  ], rfl⟩
  bodyInputs := ⟨#[
    { definition := { stage := ⟨"fixture"⟩, name := "body" }, bodyScope := ⟨[]⟩,
      node := ⟨0⟩, port := 0 },
    { definition := { stage := ⟨"fixture"⟩, name := "body" }, bodyScope := ⟨[]⟩,
      node := ⟨1⟩, port := 0 },
    { definition := { stage := ⟨"fixture"⟩, name := "body" }, bodyScope := ⟨[]⟩,
      node := ⟨2⟩, port := 0 }
  ], rfl⟩
  bodyOutputs := ⟨#[
    { fact := fixtureFamilyFactAt 0 fixtureEncodingVectorType,
      schema := .family (.parameter "width") (fixtureMatrixSchema fixtureEncodingVectorType) },
    { fact := fixtureFamilyFactAt 1 fixturePublicKeyType,
      schema := .family (.parameter "width") (fixtureMatrixSchema fixturePublicKeyType) },
    { fact := plaintextFact,
      schema := .family (.parameter "width") plaintextSchema }
  ], rfl⟩
  familyElementTemplates := [
    (.carriedInput 0,
      fixtureMatrixTemplate "fixture-encoding-element" fixtureEncodingVectorType),
    (.carriedInput 1,
      fixtureMatrixTemplate "fixture-public-key-element" fixturePublicKeyType),
    (.carriedInput 2,
      fixtureMatrixTemplate "fixture-plaintext-element" fixturePlaintextType)
  ]
  invariantInputs := []
  iterationVariable := ⟨0⟩

private def fixtureBggDecryptionRecurrence : SequentialRecurrenceSource :=
  fixtureBggDecryptionRecurrenceWithPlaintext
    (fixtureFamilyFactAt 2 fixturePlaintextType)
    (fixtureMatrixSchema fixturePlaintextType)

private def fixtureNonMatrixPlaintextRecurrence : SequentialRecurrenceSource :=
  fixtureBggDecryptionRecurrenceWithPlaintext
    (fixtureFamilyFactWithSchemaAt 2 .boolean) .boolean

private def fixtureNonScalarPlaintextRecurrence : SequentialRecurrenceSource :=
  fixtureBggDecryptionRecurrenceWithPlaintext
    (fixtureFamilyFactAt 2 fixtureOtherPublicKeyType)
    (fixtureMatrixSchema fixtureOtherPublicKeyType)

private def fixtureLeftRecurrenceRef : SequentialRecurrenceInstanceRef := {
  recurrence := ⟨(fixtureRecurrence fixturePublicKeyType).loop.site⟩
  path := []
}

private def fixtureRightRecurrenceRef : SequentialRecurrenceInstanceRef := {
  recurrence := ⟨fixtureBggDecryptionRecurrence.loop.site⟩
  path := []
}

private def fixtureLeftTransfer : SymbolicRecurrenceTransfer :=
  ((((fixtureRecurrence fixturePublicKeyType).constructSymbolicTransfer
    fixtureLeftRecurrenceRef []).toOption).get (by native_decide)).transfer

private def fixtureRightTransfer : SymbolicRecurrenceTransfer :=
  (((fixtureBggDecryptionRecurrence.constructSymbolicTransfer
    fixtureRightRecurrenceRef []).toOption).get (by native_decide)).transfer

private def fixtureCouplingAnalysis : AnalysisResult where
  facts := []
  families := []
  symbolicRecurrences := [fixtureLeftTransfer, fixtureRightTransfer]
  staticObligations := []
  inputObligations := []
  semanticObligations := []
  endpointFacts := []
  usedRules := []

private def fixtureCheckedBggSlots : CheckedBggEncodingSlots := {
  encryptionPublicKeys := ⟨0, 0⟩
  encodingVectors := ⟨0, 0⟩
  encodingPublicKeys := ⟨1, 0⟩
  plaintextMatrices := ⟨2, 0⟩
  gadgetDecomposition := fixtureCheckedGadget
  encryptionPublicKeyType := fixturePublicKeyType
  encodingVectorType := fixtureEncodingVectorType
  plaintextMatrixType := fixturePlaintextType
}

example : (fixtureRecurrence fixturePublicKeyType).resolveFamilyMatrixType ⟨0, 0⟩ =
    some fixturePublicKeyType := rfl

example : (fixtureRecurrence fixturePublicKeyType).resolveFamilyMatrixType ⟨1, 0⟩ = none := rfl

example : (matchCanonicalGadgetDecompositions fixtureDecompositionNode
    fixtureDecompositionNode).isOk = true := rfl

example : checkIntExprOriginEquality (.parameter "depth") (.parameter "other") = none := rfl

example : checkIntExprOriginEquality (.loopIndex 0) (.loopIndex 0) = none := rfl

example : (checkBggEncodingSlots (fixtureRecurrence fixturePublicKeyType)
    fixtureBggDecryptionRecurrence ⟨0, 0⟩ ⟨0, 0⟩ ⟨1, 0⟩ ⟨2, 0⟩
    fixtureCheckedGadget).isOk = true := rfl

example : checkBggEncodingSlots (fixtureRecurrence fixturePublicKeyType)
    fixtureBggDecryptionRecurrence ⟨0, 0⟩ ⟨0, 0⟩ ⟨0, 0⟩ ⟨2, 0⟩
    fixtureCheckedGadget =
    .error .duplicateDecryptionRole := by
  rfl

example : checkBggEncodingSlots (fixtureRecurrence fixturePublicKeyType)
    fixtureNonMatrixPlaintextRecurrence ⟨0, 0⟩ ⟨0, 0⟩ ⟨1, 0⟩ ⟨2, 0⟩
    fixtureCheckedGadget =
    .error .invalidPlaintextMatrices := by
  rfl

example : checkBggEncodingSlots (fixtureRecurrence fixturePublicKeyType)
    fixtureNonScalarPlaintextRecurrence ⟨0, 0⟩ ⟨0, 0⟩ ⟨1, 0⟩ ⟨2, 0⟩
    fixtureCheckedGadget =
    .error .plaintextNotOneByOne := by
  rfl

example : (deriveBggEncodingCouplingFromCheckedSlots fixtureCouplingAnalysis
    fixtureLeftRecurrenceRef fixtureRightRecurrenceRef fixtureCheckedBggSlots []).isOk = true := by
  rfl

end Mxx.Certificate
