import Mxx.Certificate.Workflow

namespace Mxx.Certificate

/-!
# Frozen reverse dependency slices

These analyzer-only slices retain the exact producer/consumer wiring needed by closed recurrence
coupling. They are computed from a verified frozen program and are never serialized or supplied
by Rust. A slice does not itself assert a semantic equation; later rules combine it with local
node soundness and actual execution traces.
-/

/-- Exact call-site data retained when a dependency slice descends into a child definition. -/
structure FrozenChildCall where
  site : CoreNodeRef
  definition : String
  outputPort : Nat
  indexSlot : Option Nat
  inputModes : List Mxx.Ir.LoopInputMode

/-- A reverse dependency slice rooted at one exact scope output. Child slices retain the exact
call site rather than flattening sibling loop lanes that happen to have the same depth. -/
inductive FrozenDependencySlice where
  | scope
      (stage : StageId)
      (scope : StaticScopeId)
      (output : Mxx.Ir.WireRef)
      (wires : List Mxx.Ir.WireRef)
      (children : List (FrozenChildCall × FrozenDependencySlice))

private def appendWireIfMissing
    (wires : List Mxx.Ir.WireRef)
    (wire : Mxx.Ir.WireRef) : List Mxx.Ir.WireRef :=
  if wires.contains wire then wires else wires ++ [wire]

private def expandDependencyWires
    (scope : Mxx.Ir.Scope)
    (wires : List Mxx.Ir.WireRef) : List Mxx.Ir.WireRef :=
  wires.foldl (fun result wire =>
    match scope.nodes[wire.node]? with
    | none => result
    | some node => node.arguments.foldl appendWireIfMissing result) wires

private def dependencyClosure
    (scope : Mxx.Ir.Scope) : Nat → List Mxx.Ir.WireRef → List Mxx.Ir.WireRef
  | 0, wires => wires
  | fuel + 1, wires => dependencyClosure scope fuel (expandDependencyWires scope wires)

private def validWire (scope : Mxx.Ir.Scope) (wire : Mxx.Ir.WireRef) : Bool :=
  match scope.nodes[wire.node]? with
  | some node => wire.port < node.outputCount
  | none => false

private def childCall?
    (stage : StageId)
    (scope : StaticScopeId)
    (wire : Mxx.Ir.WireRef)
    (node : Mxx.Ir.Node) : Option FrozenChildCall :=
  match node.kind with
  | .subgraphCall definition _ => some {
      site := { stage, scope, node := ⟨wire.node⟩ }
      definition
      outputPort := wire.port
      indexSlot := none
      inputModes := []
    }
  | .parallelLoop definition _ indexSlot _ inputModes => some {
      site := { stage, scope, node := ⟨wire.node⟩ }
      definition
      outputPort := wire.port
      indexSlot := some indexSlot
      inputModes
    }
  | .sequentialLoop definition _ indexSlot _ _ => some {
      site := { stage, scope, node := ⟨wire.node⟩ }
      definition
      outputPort := wire.port
      indexSlot := some indexSlot
      inputModes := []
    }
  | _ => none

mutual

/-- Build a complete finite reverse slice. The fuel bounds definition descent only; dependency
closure within one SSA scope uses at most the number of producer nodes. -/
partial def buildFrozenDependencySlice
    (program : Mxx.Ir.Prog)
    (stage : StageId)
    (scopeId : StaticScopeId)
    (scope : Mxx.Ir.Scope)
    (output : Mxx.Ir.WireRef)
    (fuel : Nat := program.definitions.length + 1) : Option FrozenDependencySlice := do
  guard (validWire scope output)
  let wires := dependencyClosure scope scope.nodes.size [output]
  guard (wires.all (validWire scope))
  let children ← match fuel with
    | 0 =>
        if wires.any fun wire =>
          (scope.nodes[wire.node]?).any fun node => (childCall? stage scopeId wire node).isSome
        then none else some []
    | childFuel + 1 =>
        buildFrozenDependencyChildren program stage scopeId scope wires childFuel
  return .scope stage scopeId output wires children

partial def buildFrozenDependencyChildren
    (program : Mxx.Ir.Prog)
    (stage : StageId)
    (scopeId : StaticScopeId)
    (scope : Mxx.Ir.Scope)
    (wires : List Mxx.Ir.WireRef)
    (fuel : Nat) : Option (List (FrozenChildCall × FrozenDependencySlice)) := do
  let mut children : List (FrozenChildCall × FrozenDependencySlice) := []
  for wire in wires do
    let node ← scope.nodes[wire.node]?
    match childCall? stage scopeId wire node with
    | none => pure ()
    | some call =>
        let childScope ← Mxx.Ir.lookupDefinition call.definition program.definitions
        let childOutput ← childScope.outputs[call.outputPort]?.map (·.2)
        let child ← buildFrozenDependencySlice program stage
          ⟨scopeId.path ++ [call.definition]⟩ childScope childOutput fuel
        children := children ++ [(call, child)]
  return children

end

def FrozenDependencySlice.root
    (program : Mxx.Ir.Prog)
    (stage : StageId)
    (output : Mxx.Ir.WireRef) : Option FrozenDependencySlice :=
  buildFrozenDependencySlice program stage ⟨[]⟩ program.root output

partial def FrozenDependencySlice.containsSite : FrozenDependencySlice → CoreNodeRef → Bool
  | .scope stage scopeId _ wires children, site =>
      (stage = site.stage && scopeId = site.scope &&
        wires.any (fun wire => wire.node = site.node.value)) ||
        children.any (fun child => child.2.containsSite site)

/-- All exact producer sites retained by the slice, including nested child scopes.  Duplicate
sites are harmless here and are removed by role matchers before requiring uniqueness. -/
partial def FrozenDependencySlice.sites : FrozenDependencySlice → List CoreNodeRef
  | .scope stage scopeId _output wires children =>
      wires.map (fun wire => { stage, scope := scopeId, node := ⟨wire.node⟩ }) ++
        children.flatMap fun child => child.2.sites

def scopeAtStaticPath?
    (program : Mxx.Ir.Prog)
    (scopeId : StaticScopeId) : Option Mxx.Ir.Scope :=
  match scopeId.path.reverse with
  | [] => some program.root
  | definition :: _ => Mxx.Ir.lookupDefinition definition program.definitions

private def inputNameSlot?
    (scope : Mxx.Ir.Scope)
    (name : String) : Option Nat :=
  (scope.inputNames.zipIdx.find? fun entry => entry.1 = name).map (·.2)

private def wireTypeCarriesMatrixData : Mxx.Ir.WireTypeExpr → Bool
  | .matrix _ | .preimage _ | .trapdoor .. => true
  | .indexedFamily element _ => wireTypeCarriesMatrixData element
  | _ => false

private def wireCarriesMatrixData
    (scope : Mxx.Ir.Scope)
    (wire : Mxx.Ir.WireRef) : Bool :=
  (scope.nodes[wire.node]? >>= fun node => node.outputTypes[wire.port]?).any
    wireTypeCarriesMatrixData

/-- Input slots of `scope` on which the matrix value carried by one wire transitively depends.
Integer selectors and loop indices are deliberately excluded: they control which matrix is read
but are not candidate origins for that matrix's carried role. This is a frozen SSA calculation,
not a semantic claim. -/
private def scopeMatrixInputDependencies
    (scope : Mxx.Ir.Scope) : Nat → Mxx.Ir.WireRef → List Nat
  | 0, _ => []
  | fuel + 1, wire =>
      match scope.nodes[wire.node]? with
      | none => []
      | some node =>
          match node.kind with
          | .input name => (inputNameSlot? scope name).toList
          | _ => node.arguments.filter (wireCarriesMatrixData scope)
              |>.flatMap (scopeMatrixInputDependencies scope fuel)
      |>.eraseDups

/-- All formal input slots on which a wire transitively depends, including scalar and Boolean
control data.  Unlike `scopeMatrixInputDependencies`, this is used for paired binder/control
matching and therefore follows every SSA argument edge. -/
private def scopeInputDependencies
    (scope : Mxx.Ir.Scope) : Nat → Mxx.Ir.WireRef → List Nat
  | 0, _ => []
  | fuel + 1, wire =>
      match scope.nodes[wire.node]? with
      | none => []
      | some node =>
          match node.kind with
          | .input name => (inputNameSlot? scope name).toList
          | _ => node.arguments.flatMap (scopeInputDependencies scope fuel) |>.eraseDups

/-- Project a wire at `targetSite` through the exact nested call chain retained by the slice and
return the input slots of the slice's outer scope on which it depends. A missing or ambiguous
child path fails closed. -/
partial def FrozenDependencySlice.projectInputToOuterScope?
    (program : Mxx.Ir.Prog)
    (slice : FrozenDependencySlice)
    (targetSite : CoreNodeRef)
    (targetInput : Mxx.Ir.WireRef) : Option (List Nat) := do
  let .scope stage scopeId _ _ children := slice
  let scope ← scopeAtStaticPath? program scopeId
  if targetSite.stage = stage && targetSite.scope = scopeId then
    return scopeMatrixInputDependencies scope (scope.nodes.size + 1) targetInput
  let matching := children.filter fun child => child.2.containsSite targetSite
  let (call, childSlice) ← match matching with
    | [child] => some child
    | _ => none
  let childInputs ← childSlice.projectInputToOuterScope? program targetSite targetInput
  let callNode ← scope.nodes[call.site.node.value]?
  let actualInputs := childInputs.filterMap fun slot => callNode.arguments[slot]?
  guard (actualInputs.length = childInputs.length)
  return (actualInputs.filter (wireCarriesMatrixData scope)
    |>.flatMap (scopeMatrixInputDependencies scope (scope.nodes.size + 1))).eraseDups

/-- Project any scalar, Boolean, or matrix dependency through the exact frozen child-call chain.
This is the control/binder counterpart of `projectInputToOuterScope?`; it does not discard
non-matrix edges. -/
partial def FrozenDependencySlice.projectInputToOuterScopeAny?
    (program : Mxx.Ir.Prog)
    (slice : FrozenDependencySlice)
    (targetSite : CoreNodeRef)
    (targetInput : Mxx.Ir.WireRef) : Option (List Nat) := do
  let .scope stage scopeId _ _ children := slice
  let scope ← scopeAtStaticPath? program scopeId
  if targetSite.stage = stage && targetSite.scope = scopeId then
    return scopeInputDependencies scope (scope.nodes.size + 1) targetInput
  let matching := children.filter fun child => child.2.containsSite targetSite
  let (call, childSlice) ← match matching with
    | [child] => some child
    | _ => none
  let childInputs ← childSlice.projectInputToOuterScopeAny? program targetSite targetInput
  let callNode ← scope.nodes[call.site.node.value]?
  let actualInputs := childInputs.filterMap fun slot => callNode.arguments[slot]?
  guard (actualInputs.length = childInputs.length)
  return (actualInputs.flatMap (scopeInputDependencies scope (scope.nodes.size + 1))).eraseDups

/-- Analyzer-owned pointwise matrix syntax obtained by expanding frozen subgraph and parallel-loop
bodies.  Leaves retain their exact outer-scope wire identity.  This is not a second executable DAG:
it is a finite view of the already frozen DAG used only to check recurrence-body formulas. -/
inductive FrozenPointwiseMatrixFormula where
  | atom (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
  | zero (matrixType : MatrixTypeExpr)
  | identity (matrixType : MatrixTypeExpr)
  | constant (matrixType : MatrixTypeExpr) (coefficients : List IntExpr)
  | gadget (matrixType : MatrixTypeExpr) (base : IntExpr)
  | decompose (matrixType : MatrixTypeExpr) (base : IntExpr) (small : Bool) (digitCount : IntExpr)
      (input : FrozenPointwiseMatrixFormula)
  | add (left right : FrozenPointwiseMatrixFormula)
  | subtract (left right : FrozenPointwiseMatrixFormula)
  | multiply (left right : FrozenPointwiseMatrixFormula)
  | negate (input : FrozenPointwiseMatrixFormula)
  | scale (scalar : IntExpr) (input : FrozenPointwiseMatrixFormula)
  deriving BEq, DecidableEq

/-- Provenance-preserving form of `FrozenPointwiseMatrixFormula`.  Arithmetic equality deliberately
uses the erased formula above, while semantic soundness follows these exact frozen source nodes.
This is analyzer-only metadata and is neither serialized nor accepted from a certificate. -/
inductive FrozenPointwiseMatrixProgramFormula where
  | atom (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
  | inputSubstitution (scope : StaticScopeId) (wire : Mxx.Ir.WireRef) (slot : Nat)
      (value : FrozenPointwiseMatrixProgramFormula)
  | zero (scope : StaticScopeId) (wire : Mxx.Ir.WireRef) (matrixType : MatrixTypeExpr)
  | identity (scope : StaticScopeId) (wire : Mxx.Ir.WireRef) (matrixType : MatrixTypeExpr)
  | constant (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (matrixType : MatrixTypeExpr) (coefficients : List IntExpr)
  | gadget (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (matrixType : MatrixTypeExpr) (base : IntExpr)
  | decompose (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (matrixType : MatrixTypeExpr) (base : IntExpr) (small : Bool) (digitCount : IntExpr)
      (input : FrozenPointwiseMatrixProgramFormula)
  | preimage (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (matrixType : MatrixTypeExpr) (cutoff : IntExpr)
      (publicWire trapdoor target : Mxx.Ir.WireRef)
  | slice (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (rows columns : Option (IntExpr × IntExpr))
      (input : FrozenPointwiseMatrixProgramFormula)
  | concatRows (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (left right : FrozenPointwiseMatrixProgramFormula)
  | add (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (left right : FrozenPointwiseMatrixProgramFormula)
  | subtract (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (left right : FrozenPointwiseMatrixProgramFormula)
  | multiply (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (left right : FrozenPointwiseMatrixProgramFormula)
  | negate (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (input : FrozenPointwiseMatrixProgramFormula)
  | scale (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (scalar : IntExpr) (input : FrozenPointwiseMatrixProgramFormula)
  | scaleOne (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (input : FrozenPointwiseMatrixProgramFormula)
  | select (scope : StaticScopeId) (wire index : Mxx.Ir.WireRef)
      (branches : List FrozenPointwiseMatrixProgramFormula)
  | subgraphCall (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (definition : String) (outputPort : Nat)
      (arguments : List FrozenPointwiseMatrixProgramFormula)
      (output : FrozenPointwiseMatrixProgramFormula)
  | parallelLoop (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (definition : String) (outputPort : Nat)
      (arguments : List FrozenPointwiseMatrixProgramFormula)
      (output : FrozenPointwiseMatrixProgramFormula)
  deriving BEq

def FrozenPointwiseMatrixProgramFormula.erase :
    FrozenPointwiseMatrixProgramFormula → FrozenPointwiseMatrixFormula
  | .atom scope wire => .atom scope wire
  | .inputSubstitution _ _ _ value => value.erase
  | .zero _ _ matrixType => .zero matrixType
  | .identity _ _ matrixType => .identity matrixType
  | .constant _ _ matrixType coefficients => .constant matrixType coefficients
  | .gadget _ _ matrixType base => .gadget matrixType base
  | .decompose _ _ matrixType base small digitCount input =>
      .decompose matrixType base small digitCount input.erase
  | .preimage scope wire _ _ _ _ _ => .atom scope wire
  | .slice scope wire _ _ _ => .atom scope wire
  | .concatRows scope wire _ _ => .atom scope wire
  | .add _ _ left right => .add left.erase right.erase
  | .subtract _ _ left right => .subtract left.erase right.erase
  | .multiply _ _ left right => .multiply left.erase right.erase
  | .negate _ _ input => .negate input.erase
  | .scale _ _ scalar input => .scale scalar input.erase
  | .scaleOne _ _ input => input.erase
  | .select scope wire _ _ => .atom scope wire
  | .subgraphCall _ _ _ _ _ output => output.erase
  | .parallelLoop _ _ _ _ _ output => output.erase

def FrozenPointwiseMatrixProgramFormula.source :
    FrozenPointwiseMatrixProgramFormula → StaticScopeId × Mxx.Ir.WireRef
  | .atom scope wire
  | .inputSubstitution scope wire _ _
  | .zero scope wire _
  | .identity scope wire _
  | .constant scope wire _ _
  | .gadget scope wire _ _
  | .decompose scope wire _ _ _ _ _
  | .preimage scope wire _ _ _ _ _
  | .slice scope wire _ _ _
  | .concatRows scope wire _ _
  | .add scope wire _ _
  | .subtract scope wire _ _
  | .multiply scope wire _ _
  | .negate scope wire _
  | .scale scope wire _ _
  | .scaleOne scope wire _
  | .select scope wire _ _
  | .subgraphCall scope wire _ _ _ _
  | .parallelLoop scope wire _ _ _ _ => (scope, wire)

def pointwiseFormulaArgumentsMatch
    (scopeId : StaticScopeId)
    (resultWire : Mxx.Ir.WireRef)
    (expected : List Mxx.Ir.WireRef)
    (arguments : List FrozenPointwiseMatrixProgramFormula) : Bool :=
  arguments.all (fun argument => argument.source.1 == scopeId) &&
    expected == arguments.map (fun argument => argument.source.2) &&
    expected.all (fun argument => argument.node < resultWire.node)

/-- Shared frozen-node lookup for the closed pointwise provenance validator. -/
def pointwiseFormulaNodeValid
    (program : Mxx.Ir.Prog)
    (scopeId : StaticScopeId)
    (wire : Mxx.Ir.WireRef)
    (check : Mxx.Ir.Scope → Mxx.Ir.Node → Bool) : Bool :=
  match scopeAtStaticPath? program scopeId with
  | none => false
  | some scope =>
      match scope.nodes[wire.node]? with
      | none => false
      | some node => wire.port < node.outputCount && check scope node

mutual

/-- Closed provenance validator for an annotated pointwise formula.  It checks every source node,
call boundary, formal-input substitution, and output port against the frozen program. -/
def FrozenPointwiseMatrixProgramFormula.validIn
    (program : Mxx.Ir.Prog)
    (substitutions : List FrozenPointwiseMatrixProgramFormula := []) :
    FrozenPointwiseMatrixProgramFormula → Bool
  | .atom scopeId wire => pointwiseFormulaNodeValid program scopeId wire fun _ _ => true
  | .inputSubstitution scopeId wire slot value =>
      pointwiseFormulaNodeValid program scopeId wire fun scope node =>
        match node.kind, node.arguments with
        | .input name, [] =>
            inputNameSlot? scope name == some slot &&
              match substitutions[slot]? with
              | some expected => expected == value
              | none => value.validIn program []
        | _, _ => false
  | .zero scopeId wire matrixType =>
      pointwiseFormulaNodeValid program scopeId wire fun _ node =>
        match node.kind, node.arguments with
        | .zeroMatrix actualType, [] => wire.port == 0 && matrixType == actualType
        | _, _ => false
  | .identity scopeId wire matrixType =>
      pointwiseFormulaNodeValid program scopeId wire fun _ node =>
        match node.kind, node.arguments with
        | .identityMatrix actualType, [] => wire.port == 0 && matrixType == actualType
        | _, _ => false
  | .constant scopeId wire matrixType coefficients =>
      pointwiseFormulaNodeValid program scopeId wire fun _ node =>
        match node.kind, node.arguments with
        | .constantMatrix actualType actualCoefficients, [] =>
            wire.port == 0 && matrixType == actualType && coefficients == actualCoefficients
        | _, _ => false
  | .gadget scopeId wire matrixType base =>
      pointwiseFormulaNodeValid program scopeId wire fun _ node =>
        match node.kind, node.arguments with
        | .gadgetMatrix actualType actualBase, [] =>
            wire.port == 0 && matrixType == actualType && base == actualBase
        | _, _ => false
  | .decompose scopeId wire matrixType base small digitCount input =>
      pointwiseFormulaNodeValid program scopeId wire fun _ node =>
        match node.kind with
        | .gadgetDecompose actualType actualBase actualSmall actualDigitCount =>
            wire.port == 0 && matrixType == actualType && base == actualBase &&
              small == actualSmall && digitCount == actualDigitCount &&
              pointwiseFormulaArgumentsMatch scopeId wire node.arguments [input] &&
              input.validIn program substitutions
        | _ => false
  | .add scopeId wire left right =>
      pointwiseFormulaNodeValid program scopeId wire fun _ node =>
        node.kind == .matrixAdd && wire.port == 0 &&
          pointwiseFormulaArgumentsMatch scopeId wire node.arguments [left, right] &&
          left.validIn program substitutions && right.validIn program substitutions
  | .subtract scopeId wire left right =>
      pointwiseFormulaNodeValid program scopeId wire fun _ node =>
        node.kind == .matrixSubtract && wire.port == 0 &&
          pointwiseFormulaArgumentsMatch scopeId wire node.arguments [left, right] &&
          left.validIn program substitutions && right.validIn program substitutions
  | .multiply scopeId wire left right =>
      pointwiseFormulaNodeValid program scopeId wire fun _ node =>
        node.kind == .matrixMultiply && wire.port == 0 &&
          pointwiseFormulaArgumentsMatch scopeId wire node.arguments [left, right] &&
          left.validIn program substitutions && right.validIn program substitutions
  | .negate scopeId wire input =>
      pointwiseFormulaNodeValid program scopeId wire fun _ node =>
        node.kind == .matrixNegate && wire.port == 0 &&
          pointwiseFormulaArgumentsMatch scopeId wire node.arguments [input] &&
          input.validIn program substitutions
  | .preimage scopeId wire matrixType cutoff publicWire trapdoor target =>
      pointwiseFormulaNodeValid program scopeId wire fun _ node =>
        node.kind == .preimageSample matrixType cutoff && wire.port == 0 &&
          node.arguments == [publicWire, trapdoor, target] &&
          node.arguments.all (fun argument => argument.node < wire.node)
  | .slice scopeId wire rows columns input =>
      pointwiseFormulaNodeValid program scopeId wire fun _ node =>
        node.kind == .slice rows columns && wire.port == 0 &&
          pointwiseFormulaArgumentsMatch scopeId wire node.arguments [input] &&
          input.validIn program substitutions
  | .concatRows scopeId wire left right =>
      pointwiseFormulaNodeValid program scopeId wire fun _ node =>
        node.kind == .concat .rows && wire.port == 0 &&
          pointwiseFormulaArgumentsMatch scopeId wire node.arguments [left, right] &&
          left.validIn program substitutions && right.validIn program substitutions
  | .scale scopeId wire scalar input =>
      pointwiseFormulaNodeValid program scopeId wire fun _ node =>
        match node.kind with
        | .matrixScale actualScalar =>
            wire.port == 0 && scalar == actualScalar &&
              pointwiseFormulaArgumentsMatch scopeId wire node.arguments [input] &&
              input.validIn program substitutions
        | _ => false
  | .scaleOne scopeId wire input =>
      pointwiseFormulaNodeValid program scopeId wire fun _ node =>
        node.kind == .matrixScale (.constant 1) && wire.port == 0 &&
          pointwiseFormulaArgumentsMatch scopeId wire node.arguments [input] &&
          input.validIn program substitutions
  | .select scopeId wire index branches =>
      pointwiseFormulaNodeValid program scopeId wire fun _ node =>
        node.kind == .select && wire.port == 0 && index.node < wire.node &&
          pointwiseFormulaArgumentsMatch scopeId wire (node.arguments.drop 1) branches &&
          node.arguments == index :: branches.map (fun branch => branch.source.2) &&
          allPointwiseFormulasValid program substitutions branches
  | .subgraphCall scopeId wire definition outputPort callArguments output =>
      pointwiseFormulaNodeValid program scopeId wire fun _ node =>
        match node.kind with
        | .subgraphCall actualDefinition _ =>
            definition == actualDefinition && outputPort == wire.port &&
              pointwiseFormulaArgumentsMatch scopeId wire node.arguments callArguments &&
              allPointwiseFormulasValid program substitutions callArguments &&
              match Mxx.Ir.lookupDefinition definition program.definitions with
              | none => false
              | some child =>
                  match child.outputs[outputPort]? with
                  | none => false
                  | some (_, outputWire) =>
                      output.source == (⟨scopeId.path ++ [definition]⟩, outputWire) &&
                        output.validIn program callArguments
        | _ => false
  | .parallelLoop scopeId wire definition outputPort callArguments output =>
      pointwiseFormulaNodeValid program scopeId wire fun _ node =>
        match node.kind with
        | .parallelLoop actualDefinition _ _ _ _ =>
            definition == actualDefinition && outputPort == wire.port &&
              pointwiseFormulaArgumentsMatch scopeId wire node.arguments callArguments &&
              allPointwiseFormulasValid program substitutions callArguments &&
              match Mxx.Ir.lookupDefinition definition program.definitions with
              | none => false
              | some child =>
                  match child.outputs[outputPort]? with
                  | none => false
                  | some (_, outputWire) =>
                      output.source == (⟨scopeId.path ++ [definition]⟩, outputWire) &&
                        output.validIn program callArguments
        | _ => false

/-- Structural list companion of `validIn`; keeping the recursion first-order exposes usable
equation theorems for proof-side inversion without changing the Boolean validation semantics. -/
def allPointwiseFormulasValid
    (program : Mxx.Ir.Prog)
    (substitutions : List FrozenPointwiseMatrixProgramFormula) :
    List FrozenPointwiseMatrixProgramFormula → Bool
  | [] => true
  | formula :: tail => formula.validIn program substitutions &&
      allPointwiseFormulasValid program substitutions tail

end

private partial def normalizePointwiseMatrixWire
    (program : Mxx.Ir.Prog)
    (scopeId : StaticScopeId)
    (scope : Mxx.Ir.Scope)
    (substitutions : List FrozenPointwiseMatrixProgramFormula)
    (fuel : Nat)
    (wire : Mxx.Ir.WireRef) : Option FrozenPointwiseMatrixProgramFormula := do
  let node ← scope.nodes[wire.node]?
  match fuel with
  | 0 => none
  | fuel + 1 =>
      match node.kind, node.arguments with
      | .input name, [] =>
          match inputNameSlot? scope name with
          | none => some (.atom scopeId wire)
          | some slot =>
              match substitutions[slot]? with
              | some substituted => some (.inputSubstitution scopeId wire slot substituted)
              | none => some (.atom scopeId wire)
      | .zeroMatrix matrixType, [] => some (.zero scopeId wire matrixType)
      | .identityMatrix matrixType, [] => some (.identity scopeId wire matrixType)
      | .constantMatrix matrixType coefficients, [] =>
          some (.constant scopeId wire matrixType coefficients)
      | .gadgetMatrix matrixType base, [] => some (.gadget scopeId wire matrixType base)
      | .gadgetDecompose matrixType base small digitCount, [input] =>
          return .decompose scopeId wire matrixType base small digitCount
            (← normalizePointwiseMatrixWire program scopeId scope substitutions fuel input)
      | .preimageSample matrixType cutoff, [publicWire, trapdoor, target] =>
          some (.preimage scopeId wire matrixType cutoff publicWire trapdoor target)
      | .slice rows columns, [input] =>
          return .slice scopeId wire rows columns
            (← normalizePointwiseMatrixWire program scopeId scope substitutions fuel input)
      | .concat .rows, [left, right] =>
          return .concatRows scopeId wire
            (← normalizePointwiseMatrixWire program scopeId scope substitutions fuel left)
            (← normalizePointwiseMatrixWire program scopeId scope substitutions fuel right)
      | .matrixAdd, [left, right] =>
          return .add scopeId wire
            (← normalizePointwiseMatrixWire program scopeId scope substitutions fuel left)
            (← normalizePointwiseMatrixWire program scopeId scope substitutions fuel right)
      | .matrixSubtract, [left, right] =>
          return .subtract scopeId wire
            (← normalizePointwiseMatrixWire program scopeId scope substitutions fuel left)
            (← normalizePointwiseMatrixWire program scopeId scope substitutions fuel right)
      | .matrixMultiply, [left, right] =>
          return .multiply scopeId wire
            (← normalizePointwiseMatrixWire program scopeId scope substitutions fuel left)
            (← normalizePointwiseMatrixWire program scopeId scope substitutions fuel right)
      | .matrixNegate, [input] =>
          return .negate scopeId wire
            (← normalizePointwiseMatrixWire program scopeId scope substitutions fuel input)
      | .matrixScale (.constant 1), [input] =>
          return .scaleOne scopeId wire
            (← normalizePointwiseMatrixWire program scopeId scope substitutions fuel input)
      | .matrixScale scalar, [input] =>
          return .scale scopeId wire scalar
            (← normalizePointwiseMatrixWire program scopeId scope substitutions fuel input)
      | .select, index :: branches =>
          return .select scopeId wire index
            (← branches.mapM fun branch =>
              normalizePointwiseMatrixWire program scopeId scope substitutions fuel branch)
      | .subgraphCall definition _, arguments => do
          let child ← Mxx.Ir.lookupDefinition definition program.definitions
          let output ← child.outputs[wire.port]?.map (fun entry => entry.2)
          let childSubstitutions ← arguments.mapM fun argument =>
            normalizePointwiseMatrixWire program scopeId scope substitutions fuel argument
          let normalized ← normalizePointwiseMatrixWire program
            ⟨scopeId.path ++ [definition]⟩ child childSubstitutions fuel output
          some (.subgraphCall scopeId wire definition wire.port childSubstitutions normalized)
      | .parallelLoop definition _ _ _ _, arguments => do
          let child ← Mxx.Ir.lookupDefinition definition program.definitions
          let output ← child.outputs[wire.port]?.map (fun entry => entry.2)
          let childSubstitutions ← arguments.mapM fun argument =>
            normalizePointwiseMatrixWire program scopeId scope substitutions fuel argument
          let expanded ← normalizePointwiseMatrixWire program
            ⟨scopeId.path ++ [definition]⟩ child childSubstitutions fuel output
          -- A pointwise gather/pass-through is a real runtime value shared by later formulas.
          -- Retain its parent wire identity.  A computational lane, on the other hand, is
          -- expanded so the closed BGG matcher can inspect its arithmetic.
          match expanded.erase with
          | .atom _ _ => some (.atom scopeId wire)
          | _ => some (.parallelLoop scopeId wire definition wire.port childSubstitutions expanded)
      | _, _ => some (.atom scopeId wire)

/-- Normalize the selected pointwise output of a parallel loop while retaining any independent
parallel-loop values used by its body as atomic leaves.  The selected loop is the semantic lane
whose formula is being checked; recursively expanding every parallel producer would erase the
runtime wire identities needed by the trace bridge. -/
private def normalizePointwiseMatrixTargetWire
    (program : Mxx.Ir.Prog)
    (scopeId : StaticScopeId)
    (scope : Mxx.Ir.Scope)
    (substitutions : List FrozenPointwiseMatrixProgramFormula)
    (fuel : Nat)
    (wire : Mxx.Ir.WireRef) : Option FrozenPointwiseMatrixProgramFormula :=
  normalizePointwiseMatrixWire program scopeId scope substitutions fuel wire

/-- Normalize a matrix wire at an exact nested site under the substitutions induced by the unique
child-call path retained in this dependency slice. -/
partial def FrozenDependencySlice.normalizePointwiseMatrixProgramAt?
    (program : Mxx.Ir.Prog)
    (slice : FrozenDependencySlice)
    (targetSite : CoreNodeRef)
    (targetWire : Mxx.Ir.WireRef)
    (fuel : Nat := program.definitions.length + 1) :
    Option FrozenPointwiseMatrixProgramFormula := do
  let .scope stage scopeId _ _ children := slice
  let scope ← scopeAtStaticPath? program scopeId
  if targetSite.stage = stage && targetSite.scope = scopeId then
    normalizePointwiseMatrixTargetWire program scopeId scope [] (scope.nodes.size + fuel)
      targetWire
  else
    let matching := children.filter fun child => child.2.containsSite targetSite
    let (call, childSlice) ← match matching with
      | [child] => some child
      | _ => none
    let callNode ← scope.nodes[call.site.node.value]?
    let substitutions ← callNode.arguments.mapM fun argument =>
      normalizePointwiseMatrixTargetWire program scopeId scope [] (scope.nodes.size + fuel)
        argument
    let .scope childStage childScopeId _ _ _ := childSlice
    let childScope ← scopeAtStaticPath? program childScopeId
    guard (childStage = stage)
    if targetSite.scope = childScopeId then
      normalizePointwiseMatrixTargetWire program childScopeId childScope substitutions
        (childScope.nodes.size + fuel) targetWire
    else
      -- Deeper descent must retain the substitutions already induced by this call.  Rebuild a
      -- rooted view at the child and normalize its target after substituting the child's inputs.
      let rec descend
          (current : FrozenDependencySlice)
          (currentSubstitutions : List FrozenPointwiseMatrixProgramFormula) :
          Option FrozenPointwiseMatrixProgramFormula := do
        let .scope currentStage currentScopeId _ _ currentChildren := current
        let currentScope ← scopeAtStaticPath? program currentScopeId
        guard (currentStage = stage)
        if targetSite.scope = currentScopeId then
          normalizePointwiseMatrixTargetWire program currentScopeId currentScope
            currentSubstitutions
            (currentScope.nodes.size + fuel) targetWire
        else
          let nestedMatching := currentChildren.filter fun child =>
            child.2.containsSite targetSite
          let (nestedCall, nestedSlice) ← match nestedMatching with
            | [child] => some child
            | _ => none
          let nestedNode ← currentScope.nodes[nestedCall.site.node.value]?
          let nestedSubstitutions ← nestedNode.arguments.mapM fun argument =>
            normalizePointwiseMatrixTargetWire program currentScopeId currentScope
              currentSubstitutions (currentScope.nodes.size + fuel) argument
          descend nestedSlice nestedSubstitutions
      descend childSlice substitutions

/-- Erased arithmetic view used by the closed formula matcher.  Runtime soundness uses the
provenance-preserving result above, so equality checking never discards the information needed
to recover sampler outputs from the actual execution trace. -/
def FrozenDependencySlice.normalizePointwiseMatrixAt?
    (program : Mxx.Ir.Prog)
    (slice : FrozenDependencySlice)
    (targetSite : CoreNodeRef)
    (targetWire : Mxx.Ir.WireRef)
    (fuel : Nat := program.definitions.length + 1) : Option FrozenPointwiseMatrixFormula :=
  (slice.normalizePointwiseMatrixProgramAt? program targetSite targetWire fuel).map
    FrozenPointwiseMatrixProgramFormula.erase

end Mxx.Certificate
