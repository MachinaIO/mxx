import MxxIrCore.Types
import MxxIrCore.Expr

namespace Mxx
namespace IR

def scopeAt (stage : Stage) (id : ScopeId) : Option Scope :=
  stage.scopes.find? (fun scope => scope.id = id)

def nodeAt (scope : Scope) (id : NodeId) : Option Node := scope.nodes[id]?

def wireType? (scope : Scope) (wire : WireRef) : Option WireType :=
  if wire.scope = scope.id then (nodeAt scope wire.node).bind (fun n => n.outputs[wire.port]?) else none

def Stage.wireType? (stage : Stage) (wire : WireRef) : Option WireType :=
  (scopeAt stage wire.scope).bind (fun scope => Mxx.IR.wireType? scope wire)

def NodePayload.childScope? : NodePayload → Option ScopeId
  | .subgraphCall p => some p.child
  | .sequentialLoop p => some p.child
  | .parallelGrid p => some p.child
  | _ => none

def structuralExprValid : StructuralIntExpr → Prop
  | .literal _ => True
  | .structuralSlot _ => True
  | .add left right | .subtract left right | .multiply left right =>
      structuralExprValid left ∧ structuralExprValid right
  | .exactDivide left right | .roundDivide left right =>
      structuralExprValid left ∧ structuralExprValid right
  | .log2Ceil value => structuralExprValid value

def indexExprValid : IndexMapExpr → Prop
  | .literal _ | .axis _ | .structuralSlot _ => True
  | .add left right | .sub left right | .mul left right | .divide left right |
      .remainder left right | .equal left right | .less left right | .lessEqual left right =>
      indexExprValid left ∧ indexExprValid right
  | .log2Ceil value => indexExprValid value
  | .select selector branches => indexExprValid selector ∧ ∀ branch ∈ branches, indexExprValid branch

def realExprValid : RealExpr → Prop
  | .literal value => value.denominator ≠ 0
  | .fromInt value => structuralExprValid value
  | .add left right | .subtract left right | .multiply left right | .divide left right =>
      realExprValid left ∧ realExprValid right
  | .sqrt value => realExprValid value

def optionRangeValid : Option IntRange → Prop
  | none => True
  | some range => structuralExprValid range.start ∧ structuralExprValid range.stop

def shapeValid (shape : Array StructuralIntExpr) : Prop :=
  ∀ extent ∈ shape, structuralExprValid extent

def indexMapValid (map : IndexMap) : Prop :=
  map.inputIndices.size = map.sourceRank ∧ ∀ index ∈ map.inputIndices, indexExprValid index

def matrixTypeValid (matrix : MatrixType) : Prop := matrix.Valid

def uniqueStructuralSlots (slots : Array StructuralSlotDecl) : Prop :=
  ∀ (first second : Nat) (left right : StructuralSlotDecl), slots[first]? = some left →
    slots[second]? = some right → first ≠ second → left.slot ≠ right.slot

def structuralSlotsValid (slots : Array StructuralSlotDecl) : Prop :=
  uniqueStructuralSlots slots ∧ ∀ slot ∈ slots, 0 < slot.upperBound

def slotDeclared (slots : Array StructuralSlotDecl) (slot : Nat) : Prop :=
  ∃ declaration ∈ slots, declaration.slot = slot

def structuralSlotsUsed : Array StructuralSlotDecl → StructuralIntExpr → Prop
  | slots, .literal _ => True
  | slots, .structuralSlot slot => slotDeclared slots slot
  | slots, .add left right | slots, .subtract left right | slots, .multiply left right |
      slots, .exactDivide left right | slots, .roundDivide left right =>
      structuralSlotsUsed slots left ∧ structuralSlotsUsed slots right
  | slots, .log2Ceil value => structuralSlotsUsed slots value

def indexSlotsUsed : Array StructuralSlotDecl → IndexMapExpr → Prop
  | slots, .literal _ | slots, .axis _ => True
  | slots, .structuralSlot slot => slotDeclared slots slot
  | slots, .add left right | slots, .sub left right | slots, .mul left right |
      slots, .divide left right | slots, .remainder left right | slots, .equal left right |
      slots, .less left right | slots, .lessEqual left right =>
      indexSlotsUsed slots left ∧ indexSlotsUsed slots right
  | slots, .log2Ceil value => indexSlotsUsed slots value
  | slots, .select selector branches => indexSlotsUsed slots selector ∧
      ∀ branch ∈ branches, indexSlotsUsed slots branch

def realSlotsUsed (slots : Array StructuralSlotDecl) : RealExpr → Prop
  | .literal _ => True
  | .fromInt value => structuralSlotsUsed slots value
  | .add left right | .subtract left right | .multiply left right | .divide left right =>
      realSlotsUsed slots left ∧ realSlotsUsed slots right
  | .sqrt value => realSlotsUsed slots value

def rangeSlotsUsed (slots : Array StructuralSlotDecl) (range : IntRange) : Prop :=
  structuralSlotsUsed slots range.start ∧ structuralSlotsUsed slots range.stop

def mapSlotsUsed (slots : Array StructuralSlotDecl) (map : IndexMap) : Prop :=
  map.inputIndices.size = map.sourceRank ∧ ∀ index ∈ map.inputIndices, indexSlotsUsed slots index

def payloadSlotsUsed : Array StructuralSlotDecl → NodePayload → Prop
  | slots, .evaluateInt value => structuralSlotsUsed slots value
  | slots, .constantReal value => realSlotsUsed slots value
  | slots, .constantMatrix _ literal => match literal with
      | .unitRow index | .unitColumn index => structuralSlotsUsed slots index
      | .gadget base _ => structuralSlotsUsed slots base
      | .powerOfBase base exponent => structuralSlotsUsed slots base ∧ structuralSlotsUsed slots exponent
      | .rotation exponent => structuralSlotsUsed slots exponent
      | .polynomial coefficients => ∀ coefficient ∈ coefficients, structuralSlotsUsed slots coefficient
      | _ => True
  | slots, .gadgetTrapdoor _ base => structuralSlotsUsed slots base
  | slots, .uniformIntervalSample _ range => rangeSlotsUsed slots range
  | slots, .gaussianSample _ sigma bound => realSlotsUsed slots sigma ∧ structuralSlotsUsed slots bound
  | slots, .hashSample _ _ tags decimalTags u64Tags =>
      (∀ x ∈ tags, structuralSlotsUsed slots x) ∧
      (∀ x ∈ decimalTags, structuralSlotsUsed slots x) ∧
      (∀ x ∈ u64Tags, structuralSlotsUsed slots x)
  | slots, .trapdoorSample _ sigma base digits bound =>
      realSlotsUsed slots sigma ∧ structuralSlotsUsed slots base ∧
      structuralSlotsUsed slots digits ∧ structuralSlotsUsed slots bound
  | slots, .preimageSample _ bound | slots, .familyPreimageSample _ bound => structuralSlotsUsed slots bound
  | slots, .gadgetDecompose base _ digits => structuralSlotsUsed slots base ∧ structuralSlotsUsed slots digits
  | slots, .decompositionEntry row column => structuralSlotsUsed slots row ∧ structuralSlotsUsed slots column
  | slots, .extractCoefficient position _ => structuralSlotsUsed slots position
  | slots, .thresholdDecode modulus length _ => structuralSlotsUsed slots modulus ∧ structuralSlotsUsed slots length
  | slots, .crtRecompose moduli coefficients =>
      (∀ x ∈ moduli, structuralSlotsUsed slots x) ∧ (∀ x ∈ coefficients, structuralSlotsUsed slots x)
  | slots, .packPolynomialCoefficients _ bits => structuralSlotsUsed slots bits
  | slots, .matrixScale scalar => structuralSlotsUsed slots scalar
  | slots, .familyPack shape | slots, .parallelGrid { shape := shape, .. } =>
      ∀ extent ∈ shape, structuralSlotsUsed slots extent
  | slots, .familyReindex shape map =>
      (∀ extent ∈ shape, structuralSlotsUsed slots extent) ∧ mapSlotsUsed slots map
  | slots, .familyGather shape _ => ∀ extent ∈ shape, structuralSlotsUsed slots extent
  | slots, .sequentialLoop payload => structuralSlotsUsed slots payload.count
  | slots, .select count => structuralSlotsUsed slots count
  | slots, _ => True

def validWireType : WireType → Prop
  | .constantInt | .constantReal | .constantBool | .int | .real | .bool => True
  | .bytes length => 0 < length
  | .typedBlob _ _ => True
  | .matrix matrix | .preimage matrix => matrix.Valid
  | .trapdoor trapdoor =>
      trapdoor.matrix.Valid ∧ structuralExprValid trapdoor.gadgetBase ∧
        structuralExprValid trapdoor.digitCount ∧ structuralExprValid trapdoor.preimageMaxCoefficientBound ∧
        realExprValid trapdoor.sigma
  | .family shape element => (∀ extent ∈ shape, 0 < extent) ∧ validWireType element

def validPayload : NodePayload → Prop
  | .input _ | .artifactInput _ | .constantInt _ | .constantBool _ |
      .trapdoorPublic | .intBinary _ | .intCompare _ | .intToReal | .boolToInt |
      .realBinary _ | .realSqrt | .matrixBinary _ | .matrixNegate | .transpose |
      .tensor | .concat _ | .applyPreimage | .materializePreimageExact | .preimageBinary _ |
      .preimageConcatColumns | .familyGetDynamic _ | .familySelectAxis _ => True
  | .evaluateInt value => structuralExprValid value
  | .constantReal value => realExprValid value
  | .constantMatrix matrix _ | .uniformResidueSample matrix | .liftIntegerToConstantPolynomial matrix =>
      matrix.Valid
  | .uniformIntervalSample matrix range => matrix.Valid ∧ structuralExprValid range.start ∧
      structuralExprValid range.stop
  | .gaussianSample matrix sigma bound => matrix.Valid ∧ realExprValid sigma ∧ structuralExprValid bound
  | .hashSample matrix _ tags decimalTags u64Tags => matrix.Valid ∧
      (∀ x ∈ tags, structuralExprValid x) ∧ (∀ x ∈ decimalTags, structuralExprValid x) ∧
      (∀ x ∈ u64Tags, structuralExprValid x)
  | .trapdoorSample matrix sigma base digits bound => matrix.Valid ∧ realExprValid sigma ∧
      structuralExprValid base ∧ structuralExprValid digits ∧ structuralExprValid bound
  | .preimageSample matrix bound | .familyPreimageSample matrix bound =>
      matrix.Valid ∧ structuralExprValid bound
  | .gadgetTrapdoor matrix base => matrix.Valid ∧ structuralExprValid base
  | .gadgetDecompose base _ digits => structuralExprValid base ∧ structuralExprValid digits
  | .bitExtract bit => structuralExprValid bit
  | .matrixScale scalar => structuralExprValid scalar
  | .decompositionEntry row column => structuralExprValid row ∧ structuralExprValid column
  | .extractCoefficient position _ => structuralExprValid position
  | .slice rows columns => optionRangeValid rows ∧ optionRangeValid columns
  | .familyPack shape => shapeValid shape
  | .familyReindex shape map => shapeValid shape ∧ map.outputRank = shape.size ∧ indexMapValid map
  | .familyGather shape rank => shapeValid shape ∧ rank > 0
  | .sequentialLoop payload => structuralExprValid payload.count ∧ payload.carriedCount > 0 ∧
      ∀ binding ∈ payload.bindings, structuralExprValid binding.2
  | .select count => structuralExprValid count
  | .packPolynomialCoefficients matrix bits => matrix.Valid ∧ structuralExprValid bits
  | .thresholdDecode modulus length _ => structuralExprValid modulus ∧ structuralExprValid length
  | .crtRecompose moduli coefficients => moduli.size = coefficients.size ∧
      (∀ x ∈ moduli, structuralExprValid x) ∧ (∀ x ∈ coefficients, structuralExprValid x)
  | .familyGetStatic indices => ∀ index ∈ indices, indexExprValid index
  | .matrixMulAccumulate coefficients _ => ∀ coefficient ∈ coefficients, structuralExprValid coefficient
  | .subgraphCall payload => ∀ binding ∈ payload.bindings, structuralExprValid binding.2
  | .parallelGrid payload => shapeValid payload.shape ∧
      (∀ binding ∈ payload.bindings, structuralExprValid binding.2) ∧
      ∀ mode ∈ payload.inputModes, match mode with
        | { reindex := false, map := none } => True
        | { reindex := true, map := some map } => indexMapValid map
        | _ => False

def NodePayload.Valid (payload : NodePayload) : Prop := validPayload payload

def operationArityOK (payload : NodePayload) (arguments outputs : Nat) : Prop :=
  0 < outputs ∧ match payload with
  | .input _ | .artifactInput _ | .constantInt _ | .evaluateInt _ | .constantReal _ |
      .constantBool _ | .constantMatrix _ _ | .gadgetTrapdoor _ _ | .trapdoorPublic |
      .uniformResidueSample _ | .uniformIntervalSample _ _ | .gaussianSample _ _ _ |
      .hashSample _ _ _ _ _ | .trapdoorSample _ _ _ _ _ |
      .liftIntegerToConstantPolynomial _ => True
  | .matrixBinary _ | .matrixNegate | .matrixScale _ | .transpose |
      .preimageSample _ _ | .familyPreimageSample _ _ | .applyPreimage |
      .materializePreimageExact | .preimageBinary _ | .preimageConcatColumns |
      .gadgetDecompose _ _ _ | .familyPack _ | .familyGetStatic _ |
      .familyGetDynamic _ | .familySelectAxis _ | .familyReindex _ _ |
      .familyGather _ _ | .select _ | .subgraphCall _ | .parallelGrid _ |
      .sequentialLoop _ => True
  | .intBinary _ | .intCompare _ | .realBinary _ => arguments = 2
  | .matrixMulAccumulate _ _ => 0 < arguments
  | .slice _ _ | .tensor | .decompositionEntry _ _ |
      .extractCoefficient _ _ | .thresholdDecode _ _ _ | .packPolynomialCoefficients _ _ => arguments = 1
  | _ => 0 < arguments

def structuralPayloadShapeOK : NodePayload → Prop
  | .familyReindex shape map => map.outputRank = shape.size
  | .familyGather shape _ => shape.size > 0
  | .parallelGrid payload => payload.shape.size = payload.indexSlots.size
  | .sequentialLoop payload => payload.carriedCount > 0 ∧ payload.indexSlot ≥ 0
  | _ => True

def referencedTypes (scope : Scope) (wires : Array WireRef) : List (Option WireType) :=
  wires.toList.map (wireType? scope)

def sameRing (left right : MatrixType) : Prop :=
  left.modulus = right.modulus ∧ left.ringDimension = right.ringDimension

def matrixProductType (left right output : MatrixType) : Prop :=
  sameRing left right ∧ sameRing left output ∧
    if left.rows = 1 ∧ left.columns = 1 then
      output.rows = right.rows ∧ output.columns = right.columns
    else if right.rows = 1 ∧ right.columns = 1 then
      output.rows = left.rows ∧ output.columns = left.columns
    else left.columns = right.rows ∧ output.rows = left.rows ∧ output.columns = right.columns

def matrixAddType (left right output : MatrixType) : Prop :=
  left = right ∧ output = left

def preimageEquationType (source target preimage : MatrixType) : Prop :=
  sameRing source target ∧ sameRing source preimage ∧ source.columns = preimage.rows ∧
    target.rows = source.rows ∧ target.columns = preimage.columns

def structuralTypeCompatible (actual expected : WireType) : Prop :=
  actual = expected ∨ (actual = .constantInt ∧ expected = .int) ∨
    (actual = .constantBool ∧ expected = .bool) ∨
    (actual = .constantReal ∧ expected = .real)

def optionalTypesCompatible (actual expected : List (Option WireType)) : Prop :=
  actual.length = expected.length ∧ ∀ index, index < actual.length →
    ∃ actualType expectedType, actual[index]? = some (some actualType) ∧
      expected[index]? = some (some expectedType) ∧
      structuralTypeCompatible actualType expectedType

def structuralExpressionIsNat (expression : StructuralIntExpr) (value : Nat) : Prop :=
  expression.eval {} = .ok (Int.ofNat value)

def shapeExpressionIs (expression : Array StructuralIntExpr) (shape : List Nat) : Prop :=
  expression.toList.mapM (fun item => match item.eval {} with
    | .ok value => if 0 < value then some value.toNat else none
    | .error _ => none) = some shape

def integerSelectorType : WireType → Prop
  | .int | .constantInt => True
  | _ => False

def scalarIntegerType : Option WireType → Prop
  | some .int | some .constantInt => True
  | _ => False

def scalarBooleanType : Option WireType → Prop
  | some .bool | some .constantBool => True
  | _ => False

def allIntegerTypes : List (Option WireType) → Prop
  | [] => True
  | some wireType :: rest => integerSelectorType wireType ∧ allIntegerTypes rest
  | none :: _ => False

def familyElementType : WireType → Option WireType
  | .family _ element => some element
  | _ => none

def familyShape : WireType → Option (List Nat)
  | .family shape _ => some shape
  | _ => none

/- Family preimage sampling treats an ordinary matrix or trapdoor as a rank-zero family.
   This is the same element/shape projection used by the Rust validator. -/
def matrixFamilyElement? : WireType → Option (List Nat × MatrixType)
  | .matrix matrix => some ([], matrix)
  | .family shape (.matrix matrix) => some (shape, matrix)
  | _ => none

def trapdoorFamilyElement? : WireType → Option (List Nat × TrapdoorType)
  | .trapdoor trapdoor => some ([], trapdoor)
  | .family shape (.trapdoor trapdoor) => some (shape, trapdoor)
  | _ => none

def shapeProduct : List Nat → Nat
  | [] => 1
  | extent :: rest => extent * shapeProduct rest

def removeAt? {α : Type} : List α → Nat → Option (List α)
  | [], _ => none
  | _ :: tail, 0 => some tail
  | head :: tail, index + 1 => (removeAt? tail index).map (head :: ·)

inductive CheckedIndexValue where
  | invalid | symbolic | static (value : Int)
  deriving DecidableEq

def indexSlotAllowedB (slots : Array StructuralSlotDecl) (slot : Nat) : Bool :=
  slots.any fun declaration =>
    decide (declaration.slot = slot ∧ declaration.kind = .sequentialIteration)

def checkedIndexBinary (left right : CheckedIndexValue) (operation : Int → Int → Int) :
    CheckedIndexValue := match left, right with
  | .invalid, _ | _, .invalid => .invalid
  | .static left, .static right => .static (operation left right)
  | _, _ => .symbolic

/- The evaluator consumes one unit per syntax node.  Computing this measure explicitly keeps the
   diagnostic executable while avoiding an arbitrary depth limit on honest index expressions. -/
def indexExprFuel : IndexMapExpr → Nat
  | .literal _ | .axis _ | .structuralSlot _ => 1
  | .add left right | .sub left right | .mul left right | .divide left right |
      .remainder left right | .equal left right | .less left right | .lessEqual left right =>
      indexExprFuel left + indexExprFuel right + 1
  | .log2Ceil value => indexExprFuel value + 1
  | .select selector branches =>
      branches.foldl (fun count branch => count + indexExprFuel branch) (indexExprFuel selector + 1)
termination_by expression => sizeOf expression

def indexExprCheckedFuelB (outputRank : Nat) (slots : Array StructuralSlotDecl) :
    Nat → IndexMapExpr → CheckedIndexValue
  | 0, _ => .invalid
  | fuel + 1, expression => match expression with
    | .literal value => .static value
    | .axis axis => if axis < outputRank then .symbolic else .invalid
    | .structuralSlot slot => if indexSlotAllowedB slots slot then .symbolic else .invalid
    | .add left right => checkedIndexBinary
        (indexExprCheckedFuelB outputRank slots fuel left)
        (indexExprCheckedFuelB outputRank slots fuel right) (· + ·)
    | .sub left right => checkedIndexBinary
        (indexExprCheckedFuelB outputRank slots fuel left)
        (indexExprCheckedFuelB outputRank slots fuel right) (· - ·)
    | .mul left right => checkedIndexBinary
        (indexExprCheckedFuelB outputRank slots fuel left)
        (indexExprCheckedFuelB outputRank slots fuel right) (· * ·)
    | .divide left right =>
        let leftValue := indexExprCheckedFuelB outputRank slots fuel left
        let rightValue := indexExprCheckedFuelB outputRank slots fuel right
        match leftValue, rightValue with
        | .invalid, _ | _, .invalid | _, .static 0 => .invalid
        | .static left, .static right => .static (Int.tdiv left right)
        | _, _ => .symbolic
    | .remainder left right =>
        let leftValue := indexExprCheckedFuelB outputRank slots fuel left
        let rightValue := indexExprCheckedFuelB outputRank slots fuel right
        match leftValue, rightValue with
        | .invalid, _ | _, .invalid | _, .static 0 => .invalid
        | .static left, .static right => .static (Int.tmod left right)
        | _, _ => .symbolic
    | .equal left right => checkedIndexBinary
        (indexExprCheckedFuelB outputRank slots fuel left)
        (indexExprCheckedFuelB outputRank slots fuel right) fun a b => if a = b then 1 else 0
    | .less left right => checkedIndexBinary
        (indexExprCheckedFuelB outputRank slots fuel left)
        (indexExprCheckedFuelB outputRank slots fuel right) fun a b => if a < b then 1 else 0
    | .lessEqual left right => checkedIndexBinary
        (indexExprCheckedFuelB outputRank slots fuel left)
        (indexExprCheckedFuelB outputRank slots fuel right) fun a b => if a ≤ b then 1 else 0
    | .log2Ceil value => match indexExprCheckedFuelB outputRank slots fuel value with
        | .invalid => .invalid
        | .symbolic => .symbolic
        | .static value => if value ≤ 0 then .invalid else match intLog2Ceil value with
            | none => .invalid
            | some result => .static result
    | .select selector branches =>
        let selectorValue := indexExprCheckedFuelB outputRank slots fuel selector
        let branchValues := branches.map (indexExprCheckedFuelB outputRank slots fuel)
        if branches.isEmpty || branchValues.any fun value => match value with
          | .invalid => true
          | _ => false
        then .invalid else
        match selectorValue with
        | .invalid => .invalid
        | .symbolic => .symbolic
        | .static choice => if 0 ≤ choice then
            branchValues[choice.toNat]?.getD .invalid else .invalid

def indexMapCheckedB (map : IndexMap) (inputShape : List Nat)
    (slots : Array StructuralSlotDecl) : Bool :=
  decide (map.inputIndices.size = inputShape.length) &&
    (List.range map.inputIndices.size).all fun axis =>
      match map.inputIndices[axis]?, inputShape[axis]? with
      | some expression, some extent =>
          match indexExprCheckedFuelB map.outputRank slots (indexExprFuel expression) expression with
          | .invalid => false
          | .symbolic => true
          | .static value => decide (0 ≤ value ∧ value < Int.ofNat extent)
      | _, _ => false

def indexMapBounded (map : IndexMap) (inputShape : List Nat)
    (slots : Array StructuralSlotDecl) : Prop := indexMapCheckedB map inputShape slots = true

def gridInputTypeOK (outputRank : Nat) (outer inner : WireType) : GridInputMode → Prop
  | { reindex := false, map := none } => outer = inner
  | { reindex := true, map := some map } =>
      ∃ shape element, outer = .family shape element ∧ element = inner ∧
        map.outputRank = outputRank ∧ map.sourceRank = shape.length ∧
        map.inputIndices.size = shape.length ∧ indexMapBounded map shape #[]
  | _ => False

def structuralChildTypesOK (stage : Stage) (scope : Scope) (payload : NodePayload)
    (arguments : Array WireRef) (outputs : Array WireType) : Prop :=
  match payload with
  | .subgraphCall call => ∃ child,
      scopeAt stage call.child = some child ∧
      call.canonicalInputExclusiveUppers.size = arguments.size ∧
      optionalTypesCompatible (referencedTypes scope arguments) (referencedTypes child child.inputs) ∧
      optionalTypesCompatible (referencedTypes child child.outputs) (outputs.toList.map some)
  | .sequentialLoop loop => ∃ child count,
      scopeAt stage loop.child = some child ∧ structuralExpressionIsNat loop.count count ∧
      loop.carriedCount > 0 ∧
      loop.carriedCount ≤ arguments.size ∧ child.inputs.size = arguments.size ∧
      child.outputs.size = loop.carriedCount ∧ outputs.size = loop.carriedCount ∧
      optionalTypesCompatible (referencedTypes scope arguments) (referencedTypes child child.inputs) ∧
      optionalTypesCompatible (referencedTypes child child.outputs) (outputs.toList.map some) ∧
      optionalTypesCompatible ((referencedTypes scope arguments).take loop.carriedCount)
        ((referencedTypes child child.outputs).take loop.carriedCount) ∧
      ∃ declaration ∈ child.structuralSlots,
        declaration.slot = loop.indexSlot ∧ declaration.kind = .sequentialIteration ∧
        declaration.upperBound = Int.ofNat count
  | .parallelGrid grid => ∃ child shape,
      scopeAt stage grid.child = some child ∧ shapeExpressionIs grid.shape shape ∧
      grid.inputModes.size = arguments.size ∧ child.inputs.size = arguments.size ∧
      child.outputs.size = outputs.size ∧ grid.indexSlots.size = shape.length ∧
      (∀ index, index < arguments.size → ∃ outer inner mode,
        (referencedTypes scope arguments)[index]? = some (some outer) ∧
        (referencedTypes child child.inputs)[index]? = some (some inner) ∧
        grid.inputModes[index]? = some mode ∧ gridInputTypeOK shape.length outer inner mode) ∧
      (∀ index, index < outputs.size → ∃ childOutput outputShape outputElement,
        (referencedTypes child child.outputs)[index]? = some (some childOutput) ∧
        outputs[index]? = some (.family outputShape outputElement) ∧ outputShape = shape ∧
        childOutput = outputElement) ∧
      (∀ axis, axis < shape.length → ∃ slot extent declaration,
        grid.indexSlots[axis]? = some slot ∧ shape[axis]? = some extent ∧
        declaration ∈ child.structuralSlots ∧ declaration.slot = slot ∧
        declaration.kind = .gridAxis axis ∧ declaration.upperBound = Int.ofNat extent)
  | _ => True

def declarativeStructuralEnvironments : List StructuralSlotDecl → List StructuralEnv
  | [] => [{}]
  | declaration :: rest =>
      (List.range declaration.upperBound.toNat).flatMap fun value =>
        (declarativeStructuralEnvironments rest).map fun env =>
          { env with slots := env.slots.push (declaration.slot, Int.ofNat value) }

def rangeExtentIs (slots : Array StructuralSlotDecl) (range : Option IntRange)
    (inputExtent outputExtent : Nat) : Prop := match range with
  | none => outputExtent = inputExtent
  | some range =>
      (declarativeStructuralEnvironments slots.toList).mapM (fun env =>
        match range.start.eval env, range.stop.eval env with
        | .ok start, .ok stop =>
            if 0 ≤ start ∧ start ≤ stop ∧ stop ≤ Int.ofNat inputExtent then
              some (stop - start).toNat else none
        | _, _ => none) = some (List.replicate
          (declarativeStructuralEnvironments slots.toList).length outputExtent)

def matrixConcatType (axis : ConcatAxis) (inputs : List (Option WireType))
    (output : MatrixType) : Prop := ∃ first rest,
  inputs.mapM (fun input => match input with
    | some (.matrix matrix) => some matrix
    | _ => none) = some (first :: rest) ∧
  (∀ matrix ∈ rest, sameRing first matrix) ∧ sameRing first output ∧
  match axis with
  | .rows => (∀ matrix ∈ rest, matrix.columns = first.columns) ∧
      output.rows = ((first :: rest).map MatrixType.rows).sum ∧
      output.columns = first.columns
  | .columns => (∀ matrix ∈ rest, matrix.rows = first.rows) ∧
      output.rows = first.rows ∧
      output.columns = ((first :: rest).map MatrixType.columns).sum
  | .diagonal => output.rows = ((first :: rest).map MatrixType.rows).sum ∧
      output.columns = ((first :: rest).map MatrixType.columns).sum

def indexAxesValid (outputRank : Nat) : IndexMapExpr → Prop
  | .literal _ | .structuralSlot _ => True
  | .axis axis => axis < outputRank
  | .add left right | .sub left right | .mul left right | .divide left right |
      .remainder left right | .equal left right | .less left right | .lessEqual left right =>
      indexAxesValid outputRank left ∧ indexAxesValid outputRank right
  | .log2Ceil value => indexAxesValid outputRank value
  | .select selector branches => indexAxesValid outputRank selector ∧
      ∀ branch ∈ branches, indexAxesValid outputRank branch

def indexMapAxesValid (map : IndexMap) : Prop :=
  ∀ expression ∈ map.inputIndices, indexAxesValid map.outputRank expression

def operationTypesOK (stage : Stage) (scope : Scope) (payload : NodePayload)
    (arguments : Array WireRef) (outputs : Array WireType) : Prop :=
  let argumentTypes := referencedTypes scope arguments
  match payload with
  | .input _ | .artifactInput _ => argumentTypes = [] ∧ outputs.size = 1
  | .constantInt _ | .evaluateInt _ =>
      argumentTypes = [] ∧ outputs.toList = [.constantInt]
  | .constantBool _ => argumentTypes = [] ∧ outputs.toList = [.constantBool]
  | .constantMatrix matrix _ | .uniformResidueSample matrix |
      .uniformIntervalSample matrix _ | .gaussianSample matrix _ _ =>
      argumentTypes = [] ∧ outputs.toList = [.matrix matrix]
  | .hashSample matrix _ _ _ _ =>
      argumentTypes = [some (.bytes 32)] ∧ outputs.toList = [.matrix matrix]
  | .trapdoorSample matrix sigma base digits bound => ∃ trapdoor,
      argumentTypes = [] ∧ outputs.toList = [.matrix matrix, .trapdoor trapdoor] ∧
      trapdoor.matrix = matrix ∧ trapdoor.sigma = sigma ∧ trapdoor.gadgetBase = base ∧
      trapdoor.digitCount = digits ∧ trapdoor.preimageMaxCoefficientBound = bound
  | .intBinary _ => ∃ left right, argumentTypes = [left, right] ∧
      scalarIntegerType left ∧ scalarIntegerType right ∧ outputs.toList = [.int]
  | .intCompare _ => ∃ left right, argumentTypes = [left, right] ∧
      scalarIntegerType left ∧ scalarIntegerType right ∧ outputs.toList = [.bool]
  | .bitExtract _ => ∃ input, argumentTypes = [input] ∧ scalarIntegerType input ∧
      outputs.toList = [.bool]
  | .realBinary _ => argumentTypes = [some .real, some .real] ∧ outputs.toList = [.real]
  | .intToReal => argumentTypes = [some .int] ∧ outputs.toList = [.real]
  | .boolToInt => ∃ input, argumentTypes = [input] ∧ scalarBooleanType input ∧
      outputs.toList = [.int]
  | .realSqrt => argumentTypes = [some .real] ∧ outputs.toList = [.real]
  | .matrixBinary operation => ∃ left right output,
      argumentTypes = [some (.matrix left), some (.matrix right)] ∧
      outputs.toList = [.matrix output] ∧ match operation with
        | .add | .subtract => matrixAddType left right output
        | .multiply => matrixProductType left right output
  | .matrixNegate | .matrixScale _ => ∃ matrix,
      argumentTypes = [some (.matrix matrix)] ∧ outputs.toList = [.matrix matrix]
  | .transpose => ∃ input output,
      argumentTypes = [some (.matrix input)] ∧ outputs.toList = [.matrix output] ∧
      sameRing input output ∧ output.rows = input.columns ∧ output.columns = input.rows
  | .concat axis => ∃ output,
      outputs.toList = [.matrix output] ∧ matrixConcatType axis argumentTypes output
  | .slice rows columns => ∃ input output outputRows outputColumns,
      argumentTypes = [some (.matrix input)] ∧ outputs.toList = [.matrix output] ∧
      rangeExtentIs scope.structuralSlots rows input.rows outputRows ∧
      rangeExtentIs scope.structuralSlots columns input.columns outputColumns ∧
      sameRing input output ∧ output.rows = outputRows ∧ output.columns = outputColumns ∧
      0 < output.rows ∧ 0 < output.columns
  | .extractCoefficient position _ => ∃ input value,
      argumentTypes = [some (.matrix input)] ∧ outputs.toList = [.int] ∧
      position.eval {} = .ok value ∧ input.rows = 1 ∧ input.columns = 1 ∧
      0 ≤ value ∧ value < Int.ofNat input.ringDimension
  | .preimageSample preimage _ => ∃ source trapdoor target,
      argumentTypes = [some (.matrix source), some (.trapdoor trapdoor), some (.matrix target)] ∧
      outputs.toList = [.preimage preimage] ∧ trapdoor.matrix = source ∧
      preimageEquationType source target preimage
  | .familyPreimageSample preimage _ =>
      ∃ sourceType trapdoorType sourceShape trapdoorShape targetShape outputShape
          source trapdoor target output,
      argumentTypes = [some sourceType, some trapdoorType,
        some (.family targetShape (.matrix target))] ∧
      matrixFamilyElement? sourceType = some (sourceShape, source) ∧
      trapdoorFamilyElement? trapdoorType = some (trapdoorShape, trapdoor) ∧
      outputs.toList = [.family outputShape (.preimage output)] ∧
      output = preimage ∧ outputShape = targetShape ∧ sourceShape = trapdoorShape ∧
      targetShape.length = sourceShape.length + 1 ∧
      targetShape.take sourceShape.length = sourceShape ∧ trapdoor.matrix = source ∧
      preimageEquationType source target preimage
  | .applyPreimage => ∃ left preimage output,
      argumentTypes = [some (.matrix left), some (.preimage preimage)] ∧
      outputs.toList = [.matrix output] ∧ matrixProductType left preimage output
  | .materializePreimageExact => ∃ preimage,
      argumentTypes = [some (.preimage preimage)] ∧ outputs.toList = [.matrix preimage]
  | .preimageBinary operation => ∃ left right output,
      argumentTypes = [some (.preimage left), some right] ∧ outputs.toList = [.preimage output] ∧
      match operation with
      | .add => right = .preimage left ∧ output = left
      | .rightMultiplyExact => ∃ matrix, right = .matrix matrix ∧ matrixProductType left matrix output
      | .composeExactDecomposition => ∃ preimage, right = .preimage preimage ∧
          matrixProductType left preimage output
  | .gadgetDecompose _ _ digits => ∃ target preimage digitCount,
      argumentTypes = [some (.matrix target)] ∧ outputs.toList = [.preimage preimage] ∧
      structuralExpressionIsNat digits digitCount ∧ digitCount > 0 ∧ sameRing target preimage ∧
      preimage.rows = target.rows * digitCount ∧ preimage.columns = target.columns
  | .familyPack shape => ∃ concreteShape element,
      shapeExpressionIs shape concreteShape ∧ concreteShape ≠ [] ∧
      argumentTypes.length = shapeProduct concreteShape ∧
      (∀ argument ∈ argumentTypes, argument = some element) ∧
      familyElementType element = none ∧ outputs.toList = [.family concreteShape element]
  | .familyGetStatic indices => ∃ shape element,
      argumentTypes = [some (.family shape element)] ∧ indices.size = shape.length ∧
      outputs.toList = [element]
  | .familyGetDynamic rank => ∃ shape element,
      ∃ selectors, argumentTypes = some (.family shape element) :: selectors ∧
      selectors.length = rank ∧ allIntegerTypes selectors ∧
      shape.length = rank ∧ outputs.toList = [element]
  | .familySelectAxis axis => ∃ shape element selector outputShape,
      argumentTypes = [some (.family shape element), some selector] ∧
      removeAt? shape axis = some outputShape ∧
      (integerSelectorType selector ∨
        ∃ selectorElement, selector = .family outputShape selectorElement ∧
          integerSelectorType selectorElement) ∧
      outputs.toList = [if outputShape = [] then element else .family outputShape element]
  | .familyReindex shape map => ∃ inputShape outputShape element,
      argumentTypes = [some (.family inputShape element)] ∧ shapeExpressionIs shape outputShape ∧
      map.sourceRank = inputShape.length ∧ map.outputRank = outputShape.length ∧
      map.inputIndices.size = inputShape.length ∧
      indexMapBounded map inputShape scope.structuralSlots ∧
      outputs.toList = [.family outputShape element]
  | .familyGather shape rank => ∃ inputShape outputShape element,
      shapeExpressionIs shape outputShape ∧ inputShape.length = rank ∧
      ∃ selectorTypes, argumentTypes = some (.family inputShape element) :: selectorTypes ∧
      selectorTypes.length = rank ∧
      (∀ selector ∈ selectorTypes, ∃ selectorElement,
        selector = some (.family outputShape selectorElement) ∧ integerSelectorType selectorElement) ∧
      outputs.toList = [.family outputShape element]
  | .select count => ∃ branchCount branchType selectorType,
      structuralExpressionIsNat count branchCount ∧ branchCount > 0 ∧
      integerSelectorType selectorType ∧
      argumentTypes = some selectorType :: List.replicate branchCount (some branchType) ∧
      outputs.toList = [branchType]
  | .subgraphCall _ | .sequentialLoop _ | .parallelGrid _ =>
      structuralChildTypesOK stage scope payload arguments outputs
  | .constantReal _ | .gadgetTrapdoor _ _ | .trapdoorPublic |
      .matrixMulAccumulate _ _ | .tensor |
      .preimageConcatColumns | .decompositionEntry _ _ |
      .liftIntegerToConstantPolynomial _ | .thresholdDecode _ _ _ |
      .crtRecompose _ _ | .packPolynomialCoefficients _ _ => False

def previousWireValid (scope : Scope) (wire : WireRef) (index : Nat) : Prop :=
  wire.scope = scope.id ∧ wire.node < index ∧
    ∃ node, nodeAt scope wire.node = some node ∧ wire.port < node.outputs.size

def uniqueScopeIds (scopes : Array Scope) : Prop :=
  ∀ (first second : Nat) (scopeFirst scopeSecond : Scope), scopes[first]? = some scopeFirst →
    scopes[second]? = some scopeSecond → first ≠ second → scopeFirst.id ≠ scopeSecond.id

def structuralChildren (scope : Scope) : Array ScopeId :=
  scope.nodes.foldl (fun result node => match node.payload.childScope? with
    | none => result
    | some child => result.push child) #[]

def noCyclesFrom (stage : Stage) (current : ScopeId) (seen : List ScopeId) (fuel : Nat) : Prop :=
  match fuel with
  | 0 => False
  | fuel + 1 => current ∉ seen ∧ ∃ scope, scopeAt stage current = some scope ∧
      ∀ child ∈ structuralChildren scope, noCyclesFrom stage child (current :: seen) fuel

def matrixProductTypeB (left right output : MatrixType) : Bool :=
  decide (left.modulus = right.modulus) &&
    decide (left.ringDimension = right.ringDimension) &&
    decide (left.modulus = output.modulus) &&
    decide (left.ringDimension = output.ringDimension) &&
    if left.rows = 1 ∧ left.columns = 1 then
      decide (output.rows = right.rows ∧ output.columns = right.columns)
    else if right.rows = 1 ∧ right.columns = 1 then
      decide (output.rows = left.rows ∧ output.columns = left.columns)
    else decide (left.columns = right.rows ∧ output.rows = left.rows ∧
      output.columns = right.columns)

def matrixAddTypeB (left right output : MatrixType) : Bool :=
  decide (left = right) && decide (output = left)

def preimageEquationTypeB (source target preimage : MatrixType) : Bool :=
  decide (source.modulus = target.modulus) &&
    decide (source.ringDimension = target.ringDimension) &&
    decide (source.modulus = preimage.modulus) &&
    decide (source.ringDimension = preimage.ringDimension) &&
    decide (source.columns = preimage.rows) && decide (target.rows = source.rows) &&
    decide (target.columns = preimage.columns)

theorem matrixAddTypeB_sound {left right output : MatrixType}
    (valid : matrixAddTypeB left right output = true) : matrixAddType left right output := by
  simpa [matrixAddTypeB, matrixAddType, Bool.and_eq_true] using valid

theorem matrixProductTypeB_sound {left right output : MatrixType}
    (valid : matrixProductTypeB left right output = true) : matrixProductType left right output := by
  simpa [matrixProductTypeB, matrixProductType, sameRing, Bool.and_eq_true, and_assoc] using valid

theorem preimageEquationTypeB_sound {source target preimage : MatrixType}
    (valid : preimageEquationTypeB source target preimage = true) :
    preimageEquationType source target preimage := by
  simpa [preimageEquationTypeB, preimageEquationType, sameRing, Bool.and_eq_true,
    and_assoc] using valid

def integerSelectorTypeB : WireType → Bool
  | .int | .constantInt => true
  | _ => false

def shapeExpression? (expression : Array StructuralIntExpr) : Option (List Nat) :=
  expression.toList.mapM fun item => match item.eval {} with
    | .ok value => if 0 < value then some value.toNat else none
    | .error _ => none

theorem shapeExpression?_eq_some_iff (expression : Array StructuralIntExpr) (shape : List Nat) :
    shapeExpression? expression = some shape ↔ shapeExpressionIs expression shape := Iff.rfl

theorem structuralExpressionIsNat_iff (expression : StructuralIntExpr) (value : Nat) :
    structuralExpressionIsNat expression value ↔
      expression.eval {} = .ok (Int.ofNat value) := Iff.rfl

def scalarIntegerTypeB : Option WireType → Bool
  | some .int | some .constantInt => true
  | _ => false

def scalarBooleanTypeB : Option WireType → Bool
  | some .bool | some .constantBool => true
  | _ => false

def structuralTypeCompatibleB (actual expected : WireType) : Bool :=
  decide (actual = expected) || match actual, expected with
    | .constantInt, .int | .constantBool, .bool | .constantReal, .real => true
    | _, _ => false

theorem structuralTypeCompatibleB_sound {actual expected : WireType}
    (valid : structuralTypeCompatibleB actual expected = true) :
    structuralTypeCompatible actual expected := by
  cases actual <;> cases expected <;>
    simp [structuralTypeCompatibleB, structuralTypeCompatible] at valid ⊢ <;> assumption

def optionalTypesCompatibleB (actual expected : List (Option WireType)) : Bool :=
  decide (actual.length = expected.length) &&
    (List.range actual.length).all fun index => match actual[index]?, expected[index]? with
      | some (some actualType), some (some expectedType) =>
          structuralTypeCompatibleB actualType expectedType
      | _, _ => false

theorem optionalTypesCompatibleB_sound {actual expected : List (Option WireType)}
    (valid : optionalTypesCompatibleB actual expected = true) :
    optionalTypesCompatible actual expected := by
  simp only [optionalTypesCompatibleB, Bool.and_eq_true] at valid
  refine ⟨of_decide_eq_true valid.1, ?_⟩
  intro index bound
  have checked := List.all_eq_true.mp valid.2 index (List.mem_range.mpr bound)
  cases actualAt : actual[index]? with
  | none => simp [actualAt] at checked
  | some actualOption => cases actualOption with
    | none => simp [actualAt] at checked
    | some actualType =>
      cases expectedAt : expected[index]? with
      | none => simp [actualAt, expectedAt] at checked
      | some expectedOption => cases expectedOption with
        | none => simp [actualAt, expectedAt] at checked
        | some expectedType =>
          refine ⟨actualType, expectedType, rfl, rfl, ?_⟩
          exact structuralTypeCompatibleB_sound (by simpa [actualAt, expectedAt] using checked)

def matrixConcatTypeB (axis : ConcatAxis) (inputs : List (Option WireType))
    (output : MatrixType) : Bool :=
  let matrices := inputs.mapM fun input => match input with
    | some (.matrix matrix) => some matrix
    | _ => none
  match matrices with
  | none | some [] => false
  | some (first :: rest) =>
      let sameRings := rest.all fun matrix =>
        decide (matrix.modulus = first.modulus ∧ matrix.ringDimension = first.ringDimension)
      let dimensions := match axis with
        | .rows => rest.all (fun matrix => decide (matrix.columns = first.columns)) &&
            decide (output.rows = ((first :: rest).map MatrixType.rows).sum ∧
              output.columns = first.columns)
        | .columns => rest.all (fun matrix => decide (matrix.rows = first.rows)) &&
            decide (output.rows = first.rows ∧
              output.columns = ((first :: rest).map MatrixType.columns).sum)
        | .diagonal => decide (output.rows = ((first :: rest).map MatrixType.rows).sum ∧
            output.columns = ((first :: rest).map MatrixType.columns).sum)
      sameRings && dimensions && decide (output.modulus = first.modulus ∧
        output.ringDimension = first.ringDimension)

def structuralEnvironments : List StructuralSlotDecl → List StructuralEnv
  | [] => [{}]
  | declaration :: rest =>
      (List.range declaration.upperBound.toNat).flatMap fun value =>
        (structuralEnvironments rest).map fun env =>
          { env with slots := env.slots.push (declaration.slot, Int.ofNat value) }

def rangeExtent? (slots : Array StructuralSlotDecl) (range : Option IntRange)
    (inputExtent : Nat) : Option Nat :=
  match range with
  | none => some inputExtent
  | some range =>
      let extents := (structuralEnvironments slots.toList).mapM fun env =>
        match range.start.eval env, range.stop.eval env with
        | .ok start, .ok stop =>
            if 0 ≤ start ∧ start ≤ stop ∧ stop ≤ Int.ofNat inputExtent then
              some (stop - start).toNat
            else none
        | _, _ => none
      match extents with
      | some (first :: rest) => if rest.all (· = first) then some first else none
      | _ => none

def staticFamilyIndicesB (indices : Array IndexMapExpr) (shape : List Nat) : Bool :=
  decide (indices.size = shape.length) && (List.range indices.size).all fun axis =>
    match indices[axis]?, shape[axis]? with
    | some index, some extent => match index.evalFuel {} 1024 with
        | .ok value => decide (0 ≤ value ∧ value < Int.ofNat extent)
        | .error _ => false
    | _, _ => false

def binaryIntegerTypesB (inputs : List (Option WireType)) (result : List WireType)
    (output : WireType) : Bool :=
  match inputs, result with
  | [left, right], [actual] =>
      scalarIntegerTypeB left && scalarIntegerTypeB right && decide (actual = output)
  | _, _ => false

def unaryIntegerTypesB (inputs : List (Option WireType)) (result : List WireType)
    (output : WireType) : Bool :=
  match inputs, result with
  | [input], [actual] => scalarIntegerTypeB input && decide (actual = output)
  | _, _ => false

def booleanToIntegerTypesB (inputs : List (Option WireType)) (result : List WireType) : Bool :=
  match inputs, result with
  | [input], [actual] => scalarBooleanTypeB input && decide (actual = .int)
  | _, _ => false

def concatOperationTypesB (axis : ConcatAxis) (inputs : List (Option WireType))
    (result : List WireType) : Bool :=
  match result with
  | [.matrix output] => matrixConcatTypeB axis inputs output
  | _ => false

def sliceOperationTypesB (slots : Array StructuralSlotDecl) (rows columns : Option IntRange)
    (inputs : List (Option WireType)) (result : List WireType) : Bool :=
  match inputs, result with
  | [some (.matrix input)], [.matrix output] =>
      match rangeExtent? slots rows input.rows, rangeExtent? slots columns input.columns with
      | some outputRows, some outputColumns =>
          decide (input.modulus = output.modulus ∧ input.ringDimension = output.ringDimension ∧
            output.rows = outputRows ∧ output.columns = outputColumns ∧
            0 < output.rows ∧ 0 < output.columns)
      | _, _ => false
  | _, _ => false

def familyReindexOperationTypesB (slots : Array StructuralSlotDecl)
    (shape : Array StructuralIntExpr) (map : IndexMap)
    (inputs : List (Option WireType)) (result : List WireType) : Bool :=
  match inputs, result with
  | [some (.family inputShape element)], [.family outputShape outputElement] =>
      match shapeExpression? shape with
      | some concrete => decide (concrete = outputShape ∧ outputElement = element ∧
          map.sourceRank = inputShape.length ∧ map.outputRank = outputShape.length ∧
          map.inputIndices.size = inputShape.length) && indexMapCheckedB map inputShape slots
      | none => false
  | _, _ => false

def selectOperationTypesB (count : StructuralIntExpr) (inputs : List (Option WireType))
    (result : List WireType) : Bool :=
  match count.eval {}, inputs, result with
  | .ok branchCount, some selector :: branches, [output] =>
      decide (0 < branchCount ∧ branches.length = branchCount.toNat) &&
        integerSelectorTypeB selector && branches.all (· == some output)
  | _, _, _ => false

def matrixBinaryOperationTypesB (operation : MatrixBinaryOp)
    (inputs : List (Option WireType)) (result : List WireType) : Bool :=
  match inputs, result with
  | [some (.matrix left), some (.matrix right)], [.matrix output] =>
      match operation with
      | .add | .subtract => matrixAddTypeB left right output
      | .multiply => matrixProductTypeB left right output
  | _, _ => false

def applyPreimageOperationTypesB (inputs : List (Option WireType))
    (result : List WireType) : Bool :=
  match inputs, result with
  | [some (.matrix left), some (.preimage right)], [.matrix output] =>
      matrixProductTypeB left right output
  | _, _ => false

def materializePreimageOperationTypesB (inputs : List (Option WireType))
    (result : List WireType) : Bool :=
  match inputs, result with
  | [some (.preimage input)], [.matrix output] => decide (input = output)
  | _, _ => false

def preimageBinaryOperationTypesB (operation : PreimageBinaryOp)
    (inputs : List (Option WireType)) (result : List WireType) : Bool :=
  match inputs, result with
  | [some (.preimage left), some right], [.preimage output] =>
      match operation, right with
      | .add, .preimage right => decide (left = right ∧ output = left)
      | .rightMultiplyExact, .matrix right => matrixProductTypeB left right output
      | .composeExactDecomposition, .preimage right => matrixProductTypeB left right output
      | _, _ => false
  | _, _ => false

def preimageSampleOperationTypesB (preimage : MatrixType)
    (inputs : List (Option WireType)) (result : List WireType) : Bool :=
  match inputs, result with
  | [some (.matrix source), some (.trapdoor trapdoor), some (.matrix target)],
      [.preimage output] =>
      decide (output = preimage ∧ trapdoor.matrix = source) &&
        preimageEquationTypeB source target preimage
  | _, _ => false

def familyPreimageSampleOperationTypesB (preimage : MatrixType)
    (inputs : List (Option WireType)) (result : List WireType) : Bool :=
  match inputs, result with
  | [some sourceType, some trapdoorType,
      some (.family shape (.matrix target))],
      [.family outputShape (.preimage output)] =>
      match matrixFamilyElement? sourceType, trapdoorFamilyElement? trapdoorType with
      | some (sourceShape, source), some (trapdoorShape, trapdoor) =>
          decide (output = preimage ∧ outputShape = shape ∧ sourceShape = trapdoorShape ∧
            shape.length = sourceShape.length + 1 ∧
            shape.take sourceShape.length = sourceShape ∧ trapdoor.matrix = source) &&
            preimageEquationTypeB source target preimage
      | _, _ => false
  | _, _ => false

def familySelectAxisOperationTypesB (axis : Nat) (inputs : List (Option WireType))
    (result : List WireType) : Bool :=
  match inputs, result with
  | [some (.family shape element), some selector], [output] =>
      match removeAt? shape axis with
      | none => false
      | some outputShape =>
          let selectorOK := integerSelectorTypeB selector || match selector with
            | .family selectorShape selectorElement =>
                decide (selectorShape = outputShape) && integerSelectorTypeB selectorElement
            | _ => false
          let expected := if outputShape = [] then element else .family outputShape element
          selectorOK && decide (output = expected)
  | _, _ => false

def matrixIdentityOperationTypesB (inputs : List (Option WireType))
    (result : List WireType) : Bool :=
  match inputs, result with
  | [some (.matrix input)], [.matrix output] => decide (input = output)
  | _, _ => false

def transposeOperationTypesB (inputs : List (Option WireType))
    (result : List WireType) : Bool :=
  match inputs, result with
  | [some (.matrix input)], [.matrix output] =>
      decide (input.modulus = output.modulus ∧
        input.ringDimension = output.ringDimension ∧ output.rows = input.columns ∧
        output.columns = input.rows)
  | _, _ => false

def gadgetDecomposeOperationTypesB (digits : StructuralIntExpr)
    (inputs : List (Option WireType)) (result : List WireType) : Bool :=
  match inputs, result, digits.eval {} with
  | [some (.matrix target)], [.preimage preimage], .ok count =>
      decide (0 < count ∧ target.modulus = preimage.modulus ∧
        target.ringDimension = preimage.ringDimension ∧
        preimage.rows = target.rows * count.toNat ∧ preimage.columns = target.columns)
  | _, _, _ => false

def noInputSingleOutputB (inputs : List (Option WireType)) (result : List WireType)
    (expected : Option WireType) : Bool :=
  match inputs, result, expected with
  | [], [_], none => true
  | [], [actual], some expected => decide (actual = expected)
  | _, _, _ => false

def hashSampleOperationTypesB (matrix : MatrixType) (inputs : List (Option WireType))
    (result : List WireType) : Bool :=
  decide (inputs = [some (.bytes 32)] ∧ result = [.matrix matrix])

def extractCoefficientOperationTypesB (position : StructuralIntExpr)
    (inputs : List (Option WireType)) (result : List WireType) : Bool :=
  match inputs, result, position.eval {} with
  | [some (.matrix input)], [.int], .ok value =>
      decide (input.rows = 1 ∧ input.columns = 1 ∧ 0 ≤ value ∧
        value < Int.ofNat input.ringDimension)
  | _, _, _ => false

def trapdoorSampleOperationTypesB (matrix : MatrixType) (sigma : RealExpr)
    (base digits bound : StructuralIntExpr) (inputs : List (Option WireType))
    (result : List WireType) : Bool :=
  match inputs, result with
  | [], [.matrix source, .trapdoor trapdoor] => decide (matrix = source ∧
      trapdoor.matrix = matrix ∧ trapdoor.sigma = sigma ∧ trapdoor.gadgetBase = base ∧
      trapdoor.digitCount = digits ∧ trapdoor.preimageMaxCoefficientBound = bound)
  | _, _ => false

def familyGetStaticOperationTypesB (indices : Array IndexMapExpr)
    (inputs : List (Option WireType)) (result : List WireType) : Bool :=
  match inputs, result with
  | [some (.family shape element)], [output] =>
      staticFamilyIndicesB indices shape && decide (output = element)
  | _, _ => false

def familyGetDynamicOperationTypesB (rank : Nat) (inputs : List (Option WireType))
    (result : List WireType) : Bool :=
  match inputs, result with
  | some (.family shape element) :: selectors, [output] =>
      decide (shape.length = rank ∧ selectors.length = rank ∧ output = element) &&
        selectors.all fun selector => match selector with
          | some wireType => integerSelectorTypeB wireType
          | none => false
  | _, _ => false

def familyGatherOperationTypesB (shape : Array StructuralIntExpr) (rank : Nat)
    (inputs : List (Option WireType)) (result : List WireType) : Bool :=
  match inputs, result, shapeExpression? shape with
  | some (.family inputShape element) :: selectors,
      [.family outputShape outputElement], some concrete =>
      decide (concrete = outputShape ∧ inputShape.length = rank ∧
        selectors.length = rank ∧ outputElement = element) &&
      selectors.all fun selector => match selector with
        | some (.family selectorShape selectorElement) =>
            decide (selectorShape = outputShape) && integerSelectorTypeB selectorElement
        | _ => false
  | _, _, _ => false

def familyPackOperationTypesB (shape : Array StructuralIntExpr)
    (inputs : List (Option WireType)) (result : List WireType) : Bool :=
  match result, shapeExpression? shape with
  | [.family outputShape element], some concrete =>
      decide (concrete = outputShape ∧ concrete ≠ [] ∧
        inputs.length = shapeProduct concrete) && inputs.all (· == some element) &&
        decide (familyElementType element = none)
  | _, _ => false

mutual
def operationTypesOKB (stage : Stage) (scope : Scope) (payload : NodePayload)
    (arguments : Array WireRef) (outputs : Array WireType) : Bool :=
  let inputs := referencedTypes scope arguments
  let result := outputs.toList
  match payload, inputs, result with
  | .input _, inputs, result | .artifactInput _, inputs, result =>
      noInputSingleOutputB inputs result none
  | .constantInt _, inputs, result | .evaluateInt _, inputs, result =>
      noInputSingleOutputB inputs result (some .constantInt)
  | .constantBool _, inputs, result => noInputSingleOutputB inputs result (some .constantBool)
  | .intBinary _, inputs, result => binaryIntegerTypesB inputs result .int
  | .intCompare _, inputs, result => binaryIntegerTypesB inputs result .bool
  | .bitExtract _, inputs, result => unaryIntegerTypesB inputs result .bool
  | .boolToInt, inputs, result => booleanToIntegerTypesB inputs result
  | .constantMatrix matrix _, inputs, result |
      .uniformIntervalSample matrix _, inputs, result |
      .gaussianSample matrix _ _, inputs, result =>
      noInputSingleOutputB inputs result (some (.matrix matrix))
  | .hashSample matrix _ _ _ _, inputs, result => hashSampleOperationTypesB matrix inputs result
  | .trapdoorSample matrix sigma base digits bound, inputs, result =>
      trapdoorSampleOperationTypesB matrix sigma base digits bound inputs result
  | .matrixBinary operation, inputs, result =>
      matrixBinaryOperationTypesB operation inputs result
  | .matrixNegate, inputs, result => matrixIdentityOperationTypesB inputs result
  | .matrixScale _, inputs, result => matrixIdentityOperationTypesB inputs result
  | .transpose, inputs, result => transposeOperationTypesB inputs result
  | .concat axis, inputs, result => concatOperationTypesB axis inputs result
  | .slice rows columns, inputs, result =>
      sliceOperationTypesB scope.structuralSlots rows columns inputs result
  | .extractCoefficient position _, inputs, result =>
      extractCoefficientOperationTypesB position inputs result
  | .preimageSample preimage _, inputs, result =>
      preimageSampleOperationTypesB preimage inputs result
  | .familyPreimageSample preimage _, inputs, result =>
      familyPreimageSampleOperationTypesB preimage inputs result
  | .applyPreimage, inputs, result => applyPreimageOperationTypesB inputs result
  | .materializePreimageExact, inputs, result =>
      materializePreimageOperationTypesB inputs result
  | .preimageBinary operation, inputs, result =>
      preimageBinaryOperationTypesB operation inputs result
  | .gadgetDecompose _ _ digits, inputs, result =>
      gadgetDecomposeOperationTypesB digits inputs result
  | .familyPack shape, inputs, result => familyPackOperationTypesB shape inputs result
  | .familyGetStatic indices, inputs, result => familyGetStaticOperationTypesB indices inputs result
  | .familyGetDynamic rank, inputs, result => familyGetDynamicOperationTypesB rank inputs result
  | .familySelectAxis axis, inputs, result =>
      familySelectAxisOperationTypesB axis inputs result
  | .familyReindex shape map, inputs, result =>
      familyReindexOperationTypesB scope.structuralSlots shape map inputs result
  | .familyGather shape rank, inputs, result =>
      familyGatherOperationTypesB shape rank inputs result
  | .select count, inputs, result => selectOperationTypesB count inputs result
  | .subgraphCall _, _, _ =>
      structuralChildTypesOKB stage scope payload arguments outputs
  | .sequentialLoop _, _, _ =>
      structuralChildTypesOKB stage scope payload arguments outputs
  | .parallelGrid _, _, _ =>
      structuralChildTypesOKB stage scope payload arguments outputs
  | _, _, _ => false

def structuralChildTypesOKB (stage : Stage) (scope : Scope) (payload : NodePayload)
    (arguments : Array WireRef) (outputs : Array WireType) : Bool :=
  match payload with
  | .subgraphCall call => match scopeAt stage call.child with
      | none => false
      | some child => decide (call.canonicalInputExclusiveUppers.size = arguments.size) &&
          optionalTypesCompatibleB (referencedTypes scope arguments)
            (referencedTypes child child.inputs) &&
          optionalTypesCompatibleB (referencedTypes child child.outputs) (outputs.toList.map some)
  | .sequentialLoop loop => match scopeAt stage loop.child, loop.count.eval {} with
      | some child, .ok count =>
          decide (0 ≤ count ∧ loop.carriedCount > 0 ∧ loop.carriedCount ≤ arguments.size ∧
            child.inputs.size = arguments.size ∧ child.outputs.size = loop.carriedCount ∧
            outputs.size = loop.carriedCount) &&
          optionalTypesCompatibleB (referencedTypes scope arguments)
            (referencedTypes child child.inputs) &&
          optionalTypesCompatibleB (referencedTypes child child.outputs)
            (outputs.toList.map some) &&
          optionalTypesCompatibleB
            ((referencedTypes scope arguments).take loop.carriedCount)
            ((referencedTypes child child.outputs).take loop.carriedCount) &&
          child.structuralSlots.any fun declaration =>
            decide (declaration.slot = loop.indexSlot ∧ declaration.kind = .sequentialIteration ∧
              declaration.upperBound = count)
      | _, _ => false
  | .parallelGrid grid => match scopeAt stage grid.child, shapeExpression? grid.shape with
      | some child, some shape =>
          decide (grid.inputModes.size = arguments.size ∧ child.inputs.size = arguments.size ∧
            child.outputs.size = outputs.size ∧ grid.indexSlots.size = shape.length) &&
          (List.range arguments.size).all (fun index =>
            match (referencedTypes scope arguments)[index]?,
                (referencedTypes child child.inputs)[index]?, grid.inputModes[index]? with
            | some (some outer), some (some inner), some { reindex := false, map := none } =>
                decide (outer = inner)
            | some (some (.family inputShape outer)), some (some inner),
                some { reindex := true, map := some map } =>
                decide (outer = inner) &&
                  decide (map.outputRank = shape.length ∧ map.sourceRank = inputShape.length ∧
                    map.inputIndices.size = inputShape.length) &&
                  indexMapCheckedB map inputShape #[]
            | _, _, _ => false) &&
          (List.range outputs.size).all (fun index =>
            match (referencedTypes child child.outputs)[index]?, outputs[index]? with
            | some (some childOutput), some (.family outputShape outputElement) =>
                decide (outputShape = shape ∧ childOutput = outputElement)
            | _, _ => false) &&
          (List.range shape.length).all (fun axis =>
            match grid.indexSlots[axis]?, shape[axis]? with
            | some slot, some extent => child.structuralSlots.any fun declaration =>
                decide (declaration.slot = slot ∧ declaration.kind = .gridAxis axis ∧
                  declaration.upperBound = Int.ofNat extent)
            | _, _ => false)
      | _, _ => false
  | _ => false
end

theorem matrixConcatTypeB_sound {axis : ConcatAxis} {inputs : List (Option WireType)}
    {output : MatrixType} (valid : matrixConcatTypeB axis inputs output = true) :
    matrixConcatType axis inputs output := by
  change (match inputs.mapM (fun input => match input with
    | some (.matrix matrix) => some matrix
    | _ => none) with
    | none | some [] => false
    | some (first :: rest) =>
        let sameRings := rest.all fun matrix =>
          decide (matrix.modulus = first.modulus ∧ matrix.ringDimension = first.ringDimension)
        let dimensions := match axis with
          | .rows => rest.all (fun matrix => decide (matrix.columns = first.columns)) &&
              decide (output.rows = ((first :: rest).map MatrixType.rows).sum ∧
                output.columns = first.columns)
          | .columns => rest.all (fun matrix => decide (matrix.rows = first.rows)) &&
              decide (output.rows = first.rows ∧
                output.columns = ((first :: rest).map MatrixType.columns).sum)
          | .diagonal => decide (output.rows = ((first :: rest).map MatrixType.rows).sum ∧
              output.columns = ((first :: rest).map MatrixType.columns).sum)
        sameRings && dimensions && decide (output.modulus = first.modulus ∧
          output.ringDimension = first.ringDimension)) = true at valid
  cases h : inputs.mapM (fun input => match input with
    | some (.matrix matrix) => some matrix
    | _ => none) with
  | none => simp [h] at valid
  | some matrices => cases matrices with
    | nil => simp [h] at valid
    | cons first rest =>
      simp [h, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at valid
      refine ⟨first, rest, h, ?_, ⟨valid.2.1.symm, valid.2.2.symm⟩, ?_⟩
      · intro matrix matrixIn
        exact ⟨(valid.1.1 matrix matrixIn).1.symm, (valid.1.1 matrix matrixIn).2.symm⟩
      · cases axis with
        | rows =>
          simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at valid
          exact ⟨fun matrix hm => valid.1.2.1 matrix hm, valid.1.2.2.1, valid.1.2.2.2⟩
        | columns =>
          simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at valid
          exact ⟨fun matrix hm => valid.1.2.1 matrix hm, valid.1.2.2.1, valid.1.2.2.2⟩
        | diagonal =>
          simp only [Bool.and_eq_true, decide_eq_true_eq] at valid
          exact ⟨valid.1.2.1, valid.1.2.2⟩

theorem structuralEnvironments_eq (slots : List StructuralSlotDecl) :
    structuralEnvironments slots = declarativeStructuralEnvironments slots := by
  induction slots with
  | nil => rfl
  | cons declaration rest ih =>
      simp only [structuralEnvironments, declarativeStructuralEnvironments, ih]

theorem mapM_length {α β : Type} (f : α → Option β) :
    ∀ {xs : List α} {ys : List β}, xs.mapM f = some ys → ys.length = xs.length := by
  intro xs
  induction xs with
  | nil => intro ys h; simp only [List.mapM_nil] at h; cases h; rfl
  | cons x xs ih =>
      intro ys h
      rw [List.mapM_cons] at h
      cases hx : f x with
      | none => simp [hx] at h
      | some y =>
          simp only [hx] at h
          cases hr : xs.mapM f with
          | none => simp [hr] at h
          | some ys' =>
              simp only [hr] at h
              cases h
              simp only [List.length_cons]
              congr 1
              exact ih hr

theorem all_eq_replicate {α : Type} [DecidableEq α] {xs : List α} {value : α}
    (h : ∀ x ∈ xs, x = value) : xs = List.replicate xs.length value := by
  induction xs with
  | nil => simp
  | cons head tail ih =>
      have headEq := h head (by simp)
      have tailAll : ∀ x ∈ tail, x = value := by
        intro x hx
        exact h x (by simp [hx])
      simp only [List.replicate_succ, List.length_cons]
      rw [headEq, ih tailAll]
      simp

theorem rangeExtentB_sound {slots : Array StructuralSlotDecl} {range : Option IntRange}
    {inputExtent outputExtent : Nat} (valid : rangeExtent? slots range inputExtent = some outputExtent) :
    rangeExtentIs slots range inputExtent outputExtent := by
  cases range with
  | none =>
      have h : inputExtent = outputExtent := by simpa [rangeExtent?, rangeExtentIs] using valid
      exact h.symm
  | some range =>
      simp only [rangeExtent?, rangeExtentIs, structuralEnvironments_eq] at valid ⊢
      let envs := declarativeStructuralEnvironments slots.toList
      let f : StructuralEnv → Option Nat := fun env =>
        match range.start.eval env, range.stop.eval env with
        | .ok start, .ok stop =>
            if 0 ≤ start ∧ start ≤ stop ∧ stop ≤ Int.ofNat inputExtent then
              some (stop - start).toNat else none
        | _, _ => none
      change (match envs.mapM f with
        | some (first :: rest) =>
            if rest.all (fun x => decide (x = first)) = true then some first else none
        | _ => none) = some outputExtent at valid
      change envs.mapM f = some (List.replicate envs.length outputExtent)
      cases h : envs.mapM f with
      | none => simp [h] at valid
      | some values => cases values with
        | nil => simp [h] at valid
        | cons first rest =>
          simp [h] at valid
          have restEq := all_eq_replicate valid.1
          have lengths := mapM_length f h
          have lengthEq : rest.length + 1 = envs.length := by simpa using lengths
          rw [restEq, ← lengthEq, List.replicate_succ, valid.2]

theorem concatOperationTypesB_sound (axis : ConcatAxis) (inputs : List (Option WireType))
    (result : List WireType) (valid : concatOperationTypesB axis inputs result = true) :
    ∃ output, result = [.matrix output] ∧ matrixConcatType axis inputs output := by
  cases result with
  | nil => simp [concatOperationTypesB] at valid
  | cons output tail => cases tail with
    | cons _ _ => simp [concatOperationTypesB] at valid
    | nil =>
      cases output with
      | matrix outputType => exact ⟨outputType, rfl, matrixConcatTypeB_sound valid⟩
      | _ => simp [concatOperationTypesB] at valid

theorem sliceOperationTypesB_sound (slots : Array StructuralSlotDecl)
    (rows columns : Option IntRange) (inputs : List (Option WireType)) (result : List WireType)
    (valid : sliceOperationTypesB slots rows columns inputs result = true) :
    ∃ input output outputRows outputColumns,
      inputs = [some (.matrix input)] ∧ result = [.matrix output] ∧
      rangeExtentIs slots rows input.rows outputRows ∧
      rangeExtentIs slots columns input.columns outputColumns ∧
      sameRing input output ∧ output.rows = outputRows ∧ output.columns = outputColumns ∧
      0 < output.rows ∧ 0 < output.columns := by
  cases inputs with
  | nil => simp [sliceOperationTypesB] at valid
  | cons first tail => cases tail with
    | cons _ _ => simp [sliceOperationTypesB] at valid
    | nil => cases first with
      | none => simp [sliceOperationTypesB] at valid
      | some firstType => cases firstType with
        | matrix input => cases result with
          | nil => simp [sliceOperationTypesB] at valid
          | cons output tail => cases tail with
            | cons _ _ => simp [sliceOperationTypesB] at valid
            | nil => cases output with
              | matrix outputType =>
                cases rowsStored : rangeExtent? slots rows input.rows with
                | none => simp [sliceOperationTypesB, rowsStored] at valid
                | some outputRows =>
                  cases columnsStored : rangeExtent? slots columns input.columns with
                  | none => simp [sliceOperationTypesB, rowsStored, columnsStored] at valid
                  | some outputColumns =>
                    simp only [sliceOperationTypesB, rowsStored, columnsStored,
                      Bool.and_eq_true, decide_eq_true_eq] at valid
                    exact ⟨input, outputType, outputRows, outputColumns, rfl, rfl,
                      rangeExtentB_sound rowsStored, rangeExtentB_sound columnsStored,
                      ⟨valid.1, valid.2.1⟩, valid.2.2.1, valid.2.2.2.1,
                      valid.2.2.2.2.1, valid.2.2.2.2.2⟩
              | _ => simp [sliceOperationTypesB] at valid
        | _ => simp [sliceOperationTypesB] at valid

theorem concatTypesB_sound (stage : Stage) (scope : Scope) (axis : ConcatAxis)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.concat axis) arguments outputs = true) :
    operationTypesOK stage scope (.concat axis) arguments outputs := by
  simp only [operationTypesOKB] at valid
  exact concatOperationTypesB_sound _ _ _ valid

theorem sliceTypesB_sound (stage : Stage) (scope : Scope) (rows columns : Option IntRange)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.slice rows columns) arguments outputs = true) :
    operationTypesOK stage scope (.slice rows columns) arguments outputs := by
  simp only [operationTypesOKB] at valid
  exact sliceOperationTypesB_sound _ _ _ _ _ valid

theorem familyReindexTypesB_sound (stage : Stage) (scope : Scope)
    (shape : Array StructuralIntExpr) (map : IndexMap)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.familyReindex shape map) arguments outputs = true) :
    operationTypesOK stage scope (.familyReindex shape map) arguments outputs := by
  simp only [operationTypesOKB] at valid
  simp only [familyReindexOperationTypesB] at valid
  split at valid <;> try {simp_all [operationTypesOK]}
  case h_1 =>
    rename_i inputShape element outputShape outputElement inputsStored resultStored
    cases shapeStored : shapeExpression? shape with
    | none => simp [shapeStored] at valid
    | some concrete =>
      simp only [shapeStored, Bool.and_eq_true, decide_eq_true_eq] at valid
      rcases valid with ⟨fields, axes⟩
      rcases fields with ⟨shapeEq, elementEq, sourceRank, outputRank, indicesSize⟩
      have shapeValid : shapeExpressionIs shape concrete :=
        (shapeExpression?_eq_some_iff _ _).mp shapeStored
      exact ⟨inputShape, outputShape, element, inputsStored,
        shapeEq ▸ shapeValid, sourceRank, outputRank,
        indicesSize, axes, by simpa [shapeEq, elementEq] using resultStored⟩

theorem integerSelectorTypeB_sound (value : WireType)
    (valid : integerSelectorTypeB value = true) : integerSelectorType value := by
  cases value <;> simp [integerSelectorTypeB, integerSelectorType] at valid ⊢

theorem allIntegerTypesB_sound : ∀ selectors : List (Option WireType),
    selectors.all (fun selector => match selector with
      | some wireType => integerSelectorTypeB wireType | none => false) = true →
    allIntegerTypes selectors := by
  intro selectors
  induction selectors with
  | nil => simp [allIntegerTypes]
  | cons selector rest ih =>
      cases selector with
      | none => simp
      | some wireType =>
        simp only [List.all_cons, Bool.and_eq_true]
        intro valid
        exact ⟨integerSelectorTypeB_sound wireType valid.1, ih valid.2⟩

theorem familyGetStaticTypesB_sound (stage : Stage) (scope : Scope)
    (indices : Array IndexMapExpr) (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.familyGetStatic indices) arguments outputs = true) :
    operationTypesOK stage scope (.familyGetStatic indices) arguments outputs := by
  simp only [operationTypesOKB, familyGetStaticOperationTypesB] at valid
  split at valid <;> try {simp at valid}
  rename_i shape element output inputsStored resultStored
  simp only [Bool.and_eq_true, staticFamilyIndicesB, decide_eq_true_eq] at valid
  exact ⟨shape, element, inputsStored, valid.1.1, by simpa [valid.2] using resultStored⟩

theorem familyPackTypesB_sound (stage : Stage) (scope : Scope)
    (shape : Array StructuralIntExpr) (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.familyPack shape) arguments outputs = true) :
    operationTypesOK stage scope (.familyPack shape) arguments outputs := by
  simp only [operationTypesOKB, familyPackOperationTypesB] at valid
  split at valid <;> try {simp at valid}
  rename_i outputShape element concrete resultStored shapeStored
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at valid
  rcases valid with ⟨⟨⟨shapeEq, nonempty, membersLength⟩, membersValid⟩, elementValid⟩
  refine ⟨outputShape, element, (shapeExpression?_eq_some_iff _ _).mp (shapeEq ▸ shapeStored),
    shapeEq ▸ nonempty, shapeEq ▸ membersLength, ?_, elementValid, resultStored⟩
  intro member memberIn
  exact beq_iff_eq.mp (membersValid member memberIn)

theorem familyGetDynamicTypesB_sound (stage : Stage) (scope : Scope) (rank : Nat)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.familyGetDynamic rank) arguments outputs = true) :
    operationTypesOK stage scope (.familyGetDynamic rank) arguments outputs := by
  simp only [operationTypesOKB, familyGetDynamicOperationTypesB] at valid
  split at valid <;> try {simp at valid}
  rename_i shape element selectors output inputsStored resultStored
  simp only [Bool.and_eq_true, decide_eq_true_eq] at valid
  exact ⟨shape, element, selectors, inputsStored, valid.1.2.1,
    allIntegerTypesB_sound selectors valid.2, valid.1.1,
    by simpa [valid.1.2.2] using resultStored⟩

theorem familyGatherTypesB_sound (stage : Stage) (scope : Scope)
    (shape : Array StructuralIntExpr) (rank : Nat)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.familyGather shape rank) arguments outputs = true) :
    operationTypesOK stage scope (.familyGather shape rank) arguments outputs := by
  simp only [operationTypesOKB, familyGatherOperationTypesB] at valid
  split at valid <;> try {simp at valid}
  rename_i inputShape element selectors outputShape outputElement concrete
    inputsStored resultStored shapeStored
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at valid
  rcases valid with ⟨⟨shapeEq, inputRank, selectorLength, elementEq⟩, selectorsValid⟩
  refine ⟨inputShape, outputShape, element,
    (shapeExpression?_eq_some_iff _ _).mp (shapeEq ▸ shapeStored), inputRank,
    selectors, inputsStored, selectorLength, ?_, ?_⟩
  · intro selector member
    have checked := selectorsValid selector member
    cases selector with
    | none => simp at checked
    | some selectorType => cases selectorType with
      | family selectorShape selectorElement =>
        simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
        exact ⟨selectorElement, by simp [checked.1],
          integerSelectorTypeB_sound selectorElement checked.2⟩
      | _ => simp at checked
  · rw [elementEq] at resultStored
    exact resultStored

theorem familySelectAxisTypesB_sound (stage : Stage) (scope : Scope) (axis : Nat)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.familySelectAxis axis) arguments outputs = true) :
    operationTypesOK stage scope (.familySelectAxis axis) arguments outputs := by
  simp only [operationTypesOKB, familySelectAxisOperationTypesB] at valid
  split at valid <;> try {simp at valid}
  rename_i shape element selector output inputsStored resultStored
  cases shapeStored : removeAt? shape axis with
  | none => simp [shapeStored] at valid
  | some outputShape =>
    simp only [shapeStored, Bool.and_eq_true, decide_eq_true_eq] at valid
    refine ⟨shape, element, selector, outputShape, inputsStored, shapeStored, ?_, ?_⟩
    · rcases Bool.or_eq_true_iff.mp valid.1 with selectorValid | familyValid
      · exact Or.inl (integerSelectorTypeB_sound selector selectorValid)
      · cases selector <;> simp at familyValid
        case family selectorShape selectorElement =>
          exact Or.inr ⟨selectorElement, familyValid.1 ▸ rfl,
            integerSelectorTypeB_sound selectorElement familyValid.2⟩
    · simpa [valid.2] using resultStored

theorem selectTypesB_sound (stage : Stage) (scope : Scope) (count : StructuralIntExpr)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.select count) arguments outputs = true) :
    operationTypesOK stage scope (.select count) arguments outputs := by
  simp only [operationTypesOKB, selectOperationTypesB] at valid
  split at valid <;> try {simp at valid}
  rename_i branchCount selector branches output countStored inputsStored resultStored
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at valid
  rcases valid with ⟨⟨⟨positive, branchesLength⟩, selectorValid⟩, branchesValid⟩
  have countEq : Int.ofNat branchCount.toNat = branchCount :=
    Int.toNat_of_nonneg (Int.le_of_lt positive)
  have branchesEq : branches = List.replicate branchCount.toNat (some output) := by
    rw [← branchesLength]
    apply all_eq_replicate
    intro branch member
    exact beq_iff_eq.mp (branchesValid branch member)
  exact ⟨branchCount.toNat, output, selector,
    by simpa only [structuralExpressionIsNat, countEq] using countStored,
    by omega, integerSelectorTypeB_sound selector selectorValid,
    by simpa [inputsStored, branchesEq], by simpa using resultStored⟩

theorem matrixBinaryTypesB_sound (stage : Stage) (scope : Scope)
    (operation : MatrixBinaryOp) (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.matrixBinary operation) arguments outputs = true) :
    operationTypesOK stage scope (.matrixBinary operation) arguments outputs := by
  simp only [operationTypesOKB, matrixBinaryOperationTypesB] at valid
  split at valid <;> try {simp at valid}
  rename_i left right output inputsStored resultStored
  refine ⟨left, right, output, inputsStored, resultStored, ?_⟩
  cases operation with
  | add => exact matrixAddTypeB_sound valid
  | subtract => exact matrixAddTypeB_sound valid
  | multiply => exact matrixProductTypeB_sound valid

theorem matrixIdentityOperationTypesB_sound (inputs : List (Option WireType))
    (result : List WireType) (valid : matrixIdentityOperationTypesB inputs result = true) :
    ∃ matrix, inputs = [some (.matrix matrix)] ∧ result = [.matrix matrix] := by
  simp only [matrixIdentityOperationTypesB] at valid
  split at valid <;> simp_all

theorem matrixNegateTypesB_sound (stage : Stage) (scope : Scope)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope .matrixNegate arguments outputs = true) :
    operationTypesOK stage scope .matrixNegate arguments outputs := by
  simp only [operationTypesOKB] at valid
  exact matrixIdentityOperationTypesB_sound _ _ valid

theorem matrixScaleTypesB_sound (stage : Stage) (scope : Scope) (scalar : StructuralIntExpr)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.matrixScale scalar) arguments outputs = true) :
    operationTypesOK stage scope (.matrixScale scalar) arguments outputs := by
  simp only [operationTypesOKB] at valid
  exact matrixIdentityOperationTypesB_sound _ _ valid

theorem transposeTypesB_sound (stage : Stage) (scope : Scope)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope .transpose arguments outputs = true) :
    operationTypesOK stage scope .transpose arguments outputs := by
  simp only [operationTypesOKB, transposeOperationTypesB] at valid
  split at valid <;> try {simp at valid}
  rename_i input output inputsStored resultStored
  simp only [decide_eq_true_eq] at valid
  exact ⟨input, output, inputsStored, resultStored, ⟨valid.1, valid.2.1⟩,
    valid.2.2.1, valid.2.2.2⟩

theorem gadgetDecomposeTypesB_sound (stage : Stage) (scope : Scope)
    (base : StructuralIntExpr) (small : Bool) (digits : StructuralIntExpr)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.gadgetDecompose base small digits)
      arguments outputs = true) :
    operationTypesOK stage scope (.gadgetDecompose base small digits) arguments outputs := by
  simp only [operationTypesOKB, gadgetDecomposeOperationTypesB] at valid
  split at valid <;> try {simp at valid}
  rename_i target preimage count inputsStored resultStored countStored
  simp only [decide_eq_true_eq] at valid
  have countEq : Int.ofNat count.toNat = count :=
    Int.toNat_of_nonneg (Int.le_of_lt valid.1)
  exact ⟨target, preimage, count.toNat, inputsStored, resultStored,
    by simpa only [structuralExpressionIsNat, countEq] using countStored,
    by omega, ⟨valid.2.1, valid.2.2.1⟩, valid.2.2.2.1, valid.2.2.2.2⟩

theorem noInputAnyOutputB_sound (inputs : List (Option WireType)) (result : List WireType)
    (valid : noInputSingleOutputB inputs result none = true) :
    ∃ output, inputs = [] ∧ result = [output] := by
  simp only [noInputSingleOutputB] at valid
  split at valid <;> simp_all

theorem noInputExactOutputB_sound (inputs : List (Option WireType)) (result : List WireType)
    (expected : WireType) (valid : noInputSingleOutputB inputs result (some expected) = true) :
    inputs = [] ∧ result = [expected] := by
  simp only [noInputSingleOutputB] at valid
  split at valid <;> simp_all

theorem inputTypesB_sound (stage : Stage) (scope : Scope) (index : Nat)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.input index) arguments outputs = true) :
    operationTypesOK stage scope (.input index) arguments outputs := by
  simp only [operationTypesOKB] at valid
  obtain ⟨output, inputs, result⟩ := noInputAnyOutputB_sound _ _ valid
  exact ⟨inputs, by simpa using congrArg List.length result⟩

theorem artifactInputTypesB_sound (stage : Stage) (scope : Scope) (input : ArtifactInput)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.artifactInput input) arguments outputs = true) :
    operationTypesOK stage scope (.artifactInput input) arguments outputs := by
  simp only [operationTypesOKB] at valid
  obtain ⟨output, inputs, result⟩ := noInputAnyOutputB_sound _ _ valid
  exact ⟨inputs, by simpa using congrArg List.length result⟩

theorem constantIntTypesB_sound (stage : Stage) (scope : Scope) (value : Int)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.constantInt value) arguments outputs = true) :
    operationTypesOK stage scope (.constantInt value) arguments outputs := by
  simp only [operationTypesOKB] at valid
  exact noInputExactOutputB_sound _ _ .constantInt valid

theorem evaluateIntTypesB_sound (stage : Stage) (scope : Scope) (value : StructuralIntExpr)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.evaluateInt value) arguments outputs = true) :
    operationTypesOK stage scope (.evaluateInt value) arguments outputs := by
  simp only [operationTypesOKB] at valid
  exact noInputExactOutputB_sound _ _ .constantInt valid

theorem constantBoolTypesB_sound (stage : Stage) (scope : Scope) (value : Bool)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.constantBool value) arguments outputs = true) :
    operationTypesOK stage scope (.constantBool value) arguments outputs := by
  simp only [operationTypesOKB] at valid
  exact noInputExactOutputB_sound _ _ .constantBool valid

theorem constantMatrixTypesB_sound (stage : Stage) (scope : Scope) (matrix : MatrixType)
    (literal : MatrixLiteral) (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.constantMatrix matrix literal) arguments outputs = true) :
    operationTypesOK stage scope (.constantMatrix matrix literal) arguments outputs := by
  simp only [operationTypesOKB] at valid
  exact noInputExactOutputB_sound _ _ (.matrix matrix) valid

theorem uniformIntervalSampleTypesB_sound (stage : Stage) (scope : Scope) (matrix : MatrixType)
    (range : IntRange) (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.uniformIntervalSample matrix range)
      arguments outputs = true) :
    operationTypesOK stage scope (.uniformIntervalSample matrix range) arguments outputs := by
  simp only [operationTypesOKB] at valid
  exact noInputExactOutputB_sound _ _ (.matrix matrix) valid

theorem gaussianSampleTypesB_sound (stage : Stage) (scope : Scope) (matrix : MatrixType)
    (sigma : RealExpr) (bound : StructuralIntExpr)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.gaussianSample matrix sigma bound)
      arguments outputs = true) :
    operationTypesOK stage scope (.gaussianSample matrix sigma bound) arguments outputs := by
  simp only [operationTypesOKB] at valid
  exact noInputExactOutputB_sound _ _ (.matrix matrix) valid

theorem hashSampleTypesB_sound (stage : Stage) (scope : Scope) (matrix : MatrixType)
    (tagPrefix : List UInt8) (tags decimals words : Array StructuralIntExpr)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.hashSample matrix tagPrefix tags decimals words)
      arguments outputs = true) :
    operationTypesOK stage scope (.hashSample matrix tagPrefix tags decimals words) arguments outputs := by
  simp only [operationTypesOKB, hashSampleOperationTypesB, decide_eq_true_eq] at valid
  exact valid

theorem extractCoefficientTypesB_sound (stage : Stage) (scope : Scope)
    (position : StructuralIntExpr) (upper : Option Nat)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.extractCoefficient position upper)
      arguments outputs = true) :
    operationTypesOK stage scope (.extractCoefficient position upper) arguments outputs := by
  simp only [operationTypesOKB, extractCoefficientOperationTypesB] at valid
  split at valid <;> try {simp at valid}
  rename_i input value inputsStored resultStored positionStored
  simp only [decide_eq_true_eq] at valid
  exact ⟨input, value, inputsStored, resultStored, positionStored,
    valid.1, valid.2.1, valid.2.2.1, valid.2.2.2⟩

theorem trapdoorSampleTypesB_sound (stage : Stage) (scope : Scope) (matrix : MatrixType)
    (sigma : RealExpr) (base digits bound : StructuralIntExpr)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.trapdoorSample matrix sigma base digits bound)
      arguments outputs = true) :
    operationTypesOK stage scope (.trapdoorSample matrix sigma base digits bound)
      arguments outputs := by
  simp only [operationTypesOKB, trapdoorSampleOperationTypesB] at valid
  split at valid <;> try {simp at valid}
  rename_i source trapdoor inputsStored resultStored
  simp only [decide_eq_true_eq] at valid
  rw [← valid.1] at resultStored
  exact ⟨trapdoor, inputsStored, resultStored, valid.2.1, valid.2.2.1,
    valid.2.2.2.1, valid.2.2.2.2.1, valid.2.2.2.2.2⟩

theorem applyPreimageTypesB_sound (stage : Stage) (scope : Scope)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope .applyPreimage arguments outputs = true) :
    operationTypesOK stage scope .applyPreimage arguments outputs := by
  simp only [operationTypesOKB, applyPreimageOperationTypesB] at valid
  split at valid <;> try {simp at valid}
  rename_i left preimage output inputsStored resultStored
  exact ⟨left, preimage, output, inputsStored, resultStored, matrixProductTypeB_sound valid⟩

theorem materializePreimageTypesB_sound (stage : Stage) (scope : Scope)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope .materializePreimageExact arguments outputs = true) :
    operationTypesOK stage scope .materializePreimageExact arguments outputs := by
  simp only [operationTypesOKB, materializePreimageOperationTypesB] at valid
  split at valid <;> try {simp at valid}
  rename_i input output inputsStored resultStored
  have equal : input = output := of_decide_eq_true valid
  exact ⟨input, inputsStored, equal ▸ resultStored⟩

theorem preimageBinaryTypesB_sound (stage : Stage) (scope : Scope)
    (operation : PreimageBinaryOp) (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.preimageBinary operation) arguments outputs = true) :
    operationTypesOK stage scope (.preimageBinary operation) arguments outputs := by
  simp only [operationTypesOKB, preimageBinaryOperationTypesB] at valid
  split at valid <;> try {simp at valid}
  rename_i left right output inputsStored resultStored
  refine ⟨left, right, output, inputsStored, resultStored, ?_⟩
  cases operation <;> cases right <;> simp [preimageBinaryOperationTypesB] at valid ⊢
  · exact ⟨valid.1.symm, valid.2⟩
  · exact matrixProductTypeB_sound valid
  · exact matrixProductTypeB_sound valid

theorem preimageSampleTypesB_sound (stage : Stage) (scope : Scope)
    (preimage : MatrixType) (bound : StructuralIntExpr)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.preimageSample preimage bound) arguments outputs = true) :
    operationTypesOK stage scope (.preimageSample preimage bound) arguments outputs := by
  simp only [operationTypesOKB, preimageSampleOperationTypesB] at valid
  split at valid <;> try {simp at valid}
  rename_i source trapdoor target output inputsStored resultStored
  simp only [Bool.and_eq_true, decide_eq_true_eq] at valid
  exact ⟨source, trapdoor, target, inputsStored, by simpa [valid.1.1] using resultStored,
    valid.1.2, preimageEquationTypeB_sound valid.2⟩

theorem familyPreimageSampleTypesB_sound (stage : Stage) (scope : Scope)
    (preimage : MatrixType) (bound : StructuralIntExpr)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.familyPreimageSample preimage bound)
      arguments outputs = true) :
    operationTypesOK stage scope (.familyPreimageSample preimage bound) arguments outputs := by
  simp only [operationTypesOKB, familyPreimageSampleOperationTypesB] at valid
  split at valid <;> try {simp at valid}
  rename_i sourceType trapdoorType targetShape target outputShape output inputsStored resultStored
  cases sourceStored : matrixFamilyElement? sourceType with
  | none => simp [sourceStored] at valid
  | some sourcePair =>
    rcases sourcePair with ⟨sourceShape, source⟩
    cases trapdoorStored : trapdoorFamilyElement? trapdoorType with
    | none => simp [sourceStored, trapdoorStored] at valid
    | some trapdoorPair =>
      rcases trapdoorPair with ⟨trapdoorShape, trapdoor⟩
      simp only [sourceStored, trapdoorStored, Bool.and_eq_true, decide_eq_true_eq] at valid
      rcases valid with ⟨⟨outputEq, outputShapeEq, shapesEq, rankEq, prefixEq, trapdoorEq⟩,
        equation⟩
      exact ⟨sourceType, trapdoorType, sourceShape, trapdoorShape, targetShape, outputShape,
        source, trapdoor, target, output, inputsStored, sourceStored, trapdoorStored,
        resultStored, outputEq, outputShapeEq, shapesEq, rankEq, prefixEq, trapdoorEq,
        preimageEquationTypeB_sound equation⟩

theorem scalarIntegerTypeB_sound (value : Option WireType)
    (valid : scalarIntegerTypeB value = true) : scalarIntegerType value := by
  cases value with
  | none => simp [scalarIntegerTypeB] at valid
  | some wireType => cases wireType <;> simp [scalarIntegerTypeB, scalarIntegerType] at valid ⊢

theorem scalarBooleanTypeB_sound (value : Option WireType)
    (valid : scalarBooleanTypeB value = true) : scalarBooleanType value := by
  cases value with
  | none => simp [scalarBooleanTypeB] at valid
  | some wireType => cases wireType <;> simp [scalarBooleanTypeB, scalarBooleanType] at valid ⊢

theorem binaryIntegerTypesB_sound (inputs : List (Option WireType)) (result : List WireType)
    (output : WireType) (valid : binaryIntegerTypesB inputs result output = true) :
    ∃ left right, inputs = [left, right] ∧ scalarIntegerType left ∧
      scalarIntegerType right ∧ result = [output] := by
  cases inputs with
  | nil => simp [binaryIntegerTypesB] at valid
  | cons left rest => cases rest with
    | nil => simp [binaryIntegerTypesB] at valid
    | cons right tail => cases tail with
      | cons _ _ => simp [binaryIntegerTypesB] at valid
      | nil => cases result with
        | nil => simp [binaryIntegerTypesB] at valid
        | cons actual tail => cases tail with
          | cons _ _ => simp [binaryIntegerTypesB] at valid
          | nil =>
            simp only [binaryIntegerTypesB, Bool.and_eq_true, decide_eq_true_eq] at valid
            exact ⟨left, right, rfl, scalarIntegerTypeB_sound left valid.1.1,
              scalarIntegerTypeB_sound right valid.1.2, valid.2 ▸ rfl⟩

theorem unaryIntegerTypesB_sound (inputs : List (Option WireType)) (result : List WireType)
    (output : WireType) (valid : unaryIntegerTypesB inputs result output = true) :
    ∃ input, inputs = [input] ∧ scalarIntegerType input ∧ result = [output] := by
  cases inputs with
  | nil => simp [unaryIntegerTypesB] at valid
  | cons input tail => cases tail with
    | cons _ _ => simp [unaryIntegerTypesB] at valid
    | nil => cases result with
      | nil => simp [unaryIntegerTypesB] at valid
      | cons actual tail => cases tail with
        | cons _ _ => simp [unaryIntegerTypesB] at valid
        | nil =>
          simp only [unaryIntegerTypesB, Bool.and_eq_true, decide_eq_true_eq] at valid
          exact ⟨input, rfl, scalarIntegerTypeB_sound input valid.1, valid.2 ▸ rfl⟩

theorem booleanToIntegerTypesB_sound (inputs : List (Option WireType)) (result : List WireType)
    (valid : booleanToIntegerTypesB inputs result = true) :
    ∃ input, inputs = [input] ∧ scalarBooleanType input ∧ result = [.int] := by
  cases inputs with
  | nil => simp [booleanToIntegerTypesB] at valid
  | cons input tail => cases tail with
    | cons _ _ => simp [booleanToIntegerTypesB] at valid
    | nil => cases result with
      | nil => simp [booleanToIntegerTypesB] at valid
      | cons actual tail => cases tail with
        | cons _ _ => simp [booleanToIntegerTypesB] at valid
        | nil =>
          simp only [booleanToIntegerTypesB, Bool.and_eq_true, decide_eq_true_eq] at valid
          exact ⟨input, rfl, scalarBooleanTypeB_sound input valid.1, valid.2 ▸ rfl⟩

theorem intBinaryTypesB_sound (stage : Stage) (scope : Scope) (operation : IntBinaryOp)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.intBinary operation) arguments outputs = true) :
    operationTypesOK stage scope (.intBinary operation) arguments outputs := by
  simp only [operationTypesOKB] at valid
  obtain ⟨left, right, inputs, leftValid, rightValid, result⟩ :=
    binaryIntegerTypesB_sound _ _ .int valid
  exact ⟨left, right, inputs, leftValid, rightValid, result⟩

theorem intCompareTypesB_sound (stage : Stage) (scope : Scope) (operation : IntCompareOp)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.intCompare operation) arguments outputs = true) :
    operationTypesOK stage scope (.intCompare operation) arguments outputs := by
  simp only [operationTypesOKB] at valid
  obtain ⟨left, right, inputs, leftValid, rightValid, result⟩ :=
    binaryIntegerTypesB_sound _ _ .bool valid
  exact ⟨left, right, inputs, leftValid, rightValid, result⟩

theorem bitExtractTypesB_sound (stage : Stage) (scope : Scope) (bit : StructuralIntExpr)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope (.bitExtract bit) arguments outputs = true) :
    operationTypesOK stage scope (.bitExtract bit) arguments outputs := by
  simp only [operationTypesOKB] at valid
  obtain ⟨input, inputs, inputValid, result⟩ := unaryIntegerTypesB_sound _ _ .bool valid
  exact ⟨input, inputs, inputValid, result⟩

theorem boolToIntTypesB_sound (stage : Stage) (scope : Scope)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope .boolToInt arguments outputs = true) :
    operationTypesOK stage scope .boolToInt arguments outputs := by
  simp only [operationTypesOKB] at valid
  obtain ⟨input, inputs, inputValid, result⟩ := booleanToIntegerTypesB_sound _ _ valid
  exact ⟨input, inputs, inputValid, result⟩

theorem subgraphCallTypesB_sound (stage : Stage) (scope : Scope) (call : SubgraphPayload)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : structuralChildTypesOKB stage scope (.subgraphCall call) arguments outputs = true) :
    structuralChildTypesOK stage scope (.subgraphCall call) arguments outputs := by
  cases childStored : scopeAt stage call.child with
  | none => simp [structuralChildTypesOKB, childStored] at valid
  | some child =>
    simp only [structuralChildTypesOKB, childStored, Bool.and_eq_true, decide_eq_true_eq] at valid
    exact ⟨child, childStored, valid.1.1,
      optionalTypesCompatibleB_sound valid.1.2,
      optionalTypesCompatibleB_sound valid.2⟩

theorem sequentialLoopTypesB_sound (stage : Stage) (scope : Scope) (loop : LoopPayload)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : structuralChildTypesOKB stage scope (.sequentialLoop loop) arguments outputs = true) :
    structuralChildTypesOK stage scope (.sequentialLoop loop) arguments outputs := by
  cases childStored : scopeAt stage loop.child with
  | none => simp [structuralChildTypesOKB, childStored] at valid
  | some child =>
    cases countStored : loop.count.eval {} with
    | error message => simp [structuralChildTypesOKB, childStored, countStored] at valid
    | ok count =>
      simp only [structuralChildTypesOKB, childStored, countStored] at valid
      obtain ⟨beforeSlot, slotChecked⟩ := Bool.and_eq_true_iff.mp valid
      obtain ⟨beforeCarried, carriedCompatible⟩ := Bool.and_eq_true_iff.mp beforeSlot
      obtain ⟨beforeOutputs, outputCompatible⟩ := Bool.and_eq_true_iff.mp beforeCarried
      obtain ⟨header, inputCompatible⟩ := Bool.and_eq_true_iff.mp beforeOutputs
      simp only [decide_eq_true_eq] at header
      rcases header with ⟨countNonnegative, carriedPositive, carriedBound, inputsSize,
        childOutputsSize, outputsSize⟩
      have countEq : Int.ofNat count.toNat = count := Int.toNat_of_nonneg countNonnegative
      refine ⟨child, count.toNat, childStored, ?_, carriedPositive, carriedBound, inputsSize,
        childOutputsSize, outputsSize, optionalTypesCompatibleB_sound inputCompatible,
        optionalTypesCompatibleB_sound outputCompatible,
        optionalTypesCompatibleB_sound carriedCompatible, ?_⟩
      · simpa only [structuralExpressionIsNat, countEq] using countStored
      · obtain ⟨slotIndex, slotBound, slotValid⟩ := Array.any_eq_true.mp slotChecked
        refine ⟨child.structuralSlots[slotIndex], Array.getElem_mem slotBound, ?_⟩
        simp only [decide_eq_true_eq] at slotValid
        exact ⟨slotValid.1, slotValid.2.1, slotValid.2.2.trans countEq.symm⟩

theorem parallelGridTypesB_sound (stage : Stage) (scope : Scope) (grid : GridPayload)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : structuralChildTypesOKB stage scope (.parallelGrid grid) arguments outputs = true) :
    structuralChildTypesOK stage scope (.parallelGrid grid) arguments outputs := by
  cases childStored : scopeAt stage grid.child with
  | none => simp [structuralChildTypesOKB, childStored] at valid
  | some child =>
    cases shapeStored : shapeExpression? grid.shape with
    | none => simp [structuralChildTypesOKB, childStored, shapeStored] at valid
    | some shape =>
      simp only [structuralChildTypesOKB, childStored, shapeStored] at valid
      obtain ⟨beforeAxes, axesChecked⟩ := Bool.and_eq_true_iff.mp valid
      obtain ⟨beforeOutputs, outputsChecked⟩ := Bool.and_eq_true_iff.mp beforeAxes
      obtain ⟨header, argumentsChecked⟩ := Bool.and_eq_true_iff.mp beforeOutputs
      simp only [Bool.and_eq_true, decide_eq_true_eq] at header
      rcases header with ⟨inputModesSize, childInputsSize, childOutputsSize, slotsSize⟩
      refine ⟨child, shape, childStored, (shapeExpression?_eq_some_iff _ _).mp shapeStored,
        inputModesSize, childInputsSize, childOutputsSize, slotsSize, ?_, ?_, ?_⟩
      · intro index indexBound
        have checked := List.all_eq_true.mp argumentsChecked index (List.mem_range.mpr indexBound)
        cases outerStored : (referencedTypes scope arguments)[index]? with
        | none => simp [outerStored] at checked
        | some outerOption => cases outerOption with
          | none => simp [outerStored] at checked
          | some outer =>
            cases innerStored : (referencedTypes child child.inputs)[index]? with
            | none => simp [outerStored, innerStored] at checked
            | some innerOption => cases innerOption with
              | none => simp [outerStored, innerStored] at checked
              | some inner =>
                cases modeStored : grid.inputModes[index]? with
                | none => simp [outerStored, innerStored, modeStored] at checked
                | some mode =>
                  refine ⟨outer, inner, mode, rfl, rfl, rfl, ?_⟩
                  cases mode with
                  | mk reindex map =>
                    cases reindex with
                    | false =>
                      cases map with
                      | none =>
                        simpa [outerStored, innerStored, modeStored] using checked
                      | some map => simp [outerStored, innerStored, modeStored] at checked
                    | true =>
                      cases map with
                      | none => simp [outerStored, innerStored, modeStored] at checked
                      | some map =>
                        cases outer <;> simp [outerStored, innerStored, modeStored] at checked
                        case family inputShape element =>
                          rcases checked with ⟨⟨compatible, outputRank, sourceRank, inputsSize⟩, axes⟩
                          exact ⟨inputShape, element, rfl, compatible,
                            outputRank, sourceRank, inputsSize, axes⟩
      · intro index indexBound
        have checked := List.all_eq_true.mp outputsChecked index (List.mem_range.mpr indexBound)
        cases childStored : (referencedTypes child child.outputs)[index]? with
        | none => simp [childStored] at checked
        | some childOption => cases childOption with
          | none => simp [childStored] at checked
          | some childOutput =>
            cases outputStored : outputs[index]? with
            | none => simp [childStored, outputStored] at checked
            | some output =>
              cases output <;> simp [childStored, outputStored] at checked
              case family outputShape outputElement =>
                exact ⟨childOutput, outputShape, outputElement, rfl, rfl,
                  checked.1, checked.2⟩
      · intro axis axisBound
        have checked := List.all_eq_true.mp axesChecked axis (List.mem_range.mpr axisBound)
        cases slotStored : grid.indexSlots[axis]? with
        | none => simp [slotStored] at checked
        | some slot =>
          cases extentStored : shape[axis]? with
          | none => simp [slotStored, extentStored] at checked
          | some extent =>
            have found : ∃ i, ∃ h : i < child.structuralSlots.size,
                child.structuralSlots[i].slot = slot ∧
                child.structuralSlots[i].kind = .gridAxis axis ∧
                child.structuralSlots[i].upperBound = Int.ofNat extent := by
              simpa [slotStored, extentStored] using checked
            obtain ⟨declarationIndex, declarationBound, declarationChecked⟩ := found
            refine ⟨slot, extent, child.structuralSlots[declarationIndex], rfl, rfl,
              Array.getElem_mem declarationBound, declarationChecked⟩

theorem operationTypesOKB_sound (stage : Stage) (scope : Scope) (payload : NodePayload)
    (arguments : Array WireRef) (outputs : Array WireType)
    (valid : operationTypesOKB stage scope payload arguments outputs = true) :
    operationTypesOK stage scope payload arguments outputs := by
  cases payload
  case input index => exact inputTypesB_sound stage scope index arguments outputs valid
  case artifactInput input => exact artifactInputTypesB_sound stage scope input arguments outputs valid
  case constantInt value => exact constantIntTypesB_sound stage scope value arguments outputs valid
  case evaluateInt value => exact evaluateIntTypesB_sound stage scope value arguments outputs valid
  case constantBool value => exact constantBoolTypesB_sound stage scope value arguments outputs valid
  case constantMatrix matrix literal =>
    exact constantMatrixTypesB_sound stage scope matrix literal arguments outputs valid
  case intBinary op => exact intBinaryTypesB_sound stage scope op arguments outputs valid
  case intCompare op => exact intCompareTypesB_sound stage scope op arguments outputs valid
  case bitExtract bit => exact bitExtractTypesB_sound stage scope bit arguments outputs valid
  case boolToInt => exact boolToIntTypesB_sound stage scope arguments outputs valid
  case matrixBinary op => exact matrixBinaryTypesB_sound stage scope op arguments outputs valid
  case matrixNegate => exact matrixNegateTypesB_sound stage scope arguments outputs valid
  case matrixScale scalar => exact matrixScaleTypesB_sound stage scope scalar arguments outputs valid
  case transpose => exact transposeTypesB_sound stage scope arguments outputs valid
  case slice rows columns => exact sliceTypesB_sound stage scope rows columns arguments outputs valid
  case concat axis => exact concatTypesB_sound stage scope axis arguments outputs valid
  case uniformIntervalSample matrix range =>
    exact uniformIntervalSampleTypesB_sound stage scope matrix range arguments outputs valid
  case gaussianSample matrix sigma bound =>
    exact gaussianSampleTypesB_sound stage scope matrix sigma bound arguments outputs valid
  case hashSample matrix tagPrefix tags decimals words =>
    exact hashSampleTypesB_sound stage scope matrix tagPrefix tags decimals words arguments outputs valid
  case trapdoorSample matrix sigma base digits bound =>
    exact trapdoorSampleTypesB_sound stage scope matrix sigma base digits bound arguments outputs valid
  case preimageSample preimage bound =>
    exact preimageSampleTypesB_sound stage scope preimage bound arguments outputs valid
  case applyPreimage => exact applyPreimageTypesB_sound stage scope arguments outputs valid
  case materializePreimageExact =>
    exact materializePreimageTypesB_sound stage scope arguments outputs valid
  case preimageBinary op => exact preimageBinaryTypesB_sound stage scope op arguments outputs valid
  case familyPreimageSample preimage bound =>
    exact familyPreimageSampleTypesB_sound stage scope preimage bound arguments outputs valid
  case gadgetDecompose base small digits =>
    exact gadgetDecomposeTypesB_sound stage scope base small digits arguments outputs valid
  case extractCoefficient position upper =>
    exact extractCoefficientTypesB_sound stage scope position upper arguments outputs valid
  case subgraphCall call => exact subgraphCallTypesB_sound stage scope call arguments outputs valid
  case sequentialLoop loop => exact sequentialLoopTypesB_sound stage scope loop arguments outputs valid
  case familyPack shape => exact familyPackTypesB_sound stage scope shape arguments outputs valid
  case familyGetStatic indices =>
    exact familyGetStaticTypesB_sound stage scope indices arguments outputs valid
  case familyGetDynamic rank =>
    exact familyGetDynamicTypesB_sound stage scope rank arguments outputs valid
  case familySelectAxis axis =>
    exact familySelectAxisTypesB_sound stage scope axis arguments outputs valid
  case familyReindex shape map =>
    exact familyReindexTypesB_sound stage scope shape map arguments outputs valid
  case familyGather shape rank =>
    exact familyGatherTypesB_sound stage scope shape rank arguments outputs valid
  case parallelGrid grid => exact parallelGridTypesB_sound stage scope grid arguments outputs valid
  case select count => exact selectTypesB_sound stage scope count arguments outputs valid
  case constantReal => simp [operationTypesOKB] at valid
  case gadgetTrapdoor => simp [operationTypesOKB] at valid
  case trapdoorPublic => simp [operationTypesOKB] at valid
  case intToReal => simp [operationTypesOKB] at valid
  case realBinary => simp [operationTypesOKB] at valid
  case realSqrt => simp [operationTypesOKB] at valid
  case matrixMulAccumulate => simp [operationTypesOKB] at valid
  case tensor => simp [operationTypesOKB] at valid
  case uniformResidueSample => simp [operationTypesOKB] at valid
  case preimageConcatColumns => simp [operationTypesOKB] at valid
  case decompositionEntry => simp [operationTypesOKB] at valid
  case liftIntegerToConstantPolynomial => simp [operationTypesOKB] at valid
  case thresholdDecode => simp [operationTypesOKB] at valid
  case crtRecompose => simp [operationTypesOKB] at valid
  case packPolynomialCoefficients => simp [operationTypesOKB] at valid

def previousWireValidB (scope : Scope) (wire : WireRef) (index : Nat) : Bool :=
  decide (wire.scope = scope.id ∧ wire.node < index) && match nodeAt scope wire.node with
  | some node => decide (wire.port < node.outputs.size)
  | none => false

def slotDeclaredB (slots : Array StructuralSlotDecl) (slot : Nat) : Bool :=
  slots.any fun declaration => decide (declaration.slot = slot)

def structuralSlotsUsedB (slots : Array StructuralSlotDecl) : StructuralIntExpr → Bool
  | .literal _ => true
  | .structuralSlot slot => slotDeclaredB slots slot
  | .add left right | .subtract left right | .multiply left right |
      .exactDivide left right | .roundDivide left right =>
      structuralSlotsUsedB slots left && structuralSlotsUsedB slots right
  | .log2Ceil value => structuralSlotsUsedB slots value

def indexSlotsUsedFuelB (slots : Array StructuralSlotDecl) : Nat → IndexMapExpr → Bool
  | 0, _ => false
  | fuel + 1, expression => match expression with
    | .literal _ | .axis _ => true
    | .structuralSlot slot => slotDeclaredB slots slot
    | .add left right | .sub left right | .mul left right | .divide left right |
        .remainder left right | .equal left right | .less left right | .lessEqual left right =>
        indexSlotsUsedFuelB slots fuel left && indexSlotsUsedFuelB slots fuel right
    | .log2Ceil value => indexSlotsUsedFuelB slots fuel value
    | .select selector branches => indexSlotsUsedFuelB slots fuel selector &&
        branches.all (indexSlotsUsedFuelB slots fuel)

def indexSlotsUsedB (slots : Array StructuralSlotDecl) (expression : IndexMapExpr) : Bool :=
  indexSlotsUsedFuelB slots 1024 expression

def realSlotsUsedB (slots : Array StructuralSlotDecl) : RealExpr → Bool
  | .literal value => decide (value.denominator ≠ 0)
  | .fromInt value => structuralSlotsUsedB slots value
  | .add left right | .subtract left right | .multiply left right | .divide left right =>
      realSlotsUsedB slots left && realSlotsUsedB slots right
  | .sqrt value => realSlotsUsedB slots value

def rangeSlotsUsedB (slots : Array StructuralSlotDecl) : Option IntRange → Bool
  | none => true
  | some range => structuralSlotsUsedB slots range.start && structuralSlotsUsedB slots range.stop

def payloadSlotsUsedB (slots : Array StructuralSlotDecl) : NodePayload → Bool
  | .evaluateInt value => structuralSlotsUsedB slots value
  | .constantMatrix _ (.unitRow value) | .constantMatrix _ (.unitColumn value) |
      .constantMatrix _ (.rotation value) => structuralSlotsUsedB slots value
  | .constantMatrix _ (.gadget base _) => structuralSlotsUsedB slots base
  | .constantMatrix _ (.powerOfBase base exponent) =>
      structuralSlotsUsedB slots base && structuralSlotsUsedB slots exponent
  | .constantMatrix _ (.polynomial coefficients) => coefficients.all (structuralSlotsUsedB slots)
  | .uniformIntervalSample _ range =>
      structuralSlotsUsedB slots range.start && structuralSlotsUsedB slots range.stop
  | .gaussianSample _ sigma bound => realSlotsUsedB slots sigma && structuralSlotsUsedB slots bound
  | .hashSample _ _ tags decimalTags u64Tags => tags.all (structuralSlotsUsedB slots) &&
      decimalTags.all (structuralSlotsUsedB slots) && u64Tags.all (structuralSlotsUsedB slots)
  | .trapdoorSample _ sigma base digits bound => realSlotsUsedB slots sigma &&
      structuralSlotsUsedB slots base && structuralSlotsUsedB slots digits &&
      structuralSlotsUsedB slots bound
  | .preimageSample _ bound | .familyPreimageSample _ bound => structuralSlotsUsedB slots bound
  | .gadgetDecompose base _ digits =>
      structuralSlotsUsedB slots base && structuralSlotsUsedB slots digits
  | .extractCoefficient position _ | .bitExtract position => structuralSlotsUsedB slots position
  | .slice rows columns => rangeSlotsUsedB slots rows && rangeSlotsUsedB slots columns
  | .familyReindex shape map => shape.all (structuralSlotsUsedB slots) &&
      map.inputIndices.all (indexSlotsUsedB slots)
  | .parallelGrid grid => grid.shape.all (structuralSlotsUsedB slots)
  | .sequentialLoop loop => structuralSlotsUsedB slots loop.count
  | .select count => structuralSlotsUsedB slots count
  | _ => true

def structuralSlotsValidB (slots : Array StructuralSlotDecl) : Bool :=
  slots.all (fun declaration => decide (0 < declaration.upperBound)) &&
    (List.range slots.size).all (fun first =>
      (List.range slots.size).all (fun second =>
        if first = second then true else match slots[first]?, slots[second]? with
        | some left, some right => decide (left.slot ≠ right.slot)
        | _, _ => false))

def validWireTypeB : WireType → Bool
  | .bytes length => decide (0 < length)
  | .matrix matrix | .preimage matrix => decide (1 < matrix.modulus) &&
      decide (0 < matrix.ringDimension) && decide (0 < matrix.rows) && decide (0 < matrix.columns)
  | .trapdoor trapdoor => decide (1 < trapdoor.matrix.modulus) &&
      decide (0 < trapdoor.matrix.ringDimension) && decide (0 < trapdoor.matrix.rows) &&
      decide (0 < trapdoor.matrix.columns)
  | .family shape element => shape.all (fun extent => decide (0 < extent)) && validWireTypeB element
  | _ => true

def nodeValidB (stage : Stage) (scope : Scope) (index : Nat) (node : Node) : Bool :=
  decide (0 < node.outputs.size) && node.outputs.all validWireTypeB &&
    node.arguments.all (previousWireValidB scope · index) &&
    payloadSlotsUsedB scope.structuralSlots node.payload &&
    operationTypesOKB stage scope node.payload node.arguments node.outputs

def scopeHeaderValidB (scope : Scope) : Bool :=
  structuralSlotsValidB scope.structuralSlots &&
    scope.inputs.all (fun wire => decide (wire.scope = scope.id) && (wireType? scope wire).isSome) &&
    scope.outputs.all (fun wire => decide (wire.scope = scope.id) && (wireType? scope wire).isSome)

def nodeAtIndexValidB (stage : Stage) (scope : Scope) (index : Nat) : Bool :=
  match scope.nodes[index]? with
  | some node => nodeValidB stage scope index node
  | none => false

def scopeNodesValidB (stage : Stage) (scope : Scope) : Bool :=
  (List.range scope.nodes.size).all (nodeAtIndexValidB stage scope)

def scopeValidB (stage : Stage) (scope : Scope) : Bool :=
  scopeHeaderValidB scope && scopeNodesValidB stage scope

def noCyclesFromB (stage : Stage) (current : ScopeId) (seen : List ScopeId) : Nat → Bool
  | 0 => false
  | fuel + 1 => decide (current ∉ seen) && match scopeAt stage current with
      | none => false
      | some scope => (structuralChildren scope).all
          (noCyclesFromB stage · (current :: seen) fuel)

def stageHeaderValidB (stage : Stage) : Bool :=
  decide (0 < stage.scopes.size) && (scopeAt stage stage.root).isSome &&
    (List.range stage.scopes.size).all (fun first =>
      (List.range stage.scopes.size).all (fun second =>
        if first = second then true else match stage.scopes[first]?, stage.scopes[second]? with
        | some left, some right => decide (left.id ≠ right.id)
        | _, _ => false)) &&
    noCyclesFromB stage stage.root [] (stage.scopes.size + 1) &&
    stage.namedOutputs.all (fun output => (stage.wireType? output.wire).isSome)

def scopeAtIndexValidB (stage : Stage) (index : Nat) : Bool :=
  match stage.scopes[index]? with
  | some scope => scopeValidB stage scope
  | none => false

def stageScopesValidB (stage : Stage) : Bool :=
  (List.range stage.scopes.size).all (scopeAtIndexValidB stage)

def stageValidB (stage : Stage) : Bool := stageHeaderValidB stage && stageScopesValidB stage

def ArtifactLink.validB (data : ProgramData) (link : ArtifactLink) : Bool :=
  match data.stages[link.consumerStage]?, data.stages[link.producerStage]? with
  | some consumer, some producer => decide (link.producerStage < link.consumerStage) &&
      match scopeAt consumer link.consumer.scope, consumer.wireType? link.consumer,
          producer.wireType? link.producer with
      | some consumerScope, some consumerType, some producerType =>
          match nodeAt consumerScope link.consumer.node with
          | some { payload := .artifactInput input, .. } =>
            decide (data.artifactLinks[input.index]? = some link ∧
              link.argument = link.consumer.port ∧
              input.name = link.consumerArtifact ∧
              input.confidentiality = link.consumerConfidentiality ∧
              consumerType = link.consumerType ∧
              producerType = link.producerType) &&
            structuralTypeCompatibleB link.producerType link.consumerType &&
            decide (link.consumerArtifact = link.producerArtifact ∧
              link.consumerConfidentiality = link.producerConfidentiality)
          | _ => false
      | _, _, _ => false
  | _, _ => false

def artifactNodeLinkedExactlyOnceB (data : ProgramData) (stageNumber : Nat)
    (scope : Scope) (nodeIndex : Nat) : Bool := match scope.nodes[nodeIndex]? with
  | some { payload := .artifactInput _, outputs, .. } =>
      (List.range outputs.size).all fun port =>
        decide ((data.artifactLinks.toList.filter fun link =>
          link.consumerStage = stageNumber &&
          link.consumer = { scope := scope.id, node := nodeIndex, port := port } &&
          link.argument = port).length = 1)
  | some _ => true
  | none => false

def artifactScopeInputsLinkedB (data : ProgramData) (stageNumber : Nat) (scope : Scope) : Bool :=
  (List.range scope.nodes.size).all (artifactNodeLinkedExactlyOnceB data stageNumber scope)

def artifactScopeAtIndexInputsLinkedB (data : ProgramData) (stageNumber scopeIndex : Nat) : Bool :=
  match data.stages[stageNumber]? with
  | some stage => match stage.scopes[scopeIndex]? with
    | some scope => artifactScopeInputsLinkedB data stageNumber scope
    | none => false
  | none => false

def artifactStageInputsLinkedB (data : ProgramData) (stageNumber : Nat) : Bool :=
  match data.stages[stageNumber]? with
  | some stage => (List.range stage.scopes.size).all
      (artifactScopeAtIndexInputsLinkedB data stageNumber)
  | none => false

def artifactInputsLinkedExactlyOnceB (data : ProgramData) : Bool :=
  (List.range data.stages.size).all (artifactStageInputsLinkedB data)

def stageAtIndexValidB (data : ProgramData) (index : Nat) : Bool :=
  match data.stages[index]? with
  | some stage => stageValidB stage
  | none => false

def programStagesValidB (data : ProgramData) : Bool :=
  (List.range data.stages.size).all (stageAtIndexValidB data)

def artifactLinkAtIndexValidB (data : ProgramData) (index : Nat) : Bool :=
  match data.artifactLinks[index]? with
  | some link => ArtifactLink.validB data link
  | none => false

def programArtifactLinksValidB (data : ProgramData) : Bool :=
  (List.range data.artifactLinks.size).all (artifactLinkAtIndexValidB data)

def ProgramData.validate (data : ProgramData) : Bool :=
  programStagesValidB data && programArtifactLinksValidB data &&
    artifactInputsLinkedExactlyOnceB data

theorem scopeValidB_of_components {stage : Stage} {scope : Scope}
    (header : scopeHeaderValidB scope = true)
    (nodes : scopeNodesValidB stage scope = true) : scopeValidB stage scope = true := by
  simp [scopeValidB, header, nodes]

theorem scopeNodesValidB_of_each {stage : Stage} {scope : Scope}
    (each : ∀ index, index < scope.nodes.size → nodeAtIndexValidB stage scope index = true) :
    scopeNodesValidB stage scope = true := by
  rw [scopeNodesValidB, List.all_eq_true]
  intro index member
  exact each index (List.mem_range.mp member)

theorem stageValidB_of_components {stage : Stage}
    (header : stageHeaderValidB stage = true)
    (scopes : stageScopesValidB stage = true) : stageValidB stage = true := by
  simp [stageValidB, header, scopes]

theorem stageScopesValidB_of_each {stage : Stage}
    (each : ∀ index, index < stage.scopes.size → scopeAtIndexValidB stage index = true) :
    stageScopesValidB stage = true := by
  rw [stageScopesValidB, List.all_eq_true]
  intro index member
  exact each index (List.mem_range.mp member)

theorem ProgramData.validate_of_components {data : ProgramData}
    (stages : programStagesValidB data = true)
    (links : programArtifactLinksValidB data = true)
    (inputs : artifactInputsLinkedExactlyOnceB data = true) : data.validate = true := by
  simp [ProgramData.validate, stages, links, inputs]

theorem programStagesValidB_of_each {data : ProgramData}
    (each : ∀ index, index < data.stages.size → stageAtIndexValidB data index = true) :
    programStagesValidB data = true := by
  rw [programStagesValidB, List.all_eq_true]
  intro index member
  exact each index (List.mem_range.mp member)

theorem programArtifactLinksValidB_of_each {data : ProgramData}
    (each : ∀ index, index < data.artifactLinks.size →
      artifactLinkAtIndexValidB data index = true) : programArtifactLinksValidB data = true := by
  rw [programArtifactLinksValidB, List.all_eq_true]
  intro index member
  exact each index (List.mem_range.mp member)

theorem artifactScopeInputsLinkedB_of_each {data : ProgramData} {stageNumber : Nat}
    {scope : Scope}
    (each : ∀ index, index < scope.nodes.size →
      artifactNodeLinkedExactlyOnceB data stageNumber scope index = true) :
    artifactScopeInputsLinkedB data stageNumber scope = true := by
  rw [artifactScopeInputsLinkedB, List.all_eq_true]
  intro index member
  exact each index (List.mem_range.mp member)

theorem artifactStageInputsLinkedB_of_each {data : ProgramData} {stageNumber : Nat}
    {stage : Stage} (stored : data.stages[stageNumber]? = some stage)
    (each : ∀ index, index < stage.scopes.size →
      artifactScopeAtIndexInputsLinkedB data stageNumber index = true) :
    artifactStageInputsLinkedB data stageNumber = true := by
  rw [artifactStageInputsLinkedB, stored, List.all_eq_true]
  intro index member
  exact each index (List.mem_range.mp member)

theorem artifactInputsLinkedExactlyOnceB_of_each {data : ProgramData}
    (each : ∀ index, index < data.stages.size →
      artifactStageInputsLinkedB data index = true) :
    artifactInputsLinkedExactlyOnceB data = true := by
  rw [artifactInputsLinkedExactlyOnceB, List.all_eq_true]
  intro index member
  exact each index (List.mem_range.mp member)

/-! Declarative certificate boundary.  The executable validators above are diagnostics;
the structures below state the trusted invariants without referring to a Boolean result. -/

structure Node.Valid (stage : Stage) (scope : Scope) (index : Nat) (node : Node) : Prop where
  outputsNonempty : 0 < node.outputs.size
  outputTypes : ∀ output ∈ node.outputs, validWireType output
  argumentsPrevious : ∀ argument ∈ node.arguments, previousWireValid scope argument index
  payload : node.payload.Valid
  slotsUsed : payloadSlotsUsed scope.structuralSlots node.payload
  operation : operationTypesOK stage scope node.payload node.arguments node.outputs

structure StoredOutputCert (node : Node) (index : Nat) : Type where
  output : WireType
  stored : node.outputs[index]? = some output
  valid : validWireType output

inductive OutputRangeCert (node : Node) : Nat → Nat → Type
  | empty (start : Nat) : OutputRangeCert node start start
  | single (index : Nat) (output : StoredOutputCert node index) :
      OutputRangeCert node index (index + 1)
  | append {first middle last : Nat} : OutputRangeCert node first middle →
      OutputRangeCert node middle last → OutputRangeCert node first last

noncomputable def OutputRangeCert.covers {node : Node} {first last : Nat}
    (certificate : OutputRangeCert node first last) :
    ∀ index, first ≤ index → index < last → StoredOutputCert node index := by
  induction certificate with
  | empty start => omega
  | single storedIndex stored =>
      intro index lower upper
      have : index = storedIndex := by omega
      subst index
      exact stored
  | @append first middle last left right leftIH rightIH =>
      intro index lower upper
      by_cases before : index < middle
      · exact leftIH index lower before
      · exact rightIH index (by omega) upper

structure StoredArgumentCert (scope : Scope) (nodeIndex : Nat) (node : Node)
    (index : Nat) : Type where
  argument : WireRef
  stored : node.arguments[index]? = some argument
  previous : previousWireValid scope argument nodeIndex
  argumentType : WireType
  typeStored : wireType? scope argument = some argumentType

inductive ArgumentRangeCert (scope : Scope) (nodeIndex : Nat) (node : Node) : Nat → Nat → Type
  | empty (start : Nat) : ArgumentRangeCert scope nodeIndex node start start
  | single (index : Nat) (argument : StoredArgumentCert scope nodeIndex node index) :
      ArgumentRangeCert scope nodeIndex node index (index + 1)
  | append {first middle last : Nat} : ArgumentRangeCert scope nodeIndex node first middle →
      ArgumentRangeCert scope nodeIndex node middle last →
      ArgumentRangeCert scope nodeIndex node first last

noncomputable def ArgumentRangeCert.covers {scope : Scope} {nodeIndex : Nat} {node : Node}
    {first last : Nat} (certificate : ArgumentRangeCert scope nodeIndex node first last) :
    ∀ index, first ≤ index → index < last → StoredArgumentCert scope nodeIndex node index := by
  induction certificate with
  | empty start => omega
  | single storedIndex stored =>
      intro index lower upper
      have : index = storedIndex := by omega
      subst index
      exact stored
  | @append first middle last left right leftIH rightIH =>
      intro index lower upper
      by_cases before : index < middle
      · exact leftIH index lower before
      · exact rightIH index (by omega) upper

/- An operation certificate consumes argument types in wire order.  Each leaf binds one expected
   entry to the already-local wire lookup, while the balanced tree avoids unfolding a scope prefix. -/
structure StoredArgumentTypeCert (scope : Scope) (nodeIndex : Nat) (node : Node)
    (types : List (Option WireType)) (index : Nat) : Type where
  argument : StoredArgumentCert scope nodeIndex node index
  expected : types[index]? = some (some argument.argumentType)

inductive ArgumentTypeRangeCert (scope : Scope) (nodeIndex : Nat) (node : Node)
    (types : List (Option WireType)) : Nat → Nat → Type
  | empty (start : Nat) : ArgumentTypeRangeCert scope nodeIndex node types start start
  | single (index : Nat) (argument : StoredArgumentTypeCert scope nodeIndex node types index) :
      ArgumentTypeRangeCert scope nodeIndex node types index (index + 1)
  | append {first middle last : Nat} :
      ArgumentTypeRangeCert scope nodeIndex node types first middle →
      ArgumentTypeRangeCert scope nodeIndex node types middle last →
      ArgumentTypeRangeCert scope nodeIndex node types first last

noncomputable def ArgumentTypeRangeCert.covers {scope : Scope} {nodeIndex : Nat} {node : Node}
    {types : List (Option WireType)} {first last : Nat}
    (certificate : ArgumentTypeRangeCert scope nodeIndex node types first last) :
    ∀ index, first ≤ index → index < last →
      StoredArgumentTypeCert scope nodeIndex node types index := by
  induction certificate with
  | empty start => omega
  | single storedIndex stored =>
      intro index lower upper
      have : index = storedIndex := by omega
      subst index
      exact stored
  | @append first middle last left right leftIH rightIH =>
      intro index lower upper
      by_cases before : index < middle
      · exact leftIH index lower before
      · exact rightIH index (by omega) upper

theorem ArgumentTypeRangeCert.sound {scope : Scope} {nodeIndex : Nat} {node : Node}
    {types : List (Option WireType)}
    (certificate : ArgumentTypeRangeCert scope nodeIndex node types 0 node.arguments.size)
    (typesSize : types.length = node.arguments.size) :
    referencedTypes scope node.arguments = types := by
  apply List.ext_getElem?
  intro index
  by_cases bound : index < node.arguments.size
  · let stored := certificate.covers index (Nat.zero_le index) bound
    rw [stored.expected]
    simp only [referencedTypes, List.getElem?_map, Array.getElem?_toList,
      stored.argument.stored, Option.map_some]
    rw [stored.argument.typeStored]
  · have typesBound : ¬index < types.length := by omega
    have argumentsPast : node.arguments.size ≤ index := Nat.le_of_not_gt bound
    have typesPast : types.length ≤ index := Nat.le_of_not_gt typesBound
    simp [referencedTypes, Array.getElem?_toList, Array.getElem?_eq_none argumentsPast,
      List.getElem?_eq_none typesPast]

inductive FiniteRangeCert (property : Nat → Prop) : Nat → Nat → Type
  | empty (start : Nat) : FiniteRangeCert property start start
  | single (index : Nat) (proof : property index) : FiniteRangeCert property index (index + 1)
  | append {first middle last : Nat} : FiniteRangeCert property first middle →
      FiniteRangeCert property middle last → FiniteRangeCert property first last

noncomputable def FiniteRangeCert.covers {property : Nat → Prop} {first last : Nat}
    (certificate : FiniteRangeCert property first last) :
    ∀ index, first ≤ index → index < last → property index := by
  induction certificate with
  | empty start => omega
  | single storedIndex proof =>
      intro index lower upper
      have : index = storedIndex := by omega
      subst index
      exact proof
  | @append first middle last left right leftIH rightIH =>
      intro index lower upper
      by_cases before : index < middle
      · exact leftIH index lower before
      · exact rightIH index (by omega) upper

inductive DataRangeCert (property : Nat → Type) : Nat → Nat → Type
  | empty (start : Nat) : DataRangeCert property start start
  | single (index : Nat) (proof : property index) : DataRangeCert property index (index + 1)
  | append {first middle last : Nat} : DataRangeCert property first middle →
      DataRangeCert property middle last → DataRangeCert property first last

noncomputable def DataRangeCert.covers {property : Nat → Type} {first last : Nat}
    (certificate : DataRangeCert property first last) :
    ∀ index, first ≤ index → index < last → property index := by
  induction certificate with
  | empty start => omega
  | single storedIndex proof =>
      intro index lower upper
      have : index = storedIndex := by omega
      subst index
      exact proof
  | @append first middle last left right leftIH rightIH =>
      intro index lower upper
      by_cases before : index < middle
      · exact leftIH index lower before
      · exact rightIH index (by omega) upper

def optionalTypeCompatibleAt (actual expected : List (Option WireType)) (index : Nat) : Prop :=
  ∃ actualType expectedType, actual[index]? = some (some actualType) ∧
    expected[index]? = some (some expectedType) ∧ structuralTypeCompatible actualType expectedType

inductive TypeCompatibilityCert : WireType → WireType → Prop
  | exact (wireType : WireType) : TypeCompatibilityCert wireType wireType
  | constantInt : TypeCompatibilityCert .constantInt .int
  | constantBool : TypeCompatibilityCert .constantBool .bool
  | constantReal : TypeCompatibilityCert .constantReal .real

theorem TypeCompatibilityCert.sound {actual expected : WireType}
    (certificate : TypeCompatibilityCert actual expected) :
    structuralTypeCompatible actual expected := by
  cases certificate <;> simp [structuralTypeCompatible]

structure OptionalTypePairCert (actual expected : List (Option WireType)) (index : Nat) : Type where
  actualType : WireType
  expectedType : WireType
  actualStored : actual[index]? = some (some actualType)
  expectedStored : expected[index]? = some (some expectedType)
  compatible : TypeCompatibilityCert actualType expectedType

theorem OptionalTypePairCert.sound {actual expected : List (Option WireType)} {index : Nat}
    (certificate : OptionalTypePairCert actual expected index) :
    optionalTypeCompatibleAt actual expected index :=
  ⟨certificate.actualType, certificate.expectedType, certificate.actualStored,
    certificate.expectedStored, certificate.compatible.sound⟩

structure OptionalTypesCert (actual expected : List (Option WireType)) : Type where
  lengthEq : actual.length = expected.length
  entries : DataRangeCert (OptionalTypePairCert actual expected) 0 actual.length

theorem OptionalTypesCert.sound {actual expected : List (Option WireType)}
    (certificate : OptionalTypesCert actual expected) : optionalTypesCompatible actual expected := by
  refine ⟨certificate.lengthEq, ?_⟩
  intro index bound
  exact (certificate.entries.covers index (Nat.zero_le index) bound).sound

structure StructuralNatCert (expression : StructuralIntExpr) (value : Nat) : Prop where
  evaluated : expression.eval {} = .ok (Int.ofNat value)

theorem StructuralNatCert.sound {expression : StructuralIntExpr} {value : Nat}
    (certificate : StructuralNatCert expression value) :
    structuralExpressionIsNat expression value := certificate.evaluated

def StructuralNatCert.ofEval {expression : StructuralIntExpr} {value : Nat}
    (evaluated : expression.eval {} = .ok (Int.ofNat value)) :
    StructuralNatCert expression value := ⟨evaluated⟩

structure ShapeCert (expression : Array StructuralIntExpr) (shape : List Nat) : Prop where
  checked : shapeExpression? expression = some shape

theorem ShapeCert.sound {expression : Array StructuralIntExpr} {shape : List Nat}
    (certificate : ShapeCert expression shape) : shapeExpressionIs expression shape :=
  (shapeExpression?_eq_some_iff expression shape).mp certificate.checked

def ShapeCert.ofChecked {expression : Array StructuralIntExpr} {shape : List Nat}
    (checked : shapeExpression? expression = some shape) : ShapeCert expression shape := ⟨checked⟩

structure RangeExtentCert (slots : Array StructuralSlotDecl) (range : Option IntRange)
    (inputExtent outputExtent : Nat) : Prop where
  checked : rangeExtent? slots range inputExtent = some outputExtent

theorem RangeExtentCert.sound {slots : Array StructuralSlotDecl} {range : Option IntRange}
    {inputExtent outputExtent : Nat}
    (certificate : RangeExtentCert slots range inputExtent outputExtent) :
    rangeExtentIs slots range inputExtent outputExtent := rangeExtentB_sound certificate.checked

def RangeExtentCert.ofChecked {slots : Array StructuralSlotDecl} {range : Option IntRange}
    {inputExtent outputExtent : Nat}
    (checked : rangeExtent? slots range inputExtent = some outputExtent) :
    RangeExtentCert slots range inputExtent outputExtent := ⟨checked⟩

structure IndexMapCert (map : IndexMap) (inputShape : List Nat)
    (slots : Array StructuralSlotDecl) : Prop where
  checked : indexMapCheckedB map inputShape slots = true

theorem IndexMapCert.sound {map : IndexMap} {inputShape : List Nat}
    {slots : Array StructuralSlotDecl} (certificate : IndexMapCert map inputShape slots) :
    indexMapBounded map inputShape slots := certificate.checked

def IndexMapCert.ofChecked {map : IndexMap} {inputShape : List Nat}
    {slots : Array StructuralSlotDecl} (checked : indexMapCheckedB map inputShape slots = true) :
    IndexMapCert map inputShape slots := ⟨checked⟩

def gridInputAt (scope child : Scope) (grid : GridPayload) (shape : List Nat)
    (arguments : Array WireRef) (index : Nat) : Prop :=
  ∃ outer inner mode, (referencedTypes scope arguments)[index]? = some (some outer) ∧
    (referencedTypes child child.inputs)[index]? = some (some inner) ∧
    grid.inputModes[index]? = some mode ∧ gridInputTypeOK shape.length outer inner mode

def gridOutputAt (child : Scope) (outputs : Array WireType) (shape : List Nat)
    (index : Nat) : Prop :=
  ∃ childOutput outputShape outputElement,
    (referencedTypes child child.outputs)[index]? = some (some childOutput) ∧
    outputs[index]? = some (.family outputShape outputElement) ∧ outputShape = shape ∧
    childOutput = outputElement

def gridAxisAt (child : Scope) (grid : GridPayload) (shape : List Nat) (axis : Nat) : Prop :=
  ∃ slot extent declaration, grid.indexSlots[axis]? = some slot ∧ shape[axis]? = some extent ∧
    declaration ∈ child.structuralSlots ∧ declaration.slot = slot ∧
    declaration.kind = .gridAxis axis ∧ declaration.upperBound = Int.ofNat extent

inductive GridInputCert (scope child : Scope) (grid : GridPayload) (shape : List Nat)
    (arguments : Array WireRef) : Nat → Type
  | broadcast (index : Nat) (wireType : WireType)
      (outerStored : (referencedTypes scope arguments)[index]? = some (some wireType))
      (innerStored : (referencedTypes child child.inputs)[index]? = some (some wireType))
      (modeStored : grid.inputModes[index]? = some { reindex := false, map := none }) :
      GridInputCert scope child grid shape arguments index
  | reindex (index : Nat) (inputShape : List Nat) (element : WireType) (map : IndexMap)
      (outerStored : (referencedTypes scope arguments)[index]? =
        some (some (.family inputShape element)))
      (innerStored : (referencedTypes child child.inputs)[index]? = some (some element))
      (modeStored : grid.inputModes[index]? = some { reindex := true, map := some map })
      (outputRank : map.outputRank = shape.length)
      (sourceRank : map.sourceRank = inputShape.length)
      (indicesSize : map.inputIndices.size = inputShape.length)
      (indexMap : IndexMapCert map inputShape #[]) :
      GridInputCert scope child grid shape arguments index

theorem GridInputCert.sound {scope child : Scope} {grid : GridPayload} {shape : List Nat}
    {arguments : Array WireRef} {index : Nat}
    (certificate : GridInputCert scope child grid shape arguments index) :
    gridInputAt scope child grid shape arguments index := by
  cases certificate with
  | broadcast wireType outerStored innerStored modeStored =>
      exact ⟨wireType, wireType, _, outerStored, innerStored, modeStored, rfl⟩
  | reindex inputShape element map outerStored innerStored modeStored outputRank
      sourceRank indicesSize indexMap =>
      exact ⟨.family inputShape element, element, _, outerStored, innerStored, modeStored,
        inputShape, element, rfl, rfl, outputRank, sourceRank, indicesSize, indexMap.sound⟩

structure GridOutputCert (child : Scope) (outputs : Array WireType) (shape : List Nat)
    (index : Nat) : Type where
  childType : WireType
  outputElement : WireType
  childStored : (referencedTypes child child.outputs)[index]? = some (some childType)
  outputStored : outputs[index]? = some (.family shape outputElement)
  typeEq : childType = outputElement

theorem GridOutputCert.sound {child : Scope} {outputs : Array WireType} {shape : List Nat}
    {index : Nat} (certificate : GridOutputCert child outputs shape index) :
    gridOutputAt child outputs shape index :=
  ⟨certificate.childType, shape, certificate.outputElement, certificate.childStored,
    certificate.outputStored, rfl, certificate.typeEq⟩

structure GridAxisCert (child : Scope) (grid : GridPayload) (shape : List Nat)
    (axis : Nat) : Type where
  slot : Nat
  extent : Nat
  declaration : StructuralSlotDecl
  slotStored : grid.indexSlots[axis]? = some slot
  extentStored : shape[axis]? = some extent
  declarationMem : declaration ∈ child.structuralSlots
  declarationSlot : declaration.slot = slot
  declarationKind : declaration.kind = .gridAxis axis
  declarationBound : declaration.upperBound = Int.ofNat extent

theorem GridAxisCert.sound {child : Scope} {grid : GridPayload} {shape : List Nat} {axis : Nat}
    (certificate : GridAxisCert child grid shape axis) : gridAxisAt child grid shape axis :=
  ⟨certificate.slot, certificate.extent, certificate.declaration, certificate.slotStored,
    certificate.extentStored, certificate.declarationMem, certificate.declarationSlot,
    certificate.declarationKind, certificate.declarationBound⟩

inductive StructuralOperationCert (stage : Stage) (scope : Scope) :
    NodePayload → Array WireRef → Array WireType → Type
  | subgraphCall (call : SubgraphPayload) (arguments : Array WireRef) (outputs : Array WireType)
      (child : Scope) (childStored : scopeAt stage call.child = some child)
      (canonicalSize : call.canonicalInputExclusiveUppers.size = arguments.size)
      (inputs : OptionalTypesCert (referencedTypes scope arguments)
        (referencedTypes child child.inputs))
      (childOutputs : OptionalTypesCert (referencedTypes child child.outputs)
        (outputs.toList.map some)) :
      StructuralOperationCert stage scope (.subgraphCall call) arguments outputs
  | sequentialLoop (loop : LoopPayload) (arguments : Array WireRef) (outputs : Array WireType)
      (child : Scope) (count : Nat) (childStored : scopeAt stage loop.child = some child)
      (countStored : StructuralNatCert loop.count count)
      (carriedPositive : loop.carriedCount > 0) (carriedBound : loop.carriedCount ≤ arguments.size)
      (childInputsSize : child.inputs.size = arguments.size)
      (childOutputsSize : child.outputs.size = loop.carriedCount)
      (outputsSize : outputs.size = loop.carriedCount)
      (inputs : OptionalTypesCert (referencedTypes scope arguments)
        (referencedTypes child child.inputs))
      (childOutputs : OptionalTypesCert (referencedTypes child child.outputs)
        (outputs.toList.map some))
      (carried : OptionalTypesCert ((referencedTypes scope arguments).take loop.carriedCount)
        ((referencedTypes child child.outputs).take loop.carriedCount))
      (declaration : StructuralSlotDecl) (declarationMem : declaration ∈ child.structuralSlots)
      (declarationSlot : declaration.slot = loop.indexSlot)
      (declarationKind : declaration.kind = .sequentialIteration)
      (declarationBound : declaration.upperBound = Int.ofNat count) :
      StructuralOperationCert stage scope (.sequentialLoop loop) arguments outputs
  | parallelGrid (grid : GridPayload) (arguments : Array WireRef) (outputs : Array WireType)
      (child : Scope) (shape : List Nat) (childStored : scopeAt stage grid.child = some child)
      (shapeStored : ShapeCert grid.shape shape)
      (inputModesSize : grid.inputModes.size = arguments.size)
      (childInputsSize : child.inputs.size = arguments.size)
      (childOutputsSize : child.outputs.size = outputs.size)
      (indexSlotsSize : grid.indexSlots.size = shape.length)
      (inputs : DataRangeCert (GridInputCert scope child grid shape arguments) 0 arguments.size)
      (childOutputs : DataRangeCert (GridOutputCert child outputs shape) 0 outputs.size)
      (axes : DataRangeCert (GridAxisCert child grid shape) 0 shape.length) :
      StructuralOperationCert stage scope (.parallelGrid grid) arguments outputs

noncomputable def StructuralOperationCert.sound {stage : Stage} {scope : Scope}
    {payload : NodePayload} {arguments : Array WireRef} {outputs : Array WireType}
    (certificate : StructuralOperationCert stage scope payload arguments outputs) :
    operationTypesOK stage scope payload arguments outputs := by
  cases certificate with
  | subgraphCall call arguments outputs child childStored canonicalSize inputs childOutputs =>
      exact ⟨child, childStored, canonicalSize, inputs.sound, childOutputs.sound⟩
  | sequentialLoop loop arguments outputs child count childStored countStored carriedPositive
      carriedBound childInputsSize childOutputsSize outputsSize inputs childOutputs carried
      declaration declarationMem declarationSlot declarationKind declarationBound =>
      exact ⟨child, count, childStored, countStored.sound, carriedPositive, carriedBound,
        childInputsSize, childOutputsSize, outputsSize, inputs.sound, childOutputs.sound,
        carried.sound, declaration, declarationMem, declarationSlot, declarationKind,
        declarationBound⟩
  | parallelGrid grid arguments outputs child shape childStored shapeStored inputModesSize
      childInputsSize childOutputsSize indexSlotsSize inputs childOutputs axes =>
      exact ⟨child, shape, childStored, shapeStored.sound, inputModesSize, childInputsSize,
        childOutputsSize, indexSlotsSize,
        fun index bound => GridInputCert.sound
          (inputs.covers index (Nat.zero_le index) bound),
        fun index bound => GridOutputCert.sound
          (childOutputs.covers index (Nat.zero_le index) bound),
        fun axis bound => GridAxisCert.sound (axes.covers axis (Nat.zero_le axis) bound)⟩

inductive DirectOperationCert (scope : Scope) :
    NodePayload → List (Option WireType) → Array WireType → Type
  | input (index : Nat) {types outputs} (valid : types = [] ∧ outputs.size = 1) :
      DirectOperationCert scope (.input index) types outputs
  | artifactInput (input : ArtifactInput) {types outputs}
      (valid : types = [] ∧ outputs.size = 1) :
      DirectOperationCert scope (.artifactInput input) types outputs
  | constantInt (value : Int) {types outputs}
      (valid : types = [] ∧ outputs.toList = [.constantInt]) :
      DirectOperationCert scope (.constantInt value) types outputs
  | evaluateInt (value : StructuralIntExpr) {types outputs}
      (valid : types = [] ∧ outputs.toList = [.constantInt]) :
      DirectOperationCert scope (.evaluateInt value) types outputs
  | constantBool (value : Bool) {types outputs}
      (valid : types = [] ∧ outputs.toList = [.constantBool]) :
      DirectOperationCert scope (.constantBool value) types outputs
  | constantMatrix (matrix : MatrixType) (literal : MatrixLiteral) {types outputs}
      (valid : types = [] ∧ outputs.toList = [.matrix matrix]) :
      DirectOperationCert scope (.constantMatrix matrix literal) types outputs
  | uniformResidueSample (matrix : MatrixType) {types outputs}
      (valid : types = [] ∧ outputs.toList = [.matrix matrix]) :
      DirectOperationCert scope (.uniformResidueSample matrix) types outputs
  | uniformIntervalSample (matrix : MatrixType) (range : IntRange) {types outputs}
      (valid : types = [] ∧ outputs.toList = [.matrix matrix]) :
      DirectOperationCert scope (.uniformIntervalSample matrix range) types outputs
  | gaussianSample (matrix : MatrixType) (sigma : RealExpr) (bound : StructuralIntExpr)
      {types outputs} (valid : types = [] ∧ outputs.toList = [.matrix matrix]) :
      DirectOperationCert scope (.gaussianSample matrix sigma bound) types outputs
  | hashSample (matrix : MatrixType) (tagPrefix : List UInt8)
      (tags decimals words : Array StructuralIntExpr) {types outputs}
      (valid : types = [some (.bytes 32)] ∧ outputs.toList = [.matrix matrix]) :
      DirectOperationCert scope (.hashSample matrix tagPrefix tags decimals words) types outputs
  | trapdoorSample (matrix : MatrixType) (sigma : RealExpr)
      (base digits bound : StructuralIntExpr) {types outputs}
      (valid : ∃ trapdoor, types = [] ∧ outputs.toList = [.matrix matrix, .trapdoor trapdoor] ∧
        trapdoor.matrix = matrix ∧ trapdoor.sigma = sigma ∧ trapdoor.gadgetBase = base ∧
        trapdoor.digitCount = digits ∧ trapdoor.preimageMaxCoefficientBound = bound) :
      DirectOperationCert scope (.trapdoorSample matrix sigma base digits bound) types outputs
  | intBinary (operation : IntBinaryOp) {types outputs}
      (valid : ∃ left right, types = [left, right] ∧ scalarIntegerType left ∧
        scalarIntegerType right ∧ outputs.toList = [.int]) :
      DirectOperationCert scope (.intBinary operation) types outputs
  | intCompare (operation : IntCompareOp) {types outputs}
      (valid : ∃ left right, types = [left, right] ∧ scalarIntegerType left ∧
        scalarIntegerType right ∧ outputs.toList = [.bool]) :
      DirectOperationCert scope (.intCompare operation) types outputs
  | bitExtract (bit : StructuralIntExpr) {types outputs}
      (valid : ∃ input, types = [input] ∧ scalarIntegerType input ∧ outputs.toList = [.bool]) :
      DirectOperationCert scope (.bitExtract bit) types outputs
  | realBinary (operation : RealBinaryOp) {types outputs}
      (valid : types = [some .real, some .real] ∧ outputs.toList = [.real]) :
      DirectOperationCert scope (.realBinary operation) types outputs
  | intToReal {types outputs} (valid : types = [some .int] ∧ outputs.toList = [.real]) :
      DirectOperationCert scope .intToReal types outputs
  | boolToInt {types outputs}
      (valid : ∃ input, types = [input] ∧ scalarBooleanType input ∧ outputs.toList = [.int]) :
      DirectOperationCert scope .boolToInt types outputs
  | realSqrt {types outputs} (valid : types = [some .real] ∧ outputs.toList = [.real]) :
      DirectOperationCert scope .realSqrt types outputs
  | matrixBinary (operation : MatrixBinaryOp) {types outputs}
      (valid : ∃ left right output, types = [some (.matrix left), some (.matrix right)] ∧
        outputs.toList = [.matrix output] ∧ match operation with
          | .add | .subtract => matrixAddType left right output
          | .multiply => matrixProductType left right output) :
      DirectOperationCert scope (.matrixBinary operation) types outputs
  | matrixNegate {types outputs}
      (valid : ∃ matrix, types = [some (.matrix matrix)] ∧ outputs.toList = [.matrix matrix]) :
      DirectOperationCert scope .matrixNegate types outputs
  | matrixScale (scalar : StructuralIntExpr) {types outputs}
      (valid : ∃ matrix, types = [some (.matrix matrix)] ∧ outputs.toList = [.matrix matrix]) :
      DirectOperationCert scope (.matrixScale scalar) types outputs
  | transpose {types outputs}
      (valid : ∃ input output, types = [some (.matrix input)] ∧
        outputs.toList = [.matrix output] ∧ sameRing input output ∧
        output.rows = input.columns ∧ output.columns = input.rows) :
      DirectOperationCert scope .transpose types outputs
  | concat (axis : ConcatAxis) {types outputs}
      (valid : ∃ output, outputs.toList = [.matrix output] ∧ matrixConcatType axis types output) :
      DirectOperationCert scope (.concat axis) types outputs
  | slice (rows columns : Option IntRange) {types outputs}
      (valid : ∃ input output outputRows outputColumns,
        types = [some (.matrix input)] ∧ outputs.toList = [.matrix output] ∧
        rangeExtentIs scope.structuralSlots rows input.rows outputRows ∧
        rangeExtentIs scope.structuralSlots columns input.columns outputColumns ∧
        sameRing input output ∧ output.rows = outputRows ∧ output.columns = outputColumns ∧
        0 < output.rows ∧ 0 < output.columns) :
      DirectOperationCert scope (.slice rows columns) types outputs
  | extractCoefficient (position : StructuralIntExpr) (upper : Option Nat) {types outputs}
      (valid : ∃ input value, types = [some (.matrix input)] ∧ outputs.toList = [.int] ∧
        position.eval {} = .ok value ∧ input.rows = 1 ∧ input.columns = 1 ∧
        0 ≤ value ∧ value < Int.ofNat input.ringDimension) :
      DirectOperationCert scope (.extractCoefficient position upper) types outputs
  | preimageSample (preimage : MatrixType) (bound : StructuralIntExpr) {types outputs}
      (valid : ∃ source trapdoor target,
        types = [some (.matrix source), some (.trapdoor trapdoor), some (.matrix target)] ∧
        outputs.toList = [.preimage preimage] ∧ trapdoor.matrix = source ∧
        preimageEquationType source target preimage) :
      DirectOperationCert scope (.preimageSample preimage bound) types outputs
  | familyPreimageSample (preimage : MatrixType) (bound : StructuralIntExpr) {types outputs}
      (valid : ∃ sourceType trapdoorType sourceShape trapdoorShape targetShape outputShape
          source trapdoor target output,
        types = [some sourceType, some trapdoorType, some (.family targetShape (.matrix target))] ∧
        matrixFamilyElement? sourceType = some (sourceShape, source) ∧
        trapdoorFamilyElement? trapdoorType = some (trapdoorShape, trapdoor) ∧
        outputs.toList = [.family outputShape (.preimage output)] ∧ output = preimage ∧
        outputShape = targetShape ∧ sourceShape = trapdoorShape ∧
        targetShape.length = sourceShape.length + 1 ∧
        targetShape.take sourceShape.length = sourceShape ∧ trapdoor.matrix = source ∧
        preimageEquationType source target preimage) :
      DirectOperationCert scope (.familyPreimageSample preimage bound) types outputs
  | applyPreimage {types outputs}
      (valid : ∃ left preimage output,
        types = [some (.matrix left), some (.preimage preimage)] ∧
        outputs.toList = [.matrix output] ∧ matrixProductType left preimage output) :
      DirectOperationCert scope .applyPreimage types outputs
  | materializePreimageExact {types outputs}
      (valid : ∃ preimage, types = [some (.preimage preimage)] ∧
        outputs.toList = [.matrix preimage]) :
      DirectOperationCert scope .materializePreimageExact types outputs
  | preimageBinary (operation : PreimageBinaryOp) {types outputs}
      (valid : ∃ left right output, types = [some (.preimage left), some right] ∧
        outputs.toList = [.preimage output] ∧ match operation with
          | .add => right = .preimage left ∧ output = left
          | .rightMultiplyExact => ∃ matrix, right = .matrix matrix ∧
              matrixProductType left matrix output
          | .composeExactDecomposition => ∃ preimage, right = .preimage preimage ∧
              matrixProductType left preimage output) :
      DirectOperationCert scope (.preimageBinary operation) types outputs
  | gadgetDecompose (base : StructuralIntExpr) (small : Bool) (digits : StructuralIntExpr)
      {types outputs} (valid : ∃ target preimage digitCount,
        types = [some (.matrix target)] ∧ outputs.toList = [.preimage preimage] ∧
        structuralExpressionIsNat digits digitCount ∧ digitCount > 0 ∧
        sameRing target preimage ∧ preimage.rows = target.rows * digitCount ∧
        preimage.columns = target.columns) :
      DirectOperationCert scope (.gadgetDecompose base small digits) types outputs
  | familyPack (shape : Array StructuralIntExpr) {types outputs}
      (valid : ∃ concreteShape element, shapeExpressionIs shape concreteShape ∧
        concreteShape ≠ [] ∧ types.length = shapeProduct concreteShape ∧
        (∀ argument ∈ types, argument = some element) ∧ familyElementType element = none ∧
        outputs.toList = [.family concreteShape element]) :
      DirectOperationCert scope (.familyPack shape) types outputs
  | familyGetStatic (indices : Array IndexMapExpr) {types outputs}
      (valid : ∃ shape element, types = [some (.family shape element)] ∧
        indices.size = shape.length ∧ outputs.toList = [element]) :
      DirectOperationCert scope (.familyGetStatic indices) types outputs
  | familyGetDynamic (rank : Nat) {types outputs}
      (valid : ∃ shape element selectors, types = some (.family shape element) :: selectors ∧
        selectors.length = rank ∧ allIntegerTypes selectors ∧ shape.length = rank ∧
        outputs.toList = [element]) :
      DirectOperationCert scope (.familyGetDynamic rank) types outputs
  | familySelectAxis (axis : Nat) {types outputs}
      (valid : ∃ shape element selector outputShape,
        types = [some (.family shape element), some selector] ∧
        removeAt? shape axis = some outputShape ∧
        (integerSelectorType selector ∨ ∃ selectorElement,
          selector = .family outputShape selectorElement ∧ integerSelectorType selectorElement) ∧
        outputs.toList = [if outputShape = [] then element else .family outputShape element]) :
      DirectOperationCert scope (.familySelectAxis axis) types outputs
  | familyReindex (shape : Array StructuralIntExpr) (map : IndexMap) {types outputs}
      (valid : ∃ inputShape outputShape element,
        types = [some (.family inputShape element)] ∧ shapeExpressionIs shape outputShape ∧
        map.sourceRank = inputShape.length ∧ map.outputRank = outputShape.length ∧
        map.inputIndices.size = inputShape.length ∧
        indexMapBounded map inputShape scope.structuralSlots ∧
        outputs.toList = [.family outputShape element]) :
      DirectOperationCert scope (.familyReindex shape map) types outputs
  | familyGather (shape : Array StructuralIntExpr) (rank : Nat) {types outputs}
      (valid : ∃ inputShape outputShape element, shapeExpressionIs shape outputShape ∧
        inputShape.length = rank ∧ ∃ selectorTypes,
        types = some (.family inputShape element) :: selectorTypes ∧
        selectorTypes.length = rank ∧
        (∀ selector ∈ selectorTypes, ∃ selectorElement,
          selector = some (.family outputShape selectorElement) ∧
          integerSelectorType selectorElement) ∧ outputs.toList = [.family outputShape element]) :
      DirectOperationCert scope (.familyGather shape rank) types outputs
  | select (count : StructuralIntExpr) {types outputs}
      (valid : ∃ branchCount branchType selectorType,
        structuralExpressionIsNat count branchCount ∧ branchCount > 0 ∧
        integerSelectorType selectorType ∧
        types = some selectorType :: List.replicate branchCount (some branchType) ∧
        outputs.toList = [branchType]) :
      DirectOperationCert scope (.select count) types outputs

theorem DirectOperationCert.sound {scope : Scope} {payload : NodePayload}
    {types : List (Option WireType)} {outputs : Array WireType}
    (certificate : DirectOperationCert scope payload types outputs) :
    ∀ {stage : Stage} {arguments : Array WireRef}, referencedTypes scope arguments = types →
      operationTypesOK stage scope payload arguments outputs := by
  intro stage arguments typesStored
  cases certificate <;> simp only [operationTypesOK] <;> rw [typesStored] <;> assumption

inductive OperationContractCert (stage : Stage) (scope : Scope) (payload : NodePayload)
    (arguments : Array WireRef) (outputs : Array WireType)
    (argumentTypes : List (Option WireType)) : Type
  | direct (certificate : DirectOperationCert scope payload argumentTypes outputs) :
      OperationContractCert stage scope payload arguments outputs argumentTypes
  | structural (certificate : StructuralOperationCert stage scope payload arguments outputs) :
      OperationContractCert stage scope payload arguments outputs argumentTypes

/- `OperationCert` separates local wire lookup from the payload's declarative contract. Generated
   code builds the balanced argument tree once, then supplies one indexed data constructor over the
   resulting short ordered type list. Unsupported payloads have no direct constructor. -/
structure OperationCert (stage : Stage) (scope : Scope) (nodeIndex : Nat)
    (payload : NodePayload) (arguments : Array WireRef) (outputs : Array WireType) : Type where
  argumentTypes : List (Option WireType)
  argumentTypesSize : argumentTypes.length = arguments.size
  argumentsTyped : ArgumentTypeRangeCert scope nodeIndex
    { payload := payload, arguments := arguments, outputs := outputs }
    argumentTypes 0 arguments.size
  contract : OperationContractCert stage scope payload arguments outputs argumentTypes

noncomputable def OperationCert.sound {stage : Stage} {scope : Scope} {nodeIndex : Nat}
    {payload : NodePayload} {arguments : Array WireRef} {outputs : Array WireType}
    (certificate : OperationCert stage scope nodeIndex payload arguments outputs) :
    operationTypesOK stage scope payload arguments outputs :=
  match certificate.contract with
  | .direct direct => direct.sound (certificate.argumentsTyped.sound certificate.argumentTypesSize)
  | .structural structural => structural.sound

noncomputable def OperationCert.ofStructural {stage : Stage} {scope : Scope} {nodeIndex : Nat}
    {payload : NodePayload} {arguments : Array WireRef} {outputs : Array WireType}
    {argumentTypes : List (Option WireType)}
    (argumentTypesSize : argumentTypes.length = arguments.size)
    (argumentsTyped : ArgumentTypeRangeCert scope nodeIndex
      { payload := payload, arguments := arguments, outputs := outputs }
      argumentTypes 0 arguments.size)
    (structural : StructuralOperationCert stage scope payload arguments outputs) :
    OperationCert stage scope nodeIndex payload arguments outputs where
  argumentTypes := argumentTypes
  argumentTypesSize := argumentTypesSize
  argumentsTyped := argumentsTyped
  contract := .structural structural

noncomputable def OperationCert.familyReindex {stage : Stage} {scope : Scope}
    {nodeIndex : Nat} {shape : Array StructuralIntExpr} {map : IndexMap}
    {arguments : Array WireRef} {outputs : Array WireType}
    {argumentTypes : List (Option WireType)} {inputShape outputShape : List Nat}
    {element : WireType}
    (argumentTypesSize : argumentTypes.length = arguments.size)
    (argumentsTyped : ArgumentTypeRangeCert scope nodeIndex
      { payload := .familyReindex shape map, arguments := arguments, outputs := outputs }
      argumentTypes 0 arguments.size)
    (inputStored : argumentTypes = [some (.family inputShape element)])
    (shapeStored : ShapeCert shape outputShape)
    (sourceRank : map.sourceRank = inputShape.length)
    (outputRank : map.outputRank = outputShape.length)
    (indicesSize : map.inputIndices.size = inputShape.length)
    (indicesBounded : IndexMapCert map inputShape scope.structuralSlots)
    (outputStored : outputs.toList = [.family outputShape element]) :
    OperationCert stage scope nodeIndex (.familyReindex shape map) arguments outputs where
  argumentTypes := argumentTypes
  argumentTypesSize := argumentTypesSize
  argumentsTyped := argumentsTyped
  contract := .direct (.familyReindex shape map ⟨inputShape, outputShape, element, inputStored,
    shapeStored.sound, sourceRank, outputRank, indicesSize, indicesBounded.sound, outputStored⟩)

structure PayloadSlotsCert (scope : Scope) (node : Node) : Prop where
  valid : payloadSlotsUsed scope.structuralSlots node.payload

structure Node.LocalCert (stage : Stage) (scope : Scope) (index : Nat) (node : Node) : Type where
  outputsNonempty : 0 < node.outputs.size
  outputs : OutputRangeCert node 0 node.outputs.size
  arguments : ArgumentRangeCert scope index node 0 node.arguments.size
  payload : node.payload.Valid
  payloadSlots : PayloadSlotsCert scope node
  operation : OperationCert stage scope index node.payload node.arguments node.outputs

noncomputable def Node.LocalCert.sound {stage : Stage} {scope : Scope} {index : Nat} {node : Node}
    (certificate : node.LocalCert stage scope index) : node.Valid stage scope index where
  outputsNonempty := certificate.outputsNonempty
  outputTypes output member := by
    obtain ⟨outputIndex, outputBound, outputStored⟩ := Array.mem_iff_getElem.mp member
    let stored := certificate.outputs.covers outputIndex (Nat.zero_le outputIndex) outputBound
    have : stored.output = output := by
      have storedGet : node.outputs[outputIndex] = stored.output := by
        simpa [Array.getElem?_eq_getElem outputBound] using stored.stored
      exact storedGet.symm.trans outputStored
    simpa [this] using stored.valid
  argumentsPrevious argument member := by
    obtain ⟨argumentIndex, argumentBound, argumentStored⟩ := Array.mem_iff_getElem.mp member
    let stored := certificate.arguments.covers argumentIndex (Nat.zero_le argumentIndex) argumentBound
    have : stored.argument = argument := by
      have storedGet : node.arguments[argumentIndex] = stored.argument := by
        simpa [Array.getElem?_eq_getElem argumentBound] using stored.stored
      exact storedGet.symm.trans argumentStored
    simpa [this] using stored.previous
  payload := certificate.payload
  slotsUsed := certificate.payloadSlots.valid
  operation := certificate.operation.sound

/- A local reflection records agreement with the diagnostic checker, but its soundness
still comes from the independently constructed declarative proof. -/
structure Node.LocalReflection (stage : Stage) (scope : Scope) (index : Nat)
    (node : Node) : Prop where
  diagnostic : nodeValidB stage scope index node = true
  declarative : node.Valid stage scope index

theorem Node.LocalReflection.sound {stage : Stage} {scope : Scope} {index : Nat} {node : Node}
    (reflection : node.LocalReflection stage scope index) : node.Valid stage scope index :=
  reflection.declarative

structure StoredNodeCert (stage : Stage) (scope : Scope) (index : Nat) : Type where
  node : Node
  stored : scope.nodes[index]? = some node
  valid : node.Valid stage scope index

inductive NodeRangeCert (stage : Stage) (scope : Scope) : Nat → Nat → Type
  | empty (start : Nat) : NodeRangeCert stage scope start start
  | single (index : Nat) (node : StoredNodeCert stage scope index) :
      NodeRangeCert stage scope index (index + 1)
  | append {first middle last : Nat} : NodeRangeCert stage scope first middle →
      NodeRangeCert stage scope middle last → NodeRangeCert stage scope first last

noncomputable def NodeRangeCert.covers {stage : Stage} {scope : Scope} {first last : Nat}
    (certificate : NodeRangeCert stage scope first last) :
    ∀ index, first ≤ index → index < last → StoredNodeCert stage scope index := by
  induction certificate with
  | empty start => omega
  | single storedIndex stored =>
      intro index lower upper
      have : index = storedIndex := by omega
      subst index
      exact stored
  | @append first middle last left right leftIH rightIH =>
      intro index lower upper
      by_cases before : index < middle
      · exact leftIH index lower before
      · exact rightIH index (by omega) upper

structure Scope.Valid (stage : Stage) (scope : Scope) (rankOf : ScopeId → Nat) : Prop where
  slots : structuralSlotsValid scope.structuralSlots
  inputs : ∀ wire ∈ scope.inputs, wire.scope = scope.id ∧ ∃ wireType,
    Mxx.IR.wireType? scope wire = some wireType
  outputs : ∀ wire ∈ scope.outputs, wire.scope = scope.id ∧ ∃ wireType,
    Mxx.IR.wireType? scope wire = some wireType
  childrenDecrease : ∀ child ∈ structuralChildren scope,
    (∃ childScope, scopeAt stage child = some childScope) ∧ rankOf child < rankOf scope.id

structure StoredScopeCert (stage : Stage) (rankOf : ScopeId → Nat) (index : Nat) : Type where
  scope : Scope
  stored : stage.scopes[index]? = some scope
  valid : scope.Valid stage rankOf
  nodes : NodeRangeCert stage scope 0 scope.nodes.size

inductive ScopeRangeCert (stage : Stage) (rankOf : ScopeId → Nat) : Nat → Nat → Type
  | empty (start : Nat) : ScopeRangeCert stage rankOf start start
  | single (index : Nat) (scope : StoredScopeCert stage rankOf index) :
      ScopeRangeCert stage rankOf index (index + 1)
  | append {first middle last : Nat} : ScopeRangeCert stage rankOf first middle →
      ScopeRangeCert stage rankOf middle last → ScopeRangeCert stage rankOf first last

noncomputable def ScopeRangeCert.covers {stage : Stage} {rankOf : ScopeId → Nat} {first last : Nat}
    (certificate : ScopeRangeCert stage rankOf first last) :
    ∀ index, first ≤ index → index < last → StoredScopeCert stage rankOf index := by
  induction certificate with
  | empty start => omega
  | single storedIndex stored =>
      intro index lower upper
      have : index = storedIndex := by omega
      subst index
      exact stored
  | @append first middle last left right leftIH rightIH =>
      intro index lower upper
      by_cases before : index < middle
      · exact leftIH index lower before
      · exact rightIH index (by omega) upper

structure Stage.Valid (stage : Stage) (rankOf : ScopeId → Nat) : Prop where
  nonempty : 0 < stage.scopes.size
  rootStored : ∃ scope, scopeAt stage stage.root = some scope
  uniqueIds : uniqueScopeIds stage.scopes
  namedOutputs : ∀ output ∈ stage.namedOutputs, ∃ wireType,
    stage.wireType? output.wire = some wireType
  scopes : ∀ index, index < stage.scopes.size → ∃ scope,
    stage.scopes[index]? = some scope ∧ scope.Valid stage rankOf ∧
      ∀ nodeIndex, nodeIndex < scope.nodes.size → ∃ node,
        scope.nodes[nodeIndex]? = some node ∧ node.Valid stage scope nodeIndex

structure StoredStageCert (data : ProgramData) (index : Nat) : Type where
  stage : Stage
  stored : data.stages[index]? = some stage
  nonempty : 0 < stage.scopes.size
  rootStored : ∃ scope, scopeAt stage stage.root = some scope
  uniqueIds : uniqueScopeIds stage.scopes
  namedOutputs : ∀ output ∈ stage.namedOutputs, ∃ wireType,
    stage.wireType? output.wire = some wireType
  rankOf : ScopeId → Nat
  scopes : ScopeRangeCert stage rankOf 0 stage.scopes.size

noncomputable def StoredStageCert.sound {data : ProgramData} {index : Nat}
    (certificate : StoredStageCert data index) : certificate.stage.Valid certificate.rankOf where
  nonempty := certificate.nonempty
  rootStored := certificate.rootStored
  uniqueIds := certificate.uniqueIds
  namedOutputs := certificate.namedOutputs
  scopes scopeIndex scopeBound := by
    let stored := certificate.scopes.covers scopeIndex (Nat.zero_le scopeIndex) scopeBound
    refine ⟨stored.scope, stored.stored, stored.valid, ?_⟩
    intro nodeIndex nodeBound
    let node := stored.nodes.covers nodeIndex (Nat.zero_le nodeIndex) nodeBound
    exact ⟨node.node, node.stored, node.valid⟩

inductive StageRangeCert (data : ProgramData) : Nat → Nat → Type
  | empty (start : Nat) : StageRangeCert data start start
  | single (index : Nat) (stage : StoredStageCert data index) :
      StageRangeCert data index (index + 1)
  | append {first middle last : Nat} : StageRangeCert data first middle →
      StageRangeCert data middle last → StageRangeCert data first last

noncomputable def StageRangeCert.covers {data : ProgramData} {first last : Nat}
    (certificate : StageRangeCert data first last) :
    ∀ index, first ≤ index → index < last → StoredStageCert data index := by
  induction certificate with
  | empty start => omega
  | single storedIndex stored =>
      intro index lower upper
      have : index = storedIndex := by omega
      subst index
      exact stored
  | @append first middle last left right leftIH rightIH =>
      intro index lower upper
      by_cases before : index < middle
      · exact leftIH index lower before
      · exact rightIH index (by omega) upper

structure ArtifactLink.Valid (data : ProgramData) (linkIndex : Nat) (link : ArtifactLink) : Prop where
  stored : data.artifactLinks[linkIndex]? = some link
  order : link.producerStage < link.consumerStage
  consumerStage : ∃ stage, data.stages[link.consumerStage]? = some stage
  producerStage : ∃ stage, data.stages[link.producerStage]? = some stage
  consumerStored : ∃ stage scope node input,
    data.stages[link.consumerStage]? = some stage ∧
    scopeAt stage link.consumer.scope = some scope ∧
    nodeAt scope link.consumer.node = some node ∧ node.payload = .artifactInput input ∧
    input.index = linkIndex ∧
    input.name = link.consumerArtifact ∧
    input.confidentiality = link.consumerConfidentiality
  consumerTypeStored : ∃ stage,
    data.stages[link.consumerStage]? = some stage ∧
      stage.wireType? link.consumer = some link.consumerType
  producerTypeStored : ∃ stage,
    data.stages[link.producerStage]? = some stage ∧
      stage.wireType? link.producer = some link.producerType
  typeCompatible : structuralTypeCompatible link.producerType link.consumerType
  artifactName : link.consumerArtifact = link.producerArtifact
  confidentiality : link.consumerConfidentiality = link.producerConfidentiality
  argumentPort : link.argument = link.consumer.port

structure ArtifactLink.LocalReflection (data : ProgramData) (linkIndex : Nat)
    (link : ArtifactLink) : Prop where
  diagnostic : link.validB data = true
  declarative : link.Valid data linkIndex

theorem ArtifactLink.LocalReflection.sound {data : ProgramData} {linkIndex : Nat} {link : ArtifactLink}
    (reflection : link.LocalReflection data linkIndex) : link.Valid data linkIndex :=
  reflection.declarative

structure StoredLinkCert (data : ProgramData) (index : Nat) : Type where
  link : ArtifactLink
  stored : data.artifactLinks[index]? = some link
  valid : link.Valid data index

inductive LinkRangeCert (data : ProgramData) : Nat → Nat → Type
  | empty (start : Nat) : LinkRangeCert data start start
  | single (index : Nat) (link : StoredLinkCert data index) :
      LinkRangeCert data index (index + 1)
  | append {first middle last : Nat} : LinkRangeCert data first middle →
      LinkRangeCert data middle last → LinkRangeCert data first last

noncomputable def LinkRangeCert.covers {data : ProgramData} {first last : Nat}
    (certificate : LinkRangeCert data first last) :
    ∀ index, first ≤ index → index < last → StoredLinkCert data index := by
  induction certificate with
  | empty start => omega
  | single storedIndex stored =>
      intro index lower upper
      have : index = storedIndex := by omega
      subst index
      exact stored
  | @append first middle last left right leftIH rightIH =>
      intro index lower upper
      by_cases before : index < middle
      · exact leftIH index lower before
      · exact rightIH index (by omega) upper

def ArtifactCoverage.Valid (data : ProgramData) : Prop :=
  ∀ (stageIndex : Nat) (stage : Stage) (scopeIndex : Nat) (scope : Scope)
      (nodeIndex : Nat) (node : Node) (input : ArtifactInput) (port : Nat),
    stageIndex < data.stages.size → data.stages[stageIndex]? = some stage →
    scopeIndex < stage.scopes.size → stage.scopes[scopeIndex]? = some scope →
    nodeIndex < scope.nodes.size → scope.nodes[nodeIndex]? = some node → node.payload = .artifactInput input →
    port < node.outputs.size →
    (∃ link, data.artifactLinks[input.index]? = some link ∧
      link.consumerStage = stageIndex ∧
      link.consumer = { scope := scope.id, node := nodeIndex, port := port } ∧
      link.argument = port) ∧
    ∀ linkIndex link, data.artifactLinks[linkIndex]? = some link →
      link.consumerStage = stageIndex →
      link.consumer = { scope := scope.id, node := nodeIndex, port := port } →
      link.argument = port → linkIndex = input.index

structure ArtifactNodeCoverageCert (data : ProgramData) (stageIndex : Nat) (stage : Stage)
    (scopeIndex : Nat) (scope : Scope) (nodeIndex : Nat) : Type where
  node : Node
  stored : scope.nodes[nodeIndex]? = some node
  ports : ∀ input, node.payload = .artifactInput input → ∀ port, port < node.outputs.size →
    (∃ link, data.artifactLinks[input.index]? = some link ∧
      link.consumerStage = stageIndex ∧
      link.consumer = { scope := scope.id, node := nodeIndex, port := port } ∧
      link.argument = port) ∧
    ∀ linkIndex link, data.artifactLinks[linkIndex]? = some link →
      link.consumerStage = stageIndex →
      link.consumer = { scope := scope.id, node := nodeIndex, port := port } →
      link.argument = port → linkIndex = input.index

inductive ArtifactNodeRangeCert (data : ProgramData) (stageIndex : Nat) (stage : Stage)
    (scopeIndex : Nat) (scope : Scope) : Nat → Nat → Type
  | empty (start : Nat) : ArtifactNodeRangeCert data stageIndex stage scopeIndex scope start start
  | single (index : Nat)
      (node : ArtifactNodeCoverageCert data stageIndex stage scopeIndex scope index) :
      ArtifactNodeRangeCert data stageIndex stage scopeIndex scope index (index + 1)
  | append {first middle last : Nat} :
      ArtifactNodeRangeCert data stageIndex stage scopeIndex scope first middle →
      ArtifactNodeRangeCert data stageIndex stage scopeIndex scope middle last →
      ArtifactNodeRangeCert data stageIndex stage scopeIndex scope first last

noncomputable def ArtifactNodeRangeCert.covers {data : ProgramData} {stageIndex : Nat}
    {stage : Stage} {scopeIndex : Nat} {scope : Scope} {first last : Nat}
    (certificate : ArtifactNodeRangeCert data stageIndex stage scopeIndex scope first last) :
    ∀ index, first ≤ index → index < last →
      ArtifactNodeCoverageCert data stageIndex stage scopeIndex scope index := by
  induction certificate with
  | empty start => omega
  | single storedIndex stored =>
      intro index lower upper
      have : index = storedIndex := by omega
      subst index
      exact stored
  | @append first middle last left right leftIH rightIH =>
      intro index lower upper
      by_cases before : index < middle
      · exact leftIH index lower before
      · exact rightIH index (by omega) upper

structure ArtifactScopeCoverageCert (data : ProgramData) (stageIndex : Nat) (stage : Stage)
    (scopeIndex : Nat) : Type where
  scope : Scope
  stored : stage.scopes[scopeIndex]? = some scope
  nodes : ArtifactNodeRangeCert data stageIndex stage scopeIndex scope 0 scope.nodes.size

inductive ArtifactScopeRangeCert (data : ProgramData) (stageIndex : Nat) (stage : Stage) :
    Nat → Nat → Type
  | empty (start : Nat) : ArtifactScopeRangeCert data stageIndex stage start start
  | single (index : Nat) (scope : ArtifactScopeCoverageCert data stageIndex stage index) :
      ArtifactScopeRangeCert data stageIndex stage index (index + 1)
  | append {first middle last : Nat} :
      ArtifactScopeRangeCert data stageIndex stage first middle →
      ArtifactScopeRangeCert data stageIndex stage middle last →
      ArtifactScopeRangeCert data stageIndex stage first last

noncomputable def ArtifactScopeRangeCert.covers {data : ProgramData} {stageIndex : Nat}
    {stage : Stage} {first last : Nat}
    (certificate : ArtifactScopeRangeCert data stageIndex stage first last) :
    ∀ index, first ≤ index → index < last → ArtifactScopeCoverageCert data stageIndex stage index := by
  induction certificate with
  | empty start => omega
  | single storedIndex stored =>
      intro index lower upper
      have : index = storedIndex := by omega
      subst index
      exact stored
  | @append first middle last left right leftIH rightIH =>
      intro index lower upper
      by_cases before : index < middle
      · exact leftIH index lower before
      · exact rightIH index (by omega) upper

structure ArtifactStageCoverageCert (data : ProgramData) (stageIndex : Nat) : Type where
  stage : Stage
  stored : data.stages[stageIndex]? = some stage
  scopes : ArtifactScopeRangeCert data stageIndex stage 0 stage.scopes.size

inductive ArtifactStageRangeCert (data : ProgramData) : Nat → Nat → Type
  | empty (start : Nat) : ArtifactStageRangeCert data start start
  | single (index : Nat) (stage : ArtifactStageCoverageCert data index) :
      ArtifactStageRangeCert data index (index + 1)
  | append {first middle last : Nat} : ArtifactStageRangeCert data first middle →
      ArtifactStageRangeCert data middle last → ArtifactStageRangeCert data first last

noncomputable def ArtifactStageRangeCert.covers {data : ProgramData} {first last : Nat}
    (certificate : ArtifactStageRangeCert data first last) :
    ∀ index, first ≤ index → index < last → ArtifactStageCoverageCert data index := by
  induction certificate with
  | empty start => omega
  | single storedIndex stored =>
      intro index lower upper
      have : index = storedIndex := by omega
      subst index
      exact stored
  | @append first middle last left right leftIH rightIH =>
      intro index lower upper
      by_cases before : index < middle
      · exact leftIH index lower before
      · exact rightIH index (by omega) upper

structure ArtifactCoverageCert (data : ProgramData) : Type where
  stages : ArtifactStageRangeCert data 0 data.stages.size

noncomputable def ArtifactCoverageCert.sound {data : ProgramData}
    (certificate : ArtifactCoverageCert data) : ArtifactCoverage.Valid data := by
  intro stageIndex stage scopeIndex scope nodeIndex node input port stageBound stageStored
    scopeBound scopeStored nodeBound nodeStored payload portBound
  let storedStage := certificate.stages.covers stageIndex (Nat.zero_le stageIndex)
    stageBound
  have stageEq : storedStage.stage = stage := by
    rw [storedStage.stored] at stageStored
    exact Option.some.inj stageStored
  subst stage
  let storedScope := storedStage.scopes.covers scopeIndex (Nat.zero_le scopeIndex)
    scopeBound
  have scopeEq : storedScope.scope = scope := by
    rw [storedScope.stored] at scopeStored
    exact Option.some.inj scopeStored
  subst scope
  let storedNode := storedScope.nodes.covers nodeIndex (Nat.zero_le nodeIndex)
    nodeBound
  have nodeEq : storedNode.node = node := by
    rw [storedNode.stored] at nodeStored
    exact Option.some.inj nodeStored
  subst node
  exact storedNode.ports input payload port portBound

structure ProgramData.Valid (data : ProgramData) : Prop where
  stages : ∀ index, index < data.stages.size → ∃ stage,
    data.stages[index]? = some stage ∧ ∃ rankOf, stage.Valid rankOf
  links : ∀ index, index < data.artifactLinks.size → ∃ link,
    data.artifactLinks[index]? = some link ∧ link.Valid data index
  artifactCoverage : ArtifactCoverage.Valid data

structure ProgramData.Certificate (data : ProgramData) : Type where
  stages : StageRangeCert data 0 data.stages.size
  links : LinkRangeCert data 0 data.artifactLinks.size
  artifactCoverage : ArtifactCoverageCert data

theorem ProgramData.Certificate.sound {data : ProgramData} (certificate : data.Certificate) :
    data.Valid where
  stages index bound := by
    let stored := certificate.stages.covers index (Nat.zero_le index) bound
    exact ⟨stored.stage, stored.stored, stored.rankOf, stored.sound⟩
  links index bound := by
    let stored := certificate.links.covers index (Nat.zero_le index) bound
    exact ⟨stored.link, stored.stored, stored.valid⟩
  artifactCoverage := certificate.artifactCoverage.sound

structure Program where
  data : ProgramData
  valid : data.Valid

end IR
end Mxx
