import Mxx.Assumptions

namespace Mxx.Ir

def roundDiv (numerator denominator : Int) : Int :=
  (2 * numerator + denominator) / (2 * denominator)

def log2Ceil (value : Int) : Int :=
  if value ≤ 1 then 0 else Int.ofNat (Nat.log2 (value.toNat - 1) + 1)

/-- Complete normalized protocol source retained for review and hashing.
It is emitted directly as constructors and is never parsed from JSON. -/
inductive Syntax where
  | null
  | bool (value : Bool)
  | number (canonical : String)
  | string (value : String)
  | array (values : List Syntax)
  | object (fields : List (String × Syntax))

inductive ParamValue where
  | integer (value : Int)
  | rational (value : Rat)

abbrev ParamEnvironment := List (String × ParamValue)

inductive IntExpr where
  | constant (value : Int)
  | parameter (name : String)
  | loopIndex (slot : Nat)
  | add (left right : IntExpr)
  | subtract (left right : IntExpr)
  | multiply (left right : IntExpr)
  | divide (left right : IntExpr)
  | roundDivide (left right : IntExpr)
  | log2Ceil (value : IntExpr)

def lookupParam (name : String) : ParamEnvironment → Option ParamValue
  | [] => none
  | (candidate, value) :: tail => if candidate = name then some value else lookupParam name tail

def IntExpr.evaluate (environment : ParamEnvironment) : IntExpr → Option Int
  | .constant value => some value
  | .parameter name =>
      match lookupParam name environment with
      | some (.integer value) => some value
      | _ => none
  | .loopIndex slot =>
      match lookupParam s!"__loop_{slot}" environment with
      | some (.integer value) => some value
      | _ => none
  | .add left right => do return (← left.evaluate environment) + (← right.evaluate environment)
  | .subtract left right => do return (← left.evaluate environment) - (← right.evaluate environment)
  | .multiply left right => do return (← left.evaluate environment) * (← right.evaluate environment)
  | .divide left right => do
      let denominator ← right.evaluate environment
      if denominator = 0 then none else return (← left.evaluate environment) / denominator
  | .roundDivide left right => do
      let denominator ← right.evaluate environment
      if denominator = 0 then none
      else return roundDiv (← left.evaluate environment) denominator
  | .log2Ceil value => do return Mxx.Ir.log2Ceil (← value.evaluate environment)

inductive Value where
  | integer (value : Int)
  | rational (value : Rat)
  | boolean (value : Bool)
  | bytes (value : ByteArray)
  | matrix (value : Mxx.Matrix)
  | trapdoor (publicMatrix : Mxx.Matrix)
  | family (values : List Value)
  | opaque (description : String)
  | invalid (reason : String)

abbrev Environment := List (String × Value)

def lookupEnvironment (name : String) : Environment → Option Value
  | [] => none
  | (candidate, value) :: tail =>
      if candidate = name then some value else lookupEnvironment name tail

structure WireRef where
  node : Nat
  port : Nat
  deriving BEq, DecidableEq

inductive IntBinaryOp where
  | add
  | subtract
  | multiply
  | divide
  | remainder

inductive IntCompareOp where
  | equal
  | less
  | lessEqual

inductive LoopInputMode where
  | broadcast
  | zip
  | zipOffset (offset : Nat)

inductive ConcatAxis where
  | rows
  | columns
  | diagonal

structure MatrixTypeExpr where
  modulus : IntExpr
  ringDimension : IntExpr
  rows : IntExpr
  columns : IntExpr

def MatrixTypeExpr.evaluate
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (maxCoefficientBound : IntExpr := .constant 0) : Option Mxx.SamplerParams := do
  let modulus ← matrixType.modulus.evaluate environment
  let ringDimension ← matrixType.ringDimension.evaluate environment
  let rows ← matrixType.rows.evaluate environment
  let columns ← matrixType.columns.evaluate environment
  let maxCoefficientBound ← maxCoefficientBound.evaluate environment
  if ringDimension < 0 ∨ rows < 0 ∨ columns < 0 ∨ maxCoefficientBound < 0 then none
  else
    return {
      maxCoefficientBound := maxCoefficientBound.toNat
      modulus
      ringDimension := ringDimension.toNat
      rows := rows.toNat
      columns := columns.toNat
    }

inductive NodeKind where
  | input (name : String)
  | constantInt (value : Int)
  | evaluateInt (value : IntExpr)
  | constantBool (value : Bool)
  | zeroMatrix (matrixType : MatrixTypeExpr)
  | identityMatrix (matrixType : MatrixTypeExpr)
  | constantMatrix (matrixType : MatrixTypeExpr) (coefficients : List IntExpr)
  | boolToInt
  | intBinary (operation : IntBinaryOp)
  | intCompare (operation : IntCompareOp)
  | bitExtract (bit : IntExpr)
  | extractCoefficient (position : IntExpr)
  | select
  | uniformSample (matrixType : MatrixTypeExpr) (minimum maximum : IntExpr)
  | gaussianSample (matrixType : MatrixTypeExpr) (maxCoefficientBound : IntExpr)
  | trapdoorSample (matrixType : MatrixTypeExpr) (maxCoefficientBound : IntExpr)
  | trapdoorPublic
  | preimageSample (matrixType : MatrixTypeExpr) (maxCoefficientBound : IntExpr)
  | matrixAdd
  | matrixSubtract
  | matrixMultiply
  | matrixNegate
  | matrixScale (scalar : IntExpr)
  | concat (axis : ConcatAxis)
  | thresholdDecodeBool
      (ciphertextModulus plaintextModulus length : IntExpr)
  | familyPack
  | familyGetStatic (index : IntExpr)
  | familyGetDynamic
  | subgraphCall (definition : String) (bindings : List (String × IntExpr))
  | parallelLoop
      (definition : String)
      (count : IntExpr)
      (indexSlot : Nat)
      (bindings : List (String × IntExpr))
      (inputModes : List LoopInputMode)

structure Node where
  kind : NodeKind
  arguments : List WireRef
  outputCount : Nat := 1

structure Scope where
  nodes : List Node
  outputs : List (String × WireRef)
  inputNames : List String

structure Prog where
  root : Scope
  definitions : List (String × Scope) := []

abbrev WireEnvironment := List (WireRef × Value)

def lookupWire (wire : WireRef) : WireEnvironment → Option Value
  | [] => none
  | (candidate, value) :: tail => if candidate = wire then some value else lookupWire wire tail

def arguments (node : Node) (environment : WireEnvironment) : Option (List Value) :=
  node.arguments.mapM (fun wire => lookupWire wire environment)

def addCoefficients : List Int → List Int → List Int
  | [], right => right
  | left, [] => left
  | left :: leftTail, right :: rightTail =>
      (left + right) :: addCoefficients leftTail rightTail

def subtractCoefficients : List Int → List Int → List Int
  | [], right => right.map (-·)
  | left, [] => left
  | left :: leftTail, right :: rightTail =>
      (left - right) :: subtractCoefficients leftTail rightTail

def evaluateIntBinary (operation : IntBinaryOp) (left right : Int) : Option Int :=
  match operation with
  | .add => some (left + right)
  | .subtract => some (left - right)
  | .multiply => some (left * right)
  | .divide => if right = 0 then none else some (left / right)
  | .remainder => if right = 0 then none else some (left % right)

def evaluateIntCompare (operation : IntCompareOp) (left right : Int) : Bool :=
  match operation with
  | .equal => decide (left = right)
  | .less => decide (left < right)
  | .lessEqual => decide (left ≤ right)

def centeredRepresentative (modulus value : Int) : Int :=
  if modulus ≤ 0 then value
  else
    let residue := value % modulus
    if 2 * residue > modulus then residue - modulus else residue

def thresholdDecodeBool (ciphertextModulus plaintextModulus value : Int) : Bool :=
  if plaintextModulus ≠ 2 ∨ ciphertextModulus ≤ 0 then false
  else
    let centered := centeredRepresentative ciphertextModulus value
    decide (¬ (-(ciphertextModulus / 4) < centered ∧ centered < ciphertextModulus / 4))

def evaluateBindings (params : ParamEnvironment) :
    List (String × IntExpr) → Option ParamEnvironment
  | [] => some []
  | (name, expression) :: tail => do
      let value ← expression.evaluate params
      return (name, .integer value) :: (← evaluateBindings params tail)

def loopArgument (mode : LoopInputMode) (index : Nat) (value : Value) : Value :=
  match mode, value with
  | .broadcast, value => value
  | .zip, .family values => values[index]?.getD (.invalid "parallel zip index out of range")
  | .zipOffset offset, .family values =>
      values[index + offset]?.getD (.invalid "parallel zip-offset index out of range")
  | _, _ => .invalid "parallel-loop input mode mismatch"

def appendPortValues : List (List Value) → List Value → List (List Value)
  | [], [] => []
  | accumulated :: accumulatedTail, value :: valueTail =>
      (accumulated ++ [value]) :: appendPortValues accumulatedTail valueTail
  | _, _ => []

def integerRange (minimum maximum : Int) : List Int :=
  if maximum < minimum then []
  else (List.range (maximum - minimum + 1).toNat).map fun offset => minimum + offset

def coefficientVectors (values : List Int) : Nat → List (List Int)
  | 0 => [[]]
  | count + 1 =>
      values.flatMap fun value =>
        (coefficientVectors values count).map fun tail => value :: tail

def uniformMatrixSupport (params : Mxx.SamplerParams) (minimum maximum : Int) : List Mxx.Matrix :=
  let count := params.rows * params.columns * params.ringDimension
  (coefficientVectors (integerRange minimum maximum) count).map fun coefficients =>
    Mxx.Matrix.withSamplerParams { coefficients } params

abbrev ChildRunner := String → ParamEnvironment → List Value → List (List Value)

def evaluateParallelIterations
    (runChild : ChildRunner)
    (definition : String)
    (params : ParamEnvironment)
    (indexSlot : Nat)
    (bindings : List (String × IntExpr))
    (modes : List LoopInputMode)
    (arguments : List Value)
    (outputCount : Nat) :
    List Nat → List (List (List Value)) → List (List (List Value))
  | [], states => states
  | index :: tail, states =>
      let iterationParams := (s!"__loop_{indexSlot}", .integer index) :: params
      let next := match evaluateBindings iterationParams bindings with
        | none => []
        | some evaluatedBindings =>
            let childArguments := (modes.zip arguments).map fun (mode, value) =>
              loopArgument mode index value
            states.flatMap fun state =>
              (runChild definition (evaluatedBindings ++ iterationParams) childArguments).map fun values =>
                appendPortValues state values
      evaluateParallelIterations runChild definition params indexSlot bindings modes arguments
        outputCount tail next

def evaluateNode
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (params : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (node : Node) : List (List Value) :=
  match node.kind with
  | .input name =>
      [List.replicate node.outputCount
        (lookupEnvironment name inputs |>.getD (.invalid s!"missing input {name}"))]
  | .constantInt value => [[.integer value]]
  | .evaluateInt value =>
      match value.evaluate params with
      | some value => [[.integer value]]
      | none => [[.invalid "integer-expression evaluation failed"]]
  | .constantBool value => [[.boolean value]]
  | .zeroMatrix matrixType =>
      match matrixType.evaluate params with
      | some matrixParams =>
          let count := matrixParams.rows * matrixParams.columns * matrixParams.ringDimension
          [[.matrix (Mxx.Matrix.withSamplerParams
            { coefficients := List.replicate count 0 } matrixParams)]]
      | none => [[.invalid "zero-matrix type evaluation failed"]]
  | .identityMatrix matrixType =>
      match matrixType.evaluate params with
      | some matrixParams =>
          let coefficients :=
            (List.range matrixParams.rows).flatMap fun row =>
              (List.range matrixParams.columns).flatMap fun column =>
                (List.range matrixParams.ringDimension).map fun coefficient =>
                  if row = column ∧ coefficient = 0 then 1 else 0
          [[.matrix (Mxx.Matrix.withSamplerParams { coefficients } matrixParams)]]
      | none => [[.invalid "identity-matrix type evaluation failed"]]
  | .constantMatrix matrixType coefficients =>
      match matrixType.evaluate params, coefficients.mapM (IntExpr.evaluate params) with
      | some matrixParams, some values =>
          [[.matrix (Mxx.Matrix.withSamplerParams
            { coefficients := values.map (Mxx.reduceCoefficient matrixParams.modulus) }
            matrixParams)]]
      | _, _ => [[.invalid "constant-matrix evaluation failed"]]
  | .boolToInt =>
      match arguments node wires with
      | some [.boolean value] => [[.integer (if value then 1 else 0)]]
      | _ => [[.invalid "BoolToInt argument mismatch"]]
  | .intBinary operation =>
      match arguments node wires with
      | some [.integer left, .integer right] =>
          match evaluateIntBinary operation left right with
          | some value => [[.integer value]]
          | none => [[.invalid "integer division by zero"]]
      | _ => [[.invalid "integer binary-operation argument mismatch"]]
  | .intCompare operation =>
      match arguments node wires with
      | some [.integer left, .integer right] =>
          [[.boolean (evaluateIntCompare operation left right)]]
      | _ => [[.invalid "integer comparison argument mismatch"]]
  | .bitExtract bit =>
      match arguments node wires, bit.evaluate params with
      | some [.integer value], some bit =>
          if bit < 0 then [[.invalid "negative bit position"]]
          else [[.boolean (((value / (2 ^ bit.toNat)) % 2) ≠ 0)]]
      | _, _ => [[.invalid "bit-extraction argument mismatch"]]
  | .extractCoefficient position =>
      match arguments node wires, position.evaluate params with
      | some [.matrix matrix], some position =>
          [[.integer (matrix.coefficients.getD position.toNat 0)]]
      | _, _ => [[.invalid "coefficient-extraction argument mismatch"]]
  | .select =>
      match arguments node wires with
      | some (.integer index :: branches) =>
          [[branches[index.toNat]?.getD (.invalid "Select index out of range")]]
      | _ => [[.invalid "Select argument mismatch"]]
  | .uniformSample matrixType minimum maximum =>
      match matrixType.evaluate params, minimum.evaluate params, maximum.evaluate params with
      | some matrixParams, some minimum, some maximum =>
          (uniformMatrixSupport matrixParams minimum maximum).map (fun sample => [.matrix sample])
      | _, _, _ => [[.invalid "uniform-sample parameter evaluation failed"]]
  | .gaussianSample matrixType cutoff =>
      match matrixType.evaluate params cutoff with
      | some matrixParams =>
          (samplers.gaussianSample matrixParams).map (fun sample =>
            [.matrix (sample.withSamplerParams matrixParams)])
      | none => [[.invalid "Gaussian parameter evaluation failed"]]
  | .trapdoorSample matrixType cutoff =>
      match matrixType.evaluate params cutoff with
      | some matrixParams =>
          (samplers.trapdoorSample matrixParams).map fun publicMatrix =>
            let publicMatrix := publicMatrix.withSamplerParams matrixParams
            [.matrix publicMatrix, .trapdoor publicMatrix]
      | none => [[.invalid "trapdoor parameter evaluation failed"]]
  | .trapdoorPublic =>
      match arguments node wires with
      | some [.trapdoor publicMatrix] => [[.matrix publicMatrix]]
      | _ => [[.invalid "TrapdoorPublic argument mismatch"]]
  | .preimageSample matrixType cutoff =>
      match arguments node wires, matrixType.evaluate params cutoff with
      | some [.matrix publicMatrix, .trapdoor trapdoorPublic, .matrix target], some matrixParams =>
          if publicMatrix != trapdoorPublic then
            [[.invalid "preimage trapdoor/public-matrix mismatch"]]
          else
            (samplers.samplePreimage matrixParams publicMatrix target).map (fun sample =>
              [.matrix (sample.withSamplerParams matrixParams)])
      | _, _ => [[.invalid "preimage-sample argument mismatch"]]
  | .matrixAdd =>
      match arguments node wires with
      | some [.matrix left, .matrix right] =>
          let coefficients := List.map (Mxx.reduceCoefficient left.modulus)
            (addCoefficients left.coefficients right.coefficients)
          [[.matrix { left with coefficients }]]
      | _ => [[.invalid "matrix addition argument mismatch"]]
  | .matrixSubtract =>
      match arguments node wires with
      | some [.matrix left, .matrix right] =>
          let coefficients := List.map (Mxx.reduceCoefficient left.modulus)
            (subtractCoefficients left.coefficients right.coefficients)
          [[.matrix { left with coefficients }]]
      | _ => [[.invalid "matrix subtraction argument mismatch"]]
  | .matrixMultiply =>
      match arguments node wires with
      | some [.matrix left, .matrix right] => [[.matrix (Mxx.matrixMul left right)]]
      | _ => [[.invalid "matrix multiplication argument mismatch"]]
  | .matrixNegate =>
      match arguments node wires with
      | some [.matrix value] =>
          let coefficients := value.coefficients.map fun coefficient =>
            Mxx.reduceCoefficient value.modulus (-coefficient)
          [[.matrix { value with coefficients }]]
      | _ => [[.invalid "matrix negation argument mismatch"]]
  | .matrixScale scalar =>
      match arguments node wires, scalar.evaluate params with
      | some [.matrix value], some scalar =>
          let coefficients := value.coefficients.map fun coefficient =>
            Mxx.reduceCoefficient value.modulus (scalar * coefficient)
          [[.matrix { value with coefficients }]]
      | _, _ => [[.invalid "matrix scaling argument mismatch"]]
  | .concat axis =>
      match arguments node wires with
      | some values =>
          let matrices := values.filterMap fun value =>
            match value with
            | .matrix matrix => some matrix
            | _ => none
          if matrices.length != values.length then [[.invalid "Concat argument mismatch"]]
          else
            let output := match axis with
              | .rows => Mxx.matrixConcatRows matrices
              | .columns => Mxx.matrixConcatColumns matrices
              | .diagonal => Mxx.matrixConcatDiagonal matrices
            [[.matrix output]]
      | none => [[.invalid "Concat argument mismatch"]]
  | .thresholdDecodeBool ciphertextModulus plaintextModulus length =>
      match arguments node wires, ciphertextModulus.evaluate params,
          plaintextModulus.evaluate params, length.evaluate params with
      | some [.matrix value], some ciphertextModulus, some plaintextModulus, some length =>
          if length < 0 then [[.invalid "negative threshold decode length"]]
          else [((value.coefficients.take length.toNat).map fun coefficient =>
            .boolean (thresholdDecodeBool ciphertextModulus plaintextModulus coefficient))]
      | _, _, _, _ => [[.invalid "threshold decode argument mismatch"]]
  | .familyPack =>
      match arguments node wires with
      | some values => [[.family values]]
      | none => [[.invalid "FamilyPack argument mismatch"]]
  | .familyGetStatic index =>
      match arguments node wires, index.evaluate params with
      | some [.family values], some index =>
          [[values[index.toNat]?.getD (.invalid "FamilyGetStatic index out of range")]]
      | _, _ => [[.invalid "FamilyGetStatic argument mismatch"]]
  | .familyGetDynamic =>
      match arguments node wires with
      | some [.family values, .integer index] =>
          [[values[index.toNat]?.getD (.invalid "FamilyGetDynamic index out of range")]]
      | _ => [[.invalid "FamilyGetDynamic argument mismatch"]]
  | .subgraphCall definition bindings =>
      match arguments node wires, evaluateBindings params bindings with
      | some values, some evaluatedBindings => runChild definition (evaluatedBindings ++ params) values
      | _, _ => [[.invalid "SubgraphCall argument mismatch"]]
  | .parallelLoop definition count indexSlot bindings modes =>
      match arguments node wires, count.evaluate params with
      | some values, some count =>
          let initial := [List.replicate node.outputCount []]
          (evaluateParallelIterations runChild definition params indexSlot bindings modes values
            node.outputCount (List.range count.toNat) initial).map fun outputs =>
              outputs.map Value.family
      | _, _ => [[.invalid "ParallelLoop argument mismatch"]]

def bindOutputs (nodeId : Nat) (values : List Value) : WireEnvironment :=
  values.zipIdx.map (fun (value, port) => (⟨nodeId, port⟩, value))

def evaluateNodes
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (params : ParamEnvironment)
    (inputs : Environment) :
    (nodes : List Node) → Nat → List WireEnvironment → List WireEnvironment
  | [], _, states => states
  | node :: tail, nodeId, states =>
      let next := states.flatMap fun state =>
        (evaluateNode runChild samplers params inputs state node).map fun values =>
          state ++ bindOutputs nodeId values
      evaluateNodes runChild samplers params inputs tail (nodeId + 1) next

def collectOutputs (outputs : List (String × WireRef)) (wires : WireEnvironment) : Environment :=
  outputs.map fun (name, wire) =>
    (name, lookupWire wire wires |>.getD (.invalid s!"missing output {name}"))

def lookupDefinition (name : String) : List (String × Scope) → Option Scope
  | [] => none
  | (candidate, scope) :: tail =>
      if candidate = name then some scope else lookupDefinition name tail

def denoteScopeWithFuel
    (samplers : MxxSamplerFamily)
    (program : Prog) :
    Nat → Scope → ParamEnvironment → Environment → List Environment
  | 0, _, _, _ => [[("__scope_error", .invalid "child-scope recursion fuel exhausted")]]
  | fuel + 1, scope, params, inputs =>
      let runChild : ChildRunner := fun definition childParams childValues =>
        match lookupDefinition definition program.definitions with
        | none => [[.invalid s!"missing child scope {definition}"]]
        | some child =>
            (denoteScopeWithFuel samplers program fuel child childParams
              (child.inputNames.zip childValues)).map (fun environment => environment.map Prod.snd)
      (evaluateNodes runChild samplers params inputs scope.nodes 0 [[]]).map
        (collectOutputs scope.outputs)

/-- The named form of the recursive child runner used by `denoteScopeWithFuel`.
Protocol proofs can establish a child scope's support once and rewrite loop
iterations without duplicating the complete program definition. -/
def childRunnerWithFuel
    (samplers : MxxSamplerFamily)
    (program : Prog)
    (fuel : Nat) : ChildRunner :=
  fun definition childParams childValues =>
    match lookupDefinition definition program.definitions with
    | none => [[.invalid s!"missing child scope {definition}"]]
    | some child =>
        (denoteScopeWithFuel samplers program fuel child childParams
          (child.inputNames.zip childValues)).map (fun environment => environment.map Prod.snd)

theorem denoteScopeWithFuel_succ
    (samplers : MxxSamplerFamily)
    (program : Prog)
    (fuel : Nat)
    (scope : Scope)
    (params : ParamEnvironment)
    (inputs : Environment) :
    denoteScopeWithFuel samplers program (fuel + 1) scope params inputs =
      (evaluateNodes (childRunnerWithFuel samplers program fuel) samplers params inputs
        scope.nodes 0 [[]]).map (collectOutputs scope.outputs) := by
  rfl

def denote (samplers : MxxSamplerFamily) (program : Prog)
    (params : ParamEnvironment) (inputs : Environment) : List Environment :=
  denoteScopeWithFuel samplers program (program.definitions.length + 1)
    program.root params inputs

def emptySamplerFamily : MxxSamplerFamily where
  gaussianSample := fun _ => []
  trapdoorSample := fun _ => []
  samplePreimage := fun _ _ _ => []

def denotePure (program : Prog) (params : ParamEnvironment)
    (inputs : Environment) : Option Environment :=
  match denote emptySamplerFamily program params inputs with
  | [output] => some output
  | _ => none

inductive InputSource where
  | protocol (name : String)
  | artifact (stage output : String)

structure Stage where
  id : String
  program : Prog
  inputs : List (String × InputSource)

structure Workflow where
  stages : List Stage
  entrypoint : String

abbrev StageEnvironment := List (String × Environment)

def lookupStage (name : String) : StageEnvironment → Option Environment
  | [] => none
  | (candidate, value) :: tail => if candidate = name then some value else lookupStage name tail

def resolveStageInput
    (protocolInputs : Environment)
    (stages : StageEnvironment)
    (source : InputSource) : Value :=
  match source with
  | .protocol name =>
      lookupEnvironment name protocolInputs |>.getD (.invalid s!"missing protocol input {name}")
  | .artifact stage output =>
      (lookupStage stage stages >>= lookupEnvironment output) |>.getD
        (.invalid s!"missing artifact {stage}.{output}")

def stageInputs
    (protocolInputs : Environment)
    (stages : StageEnvironment)
    (stage : Stage) : Environment :=
  stage.inputs.map fun (name, source) =>
    (name, resolveStageInput protocolInputs stages source)

def evaluateStages
    (samplers : MxxSamplerFamily)
    (params : ParamEnvironment)
    (protocolInputs : Environment) :
    List Stage → List StageEnvironment → List StageEnvironment
  | [], states => states
  | stage :: tail, states =>
      let next := states.flatMap fun state =>
        (denote samplers stage.program params (stageInputs protocolInputs state stage)).map fun output =>
          state ++ [(stage.id, output)]
      evaluateStages samplers params protocolInputs tail next

def denoteWorkflow (samplers : MxxSamplerFamily) (workflow : Workflow)
    (params : ParamEnvironment) (inputs : Environment) : List Environment :=
  (evaluateStages samplers params inputs workflow.stages [[]]).map fun stages =>
    lookupStage workflow.entrypoint stages |>.getD
      [("__workflow_error", .invalid "entrypoint did not execute")]

def environmentValues (environment : Environment) : List Value :=
  environment.map Prod.snd

def Value.isValid : Value → Bool
  | .invalid _ => false
  | .family values => values.all fun value =>
      match value with
      | .invalid _ => false
      | _ => true
  | _ => true

def Value.equal : Value → Value → Bool
  | .integer left, .integer right => decide (left = right)
  | .rational left, .rational right => decide (left = right)
  | .boolean left, .boolean right => decide (left = right)
  | .bytes left, .bytes right => decide (left = right)
  | .matrix left, .matrix right => decide (left = right)
  | .trapdoor left, .trapdoor right => decide (left = right)
  | .opaque left, .opaque right => decide (left = right)
  | _, _ => false

def valuesEqual : List Value → List Value → Bool
  | [], [] => true
  | left :: leftTail, right :: rightTail =>
      left.equal right && valuesEqual leftTail rightTail
  | _, _ => false

def coefficientDistance : List Int → List Int → Option Nat
  | [], [] => some 0
  | left :: leftTail, right :: rightTail => do
      let tail ← coefficientDistance leftTail rightTail
      return max (left - right).natAbs tail
  | _, _ => none

def Value.distance : Value → Value → Option Nat
  | .integer left, .integer right => some (left - right).natAbs
  | .rational left, .rational right => if left = right then some 0 else none
  | .boolean left, .boolean right => some (if left = right then 0 else 1)
  | .bytes left, .bytes right => if left = right then some 0 else none
  | .matrix left, .matrix right => coefficientDistance left.coefficients right.coefficients
  | .opaque left, .opaque right => if left = right then some 0 else none
  | _, _ => none

def valuesDistance : List Value → List Value → Option Nat
  | [], [] => some 0
  | left :: leftTail, right :: rightTail => do
      let head ← left.distance right
      let tail ← valuesDistance leftTail rightTail
      return max head tail
  | _, _ => none

def environmentValid (environment : Environment) : Bool :=
  environment.all fun entry => entry.2.isValid

def projectOutputs (names : List String) (environment : Environment) : Environment :=
  names.map fun name =>
    (name, lookupEnvironment name environment |>.getD (.invalid s!"missing compared output {name}"))

def singleBooleanOutput (environment : Environment) : Option Bool :=
  match environmentValues environment with
  | [.boolean value] => some value
  | _ => none

def rebindInputs (program : Prog) (values : List Value) : Environment :=
  program.root.inputNames.zip values

end Mxx.Ir
