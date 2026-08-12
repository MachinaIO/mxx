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
  deriving BEq, DecidableEq, Repr

inductive ParamKey where
  | parameter (name : String)
  | loopIndex (slot : Nat)
  deriving BEq, DecidableEq, Repr

instance : Coe String ParamKey := ⟨ParamKey.parameter⟩

abbrev ParamEnvironment := List (ParamKey × ParamValue)

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
  deriving BEq, DecidableEq, Repr

/-- Lossless syntax mirror of the compile-time real expressions carried by Graph IR types.
The correctness analyzer transports this syntax for type identity; hard-bound evaluation uses the
integer cutoff recorded by sampler nodes and never re-evaluates a Gaussian sigma. -/
inductive RealExpr where
  | rational (value : Rat)
  | parameter (name : String)
  | fromInt (value : IntExpr)
  | add (left right : RealExpr)
  | subtract (left right : RealExpr)
  | multiply (left right : RealExpr)
  | divide (left right : RealExpr)
  | sqrt (value : RealExpr)
  deriving BEq, DecidableEq

def lookupParam (name : String) : ParamEnvironment → Option ParamValue
  | [] => none
  | (.parameter candidate, value) :: tail =>
      if candidate = name then some value else lookupParam name tail
  | (.loopIndex _, _) :: tail => lookupParam name tail

def lookupLoopIndex (slot : Nat) : ParamEnvironment → Option Int
  | [] => none
  | (.loopIndex candidate, .integer value) :: tail =>
      if candidate = slot then some value else lookupLoopIndex slot tail
  | _ :: tail => lookupLoopIndex slot tail

def IntExpr.evaluate (environment : ParamEnvironment) : IntExpr → Option Int
  | .constant value => some value
  | .parameter name =>
      match lookupParam name environment with
      | some (.integer value) => some value
      | _ => none
  | .loopIndex slot => lookupLoopIndex slot environment
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

theorem loopIndexNamespace_is_disjoint_from_userParameters :
    let environment : ParamEnvironment :=
      [(.loopIndex 0, .integer 3), (.parameter "__loop_0", .integer 99)]
    (IntExpr.loopIndex 0).evaluate environment = some 3 ∧
      (IntExpr.parameter "__loop_0").evaluate environment = some 99 := by
  decide

def evaluateOptionalIntExpr (environment : ParamEnvironment) :
    Option IntExpr → Option (Option Int)
  | none => some none
  | some expression => expression.evaluate environment |>.map some

inductive Value where
  | integer (value : Int)
  | rational (value : Rat)
  | boolean (value : Bool)
  | bytes (value : ByteArray)
  | matrix (value : Mxx.Matrix)
  | trapdoor (publicMatrix : Mxx.Matrix) (origin : Mxx.TrapdoorOrigin)
  | family (values : List Value)
  | opaque (description : String)
  | invalid (reason : String)

def hashArguments : List Value → Option (ByteArray × List Int)
  | .bytes key :: tail => do
      let values ← tail.mapM fun value => match value with
        | .integer value => some value
        | _ => none
      pure (key, values)
  | _ => none

abbrev Environment := List (String × Value)

def lookupEnvironment (name : String) : Environment → Option Value
  | [] => none
  | (candidate, value) :: tail =>
      if candidate = name then some value else lookupEnvironment name tail

structure WireRef where
  node : Nat
  port : Nat
  deriving BEq, DecidableEq, Repr

inductive IntBinaryOp where
  | add
  | subtract
  | multiply
  | divide
  | remainder
  deriving BEq, DecidableEq

inductive IntCompareOp where
  | equal
  | less
  | lessEqual
  deriving BEq, DecidableEq

inductive RealBinaryOp where
  | add
  | subtract
  | multiply
  | divide
  deriving BEq, DecidableEq

inductive LoopInputMode where
  | broadcast
  | zip
  | zipOffset (offset : Nat)
  deriving BEq, DecidableEq

inductive ConcatAxis where
  | rows
  | columns
  | diagonal
  deriving BEq, DecidableEq, Repr

structure MatrixTypeExpr where
  modulus : IntExpr
  ringDimension : IntExpr
  rows : IntExpr
  columns : IntExpr
  deriving BEq, DecidableEq, Repr

/-- Exact transport of the Rust Graph IR wire type. Generated correctness modules populate this
for every node output. The default exists only so pre-redesign hand-written theorem fixtures keep
compiling; the certificate analyzer rejects missing output types in generated workflows. -/
inductive WireTypeExpr where
  | constantInt
  | constantReal
  | constantBool
  | integer
  | real
  | boolean
  | bytes (length : IntExpr)
  | typedBlob (typeName : String) (schemaHash : List Nat)
  | matrix (type : MatrixTypeExpr)
  | trapdoor
      (matrix : MatrixTypeExpr)
      (sigma : RealExpr)
      (gadgetBase digitCount preimageMaxCoefficientBound : IntExpr)
  | preimage (type : MatrixTypeExpr)
  | indexedFamily (element : WireTypeExpr) (count : IntExpr)
  deriving BEq, DecidableEq

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
  | constantReal (value : RealExpr)
  | constantBool (value : Bool)
  | zeroMatrix (matrixType : MatrixTypeExpr)
  | identityMatrix (matrixType : MatrixTypeExpr)
  | constantMatrix (matrixType : MatrixTypeExpr) (coefficients : List IntExpr)
  | unitRowMatrix (matrixType : MatrixTypeExpr) (index : IntExpr)
  | unitColumnMatrix (matrixType : MatrixTypeExpr) (index : IntExpr)
  | gadgetMatrix (matrixType : MatrixTypeExpr) (base : IntExpr)
  | smallGadgetMatrix (matrixType : MatrixTypeExpr) (base : IntExpr)
  | powerOfBaseMatrix (matrixType : MatrixTypeExpr) (base exponent : IntExpr)
  | rotationMatrix (matrixType : MatrixTypeExpr) (exponent : IntExpr)
  | gadgetTrapdoor (matrixType : MatrixTypeExpr) (base : IntExpr)
  | boolToInt
  | intToReal
  | intBinary (operation : IntBinaryOp)
  | realBinary (operation : RealBinaryOp)
  | realSqrt
  | intCompare (operation : IntCompareOp)
  | bitExtract (bit : IntExpr)
  | extractCoefficient (position : IntExpr)
  | liftIntegerToConstantPolynomial (matrixType : MatrixTypeExpr)
  | select
  | uniformResidueSample (matrixType : MatrixTypeExpr)
  | uniformIntervalSample (matrixType : MatrixTypeExpr) (minimum maximum : IntExpr)
  | gaussianSample (matrixType : MatrixTypeExpr) (maxCoefficientBound : IntExpr)
  | hashSample
      (matrixType : MatrixTypeExpr)
      (variant : Mxx.HashVariant)
      (tagPrefix : List Nat)
      (tagExpressions tagDecimalExpressions tagU64LeExpressions : List IntExpr)
      (base digitCount : Option IntExpr)
  | gadgetDecompose
      (matrixType : MatrixTypeExpr) (base : IntExpr) (small : Bool) (digitCount : IntExpr)
  | trapdoorSample (matrixType : MatrixTypeExpr) (maxCoefficientBound : IntExpr)
  | trapdoorPublic
  | preimageSample (matrixType : MatrixTypeExpr) (maxCoefficientBound : IntExpr)
  | matrixAdd
  | matrixSubtract
  | matrixMultiply
  | matrixNegate
  | matrixScale (scalar : IntExpr)
  | transpose
  | slice (rows columns : Option (IntExpr × IntExpr))
  | tensor
  | concat (axis : ConcatAxis)
  | thresholdDecodeBool
      (ciphertextModulus plaintextModulus length : IntExpr)
  | thresholdDecodeInt
      (ciphertextModulus plaintextModulus length : IntExpr)
  | crtRecompose (plaintextModuli reconstructionCoefficients : List IntExpr)
  | packPolynomialCoefficients (matrixType : MatrixTypeExpr) (coefficientBits : IntExpr)
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
  | sequentialLoop
      (definition : String)
      (count : IntExpr)
      (indexSlot : Nat)
      (bindings : List (String × IntExpr))
      (carriedCount : Nat)
  deriving BEq, DecidableEq

structure Node where
  kind : NodeKind
  arguments : List WireRef
  outputCount : Nat := 1
  outputTypes : List WireTypeExpr := []
  deriving BEq, DecidableEq

/-- Compact constructors used by generated protocol modules. They only change the transport
syntax; `n kind arguments outputCount outputTypes` elaborates to the same `Node` value as the
corresponding structure literal. -/
def w (node : Nat) (port : Nat := 0) : WireRef := { node, port }

def n (kind : NodeKind) (arguments : Array WireRef) (outputCount : Nat)
    (outputTypes : Array WireTypeExpr) : Node :=
  { kind, arguments := arguments.toList, outputCount, outputTypes := outputTypes.toList }

theorem n_eq_structure_literal (kind : NodeKind) (arguments : Array WireRef) (outputCount : Nat)
    (outputTypes : Array WireTypeExpr) :
    n kind arguments outputCount outputTypes =
      { kind, arguments := arguments.toList, outputCount, outputTypes := outputTypes.toList } :=
  rfl

structure Scope where
  nodes : Array Node
  outputs : List (String × WireRef)
  inputNames : List String
  deriving BEq, DecidableEq

structure Prog where
  root : Scope
  definitions : List (String × Scope) := []
  deriving BEq, DecidableEq

abbrev WireEnvironment := List (WireRef × Value)

def lookupWire (wire : WireRef) : WireEnvironment → Option Value
  | [] => none
  | (candidate, value) :: tail => if candidate = wire then some value else lookupWire wire tail

def arguments (node : Node) (environment : WireEnvironment) : Option (List Value) :=
  node.arguments.mapM (fun wire => lookupWire wire environment)

abbrev addCoefficients := Mxx.addCoefficients

abbrev subtractCoefficients := Mxx.subtractCoefficients

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
  | bindings => do
      let evaluated ← bindings.mapM fun (name, expression) => do
        let value ← expression.evaluate params
        pure (ParamKey.parameter name, ParamValue.integer value)
      pure <| evaluated.foldl (fun environment binding =>
        binding :: environment.filter fun candidate => candidate.1 != binding.1) []

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
      let iterationParams := (.loopIndex indexSlot, .integer index) :: params
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

def evaluateSequentialIterations
    (runChild : ChildRunner)
    (definition : String)
    (params : ParamEnvironment)
    (indexSlot : Nat)
    (bindings : List (String × IntExpr))
    (invariantArguments : List Value) :
    List Nat → List (List Value) → List (List Value)
  | [], states => states
  | index :: tail, states =>
      let iterationParams := (.loopIndex indexSlot, .integer index) :: params
      let next := match evaluateBindings iterationParams bindings with
        | none => []
        | some evaluatedBindings =>
            states.flatMap fun state =>
              runChild definition (evaluatedBindings ++ iterationParams)
                (state ++ invariantArguments)
      evaluateSequentialIterations runChild definition params indexSlot bindings
        invariantArguments tail next

@[simp] theorem evaluateSequentialIterations_empty
    (runChild : ChildRunner)
    (definition : String)
    (params : ParamEnvironment)
    (indexSlot : Nat)
    (bindings : List (String × IntExpr))
    (invariantArguments : List Value)
    (indices : List Nat) :
    evaluateSequentialIterations runChild definition params indexSlot bindings
      invariantArguments indices [] = [] := by
  induction indices with
  | nil => rfl
  | cons index tail induction =>
      simp only [evaluateSequentialIterations]
      split <;> simp_all

@[simp] theorem evaluateParallelIterations_empty
    (runChild : ChildRunner)
    (definition : String)
    (params : ParamEnvironment)
    (indexSlot : Nat)
    (bindings : List (String × IntExpr))
    (modes : List LoopInputMode)
    (arguments : List Value)
    (outputCount : Nat)
    (indices : List Nat) :
    evaluateParallelIterations runChild definition params indexSlot bindings modes arguments
      outputCount indices [] = [] := by
  induction indices with
  | nil => rfl
  | cons index tail induction =>
      simp only [evaluateParallelIterations]
      split <;> simp_all

/-- A single nondeterministic execution trace through a sequential loop.  Every constructor
records the evaluated parameter bindings and the concrete child-support member selected at that
iteration, without materializing the Cartesian product of all sampler supports. -/
inductive SequentialIterationsTrace
    (runChild : ChildRunner)
    (definition : String)
    (params : ParamEnvironment)
    (indexSlot : Nat)
    (bindings : List (String × IntExpr))
    (invariantArguments : List Value) :
    List Nat → List Value → List Value → Prop
  | nil (state) :
      SequentialIterationsTrace runChild definition params indexSlot bindings
        invariantArguments [] state state
  | cons (index : Nat) (tail) (state) (evaluatedBindings) (next) (final)
      (bindingsEvaluate :
        evaluateBindings ((.loopIndex indexSlot, .integer index) :: params) bindings =
          some evaluatedBindings)
      (childMember : next ∈
        runChild definition
          (evaluatedBindings ++ ((.loopIndex indexSlot, .integer index) :: params))
          (state ++ invariantArguments))
      (rest : SequentialIterationsTrace runChild definition params indexSlot bindings
        invariantArguments tail next final) :
      SequentialIterationsTrace runChild definition params indexSlot bindings
        invariantArguments (index :: tail) state final

theorem mem_evaluateSequentialIterations_iff_exists_trace
    (runChild : ChildRunner)
    (definition : String)
    (params : ParamEnvironment)
    (indexSlot : Nat)
    (bindings : List (String × IntExpr))
    (invariantArguments : List Value)
    (indices : List Nat)
    (states : List (List Value))
    (final : List Value) :
    final ∈ evaluateSequentialIterations runChild definition params indexSlot bindings
      invariantArguments indices states ↔
      ∃ initial ∈ states,
        SequentialIterationsTrace runChild definition params indexSlot bindings
          invariantArguments indices initial final := by
  induction indices generalizing states final with
  | nil =>
      constructor
      · intro member
        exact ⟨final, member, .nil final⟩
      · rintro ⟨initial, member, trace⟩
        cases trace
        exact member
  | cons index tail induction =>
      simp only [evaluateSequentialIterations]
      let iterationParams := (.loopIndex indexSlot, .integer index) :: params
      cases bindingsResult : evaluateBindings iterationParams bindings with
      | none =>
          simp only
          simp only [evaluateSequentialIterations_empty, List.not_mem_nil, false_iff]
          rintro ⟨initial, initialMember, trace⟩
          cases trace with
          | cons _ _ _ evaluatedBindings _ _ bindingsEvaluate _ _ =>
              rw [bindingsResult] at bindingsEvaluate
              contradiction
      | some evaluatedBindings =>
          simp only
          rw [induction]
          constructor
          · rintro ⟨next, nextMember, rest⟩
            simp only [List.mem_flatMap] at nextMember
            obtain ⟨initial, initialMember, childMember⟩ := nextMember
            exact ⟨initial, initialMember,
              .cons index tail initial evaluatedBindings next final bindingsResult childMember rest⟩
          · rintro ⟨initial, initialMember, trace⟩
            cases trace with
            | cons _ _ _ chosenBindings next _ bindingsEvaluate childMember rest =>
                rw [bindingsResult] at bindingsEvaluate
                cases bindingsEvaluate
                refine ⟨next, ?_, rest⟩
                simp only [List.mem_flatMap]
                exact ⟨initial, initialMember, childMember⟩

theorem SequentialIterationsTrace.invariant
    {runChild : ChildRunner}
    {definition : String}
    {params : ParamEnvironment}
    {indexSlot : Nat}
    {bindings : List (String × IntExpr)}
    {invariantArguments : List Value}
    (predicate : List Value → Prop)
    (preserved : ∀ (index : Nat) evaluatedBindings state next,
      evaluateBindings ((.loopIndex indexSlot, .integer index) :: params) bindings =
        some evaluatedBindings →
      next ∈ runChild definition
        (evaluatedBindings ++ ((.loopIndex indexSlot, .integer index) :: params))
        (state ++ invariantArguments) →
      predicate state → predicate next) :
    ∀ {indices initial final},
      SequentialIterationsTrace runChild definition params indexSlot bindings
        invariantArguments indices initial final →
      predicate initial → predicate final := by
  intro indices initial final trace initialProperty
  induction trace with
  | nil => exact initialProperty
  | cons index _ state evaluatedBindings next _ bindingsEvaluate childMember _ induction =>
      exact induction (preserved index evaluatedBindings state next bindingsEvaluate childMember
        initialProperty)

theorem evaluateSequentialIterations_invariant
    (runChild : ChildRunner)
    (definition : String)
    (params : ParamEnvironment)
    (indexSlot : Nat)
    (bindings : List (String × IntExpr))
    (invariantArguments : List Value)
    (indices : List Nat)
    (states : List (List Value))
    (predicate : List Value → Prop)
    (initial : ∀ state ∈ states, predicate state)
    (preserved : ∀ (index : Nat) evaluatedBindings state next,
      evaluateBindings ((.loopIndex indexSlot, .integer index) :: params) bindings =
        some evaluatedBindings →
      next ∈ runChild definition
        (evaluatedBindings ++ ((.loopIndex indexSlot, .integer index) :: params))
        (state ++ invariantArguments) →
      predicate state → predicate next)
    {final : List Value}
    (finalMember : final ∈
      evaluateSequentialIterations runChild definition params indexSlot bindings
        invariantArguments indices states) :
    predicate final := by
  obtain ⟨first, firstMember, trace⟩ :=
    (mem_evaluateSequentialIterations_iff_exists_trace runChild definition params indexSlot
      bindings invariantArguments indices states final).mp finalMember
  exact trace.invariant predicate preserved (initial first firstMember)

/-- A concrete parallel-loop accumulator trace.  `next` is exactly the port-wise append of the
selected child output to the previous accumulator, so the relation preserves iteration order. -/
inductive ParallelIterationsTrace
    (runChild : ChildRunner)
    (definition : String)
    (params : ParamEnvironment)
    (indexSlot : Nat)
    (bindings : List (String × IntExpr))
    (modes : List LoopInputMode)
    (arguments : List Value) :
    List Nat → List (List Value) → List (List Value) → Prop
  | nil (state) :
      ParallelIterationsTrace runChild definition params indexSlot bindings modes arguments
        [] state state
  | cons (index : Nat) (tail) (state) (evaluatedBindings) (childValues) (final)
      (bindingsEvaluate :
        evaluateBindings ((.loopIndex indexSlot, .integer index) :: params) bindings =
          some evaluatedBindings)
      (childMember : childValues ∈
        runChild definition
          (evaluatedBindings ++ ((.loopIndex indexSlot, .integer index) :: params))
          ((modes.zip arguments).map fun (mode, value) => loopArgument mode index value))
      (rest : ParallelIterationsTrace runChild definition params indexSlot bindings modes arguments
        tail (appendPortValues state childValues) final) :
      ParallelIterationsTrace runChild definition params indexSlot bindings modes arguments
        (index :: tail) state final

theorem mem_evaluateParallelIterations_iff_exists_trace
    (runChild : ChildRunner)
    (definition : String)
    (params : ParamEnvironment)
    (indexSlot : Nat)
    (bindings : List (String × IntExpr))
    (modes : List LoopInputMode)
    (arguments : List Value)
    (outputCount : Nat)
    (indices : List Nat)
    (states : List (List (List Value)))
    (final : List (List Value)) :
    final ∈ evaluateParallelIterations runChild definition params indexSlot bindings modes
      arguments outputCount indices states ↔
      ∃ initial ∈ states,
        ParallelIterationsTrace runChild definition params indexSlot bindings modes arguments
          indices initial final := by
  induction indices generalizing states final with
  | nil =>
      constructor
      · intro member
        exact ⟨final, member, .nil final⟩
      · rintro ⟨initial, member, trace⟩
        cases trace
        exact member
  | cons index tail induction =>
      simp only [evaluateParallelIterations]
      let iterationParams := (.loopIndex indexSlot, .integer index) :: params
      cases bindingsResult : evaluateBindings iterationParams bindings with
      | none =>
          simp only
          simp only [evaluateParallelIterations_empty, List.not_mem_nil, false_iff]
          rintro ⟨initial, initialMember, trace⟩
          cases trace with
          | cons _ _ _ evaluatedBindings _ _ bindingsEvaluate _ _ =>
              rw [bindingsResult] at bindingsEvaluate
              contradiction
      | some evaluatedBindings =>
          simp only
          rw [induction]
          constructor
          · rintro ⟨next, nextMember, rest⟩
            simp only [List.mem_flatMap, List.mem_map] at nextMember
            obtain ⟨initial, initialMember, childValues, childMember, rfl⟩ := nextMember
            exact ⟨initial, initialMember,
              .cons index tail initial evaluatedBindings childValues final bindingsResult
                childMember rest⟩
          · rintro ⟨initial, initialMember, trace⟩
            cases trace with
            | cons _ _ _ chosenBindings childValues _ bindingsEvaluate childMember rest =>
                rw [bindingsResult] at bindingsEvaluate
                cases bindingsEvaluate
                refine ⟨appendPortValues initial childValues, ?_, rest⟩
                simp only [List.mem_flatMap, List.mem_map]
                exact ⟨initial, initialMember, childValues, childMember, rfl⟩

theorem ParallelIterationsTrace.invariant
    {runChild : ChildRunner}
    {definition : String}
    {params : ParamEnvironment}
    {indexSlot : Nat}
    {bindings : List (String × IntExpr)}
    {modes : List LoopInputMode}
    {arguments : List Value}
    (predicate : List (List Value) → Prop)
    (preserved : ∀ (index : Nat) evaluatedBindings state childValues,
      evaluateBindings ((.loopIndex indexSlot, .integer index) :: params) bindings =
        some evaluatedBindings →
      childValues ∈ runChild definition
        (evaluatedBindings ++ ((.loopIndex indexSlot, .integer index) :: params))
        ((modes.zip arguments).map fun (mode, value) => loopArgument mode index value) →
      predicate state → predicate (appendPortValues state childValues)) :
    ∀ {indices initial final},
      ParallelIterationsTrace runChild definition params indexSlot bindings modes arguments
        indices initial final →
      predicate initial → predicate final := by
  intro indices initial final trace initialProperty
  induction trace with
  | nil => exact initialProperty
  | cons index _ state evaluatedBindings childValues _ bindingsEvaluate childMember _ induction =>
      exact induction (preserved index evaluatedBindings state childValues bindingsEvaluate
        childMember initialProperty)

theorem evaluateParallelIterations_invariant
    (runChild : ChildRunner)
    (definition : String)
    (params : ParamEnvironment)
    (indexSlot : Nat)
    (bindings : List (String × IntExpr))
    (modes : List LoopInputMode)
    (arguments : List Value)
    (outputCount : Nat)
    (indices : List Nat)
    (states : List (List (List Value)))
    (predicate : List (List Value) → Prop)
    (initial : ∀ state ∈ states, predicate state)
    (preserved : ∀ (index : Nat) evaluatedBindings state childValues,
      evaluateBindings ((.loopIndex indexSlot, .integer index) :: params) bindings =
        some evaluatedBindings →
      childValues ∈ runChild definition
        (evaluatedBindings ++ ((.loopIndex indexSlot, .integer index) :: params))
        ((modes.zip arguments).map fun (mode, value) => loopArgument mode index value) →
      predicate state → predicate (appendPortValues state childValues))
    {final : List (List Value)}
    (finalMember : final ∈
      evaluateParallelIterations runChild definition params indexSlot bindings modes arguments
        outputCount indices states) :
    predicate final := by
  obtain ⟨first, firstMember, trace⟩ :=
    (mem_evaluateParallelIterations_iff_exists_trace runChild definition params indexSlot bindings
      modes arguments outputCount indices states final).mp finalMember
  exact trace.invariant predicate preserved (initial first firstMember)

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
  | .constantReal _ => [[.invalid "exact real-expression execution is unavailable"]]
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
  | .unitRowMatrix _ _ | .unitColumnMatrix _ _ | .powerOfBaseMatrix _ _ _ | .rotationMatrix _ _ =>
      [[.invalid "constant-matrix variant execution is unavailable"]]
  | .gadgetMatrix matrixType base =>
      match matrixType.evaluate params, base.evaluate params with
      | some matrixParams, some base =>
          let digits := if matrixParams.rows = 0 then 0 else matrixParams.columns / matrixParams.rows
          match samplers.layoutId matrixParams with
          | some paramsId =>
              match samplers.gadgetPublicMatrix paramsId matrixParams matrixParams.rows base false digits with
              | some matrix => [[.matrix (matrix.withSamplerParams matrixParams)]]
              | none => [[.invalid "gadget-matrix layout is invalid"]]
          | none => [[.invalid "gadget-matrix layout is unavailable"]]
      | _, _ => [[.invalid "gadget-matrix evaluation failed"]]
  | .smallGadgetMatrix matrixType base =>
      match matrixType.evaluate params, base.evaluate params with
      | some matrixParams, some base =>
          let digits := if matrixParams.rows = 0 then 0 else matrixParams.columns / matrixParams.rows
          match samplers.layoutId matrixParams with
          | some paramsId =>
              match samplers.gadgetPublicMatrix paramsId matrixParams matrixParams.rows base true digits with
              | some matrix => [[.matrix (matrix.withSamplerParams matrixParams)]]
              | none => [[.invalid "small-gadget-matrix layout is invalid"]]
          | none => [[.invalid "small-gadget-matrix layout is unavailable"]]
      | _, _ => [[.invalid "small-gadget-matrix evaluation failed"]]
  | .gadgetTrapdoor matrixType base =>
      match matrixType.evaluate params (.constant 0), base.evaluate params with
      | some matrixParams, some base =>
          let digits := if matrixParams.rows = 0 then 0 else matrixParams.columns / matrixParams.rows
          match samplers.layoutId matrixParams with
          | some paramsId =>
              match samplers.gadgetPublicMatrix paramsId matrixParams matrixParams.rows base false digits with
              | some publicMatrix =>
                  let publicMatrix := publicMatrix.withSamplerParams matrixParams
                  [[.matrix publicMatrix, .trapdoor publicMatrix (.gadget paramsId base false digits)]]
              | none => [[.invalid "gadget-trapdoor layout is invalid"]]
          | none => [[.invalid "gadget-trapdoor layout is unavailable"]]
      | _, _ => [[.invalid "gadget-trapdoor parameter evaluation failed"]]
  | .boolToInt =>
      match arguments node wires with
      | some [.boolean value] => [[.integer (if value then 1 else 0)]]
      | _ => [[.invalid "BoolToInt argument mismatch"]]
  | .intToReal | .realBinary _ | .realSqrt =>
      [[.invalid "exact real-expression execution is unavailable"]]
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
          [[.integer (Mxx.reduceCoefficient matrix.modulus
            (matrix.coefficients.getD position.toNat 0))]]
      | _, _ => [[.invalid "coefficient-extraction argument mismatch"]]
  | .liftIntegerToConstantPolynomial matrixType =>
      match arguments node wires, matrixType.evaluate params with
      | some [.integer value], some matrixParams =>
          [[.matrix (Mxx.Matrix.withSamplerParams
            ({ coefficients := [value % matrixParams.modulus] } : Mxx.Matrix) matrixParams)]]
      | _, _ => [[.invalid "constant-polynomial lift argument mismatch"]]
  | .select =>
      match arguments node wires with
      | some (.integer index :: branches) =>
          [[branches[index.toNat]?.getD (.invalid "Select index out of range")]]
      | _ => [[.invalid "Select argument mismatch"]]
  | .uniformResidueSample matrixType =>
      match matrixType.evaluate params with
      | some matrixParams =>
          (uniformMatrixSupport matrixParams 0 (matrixParams.modulus - 1)).map
            (fun sample => [.matrix sample])
      | none => [[.invalid "uniform-residue-sample parameter evaluation failed"]]
  | .uniformIntervalSample matrixType minimum maximum =>
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
  | .hashSample matrixType variant tagPrefix tagExpressions tagDecimalExpressions
      tagU64LeExpressions base digitCount =>
      match (arguments node wires).bind hashArguments, matrixType.evaluate params (.constant 0),
          tagExpressions.mapM (IntExpr.evaluate params),
          tagDecimalExpressions.mapM (IntExpr.evaluate params),
          tagU64LeExpressions.mapM (IntExpr.evaluate params),
          evaluateOptionalIntExpr params base, evaluateOptionalIntExpr params digitCount with
      | some (key, trailingIntegerTagValues), some matrixParams, some tagValues, some tagDecimalValues,
          some tagU64LeValues, some base, some digitCount =>
          let query : Mxx.HashQuery := {
            params := matrixParams
            key
            variant
            tagPrefix
            tagValues
            tagDecimalValues
            tagU64LeValues
            trailingIntegerTagValues
            base
            digitCount
          }
          [[.matrix ((samplers.hashSample query).withSamplerParams matrixParams)]]
      | _, _, _, _, _, _, _ => [[.invalid "hash-sample parameter evaluation failed"]]
  | .gadgetDecompose matrixType base small digitCount =>
      match arguments node wires, matrixType.evaluate params (.constant 0),
          base.evaluate params, digitCount.evaluate params with
      | some [.matrix value], some matrixParams, some base, some digitCount =>
          if base <= 1 || digitCount <= 0 then
            [[.invalid "gadget-decomposition layout is invalid"]]
          else
            match samplers.layoutId matrixParams with
            | some paramsId =>
                match samplers.gadgetDecompose paramsId matrixParams base small digitCount.toNat value with
                | some output => [[.matrix (output.withSamplerParams matrixParams)]]
                | none => [[.invalid "gadget-decomposition layout is invalid"]]
            | none => [[.invalid "gadget-decomposition layout is unavailable"]]
      | _, _, _, _ => [[.invalid "gadget-decomposition argument mismatch"]]
  | .trapdoorSample matrixType cutoff =>
      match matrixType.evaluate params cutoff with
      | some matrixParams =>
          (samplers.trapdoorSample matrixParams).map fun publicMatrix =>
            let publicMatrix := publicMatrix.withSamplerParams matrixParams
            [.matrix publicMatrix, .trapdoor publicMatrix .sampled]
      | none => [[.invalid "trapdoor parameter evaluation failed"]]
  | .trapdoorPublic =>
      match arguments node wires with
      | some [.trapdoor publicMatrix _] => [[.matrix publicMatrix]]
      | _ => [[.invalid "TrapdoorPublic argument mismatch"]]
  | .preimageSample matrixType cutoff =>
      match arguments node wires, matrixType.evaluate params cutoff with
      | some [.matrix publicMatrix, .trapdoor trapdoorPublic origin, .matrix target], some matrixParams =>
          if publicMatrix != trapdoorPublic then
            [[.invalid "preimage trapdoor/public-matrix mismatch"]]
          else
            match origin with
            | .sampled =>
                (samplers.samplePreimage matrixParams publicMatrix target).map (fun sample =>
                  [.matrix (sample.withSamplerParams matrixParams)])
            | .gadget paramsId base small digitCount =>
                if base <= 1 || digitCount = 0 then
                  [[.invalid "gadget preimage decomposition layout is invalid"]]
                else
                  match samplers.gadgetDecompose paramsId matrixParams base small digitCount target with
                  | some output => [[.matrix (output.withSamplerParams matrixParams)]]
                  | none => [[.invalid "gadget preimage decomposition layout is invalid"]]
      | _, _ => [[.invalid "preimage-sample argument mismatch"]]
  | .matrixAdd =>
      match arguments node wires with
      | some [.matrix left, .matrix right] => [[.matrix (Mxx.matrixAdd left right)]]
      | _ => [[.invalid "matrix addition argument mismatch"]]
  | .matrixSubtract =>
      match arguments node wires with
      | some [.matrix left, .matrix right] => [[.matrix (Mxx.matrixSubtract left right)]]
      | _ => [[.invalid "matrix subtraction argument mismatch"]]
  | .matrixMultiply =>
      match arguments node wires with
      | some [.matrix left, .matrix right] => [[.matrix (Mxx.matrixMultiply left right)]]
      | _ => [[.invalid "matrix multiplication argument mismatch"]]
  | .matrixNegate =>
      match arguments node wires with
      | some [.matrix value] => [[.matrix (Mxx.matrixNegate value)]]
      | _ => [[.invalid "matrix negation argument mismatch"]]
  | .matrixScale scalar =>
      match arguments node wires, scalar.evaluate params with
      | some [.matrix value], some scalar => [[.matrix (Mxx.matrixScale scalar value)]]
      | _, _ => [[.invalid "matrix scaling argument mismatch"]]
  | .transpose | .tensor => [[.invalid "matrix transform execution is unavailable"]]
  | .slice rows columns =>
      match arguments node wires with
      | some [.matrix value] =>
          let evaluateRange (range : Option (IntExpr × IntExpr)) (length : Nat) :=
            match range with
            | none => some (0, length)
            | some (start, stop) => do
                let start ← start.evaluate params
                let stop ← stop.evaluate params
                if start < 0 ∨ stop < start then none else some (start.toNat, stop.toNat)
          match evaluateRange rows value.rows, evaluateRange columns value.columns with
          | some (rowStart, rowEnd), some (columnStart, columnEnd) =>
              [[.matrix (Mxx.matrixSlice value rowStart rowEnd columnStart columnEnd)]]
          | _, _ => [[.invalid "Slice range evaluation failed"]]
      | _ => [[.invalid "Slice argument mismatch"]]
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
  | .thresholdDecodeInt _ _ _ | .crtRecompose _ _ | .packPolynomialCoefficients _ _ =>
      [[.invalid "matrix transform execution is unavailable"]]
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
  | .sequentialLoop definition count indexSlot bindings carriedCount =>
      match arguments node wires, count.evaluate params with
      | some values, some count =>
          evaluateSequentialIterations runChild definition params indexSlot bindings
            (values.drop carriedCount) (List.range count.toNat) [values.take carriedCount]
      | _, _ => [[.invalid "SequentialLoop argument mismatch"]]

theorem evaluateNode_parallelLoop_of_arguments
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (params : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (definition : String)
    (count : IntExpr)
    (indexSlot : Nat)
    (bindings : List (String × IntExpr))
    (modes : List LoopInputMode)
    (argumentRefs : List WireRef)
    (outputCount : Nat)
    (values : List Value)
    (evaluatedCount : Int)
    (argumentsEvaluate : argumentRefs.mapM (fun wire => lookupWire wire wires) = some values)
    (countEvaluate : count.evaluate params = some evaluatedCount) :
    evaluateNode runChild samplers params inputs wires {
        kind := .parallelLoop definition count indexSlot bindings modes
        arguments := argumentRefs
        outputCount
      } =
      (evaluateParallelIterations runChild definition params indexSlot bindings modes values
        outputCount (List.range evaluatedCount.toNat)
        [List.replicate outputCount []]).map fun outputs => outputs.map Value.family := by
  simp [evaluateNode, arguments, argumentsEvaluate, countEvaluate]

theorem evaluateNode_sequentialLoop_of_arguments
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (params : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (definition : String)
    (count : IntExpr)
    (indexSlot : Nat)
    (bindings : List (String × IntExpr))
    (carriedCount : Nat)
    (argumentRefs : List WireRef)
    (outputCount : Nat)
    (values : List Value)
    (evaluatedCount : Int)
    (argumentsEvaluate : argumentRefs.mapM (fun wire => lookupWire wire wires) = some values)
    (countEvaluate : count.evaluate params = some evaluatedCount) :
    evaluateNode runChild samplers params inputs wires {
        kind := .sequentialLoop definition count indexSlot bindings carriedCount
        arguments := argumentRefs
        outputCount
      } =
      evaluateSequentialIterations runChild definition params indexSlot bindings
        (values.drop carriedCount) (List.range evaluatedCount.toNat) [values.take carriedCount] := by
  simp [evaluateNode, arguments, argumentsEvaluate, countEvaluate]

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

theorem evaluateNodes_append
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (params : ParamEnvironment)
    (inputs : Environment)
    (left right : List Node)
    (nodeId : Nat)
    (states : List WireEnvironment) :
    evaluateNodes runChild samplers params inputs (left ++ right) nodeId states =
      evaluateNodes runChild samplers params inputs right (nodeId + left.length)
        (evaluateNodes runChild samplers params inputs left nodeId states) := by
  induction left generalizing nodeId states with
  | nil => simp [evaluateNodes]
  | cons head tail induction =>
      simp only [List.cons_append, evaluateNodes, List.length_cons]
      rw [induction]
      congr 1
      omega

/-- One concrete execution path through a node list. Unlike `evaluateNodes`, this relation does
not materialize the Cartesian product of every sampler support and is therefore suitable for
proofs that start from membership in the executable support. -/
inductive EvaluatesNodesPath
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (params : ParamEnvironment)
    (inputs : Environment) :
    Nat → List Node → WireEnvironment → WireEnvironment → Prop
  | nil (nodeId state) : EvaluatesNodesPath runChild samplers params inputs nodeId [] state state
  | cons (nodeId node nodes state values output)
      (valuesMember : values ∈ evaluateNode runChild samplers params inputs state node)
      (tail : EvaluatesNodesPath runChild samplers params inputs (nodeId + 1) nodes
        (state ++ bindOutputs nodeId values) output) :
      EvaluatesNodesPath runChild samplers params inputs nodeId (node :: nodes) state output

theorem evaluatesNodesPath_cons_iff
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (params : ParamEnvironment)
    (inputs : Environment)
    (nodeId : Nat)
    (node : Node)
    (nodes : List Node)
    (state output : WireEnvironment) :
    EvaluatesNodesPath runChild samplers params inputs nodeId (node :: nodes) state output ↔
      ∃ values ∈ evaluateNode runChild samplers params inputs state node,
        EvaluatesNodesPath runChild samplers params inputs (nodeId + 1) nodes
          (state ++ bindOutputs nodeId values) output := by
  constructor
  · intro path
    cases path with
    | cons _ _ _ _ values _ valuesMember tail => exact ⟨values, valuesMember, tail⟩
  · rintro ⟨values, valuesMember, tail⟩
    exact .cons _ _ _ _ _ _ valuesMember tail

theorem mem_evaluateNodes_iff_exists_path
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (params : ParamEnvironment)
    (inputs : Environment)
    (nodes : List Node)
    (nodeId : Nat)
    (states : List WireEnvironment)
    (output : WireEnvironment) :
    output ∈ evaluateNodes runChild samplers params inputs nodes nodeId states ↔
      ∃ initial ∈ states,
        EvaluatesNodesPath runChild samplers params inputs nodeId nodes initial output := by
  induction nodes generalizing nodeId states output with
  | nil =>
      constructor
      · intro member
        exact ⟨output, member, .nil _ _⟩
      · rintro ⟨initial, member, path⟩
        cases path
        exact member
  | cons node nodes induction =>
      rw [evaluateNodes, induction]
      constructor
      · rintro ⟨next, nextMember, path⟩
        simp only [List.mem_flatMap, List.mem_map] at nextMember
        obtain ⟨initial, initialMember, values, valuesMember, rfl⟩ := nextMember
        exact ⟨initial, initialMember, .cons _ _ _ _ _ _ valuesMember path⟩
      · rintro ⟨initial, initialMember, path⟩
        cases path with
        | cons _ _ _ _ values _ valuesMember tail =>
            refine ⟨_, ?_, tail⟩
            simp only [List.mem_flatMap, List.mem_map]
            exact ⟨initial, initialMember, values, valuesMember, rfl⟩

/-- Split a concrete IR execution path at a syntactic node-list boundary. Generated protocol
proofs use this lemma instead of destructing every preceding node by hand. -/
theorem EvaluatesNodesPath.split
    {runChild : ChildRunner}
    {samplers : MxxSamplerFamily}
    {params : ParamEnvironment}
    {inputs : Environment}
    {nodeId : Nat}
    {left right : List Node}
    {initial output : WireEnvironment}
    (path : EvaluatesNodesPath runChild samplers params inputs nodeId
      (left ++ right) initial output) :
    ∃ middle,
      EvaluatesNodesPath runChild samplers params inputs nodeId left initial middle ∧
      EvaluatesNodesPath runChild samplers params inputs (nodeId + left.length) right
        middle output := by
  induction left generalizing nodeId initial with
  | nil =>
      exact ⟨initial, .nil nodeId initial, by simpa using path⟩
  | cons node tail induction =>
      cases path with
      | cons _ _ _ _ values _ valuesMember rest =>
          obtain ⟨middle, leftPath, rightPath⟩ := induction rest
          refine ⟨middle, .cons _ _ _ _ _ _ valuesMember leftPath, ?_⟩
          have nodeIdEq : nodeId + 1 + tail.length = nodeId + (node :: tail).length := by
            simp
            omega
          rw [← nodeIdEq]
          exact rightPath

/-- Invert only a selected generated node. The returned prefix state is the exact wire
environment passed to that node, and `valuesMember` retains its sampler-support obligation. -/
theorem EvaluatesNodesPath.atNode
    {runChild : ChildRunner}
    {samplers : MxxSamplerFamily}
    {params : ParamEnvironment}
    {inputs : Environment}
    {nodeId : Nat}
    {preNodes postNodes : List Node}
    {node : Node}
    {initial output : WireEnvironment}
    (path : EvaluatesNodesPath runChild samplers params inputs nodeId
      (preNodes ++ node :: postNodes) initial output) :
    ∃ before values,
      EvaluatesNodesPath runChild samplers params inputs nodeId preNodes initial before ∧
      values ∈ evaluateNode runChild samplers params inputs before node ∧
      EvaluatesNodesPath runChild samplers params inputs (nodeId + preNodes.length + 1)
        postNodes (before ++ bindOutputs (nodeId + preNodes.length) values) output := by
  obtain ⟨before, prefixPath, afterPath⟩ :=
    path.split (left := preNodes) (right := node :: postNodes)
  cases afterPath with
  | cons _ _ _ _ values _ valuesMember suffixPath =>
      exact ⟨before, values, prefixPath, valuesMember, by simpa [Nat.add_assoc] using suffixPath⟩

/-- Index-based form of `atNode`, convenient for large generated node lists. -/
theorem EvaluatesNodesPath.atNodeIndex
    {runChild : ChildRunner}
    {samplers : MxxSamplerFamily}
    {params : ParamEnvironment}
    {inputs : Environment}
    {nodeId : Nat}
    {nodes : List Node}
    {initial output : WireEnvironment}
    (path : EvaluatesNodesPath runChild samplers params inputs nodeId nodes initial output)
    (index : Nat)
    (inBounds : index < nodes.length) :
    ∃ before values,
      EvaluatesNodesPath runChild samplers params inputs nodeId (nodes.take index) initial before ∧
      values ∈ evaluateNode runChild samplers params inputs before nodes[index] ∧
      EvaluatesNodesPath runChild samplers params inputs (nodeId + index + 1)
        (nodes.drop (index + 1))
        (before ++ bindOutputs (nodeId + index) values) output := by
  have decomposition : nodes = nodes.take index ++ nodes[index] :: nodes.drop (index + 1) := by
    calc
      nodes = nodes.take index ++ nodes.drop index := (List.take_append_drop index nodes).symm
      _ = nodes.take index ++ nodes[index] :: nodes.drop (index + 1) := by
        rw [List.cons_getElem_drop_succ (l := nodes) (n := index)]
  rw [decomposition] at path
  simpa [Nat.min_eq_left (Nat.le_of_lt inBounds)] using path.atNode

/-- Appending new SSA bindings cannot change an already-resolved wire. -/
theorem lookupWire_append_of_eq_some
    {wire : WireRef} {value : Value} {left right : WireEnvironment}
    (resolved : lookupWire wire left = some value) :
    lookupWire wire (left ++ right) = some value := by
  induction left with
  | nil => simp [lookupWire] at resolved
  | cons head tail induction =>
      rcases head with ⟨candidate, candidateValue⟩
      by_cases same : candidate = wire
      · rw [lookupWire, if_pos same] at resolved
        rw [List.cons_append, lookupWire, if_pos same]
        exact resolved
      · rw [lookupWire, if_neg same] at resolved
        rw [List.cons_append, lookupWire, if_neg same]
        exact induction resolved

/-- Once an SSA wire is present, every later node on the same path preserves its value. -/
theorem EvaluatesNodesPath.lookupWire_preserved
    {runChild : ChildRunner}
    {samplers : MxxSamplerFamily}
    {params : ParamEnvironment}
    {inputs : Environment}
    {nodeId : Nat}
    {nodes : List Node}
    {initial output : WireEnvironment}
    {wire : WireRef}
    {value : Value}
    (path : EvaluatesNodesPath runChild samplers params inputs nodeId nodes initial output)
    (resolved : lookupWire wire initial = some value) :
    lookupWire wire output = some value := by
  induction path with
  | nil => exact resolved
  | cons _ _ _ _ values _ _ _ induction =>
      exact induction (lookupWire_append_of_eq_some resolved)

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
      (evaluateNodes runChild samplers params inputs scope.nodes.toList 0 [[]]).map
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
        scope.nodes.toList 0 [[]]).map (collectOutputs scope.outputs) := by
  rfl

def denote (samplers : MxxSamplerFamily) (program : Prog)
    (params : ParamEnvironment) (inputs : Environment) : List Environment :=
  denoteScopeWithFuel samplers program (program.definitions.length + 1)
    program.root params inputs

def emptySamplerFamily : MxxSamplerFamily where
  gaussianSample := fun _ => []
  hashSample := fun _ => { coefficients := [] }
  layoutId := fun _ => none
  gadgetPublicMatrix := fun _ _ _ _ _ _ => none
  gadgetDecompose := fun _ _ _ _ _ _ => none
  smallDecompositionInputLimit := fun _ _ => none
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
  | .trapdoor left leftOrigin, .trapdoor right rightOrigin =>
      decide (left = right ∧ leftOrigin = rightOrigin)
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
