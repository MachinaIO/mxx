/- A small typed Boolean circuit model. Inputs use `Fin inputCount`, so malformed input
   references cannot be represented. The recursive `WellFormed` proof is retained in `BoolCircuit`
   to make validity explicit in generated statements. -/
namespace Mxx.Gadgets

inductive BoolExpr (inputCount : Nat) where
  | input (index : Fin inputCount)
  | constant (value : Bool)
  | not (argument : BoolExpr inputCount)
  | and (left right : BoolExpr inputCount)
  | xor (left right : BoolExpr inputCount)

inductive BoolExpr.WellFormed {inputCount : Nat} : BoolExpr inputCount → Prop where
  | input (index : Fin inputCount) : WellFormed (.input index)
  | constant (value : Bool) : WellFormed (.constant value)
  | not {argument : BoolExpr inputCount} : WellFormed argument → WellFormed (.not argument)
  | and {left right : BoolExpr inputCount} :
      WellFormed left → WellFormed right → WellFormed (.and left right)
  | xor {left right : BoolExpr inputCount} :
      WellFormed left → WellFormed right → WellFormed (.xor left right)

def BoolExpr.eval {inputCount : Nat}
    (inputs : Fin inputCount → Bool) : BoolExpr inputCount → Bool
  | .input index => inputs index
  | .constant value => value
  | .not argument => !(argument.eval inputs)
  | .and left right => left.eval inputs && right.eval inputs
  | .xor left right =>
      (left.eval inputs && !(right.eval inputs)) || (!(left.eval inputs) && right.eval inputs)

structure BoolCircuit (inputCount : Nat) where
  output : BoolExpr inputCount
  valid : output.WellFormed

def BoolCircuit.eval {inputCount : Nat} (circuit : BoolCircuit inputCount)
    (inputs : Fin inputCount → Bool) : Bool :=
  circuit.output.eval inputs

theorem BoolExpr.wellFormed {inputCount : Nat} (expression : BoolExpr inputCount) :
    expression.WellFormed := by
  induction expression with
  | input index => exact .input index
  | constant value => exact .constant value
  | not argument ih => exact BoolExpr.WellFormed.not ih
  | and left right ihLeft ihRight => exact BoolExpr.WellFormed.and ihLeft ihRight
  | xor left right ihLeft ihRight => exact BoolExpr.WellFormed.xor ihLeft ihRight

@[simp] theorem BoolExpr.eval_input {inputCount : Nat} (inputs : Fin inputCount → Bool)
    (index : Fin inputCount) :
    (.input index : BoolExpr inputCount).eval inputs = inputs index := rfl

@[simp] theorem BoolExpr.eval_constant {inputCount : Nat} (inputs : Fin inputCount → Bool)
    (value : Bool) :
    (.constant value : BoolExpr inputCount).eval inputs = value := rfl

@[simp] theorem BoolExpr.eval_not {inputCount : Nat} (inputs : Fin inputCount → Bool)
    (argument : BoolExpr inputCount) :
    (.not argument : BoolExpr inputCount).eval inputs = !(argument.eval inputs) := rfl

@[simp] theorem BoolExpr.eval_and {inputCount : Nat} (inputs : Fin inputCount → Bool)
    (left right : BoolExpr inputCount) :
    (.and left right : BoolExpr inputCount).eval inputs =
      (left.eval inputs && right.eval inputs) := rfl

@[simp] theorem BoolExpr.eval_xor {inputCount : Nat} (inputs : Fin inputCount → Bool)
    (left right : BoolExpr inputCount) :
    (.xor left right : BoolExpr inputCount).eval inputs =
      ((left.eval inputs && !(right.eval inputs)) ||
        (!(left.eval inputs) && right.eval inputs)) := rfl

theorem BoolCircuit.eval_output {inputCount : Nat} (circuit : BoolCircuit inputCount)
    (inputs : Fin inputCount → Bool) :
    circuit.eval inputs = circuit.output.eval inputs := rfl

theorem BoolCircuit.accepts_iff {inputCount : Nat} (circuit : BoolCircuit inputCount)
    (inputs : Fin inputCount → Bool) :
    circuit.eval inputs = true ↔ circuit.output.eval inputs = true := Iff.rfl

/- The runtime circuit format is rectangular: every layer reserves `maxLayerWidth` records, while
   `activeGateCounts` says how many records are live.  All integer-valued fields below are kept as
   integers because that is their representation at the IR boundary; `Valid` rejects negative and
   out-of-range values before evaluation. -/
structure LayeredBoolCircuitShape where
  instanceWidth : Nat
  witnessWidth : Nat
  depth : Nat
  maxLayerWidth : Nat
  deriving Repr, DecidableEq

def LayeredBoolCircuitShape.inputWidth (shape : LayeredBoolCircuitShape) : Nat :=
  shape.instanceWidth + shape.witnessWidth

def maxUInt32 : Nat := 4294967295

def LayeredBoolCircuitShape.Valid (shape : LayeredBoolCircuitShape) : Prop :=
  0 < shape.inputWidth ∧
    shape.inputWidth ≤ maxUInt32 ∧
    0 < shape.depth ∧
    0 < shape.maxLayerWidth ∧
    shape.maxLayerWidth ≤ maxUInt32 ∧
    shape.inputWidth ≤ shape.maxLayerWidth

structure LayeredBoolCircuit (shape : LayeredBoolCircuitShape) where
  activeGateCounts : Fin shape.depth → Int
  gateKinds : Fin (shape.depth * shape.maxLayerWidth) → Int
  leftSources : Fin (shape.depth * shape.maxLayerWidth) → Int
  rightSources : Fin (shape.depth * shape.maxLayerWidth) → Int
  outputSource : Int

/- The runtime stores active widths as signed integers.  This normalization is the only place
   where those integers enter the dependent circuit proof; validity guarantees that `toNat` is
   the intended finite width. -/
def LayeredBoolCircuit.activeWidth {shape : LayeredBoolCircuitShape}
    (circuit : LayeredBoolCircuit shape) (layer : Fin shape.depth) : Nat :=
  (circuit.activeGateCounts layer).toNat

def LayeredBoolCircuit.previousNatWidth {shape : LayeredBoolCircuitShape}
    (circuit : LayeredBoolCircuit shape) (layer : Fin shape.depth) : Nat :=
  if layer.val = 0 then shape.inputWidth
  else circuit.activeWidth ⟨layer.val - 1, by omega⟩

namespace LayeredBoolCircuit

def ofFamilies {shape : LayeredBoolCircuitShape}
    (activeGateCounts : Fin shape.depth → Int)
    (gateKinds leftSources rightSources : Fin (shape.depth * shape.maxLayerWidth) → Int)
    (outputSources : Fin 1 → Int) : LayeredBoolCircuit shape where
  activeGateCounts := activeGateCounts
  gateKinds := gateKinds
  leftSources := leftSources
  rightSources := rightSources
  outputSource := outputSources 0

def flatIndex {shape : LayeredBoolCircuitShape} (layer : Fin shape.depth)
    (slot : Fin shape.maxLayerWidth) : Fin (shape.depth * shape.maxLayerWidth) :=
  ⟨layer.val * shape.maxLayerWidth + slot.val, by
    calc
      layer.val * shape.maxLayerWidth + slot.val <
          layer.val * shape.maxLayerWidth + shape.maxLayerWidth :=
        Nat.add_lt_add_left slot.isLt _
      _ = (layer.val + 1) * shape.maxLayerWidth := by rw [Nat.succ_mul]
      _ ≤ shape.depth * shape.maxLayerWidth :=
        Nat.mul_le_mul_right _ (Nat.succ_le_of_lt layer.isLt)⟩

def previousWidth {shape : LayeredBoolCircuitShape} (circuit : LayeredBoolCircuit shape)
    (layer : Fin shape.depth) : Int :=
  if layer.val = 0 then shape.inputWidth
  else circuit.activeGateCounts
    ⟨layer.val - 1, Nat.lt_of_le_of_lt (Nat.sub_le layer.val 1) layer.isLt⟩

def sourceValid (source previousWidth : Int) : Prop :=
  0 ≤ source ∧ source < previousWidth

def gateRecordValid (kind left right previousWidth : Int) : Prop :=
  (kind = 0 ∧ left = 0 ∧ right = 0) ∨
    (kind = 1 ∧ left = 0 ∧ right = 0) ∨
    (kind = 2 ∧ sourceValid left previousWidth ∧ right = 0) ∨
    (kind = 3 ∧ sourceValid left previousWidth ∧ right = 0) ∨
    (kind = 4 ∧ sourceValid left previousWidth ∧ sourceValid right previousWidth) ∨
    (kind = 5 ∧ sourceValid left previousWidth ∧ sourceValid right previousWidth)

def finalActiveCount {shape : LayeredBoolCircuitShape} (circuit : LayeredBoolCircuit shape) : Int :=
  if empty : shape.depth = 0 then 0
  else circuit.activeGateCounts
    ⟨shape.depth - 1, Nat.sub_lt (Nat.zero_lt_of_ne_zero empty) Nat.zero_lt_one⟩

def Valid {shape : LayeredBoolCircuitShape} (circuit : LayeredBoolCircuit shape) : Prop :=
  shape.Valid ∧
    (∀ layer, 1 ≤ circuit.activeGateCounts layer ∧
      circuit.activeGateCounts layer ≤ shape.maxLayerWidth) ∧
    (∀ layer slot,
      let index := flatIndex layer slot
      let kind := circuit.gateKinds index
      let left := circuit.leftSources index
      let right := circuit.rightSources index
      if (slot.val : Int) < circuit.activeGateCounts layer then
        gateRecordValid kind left right (circuit.previousWidth layer)
      else kind = 0 ∧ left = 0 ∧ right = 0) ∧
    0 ≤ circuit.outputSource ∧ circuit.outputSource < circuit.finalActiveCount

theorem previousNatWidth_eq_previousWidth_toNat {shape : LayeredBoolCircuitShape}
    (circuit : LayeredBoolCircuit shape) (layer : Fin shape.depth) :
    circuit.previousNatWidth layer = (circuit.previousWidth layer).toNat := by
  by_cases h : layer.val = 0
  · simp [previousNatWidth, previousWidth, h, LayeredBoolCircuitShape.inputWidth]
    omega
  · simp [previousNatWidth, previousWidth, h, activeWidth]

theorem activeWidth_pos {shape : LayeredBoolCircuitShape}
    {circuit : LayeredBoolCircuit shape} (valid : circuit.Valid) (layer : Fin shape.depth) :
    0 < circuit.activeWidth layer := by
  have h := valid.2.1 layer
  simp [activeWidth]
  omega

theorem activeWidth_le_max {shape : LayeredBoolCircuitShape}
    {circuit : LayeredBoolCircuit shape} (valid : circuit.Valid) (layer : Fin shape.depth) :
    circuit.activeWidth layer ≤ shape.maxLayerWidth := by
  have h := valid.2.1 layer
  simp [activeWidth]
  omega

theorem previousNatWidth_pos {shape : LayeredBoolCircuitShape}
    {circuit : LayeredBoolCircuit shape} (valid : circuit.Valid) (layer : Fin shape.depth) :
    0 < circuit.previousNatWidth layer := by
  by_cases h : layer.val = 0
  · simp [previousNatWidth, h, LayeredBoolCircuitShape.inputWidth]
    exact valid.1.1
  · simp [previousNatWidth, h]
    exact activeWidth_pos valid ⟨layer.val - 1, by omega⟩

def gateAt? {shape : LayeredBoolCircuitShape} (circuit : LayeredBoolCircuit shape)
    (layer slot : Nat) : Option (Int × Int × Int) :=
  if layerValid : layer < shape.depth then
    if slotValid : slot < shape.maxLayerWidth then
      let index := flatIndex ⟨layer, layerValid⟩ ⟨slot, slotValid⟩
      some (circuit.gateKinds index, circuit.leftSources index, circuit.rightSources index)
    else none
  else none

def valueAt? (values : Array Bool) (source : Int) : Option Bool :=
  if 0 ≤ source then values[source.toNat]?
  else none

def evaluateGate? (kind left right : Int) (previous : Array Bool) : Option Bool :=
  match kind with
  | 0 => some false
  | 1 => some true
  | 2 => valueAt? previous left
  | 3 => (!·) <$> valueAt? previous left
  | 4 => (· && ·) <$> valueAt? previous left <*> valueAt? previous right
  | 5 => (· != ·) <$> valueAt? previous left <*> valueAt? previous right
  | _ => none

/- A decoded active record has finite predecessor indices.  This is the shared
   semantic form consumed by protocol-specific correctness proofs; decoding is
   separate from evaluation so no later proof needs to reason about signed IR
   indices. -/
inductive ActiveGateSpec (previousCount : Nat) where
  | zero
  | one
  | copy (left : Fin previousCount)
  | not (left : Fin previousCount)
  | and (left right : Fin previousCount)
  | xor (left right : Fin previousCount)

def ActiveGateSpec.outputBit {previousCount : Nat}
    (spec : ActiveGateSpec previousCount) (previous : Fin previousCount → Bool) : Bool :=
  match spec with
  | .zero => false
  | .one => true
  | .copy left => previous left
  | .not left => !(previous left)
  | .and left right => previous left && previous right
  | .xor left right => previous left != previous right

def ActiveGateSpec.runtimeRecord {previousCount : Nat}
    (spec : ActiveGateSpec previousCount) : Int × Int × Int :=
  match spec with
  | .zero => (0, 0, 0)
  | .one => (1, 0, 0)
  | .copy left => (2, left.val, 0)
  | .not left => (3, left.val, 0)
  | .and left right => (4, left.val, right.val)
  | .xor left right => (5, left.val, right.val)

def ActiveGateSpec.evaluateArray {previousCount : Nat}
    (spec : ActiveGateSpec previousCount) (previous : Array Bool) : Option Bool :=
  match spec with
  | .zero => some false
  | .one => some true
  | .copy left => previous[left.val]?
  | .not left => (!·) <$> previous[left.val]?
  | .and left right => (· && ·) <$> previous[left.val]? <*> previous[right.val]?
  | .xor left right => (· != ·) <$> previous[left.val]? <*> previous[right.val]?

theorem evaluateGate?_runtimeRecord {previousCount : Nat}
    (spec : ActiveGateSpec previousCount) (previous : Array Bool) :
    let record := spec.runtimeRecord
    evaluateGate? record.1 record.2.1 record.2.2 previous = spec.evaluateArray previous := by
  cases spec <;> simp [ActiveGateSpec.runtimeRecord, ActiveGateSpec.evaluateArray, evaluateGate?,
    valueAt?]

theorem ActiveGateSpec.evaluateArray_ofFn {previousCount : Nat}
    (spec : ActiveGateSpec previousCount) (previous : Fin previousCount → Bool) :
    spec.evaluateArray (Array.ofFn previous) = some (spec.outputBit previous) := by
  cases spec <;> simp [ActiveGateSpec.evaluateArray, ActiveGateSpec.outputBit]
  all_goals rfl

def decodeActiveGate? (previousCount : Nat) (kind left right : Int) :
    Option (ActiveGateSpec previousCount) :=
  match kind with
  | 0 => if left = 0 ∧ right = 0 then some .zero else none
  | 1 => if left = 0 ∧ right = 0 then some .one else none
  | 2 => if hleft : 0 ≤ left ∧ left < previousCount ∧ right = 0 then
      some (.copy ⟨left.toNat, by omega⟩)
    else none
  | 3 => if hleft : 0 ≤ left ∧ left < previousCount ∧ right = 0 then
      some (.not ⟨left.toNat, by omega⟩)
    else none
  | 4 => if hleft : 0 ≤ left ∧ left < previousCount ∧ 0 ≤ right ∧ right < previousCount then
      some (.and ⟨left.toNat, by omega⟩ ⟨right.toNat, by omega⟩)
    else none
  | 5 => if hleft : 0 ≤ left ∧ left < previousCount ∧ 0 ≤ right ∧ right < previousCount then
      some (.xor ⟨left.toNat, by omega⟩ ⟨right.toNat, by omega⟩)
    else none
  | _ => none

/- Validity is exactly the condition needed to decode a signed runtime record
   into a finite active specification.  This is a conversion theorem, not an
   evaluator assumption: the resulting record is definitionally the input
   record. -/
theorem exists_decodeActiveGate?_of_gateRecordValid {previousCount : Nat}
    (previousWidth kind left right : Int) (width_eq : previousWidth = previousCount)
    (record_valid : gateRecordValid kind left right previousWidth) :
    ∃ spec : ActiveGateSpec previousCount,
      decodeActiveGate? previousCount kind left right = some spec ∧
        spec.runtimeRecord = (kind, left, right) := by
  rcases record_valid with zero | one | copy | not | and | xor
  · rcases zero with ⟨rfl, rfl, rfl⟩
    exact ⟨.zero, by simp [decodeActiveGate?], by simp [ActiveGateSpec.runtimeRecord]⟩
  · rcases one with ⟨rfl, rfl, rfl⟩
    exact ⟨.one, by simp [decodeActiveGate?], by simp [ActiveGateSpec.runtimeRecord]⟩
  · rcases copy with ⟨rfl, left_valid, rfl⟩
    rw [width_eq] at left_valid
    have left_lt : left < (previousCount : Int) := left_valid.2
    let source : Fin previousCount :=
      ⟨left.toNat, (Int.toNat_lt left_valid.1).2 left_lt⟩
    refine ⟨.copy source, ?_, ?_⟩
    · simp [decodeActiveGate?, source, left_valid.1, left_lt]
    · simp [ActiveGateSpec.runtimeRecord, source, Int.toNat_of_nonneg left_valid.1]
  · rcases not with ⟨rfl, left_valid, rfl⟩
    rw [width_eq] at left_valid
    have left_lt : left < (previousCount : Int) := left_valid.2
    let source : Fin previousCount :=
      ⟨left.toNat, (Int.toNat_lt left_valid.1).2 left_lt⟩
    refine ⟨.not source, ?_, ?_⟩
    · simp [decodeActiveGate?, source, left_valid.1, left_lt]
    · simp [ActiveGateSpec.runtimeRecord, source, Int.toNat_of_nonneg left_valid.1]
  · rcases and with ⟨rfl, left_valid, right_valid⟩
    rw [width_eq] at left_valid right_valid
    have left_lt : left < (previousCount : Int) := left_valid.2
    have right_lt : right < (previousCount : Int) := right_valid.2
    let leftSource : Fin previousCount :=
      ⟨left.toNat, (Int.toNat_lt left_valid.1).2 left_lt⟩
    let rightSource : Fin previousCount :=
      ⟨right.toNat, (Int.toNat_lt right_valid.1).2 right_lt⟩
    refine ⟨.and leftSource rightSource, ?_, ?_⟩
    · simp [decodeActiveGate?, leftSource, rightSource, left_valid.1, left_lt,
        right_valid.1, right_lt]
    · simp [ActiveGateSpec.runtimeRecord, leftSource, rightSource,
        Int.toNat_of_nonneg left_valid.1, Int.toNat_of_nonneg right_valid.1]
  · rcases xor with ⟨rfl, left_valid, right_valid⟩
    rw [width_eq] at left_valid right_valid
    have left_lt : left < (previousCount : Int) := left_valid.2
    have right_lt : right < (previousCount : Int) := right_valid.2
    let leftSource : Fin previousCount :=
      ⟨left.toNat, (Int.toNat_lt left_valid.1).2 left_lt⟩
    let rightSource : Fin previousCount :=
      ⟨right.toNat, (Int.toNat_lt right_valid.1).2 right_lt⟩
    refine ⟨.xor leftSource rightSource, ?_, ?_⟩
    · simp [decodeActiveGate?, leftSource, rightSource, left_valid.1, left_lt,
        right_valid.1, right_lt]
    · simp [ActiveGateSpec.runtimeRecord, leftSource, rightSource,
        Int.toNat_of_nonneg left_valid.1, Int.toNat_of_nonneg right_valid.1]

/- The following projections isolate the live prefix of a rectangular layer.
   They are parameterized by the width proof so callers retain the exact
   runtime slot rather than an approximation based on traversal position. -/
def activeSlot {shape : LayeredBoolCircuitShape} (circuit : LayeredBoolCircuit shape)
    (layer : Fin shape.depth) (slot : Fin (circuit.activeWidth layer))
    (active_le_max : circuit.activeWidth layer ≤ shape.maxLayerWidth) : Fin shape.maxLayerWidth :=
  ⟨slot.val, Nat.lt_of_lt_of_le slot.isLt active_le_max⟩

def activeGateRecord {shape : LayeredBoolCircuitShape} (circuit : LayeredBoolCircuit shape)
    (layer : Fin shape.depth) (slot : Fin (circuit.activeWidth layer))
    (active_le_max : circuit.activeWidth layer ≤ shape.maxLayerWidth) : Int × Int × Int :=
  let index := flatIndex layer (activeSlot circuit layer slot active_le_max)
  (circuit.gateKinds index, circuit.leftSources index, circuit.rightSources index)

theorem previousWidth_nonnegative {shape : LayeredBoolCircuitShape}
    {circuit : LayeredBoolCircuit shape} (valid : circuit.Valid) (layer : Fin shape.depth) :
    0 ≤ circuit.previousWidth layer := by
  by_cases initial : layer.val = 0
  · simp [previousWidth, initial]
  · rw [previousWidth, if_neg initial]
    have positive := (valid.2.1 ⟨layer.val - 1, by omega⟩).1
    omega

theorem previousWidth_eq_previousNatWidth {shape : LayeredBoolCircuitShape}
    {circuit : LayeredBoolCircuit shape} (valid : circuit.Valid) (layer : Fin shape.depth) :
    circuit.previousWidth layer = circuit.previousNatWidth layer := by
  rw [previousNatWidth_eq_previousWidth_toNat]
  symm
  exact Int.toNat_of_nonneg (previousWidth_nonnegative valid layer)

theorem valid_activeGateRecord {shape : LayeredBoolCircuitShape}
    {circuit : LayeredBoolCircuit shape} (valid : circuit.Valid) (layer : Fin shape.depth)
    (slot : Fin (circuit.activeWidth layer)) :
    let active_le_max := activeWidth_le_max valid layer
    let record := circuit.activeGateRecord layer slot active_le_max
    gateRecordValid record.1 record.2.1 record.2.2 (circuit.previousWidth layer) := by
  dsimp
  let active_le_max := activeWidth_le_max valid layer
  let max_slot := activeSlot circuit layer slot active_le_max
  have active_positive : 0 < circuit.activeGateCounts layer := by
    exact (valid.2.1 layer).1
  have slot_lt : (max_slot.val : Int) < circuit.activeGateCounts layer := by
    apply (Int.toNat_lt_toNat active_positive).mp
    change slot.val < (circuit.activeGateCounts layer).toNat
    exact slot.isLt
  have record_valid := valid.2.2.1 layer max_slot
  dsimp only at record_valid
  rw [if_pos slot_lt] at record_valid
  exact record_valid

theorem valid_activeGate_decodes {shape : LayeredBoolCircuitShape}
    {circuit : LayeredBoolCircuit shape} (valid : circuit.Valid) (layer : Fin shape.depth)
    (slot : Fin (circuit.activeWidth layer)) :
    let active_le_max := activeWidth_le_max valid layer
    let record := circuit.activeGateRecord layer slot active_le_max
    ∃ spec : ActiveGateSpec (circuit.previousNatWidth layer),
      decodeActiveGate? (circuit.previousNatWidth layer) record.1 record.2.1 record.2.2 =
        some spec ∧ spec.runtimeRecord = record := by
  dsimp
  exact exists_decodeActiveGate?_of_gateRecordValid (circuit.previousWidth layer) _ _ _
    (previousWidth_eq_previousNatWidth valid layer) (valid_activeGateRecord valid layer slot)

noncomputable def activeGateSpec {shape : LayeredBoolCircuitShape}
    {circuit : LayeredBoolCircuit shape} (valid : circuit.Valid) (layer : Fin shape.depth)
    (slot : Fin (circuit.activeWidth layer)) : ActiveGateSpec (circuit.previousNatWidth layer) :=
  Classical.choose (valid_activeGate_decodes valid layer slot)

theorem activeGateSpec_decodes {shape : LayeredBoolCircuitShape}
    {circuit : LayeredBoolCircuit shape} (valid : circuit.Valid) (layer : Fin shape.depth)
    (slot : Fin (circuit.activeWidth layer)) :
    let active_le_max := activeWidth_le_max valid layer
    let record := circuit.activeGateRecord layer slot active_le_max
    decodeActiveGate? (circuit.previousNatWidth layer) record.1 record.2.1 record.2.2 =
      some (activeGateSpec valid layer slot) ∧
      (activeGateSpec valid layer slot).runtimeRecord = record :=
  Classical.choose_spec (valid_activeGate_decodes valid layer slot)

theorem gateAt?_eq_activeGateRecord {shape : LayeredBoolCircuitShape}
    {circuit : LayeredBoolCircuit shape} (valid : circuit.Valid) (layer : Fin shape.depth)
    (slot : Fin (circuit.activeWidth layer)) :
    let active_le_max := activeWidth_le_max valid layer
    circuit.gateAt? layer.val slot.val =
      some (circuit.activeGateRecord layer slot active_le_max) := by
  dsimp
  have active_le_max := activeWidth_le_max valid layer
  have slot_in_max : slot.val < shape.maxLayerWidth :=
    Nat.lt_of_lt_of_le slot.isLt active_le_max
  simp [gateAt?, activeGateRecord, activeSlot, layer.isLt, slot_in_max]

example : evaluateGate? 0 0 0 #[true, true] = some false := by decide

example : evaluateGate? 1 0 0 #[false, false] = some true := by decide

example : evaluateGate? 2 1 0 #[false, true] = some true := by decide

example : evaluateGate? 3 0 0 #[false, true] = some true := by decide

example : evaluateGate? 4 0 1 #[true, false] = some false := by decide

example : evaluateGate? 5 0 1 #[true, false] = some true := by decide

example : evaluateGate? 2 (-1) 0 #[true] = none := by decide

example : evaluateGate? 6 0 0 #[true] = none := by decide

def evaluateLayer? {shape : LayeredBoolCircuitShape} (circuit : LayeredBoolCircuit shape)
    (layer : Nat) (previous : Array Bool) : Option (Array Bool) := do
  let activeValue ← if layerValid : layer < shape.depth then
    some (circuit.activeGateCounts ⟨layer, layerValid⟩)
  else none
  if 0 ≤ activeValue then
    (← (List.range activeValue.toNat).mapM fun slot => do
      let (kind, left, right) ← circuit.gateAt? layer slot
      evaluateGate? kind left right previous).toArray
  else none

/- Normalization used by proofs that connect the rectangular runtime layer to
   typed finite-index data. -/
theorem activeCount_toNat {shape : LayeredBoolCircuitShape}
    (circuit : LayeredBoolCircuit shape) (layer : Fin shape.depth)
    (active : Nat) (h : circuit.activeGateCounts layer = active) :
    (circuit.activeGateCounts layer).toNat = active := by
  simp [h]

theorem mapM_range_eq_some_of_pointwise {α : Type} {n : Nat}
    (f : Nat → Option α) (values : Nat → α)
    (pointwise : ∀ index, index < n → f index = some (values index)) :
    (List.range n).mapM f = some ((List.range n).map values) := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [List.range_succ]
      simp only [List.mapM_append, List.map_append]
      rw [ih (fun index hindex => pointwise index (Nat.lt_trans hindex (Nat.lt_succ_self n)))]
      simp [pointwise n (Nat.lt_succ_self n)]

def activeValuesAt (width : Nat) (values : Fin width → Bool) (index : Nat) : Bool :=
  if active : index < width then values ⟨index, active⟩ else false

theorem activeValuesAt_range_toArray {width : Nat} (values : Fin width → Bool) :
    ((List.range width).map (activeValuesAt width values)).toArray = Array.ofFn values := by
  apply Array.ext
  · simp
  · intro index left_lt right_lt
    have index_lt : index < width := by simpa using right_lt
    simp [activeValuesAt, index_lt]

/- The layer evaluator is a map over exactly the active finite slots.  The
   premise supplies successful typed evaluations, not the layer result: the
   proof reconstructs that result through the runtime `mapM` definition. -/
theorem evaluateLayer?_eq_some_of_activeGateSpec {shape : LayeredBoolCircuitShape}
    {circuit : LayeredBoolCircuit shape} (valid : circuit.Valid) (layer : Fin shape.depth)
    (previous : Array Bool) (values : Fin (circuit.activeWidth layer) → Bool)
    (evaluates : ∀ slot,
      (activeGateSpec valid layer slot).evaluateArray previous = some (values slot)) :
    circuit.evaluateLayer? layer.val previous =
      some ((List.range (circuit.activeWidth layer)).map
        (activeValuesAt (circuit.activeWidth layer) values)).toArray := by
  have active_nonnegative : 0 ≤ circuit.activeGateCounts layer := by
    have active_positive := (valid.2.1 layer).1
    omega
  have step : ∀ index, index < circuit.activeWidth layer →
      (do
        let record ← circuit.gateAt? layer.val index
        evaluateGate? record.1 record.2.1 record.2.2 previous) =
        some (activeValuesAt (circuit.activeWidth layer) values index) := by
    intro index index_lt
    let slot : Fin (circuit.activeWidth layer) := ⟨index, index_lt⟩
    rw [gateAt?_eq_activeGateRecord valid layer slot]
    have decoded := activeGateSpec_decodes valid layer slot
    rw [← decoded.2]
    change evaluateGate? (activeGateSpec valid layer slot).runtimeRecord.1
      (activeGateSpec valid layer slot).runtimeRecord.2.1
      (activeGateSpec valid layer slot).runtimeRecord.2.2 previous = _
    rw [evaluateGate?_runtimeRecord]
    simpa [activeValuesAt, index_lt] using evaluates slot
  unfold evaluateLayer?
  rw [dif_pos layer.isLt]
  simp [active_nonnegative]
  change ((List.range (circuit.activeWidth layer)).mapM (fun index => do
      let record ← circuit.gateAt? layer.val index
      evaluateGate? record.1 record.2.1 record.2.2 previous)).bind
      (fun outputs => some outputs.toArray) = _
  rw [mapM_range_eq_some_of_pointwise _ _ step]
  simp

theorem evaluateLayer?_of_activeSpecs {shape : LayeredBoolCircuitShape}
    {circuit : LayeredBoolCircuit shape} (valid : circuit.Valid) (layer : Fin shape.depth)
    (previous : Fin (circuit.previousNatWidth layer) → Bool) :
    circuit.evaluateLayer? layer.val (Array.ofFn previous) =
      some (Array.ofFn fun slot =>
        (activeGateSpec valid layer slot).outputBit previous) := by
  rw [evaluateLayer?_eq_some_of_activeGateSpec valid layer (Array.ofFn previous)
    (fun slot => (activeGateSpec valid layer slot).outputBit previous)]
  · rw [activeValuesAt_range_toArray]
  · intro slot
    exact ActiveGateSpec.evaluateArray_ofFn _ previous

theorem foldlM_range_succ {m : Type → Type} [Monad m] [LawfulMonad m] {α : Type} {n : Nat}
    (f : α → Nat → m α) (initial : α) :
    (List.range (n + 1)).foldlM f initial =
      (List.range n).foldlM f initial >>= fun value => f value n := by
  rw [show n + 1 = Nat.succ n by omega, List.range_succ]
  simp [List.foldlM]

def evaluateUnchecked? {shape : LayeredBoolCircuitShape} (circuit : LayeredBoolCircuit shape)
    (instanceBits : Fin shape.instanceWidth → Bool)
    (witness : Fin shape.witnessWidth → Bool) :
    Option Bool := do
  let initial := (Array.ofFn instanceBits).append (Array.ofFn witness)
  let final ← (List.range shape.depth).foldlM
    (fun previous layer => circuit.evaluateLayer? layer previous) initial
  valueAt? final circuit.outputSource

/- The selected output is itself a finite active-slot index.  This is the
   final bridge used after a layer-by-layer induction has constructed the last
   active family. -/
theorem valid_outputSource_slot {shape : LayeredBoolCircuitShape}
    {circuit : LayeredBoolCircuit shape} (valid : circuit.Valid) :
    circuit.outputSource.toNat < circuit.finalActiveCount.toNat := by
  have output_nonnegative := valid.2.2.2.1
  have output_lt := valid.2.2.2.2
  have final_positive : 0 < circuit.finalActiveCount := by
    omega
  exact (Int.toNat_lt_toNat final_positive).2 output_lt

def outputSourceSlot {shape : LayeredBoolCircuitShape} {circuit : LayeredBoolCircuit shape}
    (valid : circuit.Valid) : Fin circuit.finalActiveCount.toNat :=
  ⟨circuit.outputSource.toNat, valid_outputSource_slot valid⟩

theorem evaluateUnchecked?_of_finalLayer {shape : LayeredBoolCircuitShape}
    (circuit : LayeredBoolCircuit shape)
    (instanceBits : Fin shape.instanceWidth → Bool)
    (witness : Fin shape.witnessWidth → Bool) (final : Array Bool)
    (layers : (List.range shape.depth).foldlM
      (fun previous layer => circuit.evaluateLayer? layer previous)
      ((Array.ofFn instanceBits).append (Array.ofFn witness)) = some final) :
    circuit.evaluateUnchecked? instanceBits witness = valueAt? final circuit.outputSource := by
  unfold evaluateUnchecked?
  change ((List.range shape.depth).foldlM
    (fun previous layer => circuit.evaluateLayer? layer previous)
    ((Array.ofFn instanceBits).append (Array.ofFn witness))).bind
      (fun output => valueAt? output circuit.outputSource) = _
  rw [layers]
  simp

def evaluate {shape : LayeredBoolCircuitShape} (circuit : LayeredBoolCircuit shape)
    (_ : circuit.Valid)
    (instanceBits : Fin shape.instanceWidth → Bool)
    (witness : Fin shape.witnessWidth → Bool) :
    Option Bool :=
  circuit.evaluateUnchecked? instanceBits witness

private def andShape : LayeredBoolCircuitShape where
  instanceWidth := 1
  witnessWidth := 1
  depth := 1
  maxLayerWidth := 2

private def andCircuit : LayeredBoolCircuit andShape where
  activeGateCounts := fun _ => 1
  gateKinds := fun index => if index.val = 0 then 4 else 0
  leftSources := fun _ => 0
  rightSources := fun index => if index.val = 0 then 1 else 0
  outputSource := 0

private theorem andCircuitValid : andCircuit.Valid := by
  simp [Valid, LayeredBoolCircuitShape.Valid, LayeredBoolCircuitShape.inputWidth,
    maxUInt32, andCircuit, andShape, flatIndex, gateRecordValid, sourceValid, previousWidth,
    finalActiveCount]

example : andCircuit.evaluate andCircuitValid (fun _ => true) (fun _ => true) = some true := by
  decide

example : andCircuit.evaluate andCircuitValid (fun _ => true) (fun _ => false) = some false := by
  decide

private def noncanonicalPadding : LayeredBoolCircuit andShape where
  activeGateCounts := fun _ => 1
  gateKinds := fun index => if index.val = 0 then 4 else 1
  leftSources := fun _ => 0
  rightSources := fun index => if index.val = 0 then 1 else 0
  outputSource := 0

example : ¬noncanonicalPadding.Valid := by
  simp [Valid, noncanonicalPadding, andShape, flatIndex]

end LayeredBoolCircuit

end Mxx.Gadgets
