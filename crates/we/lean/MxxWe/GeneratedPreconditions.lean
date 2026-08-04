import MxxWe.Generated.DiamondWeFamily.Statement
import MxxWe.GenericBooleanLayers

open Mxx
open MxxWe.Generated.DiamondWeFamily

namespace MxxWe

def generatedCircuitData {p : DiamondWeFamilyParams}
    (x : DiamondWeFamilyInputs p) : DynamicBoolean.CircuitData := {
  activeGateCounts := x.circuitActiveGateCount
  gateKinds := x.circuitGateKind
  leftSources := x.circuitLeftSource
  rightSources := x.circuitRightSource
  outputSources := x.circuitOutputSource
}

def generatedGateKind : Int → BooleanGate := DynamicBoolean.gateKind

def generatedCircuitEntry (values : List Int) (p : DiamondWeFamilyParams)
    (layer slot : Nat) : Int :=
  DynamicBoolean.entry values p.maxLayerWidth layer slot

def generatedActiveCount {p : DiamondWeFamilyParams}
    (x : DiamondWeFamilyInputs p) (layer : Nat) : Nat :=
  DynamicBoolean.activeCount (generatedCircuitData x) layer

def generatedBooleanLayer (p : DiamondWeFamilyParams)
    (x : DiamondWeFamilyInputs p) (layer : Nat) : BooleanLayerProgram :=
  let activeCount := generatedActiveCount x layer
  { kinds := (List.range activeCount).map fun slot ↦
      generatedGateKind (generatedCircuitEntry x.circuitGateKind p layer slot)
    leftPredecessors := (List.range activeCount).map fun slot ↦
      (generatedCircuitEntry x.circuitLeftSource p layer slot).toNat
    rightPredecessors := (List.range activeCount).map fun slot ↦
      (generatedCircuitEntry x.circuitRightSource p layer slot).toNat }

def generatedBooleanLayers (p : DiamondWeFamilyParams)
    (x : DiamondWeFamilyInputs p) : List BooleanLayerProgram :=
  (List.range p.depth).map (generatedBooleanLayer p x)

@[simp] theorem generatedBooleanLayers_length (p : DiamondWeFamilyParams)
    (x : DiamondWeFamilyInputs p) :
    (generatedBooleanLayers p x).length = p.depth := by
  simp [generatedBooleanLayers]

@[simp] theorem generatedBooleanLayer_activeWidth (p : DiamondWeFamilyParams)
    (x : DiamondWeFamilyInputs p) (layer : Nat) :
    (generatedBooleanLayer p x layer).activeWidth = generatedActiveCount x layer := by
  simp [generatedBooleanLayer, BooleanLayerProgram.activeWidth]

theorem generatedGateKind_of_bounds (kind : Int) (lower : 0 ≤ kind) (upper : kind ≤ 5) :
    generatedGateKind kind =
      [.constantFalse, .constantTrue, .copy, .not, .and, .xor][kind.toNat]?.getD
        .constantFalse := by
  interval_cases kind <;> rfl

def generatedBooleanInputs (p : DiamondWeFamilyParams)
    (x : DiamondWeFamilyInputs p) : List Bool :=
  DynamicBoolean.canonicalInputs p.maxLayerWidth p.instanceWidth p.witnessWidth
    x.booleanInstance x.booleanWitness

theorem generatedBooleanInputs_length (p : DiamondWeFamilyParams)
    (x : DiamondWeFamilyInputs p) (_inputsWF : DiamondWeFamilyInputsWF p x) :
    (generatedBooleanInputs p x).length = p.maxLayerWidth := by
  exact DynamicBoolean.canonicalInputs_length p.maxLayerWidth p.instanceWidth p.witnessWidth
    x.booleanInstance x.booleanWitness

def evaluateGeneratedGate : BooleanGate → Bool → Bool → Bool
  := DynamicBoolean.evalGate

def evaluateGeneratedLayer (maxWidth : Nat) (previous : List Bool)
    (layer : BooleanLayerProgram) : List Bool :=
  (List.range maxWidth).map fun slot ↦
    if slot < layer.activeWidth then
      evaluateGeneratedGate (layer.kinds[slot]?.getD .constantFalse)
        (previous[layer.leftPredecessors[slot]?.getD 0]?.getD false)
        (previous[layer.rightPredecessors[slot]?.getD 0]?.getD false)
    else false

def evaluateGeneratedLayers (maxWidth : Nat) (initial : List Bool) :
    List BooleanLayerProgram → List Bool
  | [] => initial
  | layer :: layers =>
      evaluateGeneratedLayers maxWidth (evaluateGeneratedLayer maxWidth initial layer) layers

@[simp] theorem evaluateGeneratedLayer_length (maxWidth : Nat) (previous : List Bool)
    (layer : BooleanLayerProgram) :
    (evaluateGeneratedLayer maxWidth previous layer).length = maxWidth := by
  simp [evaluateGeneratedLayer]

theorem evaluateGeneratedLayers_length (maxWidth : Nat) (initial : List Bool)
    (initialLength : initial.length = maxWidth) (layers : List BooleanLayerProgram) :
    (evaluateGeneratedLayers maxWidth initial layers).length = maxWidth := by
  induction layers generalizing initial with
  | nil => exact initialLength
  | cons layer layers induction =>
      exact induction (evaluateGeneratedLayer maxWidth initial layer)
        (evaluateGeneratedLayer_length maxWidth initial layer)

def generatedCircuitOutput (p : DiamondWeFamilyParams)
    (x : DiamondWeFamilyInputs p) : Bool :=
  DynamicBoolean.output p.depth p.maxLayerWidth p.instanceWidth p.witnessWidth
    (generatedCircuitData x) x.booleanInstance x.booleanWitness

def generatedOutputSource {p : DiamondWeFamilyParams}
    (x : DiamondWeFamilyInputs p) : Int :=
  DynamicBoolean.outputSource (generatedCircuitData x)

structure DiamondWeCircuitFacts (p : DiamondWeFamilyParams)
    (x : DiamondWeFamilyInputs p) : Prop where
  layersLength : (generatedBooleanLayers p x).length = p.depth
  layersValid : ∀ layer ∈ generatedBooleanLayers p x,
    layer.Valid p.maxLayerWidth p.maxLayerWidth
  activeOpcodes : ∀ layer slot, layer < p.depth → slot < generatedActiveCount x layer →
    0 ≤ generatedCircuitEntry x.circuitGateKind p layer slot ∧
      generatedCircuitEntry x.circuitGateKind p layer slot ≤ 5
  activeLeftSourcesNonnegative : ∀ layer slot,
    layer < p.depth → slot < generatedActiveCount x layer →
      0 ≤ generatedCircuitEntry x.circuitLeftSource p layer slot
  activeRightSourcesNonnegative : ∀ layer slot,
    layer < p.depth → slot < generatedActiveCount x layer →
      0 ≤ generatedCircuitEntry x.circuitRightSource p layer slot
  inactivePadding : ∀ layer slot,
    layer < p.depth → generatedActiveCount x layer ≤ slot → slot < p.maxLayerWidth →
      generatedCircuitEntry x.circuitGateKind p layer slot = 0 ∧
        generatedCircuitEntry x.circuitLeftSource p layer slot = 0 ∧
        generatedCircuitEntry x.circuitRightSource p layer slot = 0
  outputSourceInFinalActive :
    0 ≤ generatedOutputSource x ∧
      (generatedOutputSource x).toNat < generatedActiveCount x (p.depth - 1)
  circuitOutputTrue : generatedCircuitOutput p x = true

theorem circuitFacts_of_dynamicBoolean {p : DiamondWeFamilyParams}
    {x : DiamondWeFamilyInputs p}
    (valid : DynamicBoolean.Valid p.depth p.maxLayerWidth (generatedCircuitData x))
    (satisfied : DynamicBoolean.Satisfied p.depth p.maxLayerWidth p.instanceWidth
      p.witnessWidth (generatedCircuitData x) x.booleanInstance x.booleanWitness) :
    DiamondWeCircuitFacts p x := by
  refine {
    layersLength := generatedBooleanLayers_length p x
    layersValid := ?_
    activeOpcodes := fun layer slot layerLt slotLt ↦
      valid.activeGateKinds layer slot layerLt slotLt
    activeLeftSourcesNonnegative := fun layer slot layerLt slotLt ↦
      (valid.activeLeftSources layer slot layerLt slotLt).1
    activeRightSourcesNonnegative := fun layer slot layerLt slotLt ↦
      (valid.activeRightSources layer slot layerLt slotLt).1
    inactivePadding := fun layer slot layerLt activeLe slotLt ↦
      valid.inactivePadding layer slot layerLt activeLe slotLt
    outputSourceInFinalActive := valid.outputSourceValid
    circuitOutputTrue := satisfied.outputTrue
  }
  intro layer member
  simp only [generatedBooleanLayers, List.mem_map, List.mem_range] at member
  obtain ⟨index, indexLt, rfl⟩ := member
  simp only [BooleanLayerProgram.Valid, generatedBooleanLayer,
    BooleanLayerProgram.activeWidth, List.length_map, List.length_range]
  refine ⟨trivial, trivial, (valid.activeCountBounds index indexLt).2, ?_, ?_⟩
  · intro predecessor predecessorMember
    simp only [List.mem_map, List.mem_range] at predecessorMember
    obtain ⟨slot, slotLt, rfl⟩ := predecessorMember
    exact (valid.activeLeftSources index slot indexLt slotLt).2
  · intro predecessor predecessorMember
    simp only [List.mem_map, List.mem_range] at predecessorMember
    obtain ⟨slot, slotLt, rfl⟩ := predecessorMember
    exact (valid.activeRightSources index slot indexLt slotLt).2

theorem DiamondWeFamilyPreconditions.circuitValid {p : DiamondWeFamilyParams}
    {x : DiamondWeFamilyInputs p} (preconditions : DiamondWeFamilyPreconditions p x) :
    DynamicBoolean.Valid p.depth p.maxLayerWidth (generatedCircuitData x) := by
  simpa [DiamondWeFamilyPreconditions, generatedCircuitData] using preconditions.1

theorem DiamondWeFamilyPreconditions.circuitSatisfied {p : DiamondWeFamilyParams}
    {x : DiamondWeFamilyInputs p} (preconditions : DiamondWeFamilyPreconditions p x) :
    DynamicBoolean.Satisfied p.depth p.maxLayerWidth p.instanceWidth p.witnessWidth
      (generatedCircuitData x) x.booleanInstance x.booleanWitness := by
  simpa [DiamondWeFamilyPreconditions, generatedCircuitData] using preconditions.2

end MxxWe
