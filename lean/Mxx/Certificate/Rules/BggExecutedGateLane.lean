import Mxx.Certificate.Rules.BggGateSelection
import Mxx.Certificate.Rules.BggLayerExecution
import Mxx.Certificate.Rules.PointwiseBggGateSemantics
import Mxx.Certificate.Rules.ScalarFormulaExecution

namespace Mxx.Certificate

/-!
# Actual source projection for an executed BGG gate lane

This module contains the source-side projection needed before the four executed candidate
semantics can be assembled into `QuotientBggExecutedGateLane`.  A source lane is recovered from
an actual family lookup; no source matrix or BGG equation is supplied independently.
-/

/-- The executable gate list uses the same positional encoding as the closed gate algebra. -/
theorem bggBooleanGates_at_gate_index (gate : BggBooleanGate) :
    bggBooleanGates[gate.index]? = some gate := by
  cases gate <;> rfl

/-- Successful trace-indexed gate selection fixes the candidate position to `gate.index`. -/
theorem CheckedRecurrenceLaneOutput.CandidateFrame.ExecutedGateSelection.selectorPosition
    {interface : FrozenSequentialRecurrenceInterface}
    {samplers : Mxx.MxxSamplerFamily}
    {fuel : Nat}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments values : List Mxx.Ir.Value}
    {lane : CheckedRecurrenceLaneOutput interface}
    {scopeExecution : ChildScopeExecutionPath samplers interface.program fuel
      interface.definition params arguments values}
    {execution : lane.Execution scopeExecution}
    {position : Nat}
    {candidateFrame : lane.CandidateFrame execution position}
    (selected : candidateFrame.ExecutedGateSelection) :
    selected.selector.value.toNat = selected.gate.index := by
  have canonical := bggBooleanGates_at_gate_index selected.gate
  have selectorInBounds := (List.getElem?_eq_some_iff.mp selected.gateFound).1
  apply (List.getElem?_inj selectorInBounds (by decide)).mp
  exact selected.gateFound.trans canonical.symm

/-- Program-preserving candidate occupying the closed gate position. -/
def CheckedSixWayMatrixProgramSkeleton.programForGate
    {programFormulas : List FrozenPointwiseMatrixProgramFormula}
    {skeleton : CheckedSixWayMatrixSkeleton}
    (programs : CheckedSixWayMatrixProgramSkeleton programFormulas skeleton) :
    BggBooleanGate → FrozenPointwiseMatrixProgramFormula
  | .zero => programs.zeroProgram
  | .one => programs.oneProgram
  | .copyLeft => programs.leftProgram
  | .notLeft => programs.notProgram
  | .and => programs.andProgram
  | .xor => programs.xorProgram

/-- Erased arithmetic candidate occupying the closed gate position. -/
def CheckedSixWayMatrixSkeleton.formulaForGate
    (skeleton : CheckedSixWayMatrixSkeleton) : BggBooleanGate → FrozenPointwiseMatrixFormula
  | .zero => skeleton.zeroFormula
  | .one => skeleton.oneFormula
  | .copyLeft => skeleton.leftFormula
  | .notLeft => skeleton.notFormula
  | .and => skeleton.andFormula
  | .xor => skeleton.xorFormula

/-- Program preservation retains the exact checked erased formula at every gate position. -/
theorem CheckedSixWayMatrixProgramSkeleton.programForGate_erases
    {programFormulas : List FrozenPointwiseMatrixProgramFormula}
    {skeleton : CheckedSixWayMatrixSkeleton}
    (programs : CheckedSixWayMatrixProgramSkeleton programFormulas skeleton)
    (gate : BggBooleanGate) :
    (programs.programForGate gate).erase = skeleton.formulaForGate gate := by
  cases gate <;> simp [CheckedSixWayMatrixProgramSkeleton.programForGate,
    CheckedSixWayMatrixSkeleton.formulaForGate, programs.zeroErases, programs.oneErases,
    programs.leftErases, programs.notErases, programs.andErases, programs.xorErases]

/-- Positional selection preserves the exact program formula, not merely its erased syntax. -/
theorem CheckedSixWayMatrixProgramSkeleton.programAtGate
    {programFormulas : List FrozenPointwiseMatrixProgramFormula}
    {skeleton : CheckedSixWayMatrixSkeleton}
    (programs : CheckedSixWayMatrixProgramSkeleton programFormulas skeleton)
    (gate : BggBooleanGate) :
    programFormulas[gate.index]? = some (programs.programForGate gate) := by
  cases programs
  cases gate <;> simp_all [CheckedSixWayMatrixProgramSkeleton.programForGate,
    BggBooleanGate.index]

/-- The candidate recovered from the actual selector is the role's exact program formula for
that gate. -/
theorem CheckedRecurrenceLaneOutput.CandidateFrame.ExecutedGateSelection.candidate_eq_program
    {interface : FrozenSequentialRecurrenceInterface}
    {samplers : Mxx.MxxSamplerFamily}
    {fuel : Nat}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments values : List Mxx.Ir.Value}
    {lane : CheckedRecurrenceLaneOutput interface}
    {scopeExecution : ChildScopeExecutionPath samplers interface.program fuel
      interface.definition params arguments values}
    {execution : lane.Execution scopeExecution}
    {position : Nat}
    {candidateFrame : lane.CandidateFrame execution position}
    (selected : candidateFrame.ExecutedGateSelection)
    {skeleton : CheckedSixWayMatrixSkeleton}
    (programs : CheckedSixWayMatrixProgramSkeleton lane.gateCandidateProgramFormulas skeleton) :
    selected.candidate = programs.programForGate selected.gate := by
  have expected := programs.programAtGate selected.gate
  rw [← selected.selectorPosition] at expected
  exact Option.some.inj (selected.candidateFound.symm.trans expected)

/-- The actual selected program candidate has the role checker's exact gate-specific arithmetic
shape. -/
theorem CheckedRecurrenceLaneOutput.CandidateFrame.ExecutedGateSelection.candidate_erases
    {interface : FrozenSequentialRecurrenceInterface}
    {samplers : Mxx.MxxSamplerFamily}
    {fuel : Nat}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments values : List Mxx.Ir.Value}
    {lane : CheckedRecurrenceLaneOutput interface}
    {scopeExecution : ChildScopeExecutionPath samplers interface.program fuel
      interface.definition params arguments values}
    {execution : lane.Execution scopeExecution}
    {position : Nat}
    {candidateFrame : lane.CandidateFrame execution position}
    (selected : candidateFrame.ExecutedGateSelection)
    {skeleton : CheckedSixWayMatrixSkeleton}
    (programs : CheckedSixWayMatrixProgramSkeleton lane.gateCandidateProgramFormulas skeleton) :
    selected.candidate.erase = skeleton.formulaForGate selected.gate := by
  rw [selected.candidate_eq_program programs]
  exact programs.programForGate_erases selected.gate

/-- Exact subtraction recovered below only the transparent program boundaries admitted by the
pointwise semantics. -/
structure FrozenPointwiseMatrixProgramFormula.SubtractWitness
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (leftErase rightErase : FrozenPointwiseMatrixFormula)
    (matrix : Mxx.Matrix) : Type where
  leftFormula : FrozenPointwiseMatrixProgramFormula
  rightFormula : FrozenPointwiseMatrixProgramFormula
  leftValue : Mxx.Matrix
  rightValue : Mxx.Matrix
  leftErases : leftFormula.erase = leftErase
  rightErases : rightFormula.erase = rightErase
  leftDenotes : @FrozenPointwiseMatrixProgramFormula.DenotationSomewhere samplers program
    leftFormula leftValue
  rightDenotes : @FrozenPointwiseMatrixProgramFormula.DenotationSomewhere samplers program
    rightFormula rightValue
  valueEq : matrix = Mxx.matrixSubtract leftValue rightValue

/-- Sound inversion of subtraction without dropping input-substitution, subgraph, or parallel
execution boundaries. -/
def FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeSubtract
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {formula : FrozenPointwiseMatrixProgramFormula}
    {matrix : Mxx.Matrix}
    {leftErase rightErase : FrozenPointwiseMatrixFormula}
    (denotes : formula.DenotesAt frame matrix)
    (erases : formula.erase = .subtract leftErase rightErase) :
    FrozenPointwiseMatrixProgramFormula.SubtractWitness (samplers := samplers)
      (program := program) leftErase rightErase matrix := by
  cases denotes with
  | inputSubstitutionSubgraph parentDenotes =>
      exact FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeSubtract parentDenotes erases
  | inputSubstitutionParallel parentDenotes =>
      exact FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeSubtract parentDenotes erases
  | scaleOne inputDenotes =>
      exact FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeSubtract inputDenotes erases
  | subgraphCall _ _ _ _ _ _ _ _ outputDenotes =>
      exact FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeSubtract outputDenotes erases
  | parallelLoop _ _ _ _ _ _ _ _ outputDenotes =>
      exact FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeSubtract outputDenotes erases
  | subtract leftDenotes rightDenotes =>
      simp only [FrozenPointwiseMatrixProgramFormula.erase,
        FrozenPointwiseMatrixFormula.subtract.injEq] at erases
      exact {
        leftFormula := _
        rightFormula := _
        leftValue := _
        rightValue := _
        leftErases := erases.1
        rightErases := erases.2
        leftDenotes := ⟨_, _, leftDenotes⟩
        rightDenotes := ⟨_, _, rightDenotes⟩
        valueEq := rfl
      }
  | _ => simp_all [FrozenPointwiseMatrixProgramFormula.erase]
termination_by sizeOf formula

/-- Arithmetic shape forced by one actual selected matrix candidate.  The direct `one`, `left`,
and role-specific `and` cases need no invented equation; subtraction witnesses are extracted from
the selected candidate's exact denotation. -/
inductive ExecutedSixWayMatrixCandidateShape
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (skeleton : CheckedSixWayMatrixSkeleton) :
    BggBooleanGate → FrozenPointwiseMatrixProgramFormula → Mxx.Matrix → Type where
  | zero
      {formula : FrozenPointwiseMatrixProgramFormula}
      {value : Mxx.Matrix}
      (witness : FrozenPointwiseMatrixProgramFormula.SubtractWitness
        (samplers := samplers) (program := program)
        skeleton.oneFormula skeleton.oneFormula value) :
      ExecutedSixWayMatrixCandidateShape skeleton .zero formula value
  | one
      {formula : FrozenPointwiseMatrixProgramFormula}
      {value : Mxx.Matrix} :
      ExecutedSixWayMatrixCandidateShape skeleton .one formula value
  | copyLeft
      {formula : FrozenPointwiseMatrixProgramFormula}
      {value : Mxx.Matrix} :
      ExecutedSixWayMatrixCandidateShape skeleton .copyLeft formula value
  | notLeft
      {formula : FrozenPointwiseMatrixProgramFormula}
      {value : Mxx.Matrix}
      (witness : FrozenPointwiseMatrixProgramFormula.SubtractWitness
        (samplers := samplers) (program := program)
        skeleton.oneFormula skeleton.leftFormula value) :
      ExecutedSixWayMatrixCandidateShape skeleton .notLeft formula value
  | and
      {formula : FrozenPointwiseMatrixProgramFormula}
      {value : Mxx.Matrix} :
      ExecutedSixWayMatrixCandidateShape skeleton .and formula value
  | xor
      {formula : FrozenPointwiseMatrixProgramFormula}
      {value : Mxx.Matrix}
      (witness : FrozenPointwiseMatrixProgramFormula.SubtractWitness
        (samplers := samplers) (program := program)
        (.add skeleton.leftFormula skeleton.rightFormula) skeleton.twiceAndFormula value) :
      ExecutedSixWayMatrixCandidateShape skeleton .xor formula value

/-- Extract the gate-specific arithmetic shape from an actual selected candidate semantic
result. -/
def CheckedRecurrenceLaneOutput.CandidateFrame.ExecutedGateSelection.matrixCandidateShape
    {interface : FrozenSequentialRecurrenceInterface}
    {samplers : Mxx.MxxSamplerFamily}
    {fuel : Nat}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments values : List Mxx.Ir.Value}
    {lane : CheckedRecurrenceLaneOutput interface}
    {scopeExecution : ChildScopeExecutionPath samplers interface.program fuel
      interface.definition params arguments values}
    {execution : lane.Execution scopeExecution}
    {position : Nat}
    {candidateFrame : lane.CandidateFrame execution position}
    (selected : candidateFrame.ExecutedGateSelection)
    {skeleton : CheckedSixWayMatrixSkeleton}
    (programs : CheckedSixWayMatrixProgramSkeleton lane.gateCandidateProgramFormulas skeleton)
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {runtimeValue : Mxx.Matrix}
    (result : selected.candidate.SemanticResultAt
      (.parallelLane candidateFrame.parent candidateFrame.edge)
      q ringDimension rows columns runtimeValue) :
    @ExecutedSixWayMatrixCandidateShape samplers interface.program skeleton selected.gate
      selected.candidate result.normalizedValue := by
  have erases := selected.candidate_erases programs
  cases gateEq : selected.gate with
  | zero =>
      have witness := result.normalizedDenotes.exposeSubtract (by
        simpa [gateEq, CheckedSixWayMatrixSkeleton.formulaForGate, skeleton.zeroMatches]
          using erases)
      exact ExecutedSixWayMatrixCandidateShape.zero witness
  | one => exact ExecutedSixWayMatrixCandidateShape.one
  | copyLeft => exact ExecutedSixWayMatrixCandidateShape.copyLeft
  | notLeft =>
      have witness := result.normalizedDenotes.exposeSubtract (by
        simpa [gateEq, CheckedSixWayMatrixSkeleton.formulaForGate, skeleton.notMatches]
          using erases)
      exact ExecutedSixWayMatrixCandidateShape.notLeft witness
  | and => exact ExecutedSixWayMatrixCandidateShape.and
  | xor =>
      have witness := result.normalizedDenotes.exposeSubtract (by
        simpa [gateEq, CheckedSixWayMatrixSkeleton.formulaForGate, skeleton.xorMatches]
          using erases)
      exact ExecutedSixWayMatrixCandidateShape.xor witness

/-- A successful public-key lookup in a synchronized BGG family determines the entire lane at
the same coordinate. -/
theorem QuotientBggFamilyRelation.laneAtOfPublicKeyFound
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type} [Fintype secretColumns]
    {secret : _root_.Matrix outputRows secretColumns R}
    {gadget : _root_.Matrix secretColumns publicColumns R}
    {publicKeys : List (_root_.Matrix secretColumns publicColumns R)}
    {vectors : List (_root_.Matrix outputRows publicColumns R)}
    {booleans : List Bool}
    (relation : QuotientBggFamilyRelation secret gadget publicKeys vectors booleans)
    (index : Nat)
    (publicKey : _root_.Matrix secretColumns publicColumns R)
    (publicKeyFound : publicKeys[index]? = some publicKey) :
    Nonempty (relation.LaneAt index) := by
  induction relation generalizing index with
  | nil => simp at publicKeyFound
  | @cons headPublicKey headVector headBoolean publicKeys vectors booleans head tail induction =>
      cases index with
      | zero =>
          simp at publicKeyFound
          subst headPublicKey
          exact ⟨{
            publicKey
            vector := headVector
            booleanValue := headBoolean
            publicKeyFound := rfl
            vectorFound := rfl
            booleanFound := rfl
            lane := head
          }⟩
      | succ index =>
          obtain ⟨lane⟩ := induction index publicKeyFound
          exact ⟨{
            lane with
            publicKeyFound := lane.publicKeyFound
            vectorFound := lane.vectorFound
            booleanFound := lane.booleanFound
          }⟩

/-- Trace-derived source selection.  The runtime public key and its layout are tied to the
actual family element; `lane` is mechanically projected from the existing quotient relation. -/
structure TraceBoundBggSourceLane
    {q ringDimension outputRows secretColumns publicColumns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {secret : _root_.Matrix (Fin outputRows) (Fin secretColumns)
      (Mxx.Toolkit.Negacyclic q ringDimension)}
    {gadget : _root_.Matrix (Fin secretColumns) (Fin publicColumns)
      (Mxx.Toolkit.Negacyclic q ringDimension)}
    {publicKeys : List Mxx.Matrix}
    {vectors : List Mxx.Matrix}
    {booleans : List Bool}
    (relation : QuotientBggFamilyRelation secret gadget
      (runtimeMatrixValues q ringDimension secretColumns publicColumns publicKeys)
      (runtimeMatrixValues q ringDimension outputRows publicColumns vectors) booleans)
    (index : Nat) : Type where
  runtimePublicKey : Mxx.Matrix
  runtimePublicKeyFound : publicKeys[index]? = some runtimePublicKey
  runtimePublicKeyLayout : Mxx.Toolkit.MatrixLayout runtimePublicKey q ringDimension
    secretColumns publicColumns
  lane : relation.LaneAt index
  lanePublicKey : lane.publicKey =
    Mxx.Toolkit.matrixValue q ringDimension secretColumns publicColumns runtimePublicKey

/-- Construct the source lane solely from the actual runtime family lookup and layout. -/
noncomputable def TraceBoundBggSourceLane.ofPublicKeyFound
    {q ringDimension outputRows secretColumns publicColumns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {secret : _root_.Matrix (Fin outputRows) (Fin secretColumns)
      (Mxx.Toolkit.Negacyclic q ringDimension)}
    {gadget : _root_.Matrix (Fin secretColumns) (Fin publicColumns)
      (Mxx.Toolkit.Negacyclic q ringDimension)}
    {publicKeys vectors : List Mxx.Matrix}
    {booleans : List Bool}
    (relation : QuotientBggFamilyRelation secret gadget
      (runtimeMatrixValues q ringDimension secretColumns publicColumns publicKeys)
      (runtimeMatrixValues q ringDimension outputRows publicColumns vectors) booleans)
    (index : Nat)
    (runtimePublicKey : Mxx.Matrix)
    (runtimePublicKeyFound : publicKeys[index]? = some runtimePublicKey)
    (runtimePublicKeyLayout : Mxx.Toolkit.MatrixLayout runtimePublicKey q ringDimension
      secretColumns publicColumns) :
    TraceBoundBggSourceLane relation index := by
  have quotientFound :
      (runtimeMatrixValues q ringDimension secretColumns publicColumns publicKeys)[index]? =
        some (Mxx.Toolkit.matrixValue q ringDimension secretColumns publicColumns
          runtimePublicKey) := by
    simp [runtimeMatrixValues, runtimePublicKeyFound]
  let lane := Classical.choice
    (relation.laneAtOfPublicKeyFound index
      (Mxx.Toolkit.matrixValue q ringDimension secretColumns publicColumns runtimePublicKey)
      quotientFound)
  exact {
    runtimePublicKey
    runtimePublicKeyFound
    runtimePublicKeyLayout
    lane
    lanePublicKey := Option.some.inj (lane.publicKeyFound.symm.trans quotientFound)
  }

end Mxx.Certificate
