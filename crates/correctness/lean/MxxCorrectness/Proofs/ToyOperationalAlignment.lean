/-
Deferred proof source retained for the later end-to-end correctness milestone.

This proof depends on the superseded operational semantic interface and is intentionally commented
out. The active Toy operational checker is `MxxCorrectness.OperationalToy`.

import MxxCorrectness.Generated.ToyExample.Ir
import Mxx.Certificate.OperationalSemantics

namespace MxxCorrectness.Proofs.ToyOperationalAlignment

open Mxx Mxx.Ir Mxx.Certificate
open MxxCorrectness.Generated.ToyExample

private def operationalParameters (cutoff : Nat) : ParamEnvironment :=
  [("cutoff", .integer cutoff)]

/-- The generated Toy encryption scope preserves execution-aligned operational evidence across
all seven executable nodes. This is the first whole-scope gate for the generic local rules. -/
theorem encryptScopeOperationalAlignment
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (contract : MxxBoundedSamplerContract samplers)
    (cutoff : Nat)
    (message : Bool)
    (output : WireEnvironment)
    (path : EvaluatesNodesPath runChild samplers (operationalParameters cutoff)
      [("message", .boolean message)] 0 ToyExample_stage_encrypt.root.nodes [] output) :
    ∃ facts entries,
      OperationalScopeAlignment samplers (operationalParameters cutoff) [] facts output entries := by
  simp only [ToyExample_stage_encrypt] at path
  rw [evaluatesNodesPath_cons_iff] at path
  obtain ⟨values0, member0, path⟩ := path
  rw [evaluatesNodesPath_cons_iff] at path
  obtain ⟨values1, member1, path⟩ := path
  rw [evaluatesNodesPath_cons_iff] at path
  obtain ⟨values2, member2, path⟩ := path
  rw [evaluatesNodesPath_cons_iff] at path
  obtain ⟨values3, member3, path⟩ := path
  rw [evaluatesNodesPath_cons_iff] at path
  obtain ⟨values4, member4, path⟩ := path
  rw [evaluatesNodesPath_cons_iff] at path
  obtain ⟨values5, member5, path⟩ := path
  rw [evaluatesNodesPath_cons_iff] at path
  obtain ⟨values6, member6, path⟩ := path
  let address : ExecutionAddress := {
    root := .workflowStage ⟨"encrypt"⟩
    frames := []
  }
  have alignment0 := OperationalScopeAlignment.nil samplers (operationalParameters cutoff) []
  obtain ⟨entries0, values0Eq, alignment1⟩ := alignment0.stepBooleanInput
    (runChild := runChild) (inputs := [("message", .boolean message)]) address
    "message" message (by simp [Mxx.Ir.lookupEnvironment]) member0
  subst values0
  obtain ⟨entries1, values1Eq, alignment2⟩ := alignment1.stepBoolToInt
    (runChild := runChild) (inputs := [("message", .boolean message)]) address
    { node := 0, port := 0 } message (by simp [Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs]) member1
  subst values1
  let matrixType : Mxx.Ir.MatrixTypeExpr := {
    modulus := .constant 256
    ringDimension := .constant 1
    rows := .constant 1
    columns := .constant 1
  }
  let zeroParams : Mxx.SamplerParams := {
    maxCoefficientBound := 0
    modulus := 256
    ringDimension := 1
    rows := 1
    columns := 1
  }
  obtain ⟨entries2, values2Eq, alignment3⟩ := alignment2.stepZeroMatrix
    (runChild := runChild) (inputs := [("message", .boolean message)]) address matrixType
    zeroParams (by simp [matrixType, zeroParams, Mxx.Ir.MatrixTypeExpr.evaluate,
      Mxx.Ir.IntExpr.evaluate]) member2
  subst values2
  let constantFactParams : Mxx.SamplerParams := {
    maxCoefficientBound := 128
    modulus := 256
    ringDimension := 1
    rows := 1
    columns := 1
  }
  obtain ⟨entries3, values3Eq, alignment4⟩ := alignment3.stepConstantMatrix
    (runChild := runChild) (inputs := [("message", .boolean message)]) address
    matrixType [.constant 128] zeroParams constantFactParams [128]
    (by simp [matrixType, zeroParams, Mxx.Ir.MatrixTypeExpr.evaluate,
      Mxx.Ir.IntExpr.evaluate])
    (by simp [operationalParameters, Mxx.Ir.IntExpr.evaluate])
    (by norm_num [zeroParams]) rfl rfl rfl rfl member3
  subst values3
  let zeroOutput := zeroConstantOutput zeroParams
  let constantOutput := Mxx.Matrix.withSamplerParams {
    coefficients := [Mxx.reduceCoefficient zeroParams.modulus 128]
  } zeroParams
  let selectedOutput := if message then constantOutput else zeroOutput
  have zeroMagnitude : OperationalMatrixMagnitude.Holds (operationalParameters cutoff) []
      (.finite (.closedInt (.constant 0))) zeroOutput := by
    refine ⟨0, by simp, by omega, ?_⟩
    simp [zeroOutput, zeroConstantOutput, zeroParams, Mxx.maxCenteredCoefficientNorm,
      Mxx.coefficientNorm, Mxx.Matrix.withSamplerParams, Mxx.centeredCoefficient,
      Mxx.reduceCoefficient]
  have constantMagnitude : OperationalMatrixMagnitude.Holds (operationalParameters cutoff) []
      (.finite (.closedInt (.constant 128))) constantOutput := by
    refine ⟨128, by simp, by omega, ?_⟩
    norm_num [constantOutput, zeroParams, Mxx.maxCenteredCoefficientNorm,
      Mxx.coefficientNorm, Mxx.Matrix.withSamplerParams, Mxx.centeredCoefficient,
      Mxx.reduceCoefficient]
  have selected : [zeroOutput, constantOutput][(if message then (1 : Int) else 0).toNat]? =
      some selectedOutput := by
    cases message <;> simp [selectedOutput]
  have selectedType : Mxx.Certificate.MatrixTypeExpr.Holds matrixType
      (operationalParameters cutoff) selectedOutput := by
    cases message <;>
      refine ⟨zeroParams, ?_, ?_, ?_, ?_, ?_, ?_⟩
    all_goals simp [selectedOutput, constantOutput, zeroOutput, zeroConstantOutput, matrixType,
      zeroParams, Mxx.Ir.MatrixTypeExpr.evaluate, Mxx.Matrix.withSamplerParams,
      Mxx.Ir.IntExpr.evaluate, Mxx.Matrix.WellFormed]
  obtain ⟨entries4, values4Eq, alignment5⟩ := alignment4.stepSelectTwoFinite
    (runChild := runChild) (inputs := [("message", .boolean message)]) address
    { node := 1, port := 0 } { node := 2, port := 0 } { node := 3, port := 0 }
    (if message then (1 : Int) else 0) zeroOutput constantOutput selectedOutput matrixType
    constantFactParams (.closedInt (.constant 0)) (.closedInt (.constant 128)) 0 128
    zeroMagnitude constantMagnitude (by simp) (by simp) selected selectedType
    (by cases message <;> simp [selectedOutput, constantOutput, zeroOutput, zeroConstantOutput,
      zeroParams, constantFactParams, Mxx.Matrix.withSamplerParams])
    (by cases message <;> simp [selectedOutput, constantOutput, zeroOutput, zeroConstantOutput,
      zeroParams, constantFactParams, Mxx.Matrix.withSamplerParams])
    (by cases message <;> simp [selectedOutput, constantOutput, zeroOutput, zeroConstantOutput,
      zeroParams, constantFactParams, Mxx.Matrix.withSamplerParams])
    (by cases message <;> simp [selectedOutput, constantOutput, zeroOutput, zeroConstantOutput,
      zeroParams, constantFactParams, Mxx.Matrix.withSamplerParams])
    (by cases message <;> norm_num [selectedOutput, constantOutput, zeroOutput,
      zeroConstantOutput, zeroParams, Mxx.Matrix.withSamplerParams])
    (by cases message <;> simp [Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs, zeroOutput,
      constantOutput, zeroConstantOutput, zeroParams, Mxx.Matrix.withSamplerParams]) member4
  subst values4
  let gaussianParams : Mxx.SamplerParams := {
    maxCoefficientBound := cutoff
    modulus := 256
    ringDimension := 1
    rows := 1
    columns := 1
  }
  let gaussianFactParams : Mxx.SamplerParams := {
    maxCoefficientBound := min 128 cutoff
    modulus := 256
    ringDimension := 1
    rows := 1
    columns := 1
  }
  obtain ⟨gaussianOutput, entries5, values5Eq, alignment6⟩ := alignment5.stepGaussian
    (runChild := runChild) (inputs := [("message", .boolean message)])
    contract address matrixType (.parameter "cutoff") gaussianParams zeroParams gaussianFactParams
    cutoff
    (by simp [matrixType, gaussianParams, operationalParameters, Mxx.Ir.MatrixTypeExpr.evaluate,
      Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupParam])
    (by simp [matrixType, zeroParams, Mxx.Ir.MatrixTypeExpr.evaluate, Mxx.Ir.IntExpr.evaluate])
    (by simp [operationalParameters, Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupParam])
    (by omega) (by norm_num [gaussianParams]) (by simp [gaussianParams])
    (by simp [gaussianParams, zeroParams]) (by simp [gaussianParams, zeroParams])
    (by simp [gaussianParams, zeroParams]) (by simp [gaussianParams, zeroParams])
    (by simp [gaussianParams, gaussianFactParams])
    (by simp [gaussianParams, gaussianFactParams])
    (by simp [gaussianParams, gaussianFactParams])
    (by simp [gaussianParams, gaussianFactParams]) member5
  subst values5
  let branchMaximum : OperationalBoundExpr :=
    .maximum (.maximum (.closedInt (.constant 0)) (.closedInt (.constant 0)))
      (.closedInt (.constant 128))
  let selectedMaximum : OperationalBoundExpr :=
    .minimum (.closedInt (.constant 128)) branchMaximum
  have branchMagnitude : OperationalMatrixMagnitude.Holds (operationalParameters cutoff) []
      (.finite branchMaximum) selectedOutput := by
    simpa [branchMaximum] using selectedTwoMagnitudeHolds zeroMagnitude constantMagnitude selected
  have selectedMagnitude : OperationalMatrixMagnitude.Holds (operationalParameters cutoff) []
      (.finite selectedMaximum) selectedOutput := by
    simpa [selectedMaximum] using branchMagnitude.minimumCentered (bound := 128)
      (modulus := 256) (by
        simp [branchMaximum, OperationalBoundExpr.evaluateWithStates]
        rfl)
      (by cases message <;> simp [selectedOutput, constantOutput, zeroOutput,
        zeroConstantOutput, zeroParams, Mxx.Matrix.withSamplerParams]) (by norm_num)
  let wires5 : WireEnvironment :=
    bindOutputs 0 [.boolean message] ++
    bindOutputs 1 [.integer (if message then 1 else 0)] ++
    bindOutputs 2 [.matrix zeroOutput] ++
    bindOutputs 3 [.matrix constantOutput] ++
    bindOutputs 4 [.matrix selectedOutput]
  have member5' : [.matrix gaussianOutput] ∈ evaluateNode runChild samplers
      (operationalParameters cutoff) [("message", .boolean message)] wires5 {
        kind := .gaussianSample matrixType (.parameter "cutoff")
        arguments := []
        outputCount := 1
        outputTypes := [.matrix matrixType]
      } := by
    simpa [wires5, zeroOutput, constantOutput] using member5
  obtain ⟨gaussianOutputAgain, gaussianValues, gaussianFactHolds⟩ :=
    gaussianNode_operational_sound runChild samplers contract (operationalParameters cutoff)
      [("message", .boolean message)] wires5 [] address { node := 5, port := 0 } matrixType
      (.parameter "cutoff") gaussianParams zeroParams gaussianFactParams cutoff 1
      (by simp [matrixType, gaussianParams, operationalParameters, Mxx.Ir.MatrixTypeExpr.evaluate,
        Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupParam])
      (by simp [matrixType, zeroParams, Mxx.Ir.MatrixTypeExpr.evaluate, Mxx.Ir.IntExpr.evaluate])
      (by simp [operationalParameters, Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupParam])
      (by omega) (by norm_num [gaussianParams]) (by simp [gaussianParams])
      (by simp [gaussianParams, zeroParams]) (by simp [gaussianParams, zeroParams])
      (by simp [gaussianParams, zeroParams]) (by simp [gaussianParams, zeroParams])
      (by simp [gaussianParams, gaussianFactParams])
      (by simp [gaussianParams, gaussianFactParams])
      (by simp [gaussianParams, gaussianFactParams])
      (by simp [gaussianParams, gaussianFactParams]) member5'
  have gaussianOutputEq : gaussianOutputAgain = gaussianOutput := by
    symm
    simpa using List.cons.inj gaussianValues
  subst gaussianOutputAgain
  rcases gaussianFactHolds with
    ⟨gaussianType, gaussianModulus, gaussianRingDimension, gaussianRows, gaussianColumns,
      _, gaussianMagnitude, _, _, _⟩
  obtain ⟨sumOutput, entries6, values6Eq, alignment7⟩ := alignment6.stepMatrixAddFinite
    (runChild := runChild) (inputs := [("message", .boolean message)])
    address { node := 4, port := 0 } { node := 5, port := 0 } selectedOutput gaussianOutput
    matrixType zeroParams constantFactParams selectedMaximum
    (.minimum (.closedInt (.constant 128))
      (.contextual .maximum (operationalParameters cutoff) [] (.parameter "cutoff")))
    (by simp [matrixType, zeroParams, Mxx.Ir.MatrixTypeExpr.evaluate, Mxx.Ir.IntExpr.evaluate])
    selectedType gaussianType.wellFormed
    (by cases message <;> simp [selectedOutput, constantOutput, zeroOutput, zeroConstantOutput,
      zeroParams, gaussianFactParams, Mxx.Matrix.withSamplerParams, gaussianModulus])
    (by cases message <;> simp [selectedOutput, constantOutput, zeroOutput, zeroConstantOutput,
      zeroParams, gaussianFactParams, Mxx.Matrix.withSamplerParams, gaussianRingDimension])
    (by cases message <;> simp [selectedOutput, constantOutput, zeroOutput, zeroConstantOutput,
      zeroParams, gaussianFactParams, Mxx.Matrix.withSamplerParams, gaussianRows])
    (by cases message <;> simp [selectedOutput, constantOutput, zeroOutput, zeroConstantOutput,
      zeroParams, gaussianFactParams, Mxx.Matrix.withSamplerParams, gaussianColumns])
    (by cases message <;> simp [selectedOutput, constantOutput, zeroOutput, zeroConstantOutput,
      zeroParams, constantFactParams, Mxx.Matrix.withSamplerParams])
    (by cases message <;> simp [selectedOutput, constantOutput, zeroOutput, zeroConstantOutput,
      zeroParams, constantFactParams, Mxx.Matrix.withSamplerParams])
    (by cases message <;> simp [selectedOutput, constantOutput, zeroOutput, zeroConstantOutput,
      zeroParams, constantFactParams, Mxx.Matrix.withSamplerParams])
    (by cases message <;> simp [selectedOutput, constantOutput, zeroOutput, zeroConstantOutput,
      zeroParams, constantFactParams, Mxx.Matrix.withSamplerParams])
    (by cases message <;> norm_num [selectedOutput, constantOutput, zeroOutput,
      zeroConstantOutput, zeroParams, Mxx.Matrix.withSamplerParams])
    selectedMagnitude gaussianMagnitude
    (by cases message <;> simp [Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs, selectedOutput,
      constantOutput, zeroOutput, zeroConstantOutput, zeroParams, Mxx.Matrix.withSamplerParams])
    member6
  subst values6
  cases path
  exact ⟨_, _, alignment7⟩

end MxxCorrectness.Proofs.ToyOperationalAlignment
-/
