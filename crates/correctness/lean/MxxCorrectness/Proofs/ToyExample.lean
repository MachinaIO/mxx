import MxxCorrectness.Generated.ToyExample.Statement
import Mathlib.Tactic

open MxxCorrectness.Generated.ToyExample
open Mxx

def toyExampleSamplerParams (cutoff : Nat) : Mxx.SamplerParams := {
  maxCoefficientBound := cutoff
  modulus := 256
  ringDimension := 1
  rows := 1
  columns := 1
}

theorem toyExampleDecodeCorrect (message : Bool) (noise : Int) (bound : noise.natAbs < 64) :
    Mxx.Ir.thresholdDecodeBool 256 2 ((if message then 128 else 0) + noise) = message := by
  have boundInt : (noise.natAbs : Int) < 64 := by exact_mod_cast bound
  have lower : -64 < noise := by
    by_cases nonnegative : 0 ≤ noise
    · omega
    · have boundNeg : ((-noise).natAbs : Int) < 64 := by
        exact_mod_cast (show (-noise).natAbs < 64 by simpa using bound)
      rw [Int.natAbs_of_nonneg (by omega)] at boundNeg
      omega
  have upper : noise < 64 := by
    by_cases nonnegative : 0 ≤ noise
    · rw [Int.natAbs_of_nonneg nonnegative] at boundInt
      omega
    · omega
  cases message
  · by_cases nonnegative : 0 ≤ noise
    · have reduced : noise % 256 = noise := Int.emod_eq_of_lt nonnegative (by omega)
      simp [Mxx.Ir.thresholdDecodeBool, Mxx.Ir.centeredRepresentative, reduced]
      omega
    · have shifted : (noise + 256) % 256 = noise + 256 :=
        Int.emod_eq_of_lt (by omega) (by omega)
      have reduced : noise % 256 = noise + 256 := by
        rw [← shifted]
        omega
      simp [Mxx.Ir.thresholdDecodeBool, Mxx.Ir.centeredRepresentative, reduced]
      omega
  · have reduced : (128 + noise) % 256 = 128 + noise :=
      Int.emod_eq_of_lt (by omega) (by omega)
    simp [Mxx.Ir.thresholdDecodeBool, Mxx.Ir.centeredRepresentative, reduced]
    omega

theorem toyExampleCenteredEmod (value : Int) :
    Mxx.centeredCoefficient 256 value % 256 = value % 256 := by
  by_cases upper : 256 < 2 * (value % 256)
  · simp [Mxx.centeredCoefficient, Mxx.reduceCoefficient, upper]
  · simp [Mxx.centeredCoefficient, Mxx.reduceCoefficient, upper]

theorem toyExampleThresholdCongruent (left right : Int)
    (congruent : left % 256 = right % 256) :
    Mxx.Ir.thresholdDecodeBool 256 2 left =
      Mxx.Ir.thresholdDecodeBool 256 2 right := by
  simp [Mxx.Ir.thresholdDecodeBool, Mxx.Ir.centeredRepresentative, congruent]

theorem toyExampleDecodeReducedCorrect (message : Bool) (noise : Int)
    (bound : (Mxx.centeredCoefficient 256 noise).natAbs < 64) :
    Mxx.Ir.thresholdDecodeBool 256 2
      (Mxx.reduceCoefficient 256 ((if message then 128 else 0) + noise)) = message := by
  have decoded := toyExampleDecodeCorrect message (Mxx.centeredCoefficient 256 noise) bound
  have congruent :
      Mxx.reduceCoefficient 256 ((if message then 128 else 0) + noise) % 256 =
        ((if message then 128 else 0) + Mxx.centeredCoefficient 256 noise) % 256 := by
    simp [Mxx.reduceCoefficient, Int.add_emod, toyExampleCenteredEmod]
  exact (toyExampleThresholdCongruent _ _ congruent).trans decoded

theorem toyExampleConcreteOutcome
    (samplers : MxxSamplerFamily) (p : ToyExampleParams) (x : ToyExampleInputs p)
    (output : Mxx.Ir.Environment)
    (member : output ∈ ToyExampleConcreteOutcomes samplers p x) :
    ∃ sample ∈ samplers.gaussianSample (toyExampleSamplerParams p.cutoff),
      [("ciphertext", Mxx.Ir.Value.matrix
        { coefficients :=
            List.map (Mxx.reduceCoefficient 256)
              (Mxx.Ir.addCoefficients (if x.message then [128] else [0])
                (sample.withSamplerParams (toyExampleSamplerParams p.cutoff)).coefficients)
          modulus := 256
          ringDimension := 1
          rows := 1
          columns := 1 })] = output := by
  rcases x with ⟨message⟩
  have cutoffNonnegative : ¬ ((p.cutoff : Int) < 0) := by omega
  cases message <;> simpa [ToyExampleConcreteOutcomes, ToyExample_workflow,
    ToyExample_stage_encrypt, ToyExample_stage_decrypt, ToyExampleParamEnvironment,
    ToyExampleInputEnvironment, Mxx.Ir.denoteWorkflow, Mxx.Ir.evaluateStages,
    Mxx.Ir.stageInputs, Mxx.Ir.resolveStageInput, Mxx.Ir.lookupStage,
    Mxx.Ir.lookupEnvironment, Mxx.Ir.denote, Mxx.Ir.evaluateNodes,
    Mxx.Ir.denoteScopeWithFuel, Mxx.Ir.lookupDefinition,
    Mxx.Ir.evaluateNode, Mxx.Ir.arguments, Mxx.Ir.lookupWire,
    Mxx.Ir.bindOutputs, Mxx.Ir.collectOutputs, Mxx.Ir.IntExpr.evaluate,
    Mxx.Ir.MatrixTypeExpr.evaluate, Mxx.Matrix.withSamplerParams,
    toyExampleSamplerParams, Mxx.Ir.lookupParam, Mxx.matrixAdd, Mxx.reduceCoefficient,
    cutoffNonnegative] using member

theorem toyExampleFailureBoolSafe
    (p : ToyExampleParams) (x : ToyExampleInputs p) (sample : Mxx.Matrix)
    (sampleModulus : sample.modulus = 256)
    (bound : Mxx.maxCenteredCoefficientNorm sample ≤ p.cutoff)
    (cutoff : p.cutoff < 64) :
    ToyExampleFailureBool p x
      [("ciphertext", Mxx.Ir.Value.matrix
        { coefficients :=
            List.map (Mxx.reduceCoefficient 256)
              (Mxx.Ir.addCoefficients (if x.message then [128] else [0]) sample.coefficients)
          modulus := 256
          ringDimension := 1
          rows := 1
          columns := 1 })] = false := by
  have noiseBound :
      (Mxx.centeredCoefficient 256 (sample.coefficients.headD 0)).natAbs < 64 := by
    rw [← sampleModulus]
    have headBound := Mxx.headD_natAbs_le_norm
      sample
    exact lt_of_le_of_lt (le_trans headBound bound) cutoff
  have decoded := toyExampleDecodeReducedCorrect x.message
    (sample.coefficients.headD 0) noiseBound
  rcases x with ⟨message⟩
  cases sample with
  | mk coefficients modulus ringDimension rows columns =>
    cases message <;> cases coefficients <;>
      simp [ToyExampleFailureBool, ToyExampleIdealOutput, ToyExample_ideal,
        ToyExample_comparatorMap, ToyExampleParamEnvironment, ToyExampleInputEnvironment,
        Mxx.Ir.denotePure, Mxx.Ir.denote, Mxx.Ir.denoteScopeWithFuel,
        Mxx.Ir.lookupDefinition, Mxx.Ir.evaluateNodes,
        Mxx.Ir.evaluateNode, Mxx.Ir.arguments, Mxx.Ir.lookupWire,
        Mxx.Ir.bindOutputs, Mxx.Ir.collectOutputs, Mxx.Ir.IntExpr.evaluate,
        Mxx.Ir.projectOutputs, Mxx.Ir.lookupEnvironment, Mxx.Ir.rebindInputs,
        Mxx.Ir.environmentValues, Mxx.Ir.environmentValid, Mxx.Ir.Value.isValid,
        Mxx.Ir.valuesEqual, Mxx.Ir.Value.equal, Mxx.Ir.addCoefficients,
        Mxx.addCoefficients, Mxx.reduceCoefficient] at decoded ⊢ <;>
      exact decoded

def toyExampleChecker (p : ToyExampleParams) : Bool :=
  decide (4 * p.cutoff < 256)

theorem toyExample_correct : ToyExampleCorrectStatement toyExampleChecker := by
  constructor
  · refine ⟨{ cutoff := 0 }, ?_, ?_⟩
    · decide
    · norm_num [ToyExampleParamsValid]
  intro samplers contract p checker _ x _ _
  have checkerFacts : 4 * p.cutoff < 256 := by
    simpa [toyExampleChecker] using checker
  have cutoff_lt : p.cutoff < 64 := by omega
  unfold ToyExampleFailureProbability
  apply Mxx.booleanFailureProbability_eq_zero
  intro output member
  obtain ⟨sample, sampleMember, outputIdentity⟩ :=
    toyExampleConcreteOutcome samplers p x output member
  have sampleBound :=
    contract.gaussianHardSupport (toyExampleSamplerParams p.cutoff) sample sampleMember
  have safe := toyExampleFailureBoolSafe p x
    (sample.withSamplerParams (toyExampleSamplerParams p.cutoff)) (by rfl) sampleBound cutoff_lt
  rw [← outputIdentity, safe]
