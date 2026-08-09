import MxxCorrectness.Proofs.ToyExampleExecution
import Mxx.Certificate.Checker
import Mxx.Certificate.Preconditions
import Mathlib.Tactic

namespace MxxCorrectness.Proofs.ToyExample

open Mxx
open Mxx.Certificate
open MxxCorrectness.Generated.ToyExample

deriving instance DecidableEq for IntExpr
deriving instance DecidableEq for MatrixTypeExpr
deriving instance DecidableEq for BoundExpr
deriving instance DecidableEq for IntBoundExpr
deriving instance DecidableEq for StaticObligation

example : checkProgramDerivation ToyExample_stage_encrypt ToyExample_stage_encrypt_derivation =
    .ok () := by decide

example : checkProgramDerivation ToyExample_stage_decrypt ToyExample_stage_decrypt_derivation =
    .ok () := by decide

example : checkProgramDerivation ToyExample_ideal ToyExample_ideal_derivation = .ok () := by
  decide


def checker (cutoff : Nat) : Bool :=
  match analyzeProtocol ToyExample_protocol { overrides := [] } with
  | .error _ => false
  | .ok analysis =>
      (checkStaticParameters analysis (parameters cutoff)).isOk

/-- The only `native_decide` use: closed Phase-A/Phase-B analyzer evaluation. -/
private theorem closedAnalyzerFacts :
    checker 63 = true ∧
      .thresholdNoise (.parameter (.parameter "cutoff")) (.constant 256) (.constant 2) ∈
        (match analyzeProtocol ToyExample_protocol { overrides := [] } with
        | .ok result => result.staticObligations
        | .error _ => []) := by native_decide

theorem checker_accepts_63 : checker 63 = true := closedAnalyzerFacts.1

theorem decodeCorrect (message : Bool) (noise : Int) (bound : noise.natAbs < 64) :
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

theorem centeredEmod (value : Int) :
    Mxx.centeredCoefficient 256 value % 256 = value % 256 := by
  by_cases upper : 256 < 2 * (value % 256)
  · simp [Mxx.centeredCoefficient, Mxx.reduceCoefficient, upper]
  · simp [Mxx.centeredCoefficient, Mxx.reduceCoefficient, upper]

theorem thresholdCongruent (left right : Int)
    (congruent : left % 256 = right % 256) :
    Mxx.Ir.thresholdDecodeBool 256 2 left = Mxx.Ir.thresholdDecodeBool 256 2 right := by
  simp [Mxx.Ir.thresholdDecodeBool, Mxx.Ir.centeredRepresentative, congruent]

theorem decodeReducedCorrect (message : Bool) (noise : Int)
    (bound : (Mxx.centeredCoefficient 256 noise).natAbs < 64) :
    Mxx.Ir.thresholdDecodeBool 256 2
      (Mxx.reduceCoefficient 256 ((if message then 128 else 0) + noise)) = message := by
  have decoded := decodeCorrect message (Mxx.centeredCoefficient 256 noise) bound
  have congruent :
      Mxx.reduceCoefficient 256 ((if message then 128 else 0) + noise) % 256 =
        ((if message then 128 else 0) + Mxx.centeredCoefficient 256 noise) % 256 := by
    simp [Mxx.reduceCoefficient, Int.add_emod, centeredEmod]
  exact (thresholdCongruent _ _ congruent).trans decoded

private theorem staticThresholdObligation
    {analysis : AnalysisResult}
    (phaseA : analyzeProtocol ToyExample_protocol { overrides := [] } = .ok analysis) :
    .thresholdNoise (.parameter (.parameter "cutoff")) (.constant 256) (.constant 2) ∈
      analysis.staticObligations := by
  have derived := closedAnalyzerFacts.2
  rw [phaseA] at derived
  exact derived

theorem correct
    (samplers : MxxSamplerFamily)
    (contract : MxxBoundedSamplerContract samplers)
    (cutoff : Nat)
    (message : Bool) :
    ToyExample_protocol.ParamsWF (parameters cutoff) →
    ToyExample_protocol.InputsWF (parameters cutoff) (inputs message) →
    ToyExample_protocol.ProtocolPreconditions (parameters cutoff) (inputs message) →
    ∀ analysis checked,
    analyzeProtocol ToyExample_protocol { overrides := [] } = .ok analysis →
    checkStaticParameters analysis (parameters cutoff) = .ok checked →
    ∀ outcome ∈ denoteProtocolBundleOutcomes samplers ToyExample_protocol.bundle
      (parameters cutoff) (inputs message),
      outcome.comparator.failure = false := by
  intro _paramsWF _inputsWF _preconditions analysis checked phaseA phaseB outcome member
  simp only [denoteProtocolBundleOutcomes, List.mem_map] at member
  obtain ⟨workflowOutput, workflowMember, rfl⟩ := member
  obtain ⟨sample, sampleMember, outputIdentity⟩ :=
    workflowOutcome samplers cutoff message workflowOutput workflowMember
  have sampleBound := contract.gaussianHardSupport (samplerParams cutoff) sample sampleMember
  have sampleBound' : Mxx.maxCenteredCoefficientNorm
      (sample.withSamplerParams (samplerParams cutoff)) ≤ cutoff := by
    simpa [samplerParams] using sampleBound
  have thresholdHolds := checkStaticParameters_sound phaseB _
    (staticThresholdObligation phaseA)
  change 0 < (256 : Int) ∧ 0 < (2 : Int) ∧ 2 * 2 * cutoff < 256 at thresholdHolds
  have cutoffLt : cutoff < 64 := by omega
  have noiseBound :
      (Mxx.centeredCoefficient 256
        ((sample.withSamplerParams (samplerParams cutoff)).coefficients.headD 0)).natAbs < 64 := by
    have headBound := Mxx.headD_natAbs_le_norm
      (sample.withSamplerParams (samplerParams cutoff))
    exact lt_of_le_of_lt (le_trans headBound sampleBound') cutoffLt
  have decoded := decodeReducedCorrect message
    ((sample.withSamplerParams (samplerParams cutoff)).coefficients.headD 0) noiseBound
  have outputIdentity' : [("decoded", .boolean message)] = workflowOutput := by
    rw [decoded] at outputIdentity
    simpa using outputIdentity
  rw [← outputIdentity']
  unfold denoteClosedProtocolOutcome
  have idealEq :
      Mxx.Ir.denotePure ToyExample_protocol.bundle.ideal (parameters cutoff)
          (idealInputEnvironment ToyExample_protocol.bundle (inputs message)) =
        some [("result", .boolean message)] := by
    have different :
        (ProtocolInputDestination.ideal "message" ==
          ProtocolInputDestination.workflowStage ⟨"encrypt"⟩ "message") = false := by decide
    have same :
        (ProtocolInputDestination.ideal "message" ==
          ProtocolInputDestination.ideal "message") = true := by decide
    cases message <;>
      simp [ToyExample_protocol, ToyExample_ideal, parameters, inputs, idealInputEnvironment,
        protocolDestinationValue, protocolInputBoundTo, protocolInputName, Mxx.Ir.denotePure,
        Mxx.Ir.denote, Mxx.Ir.denoteScopeWithFuel, Mxx.Ir.evaluateNodes,
        Mxx.Ir.evaluateNode, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs, Mxx.Ir.collectOutputs,
        Mxx.Ir.lookupEnvironment, different, same]
  rw [idealEq]
  apply denoteComparator_singleEquality_success ToyExample_protocol.bundle (parameters cutoff)
    [("decoded", .boolean message)] [("result", .boolean message)]
    {
      endpoint := .toyThresholdDecode
      actualInput := "decoded"
      idealInput := "result"
      resultOutput := "failure"
      failureValue := true
    }
    {
      specification := .toyThresholdDecode
      stage := ⟨"decrypt"⟩
      semanticAnchor := { stage := ⟨"decrypt"⟩, label := "decoded-endpoint" }
      semantics := .thresholdDecode
      workflowOutput := "decoded"
      idealOutput := "result"
    }
    message
  · rfl
  · rfl
  · simp [Mxx.Ir.lookupEnvironment]
  · simp [Mxx.Ir.lookupEnvironment]

end MxxCorrectness.Proofs.ToyExample
