import Mxx.Certificate.OperationalNoise.ToyABI

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.ToyABI

open Mxx.Certificate.OperationalNoise
open EventReplay

def fixture : ToyCertificate where
  rows := toyRows
  expressions := expectedExpressions
  statementEvents := expectedStatementEvents
  proofEvents := expectedProofEvents

theorem fixture_valid : ToyValid fixture := by
  decide

theorem fixture_sampler_contract : ToySamplerContract fixture 1 := by
  refine ⟨rfl, ?_⟩
  decide

theorem fixture_sampler_sound : (recordedFiniteContract 1).Interprets (1 : Int).natAbs :=
  fixture_sampler_contract.sound

theorem fixture_preimage_contract : ToyPreimageContract fixture 1 := by
  refine ⟨rfl, ?_⟩
  decide

theorem fixture_universal_relation : ToyUniversalRelation fixture :=
  fixture_valid.universalRelation

def ToyMonomial.toCore (value : ToyMonomial) : MonomialKey :=
  { centralFactors := [],
    orderedFactors := value.ordered.map (fun factor => factor.expression.row) }

def ToyTerm.toCore (value : ToyTerm) : ExactTerm :=
  { coefficient := value.coefficient, key := value.monomial.toCore }

def relationContext : MonomialContext :=
  { exteriorCentral := [], prefixFactors := [], suffixFactors := [] }

def relationValuation (key : MonomialKey) : Int :=
  if key.orderedFactors = relationLeftTerm.toCore.key.orderedFactors ∨
      key.orderedFactors = targetTerm.toCore.key.orderedFactors then 1 else 0

theorem fixture_base_relation :
    evaluatePolynomial relationValuation [relationLeftTerm.toCore] =
      evaluatePolynomial relationValuation [targetTerm.toCore] := by
  decide

theorem fixture_relation_reconstruction :
    evaluatePolynomial relationValuation
        (relationReplacement relationContext (-1) [relationLeftTerm.toCore]) =
      evaluatePolynomial relationValuation
        (relationReplacement relationContext (-1) [targetTerm.toCore]) := by
  apply relationReplacement_congruent relationValuation relationContext 1 (-1)
  · intro key
    simp [relationContext, MonomialContext.plug, relationValuation]
  · exact fixture_base_relation

def mergedPolynomial : Polynomial :=
  [targetTerm.toCore, targetCancellation.toCore, noiseTerm.toCore]

theorem fixture_relation_merge_cancels :
    coefficient targetTerm.toCore.key mergedPolynomial = 0 := by
  decide

theorem fixture_operator_merge_survives :
    coefficient noiseTerm.toCore.key mergedPolynomial = 1 := by
  decide

theorem fixture_preimage_transfer :
    (recordedFiniteContract 1).Interprets (1 : Int).natAbs :=
  fixture_preimage_contract.sound

theorem fixture_product_transfer : 1 * 1 ≤ 1 * 1 :=
  boundTransfer_product (by decide) (by decide)

theorem fixture_monomial_product :
    productNonempty (recordedFiniteContract 1) [] = recordedFiniteContract 1 := by
  rfl

theorem fixture_sum_transfer : 0 + 1 ≤ 0 + 1 :=
  boundTransfer_sum (by decide) (by decide)

theorem fixture_survivor_fold : [1].sum ≤ [1].sum := by
  exact survivorFold_sound (.cons (by decide) .nil)

theorem fixture_invocation_end : 0 + [1].sum ≤ 0 + [1].sum := by
  exact preFold_to_invocationEnd (by decide) (.cons (by decide) .nil)

theorem fixture_centered_bridge :
    centeredCoefficient 257 ((1 : Int) - 1 + 1) = centeredCoefficient 257 1 := by
  exact centeredCoefficient_add_relation (by decide) (by decide)

theorem fixture_lifted_norm :
    (liftCoefficient 1).maxCenteredCoefficientNorm 257 = 1 := by
  rw [liftCoefficient_norm]
  decide

theorem fixture_final_inequality : 2 * 2 * centeredNorm 257 1 = 4 ∧ 4 < 257 := by
  decide

theorem fixture_operational_claim : ToyOperationalClaim 1 := by
  unfold ToyOperationalClaim centeredNorm centeredCoefficient
  decide

/-- The fixed ABI composes statement-row contracts, typed universal replay, exact merge
    cancellation, owner-local transfer/fold arithmetic, modular centering, and the final strict
    operational inequality. -/
theorem toy_event_replay :
    ToyValid fixture ∧
      ToySamplerContract fixture 1 ∧
      ToyPreimageContract fixture 1 ∧
      ToyUniversalRelation fixture ∧
      evaluatePolynomial relationValuation
          (relationReplacement relationContext (-1) [relationLeftTerm.toCore]) =
        evaluatePolynomial relationValuation
          (relationReplacement relationContext (-1) [targetTerm.toCore]) ∧
      coefficient targetTerm.toCore.key mergedPolynomial = 0 ∧
      coefficient noiseTerm.toCore.key mergedPolynomial = 1 ∧
      (recordedFiniteContract 1).Interprets (1 : Int).natAbs ∧
      productNonempty (recordedFiniteContract 1) [] = recordedFiniteContract 1 ∧
      1 * 1 ≤ 1 * 1 ∧
      0 + 1 ≤ 0 + 1 ∧
      [1].sum ≤ [1].sum ∧
      0 + [1].sum ≤ 0 + [1].sum ∧
      centeredCoefficient 257 ((1 : Int) - 1 + 1) = centeredCoefficient 257 1 ∧
      (liftCoefficient 1).maxCenteredCoefficientNorm 257 = 1 ∧
      2 * 2 * centeredNorm 257 1 = 4 ∧
      4 < 257 ∧
      ToyOperationalClaim 1 := by
  exact ⟨fixture_valid, fixture_sampler_contract, fixture_preimage_contract,
    fixture_universal_relation,
    fixture_relation_reconstruction, fixture_relation_merge_cancels,
    fixture_operator_merge_survives, fixture_sampler_sound, fixture_monomial_product,
    fixture_product_transfer, fixture_sum_transfer, fixture_survivor_fold,
    fixture_invocation_end, fixture_centered_bridge, fixture_lifted_norm,
    fixture_final_inequality.1, fixture_final_inequality.2, fixture_operational_claim⟩

#print axioms toy_event_replay
#print axioms ToyValid.universalRelation
#print axioms centeredCoefficient_add_relation

end Mxx.Certificate.OperationalNoise.ToyABI
