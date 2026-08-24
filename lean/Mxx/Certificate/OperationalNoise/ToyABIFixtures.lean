import Mxx.Certificate.OperationalNoise.ToyABI

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.ToyABI

open Mxx.Certificate.OperationalNoise
open EventReplay

def fixtureCertificate : ToyCertificate where
  expressions :=
    [publicRow, preimageRow toyRows, targetRow toyRows, noiseRow toyRows, rootRow toyRows]
  statementEvents := [preimageStatementRow toyRows, noiseStatementRow toyRows]

/-- The event inventory is authored once. `ToyValid` independently checks each fixed position. -/
def fixtureEvents : List ToyEvent :=
  [ .invocationStart (owner toyRows toyRows.root),
    .invocationStart (owner toyRows toyRows.publicExpression),
    .result (owner toyRows toyRows.publicExpression) (publicValue toyRows),
    .invocationEnd (owner toyRows toyRows.publicExpression) (publicValue toyRows),
    .invocationStart (owner toyRows toyRows.preimageExpression),
    .result (owner toyRows toyRows.preimageExpression) (preimageValue toyRows),
    .invocationEnd (owner toyRows toyRows.preimageExpression) (preimageValue toyRows),
    .invocationStart (owner toyRows toyRows.targetExpression),
    .predecessor (owner toyRows toyRows.targetExpression) 0 toyRows.publicExpression 2,
    .predecessor (owner toyRows toyRows.targetExpression) 1 toyRows.preimageExpression 5,
    .result (owner toyRows toyRows.targetExpression) (targetValue toyRows),
    .invocationEnd (owner toyRows toyRows.targetExpression) (targetValue toyRows),
    .invocationStart (owner toyRows toyRows.noiseExpression),
    .result (owner toyRows toyRows.noiseExpression) (noiseValue toyRows),
    .invocationEnd (owner toyRows toyRows.noiseExpression) (noiseValue toyRows),
    .predecessor (owner toyRows toyRows.root) 0 toyRows.targetExpression 10,
    .predecessor (owner toyRows toyRows.root) 1 toyRows.noiseExpression 13,
    .specializationComputed (owner toyRows toyRows.root) 7 12 0 10 11,
    .appliedUniversal (owner toyRows toyRows.root) 17 toyRows.preimageEvent
      (relationLeftTerm toyRows).monomial (-1) 0 2 (relationLeftTerm toyRows)
      (targetTerm toyRows) 10 0,
    .boundTransfer (owner toyRows toyRows.root)
      (.authorityRelationPreimageSource toyRows.preimageEvent) 8,
    .boundTransfer (owner toyRows toyRows.root)
      (.authorityNoiseOperator toyRows.noiseEvent) 1,
    .boundTransfer (owner toyRows toyRows.root) (.monomialProduct [19]) 8,
    .boundTransfer (owner toyRows toyRows.root) (.product 19 20) 8,
    .boundTransfer (owner toyRows toyRows.root) (.sum [19, 20]) 9,
    .coefficientMerge (owner toyRows toyRows.root)
      (.relation 18 relationContributionOrdinal (targetCancellation toyRows)),
    .coefficientMerge (owner toyRows toyRows.root)
      (.operator ⟨10, 0⟩ ⟨13, 0⟩ (noiseTerm toyRows)),
    .preFold (owner toyRows toyRows.root)
      [targetTerm toyRows, targetCancellation toyRows] 0,
    .survivorFold 1 20,
    .result (owner toyRows toyRows.root) (rootValue toyRows),
    .invocationEnd (owner toyRows toyRows.root) (rootValue toyRows) ]

theorem fixture_valid : ToyValid fixtureCertificate toyRows fixtureEvents := by
  decide

theorem fixture_sampler_contract : ToySamplerContract fixtureCertificate toyRows 1 := by
  exact ⟨1, fixture_valid.noiseCutoff, by decide⟩

theorem fixture_preimage_contract : ToyPreimageContract fixtureCertificate toyRows 8 := by
  exact ⟨8, fixture_valid.preimageCutoff, by decide⟩

theorem fixture_sampler_sound : (recordedFiniteContract 1).Interprets (1 : Int).natAbs :=
  fixture_sampler_contract.sound fixture_valid

theorem fixture_preimage_sound : (recordedFiniteContract 8).Interprets (8 : Int).natAbs :=
  fixture_preimage_contract.sound fixture_valid

theorem fixture_universal_relation :
    ToyUniversalRelation fixtureCertificate toyRows fixtureEvents 1 1 :=
  fixture_valid.universalRelation

def ToyMonomial.toCore (value : ToyMonomial) : MonomialKey :=
  { centralFactors := [],
    orderedFactors := value.ordered.map (fun factor => factor.expression.row) }

def ToyTerm.toCore (value : ToyTerm) : ExactTerm :=
  { coefficient := value.coefficient, key := value.monomial.toCore }

def relationContext : MonomialContext :=
  { exteriorCentral := [], prefixFactors := [], suffixFactors := [] }

def relationValuation (key : MonomialKey) : Int :=
  if key.orderedFactors = (relationLeftTerm toyRows).toCore.key.orderedFactors ∨
      key.orderedFactors = (targetTerm toyRows).toCore.key.orderedFactors then 1 else 0

theorem fixture_base_relation :
    evaluatePolynomial relationValuation [(relationLeftTerm toyRows).toCore] =
      evaluatePolynomial relationValuation [(targetTerm toyRows).toCore] := by
  decide

theorem fixture_relation_reconstruction :
    evaluatePolynomial relationValuation
        (relationReplacement relationContext (-1) [(relationLeftTerm toyRows).toCore]) =
      evaluatePolynomial relationValuation
        (relationReplacement relationContext (-1) [(targetTerm toyRows).toCore]) := by
  apply relationReplacement_congruent relationValuation relationContext 1 (-1)
  · intro key
    simp [relationContext, MonomialContext.plug, relationValuation]
  · exact fixture_base_relation

def mergedPolynomial : Polynomial :=
  [(targetTerm toyRows).toCore, (targetCancellation toyRows).toCore,
    (noiseTerm toyRows).toCore]

theorem fixture_relation_merge_cancels :
    coefficient (targetTerm toyRows).toCore.key mergedPolynomial = 0 := by
  decide

theorem fixture_operator_merge_survives :
    coefficient (noiseTerm toyRows).toCore.key mergedPolynomial = 1 := by
  decide

theorem fixture_product_transfer : 8 * 1 ≤ 8 * 1 :=
  boundTransfer_product (by decide) (by decide)

theorem fixture_monomial_product :
    productNonempty (recordedFiniteContract 8) [] = recordedFiniteContract 8 := by
  rfl

theorem fixture_sum_transfer : 0 + 1 ≤ 0 + 1 :=
  boundTransfer_sum (by decide) (by decide)

theorem fixture_survivor_fold : [1].sum ≤ [1].sum := by
  exact survivorFold_sound (.cons (by decide) .nil)

theorem fixture_invocation_end : 0 + [1].sum ≤ 0 + [1].sum := by
  exact preFold_to_invocationEnd (by decide) (.cons (by decide) .nil)

theorem fixture_operational_proof :
    ToyOperationalClaim fixtureCertificate toyRows fixtureEvents ((1 : Int) - 1 + 1) :=
  operationalProof fixture_valid fixture_sampler_contract fixture_universal_relation

theorem fixture_lifted_norm :
    (liftCoefficient ((1 : Int) - 1 + 1)).maxCenteredCoefficientNorm 257 = 1 := by
  rw [liftCoefficient_norm]
  decide

/-- The fixed ABI composes structural validation, row-derived sampler bounds, typed universal
    replay, exact merge cancellation, transfer/fold arithmetic, and the final strict inequality. -/
theorem toy_event_replay :
    ToyValid fixtureCertificate toyRows fixtureEvents ∧
      ToySamplerContract fixtureCertificate toyRows 1 ∧
      ToyPreimageContract fixtureCertificate toyRows 8 ∧
      ToyUniversalRelation fixtureCertificate toyRows fixtureEvents 1 1 ∧
      evaluatePolynomial relationValuation
          (relationReplacement relationContext (-1) [(relationLeftTerm toyRows).toCore]) =
        evaluatePolynomial relationValuation
          (relationReplacement relationContext (-1) [(targetTerm toyRows).toCore]) ∧
      coefficient (targetTerm toyRows).toCore.key mergedPolynomial = 0 ∧
      coefficient (noiseTerm toyRows).toCore.key mergedPolynomial = 1 ∧
      (recordedFiniteContract 1).Interprets (1 : Int).natAbs ∧
      (recordedFiniteContract 8).Interprets (8 : Int).natAbs ∧
      productNonempty (recordedFiniteContract 8) [] = recordedFiniteContract 8 ∧
      8 * 1 ≤ 8 * 1 ∧
      0 + 1 ≤ 0 + 1 ∧
      [1].sum ≤ [1].sum ∧
      0 + [1].sum ≤ 0 + [1].sum ∧
      ToyOperationalClaim fixtureCertificate toyRows fixtureEvents ((1 : Int) - 1 + 1) ∧
      (liftCoefficient ((1 : Int) - 1 + 1)).maxCenteredCoefficientNorm 257 = 1 := by
  exact ⟨fixture_valid, fixture_sampler_contract, fixture_preimage_contract,
    fixture_universal_relation, fixture_relation_reconstruction,
    fixture_relation_merge_cancels, fixture_operator_merge_survives,
    fixture_sampler_sound, fixture_preimage_sound, fixture_monomial_product,
    fixture_product_transfer, fixture_sum_transfer, fixture_survivor_fold,
    fixture_invocation_end, fixture_operational_proof, fixture_lifted_norm⟩

#print axioms toy_event_replay
#print axioms operationalProof
#print axioms ToyValid.universalRelation

end Mxx.Certificate.OperationalNoise.ToyABI
