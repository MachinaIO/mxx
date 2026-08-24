import Mxx.Certificate.OperationalNoise.BoundReplay

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.EventReplay

open Mxx.Certificate.OperationalNoise

def g2LeftKey : MonomialKey :=
  { centralFactors := [7], orderedFactors := [11, 12] }

def g2RightKey : MonomialKey :=
  { centralFactors := [3], orderedFactors := [21, 22] }

def g2LeftTerm : ExactTerm := { coefficient := -2, key := g2LeftKey }
def g2RightTerm : ExactTerm := { coefficient := 3, key := g2RightKey }

theorem g2_left_scalar_key :
    scalarProductKey g2LeftKey g2RightKey true false =
      { centralFactors := [3, 7, 11, 12], orderedFactors := [21, 22] } := by
  decide

theorem g2_right_scalar_key :
    scalarProductKey g2LeftKey g2RightKey false true =
      { centralFactors := [3, 7, 21, 22], orderedFactors := [11, 12] } := by
  decide

theorem g2_both_scalar_key :
    scalarProductKey g2LeftKey g2RightKey true true =
      { centralFactors := [3, 7], orderedFactors := [11, 12, 21, 22] } := by
  decide

theorem g2_neither_scalar_key :
    scalarProductKey g2LeftKey g2RightKey false false =
      { centralFactors := [3, 7], orderedFactors := [11, 12, 21, 22] } := by
  decide

def g2LeftScalarContribution : ExactTerm :=
  operatorProductContribution g2LeftTerm g2RightTerm true false

def g2CancellationTerm : ExactTerm :=
  { coefficient := 6, key := g2LeftScalarContribution.key }

def g2MergedPolynomial : Polynomial := [g2LeftScalarContribution, g2CancellationTerm]

theorem g2_negative_product_coefficient : g2LeftScalarContribution.coefficient = -6 := by
  decide

theorem g2_duplicate_key_cancels :
    coefficient g2LeftScalarContribution.key g2MergedPolynomial = 0 := by
  decide

theorem g2_product_bound : g2LeftScalarContribution.coefficient.natAbs ≤ 2 * 3 := by
  exact operatorProductContribution_natAbs_le g2LeftTerm g2RightTerm true false 2 3
    (by decide) (by decide)

def g2RelationContext : MonomialContext :=
  { exteriorCentral := [5], prefixFactors := [31], suffixFactors := [41] }

def g2RelationSourceKey : MonomialKey :=
  { centralFactors := [13], orderedFactors := [51, 52] }

def g2RelationRhs : Polynomial :=
  [{ coefficient := -3, key := g2RelationSourceKey }]

def g2RelationReplacement : Polynomial :=
  relationReplacement g2RelationContext 2 g2RelationRhs

theorem g2_relation_context_preserves_order :
    g2RelationContext.plug g2RelationSourceKey =
      { centralFactors := [5, 13], orderedFactors := [31, 51, 52, 41] } := by
  decide

theorem g2_relation_replacement_preserves_context :
    g2RelationReplacement =
      [{ coefficient := -6,
         key := { centralFactors := [5, 13], orderedFactors := [31, 51, 52, 41] } }] := by
  decide

theorem g2_four_role_product_kernel :
    scalarProductKey g2LeftKey g2RightKey true false =
        { centralFactors := [3, 7, 11, 12], orderedFactors := [21, 22] } ∧
      scalarProductKey g2LeftKey g2RightKey false true =
        { centralFactors := [3, 7, 21, 22], orderedFactors := [11, 12] } ∧
      scalarProductKey g2LeftKey g2RightKey true true =
        { centralFactors := [3, 7], orderedFactors := [11, 12, 21, 22] } ∧
      scalarProductKey g2LeftKey g2RightKey false false =
        { centralFactors := [3, 7], orderedFactors := [11, 12, 21, 22] } ∧
      g2LeftScalarContribution.coefficient = -6 ∧
      coefficient g2LeftScalarContribution.key g2MergedPolynomial = 0 ∧
      g2LeftScalarContribution.coefficient.natAbs ≤ 2 * 3 ∧
      g2RelationContext.plug g2RelationSourceKey =
        { centralFactors := [5, 13], orderedFactors := [31, 51, 52, 41] } ∧
      g2RelationReplacement =
        [{ coefficient := -6,
           key := { centralFactors := [5, 13], orderedFactors := [31, 51, 52, 41] } }] := by
  exact ⟨g2_left_scalar_key, g2_right_scalar_key, g2_both_scalar_key,
    g2_neither_scalar_key, g2_negative_product_coefficient, g2_duplicate_key_cancels,
    g2_product_bound, g2_relation_context_preserves_order,
    g2_relation_replacement_preserves_context⟩

/-! A single owner-local algebra chain using the left-scalar product role. The replacement and
    product contribution meet at one contextualized key, cancel there, and leave one survivor
    whose resolved transfer is folded into the invocation-end bound. -/

def g2ReplayRightTerm : ExactTerm :=
  { coefficient := 3
    key := { centralFactors := [3], orderedFactors := [31, 51, 41] } }

def g2ReplayProduct : ExactTerm :=
  operatorProductContribution g2LeftTerm g2ReplayRightTerm true false

def g2ReplayContext : MonomialContext :=
  { exteriorCentral := [7, 11, 12], prefixFactors := [31], suffixFactors := [41] }

def g2ReplaySourceA : MonomialKey :=
  { centralFactors := [3], orderedFactors := [51] }

def g2ReplaySourceB : MonomialKey :=
  { centralFactors := [4], orderedFactors := [61] }

def g2ReplayReplacement : Polynomial :=
  relationReplacement g2ReplayContext 2
    [ { coefficient := 3, key := g2ReplaySourceA },
      { coefficient := -1, key := g2ReplaySourceB } ]

def g2ReplayMerged : Polynomial := add g2ReplayReplacement [g2ReplayProduct]
def g2ReplaySubtracted : Polynomial := subtract g2ReplayReplacement [g2ReplayProduct]

theorem g2_replay_composition :
    g2ReplayProduct =
        { coefficient := -6
          key := { centralFactors := [3, 7, 11, 12], orderedFactors := [31, 51, 41] } } ∧
      g2ReplayContext.plug g2ReplaySourceA = g2ReplayProduct.key ∧
      g2ReplayReplacement =
        [ { coefficient := 6
            key := { centralFactors := [3, 7, 11, 12], orderedFactors := [31, 51, 41] } },
          { coefficient := -2
            key := { centralFactors := [4, 7, 11, 12], orderedFactors := [31, 61, 41] } } ] ∧
      coefficient (g2ReplayContext.plug g2ReplaySourceA) g2ReplayMerged = 0 ∧
      coefficient (g2ReplayContext.plug g2ReplaySourceB) g2ReplayMerged = -2 ∧
      coefficient (g2ReplayContext.plug g2ReplaySourceA) g2ReplaySubtracted = 12 ∧
      g2ReplayProduct.coefficient.natAbs ≤ 2 * 3 ∧
      (coefficient (g2ReplayContext.plug g2ReplaySourceB) g2ReplayMerged).natAbs = 2 ∧
      [2 * 4].sum ≤ [2 * 5].sum ∧
      7 + [2 * 4].sum ≤ 9 + [2 * 5].sum := by
  have productBound : g2ReplayProduct.coefficient.natAbs ≤ 2 * 3 := by
    exact operatorProductContribution_natAbs_le g2LeftTerm g2ReplayRightTerm true false 2 3
      (by decide) (by decide)
  have addCancellation :
      coefficient (g2ReplayContext.plug g2ReplaySourceA) g2ReplayMerged = 0 := by
    unfold g2ReplayMerged
    rw [coefficient_add]
    decide
  have addSurvivor :
      coefficient (g2ReplayContext.plug g2ReplaySourceB) g2ReplayMerged = -2 := by
    unfold g2ReplayMerged
    rw [coefficient_add]
    decide
  have subtractContribution :
      coefficient (g2ReplayContext.plug g2ReplaySourceA) g2ReplaySubtracted = 12 := by
    unfold g2ReplaySubtracted
    rw [coefficient_subtract]
    decide
  have survivorTransfer : 2 * 4 ≤ 2 * 5 := boundTransfer_scale (by decide)
  have survivorTransfers :
      List.Forall₂ (fun value bound => value ≤ bound) [2 * 4] [2 * 5] :=
    .cons survivorTransfer .nil
  have survivorFold : [2 * 4].sum ≤ [2 * 5].sum :=
    survivorFold_sound survivorTransfers
  have invocationEnd : 7 + [2 * 4].sum ≤ 9 + [2 * 5].sum :=
    preFold_to_invocationEnd (by decide) survivorTransfers
  exact ⟨by decide, by decide, by decide, addCancellation, addSurvivor, subtractContribution,
    productBound, by rw [addSurvivor]; decide, survivorFold, invocationEnd⟩

def g2aFinite2 : CoeffClass := .finite ⟨2, by decide⟩
def g2aFinite3 : CoeffClass := .finite ⟨3, by decide⟩
def g2aFinite5 : CoeffClass := .finite ⟨5, by decide⟩
def g2aFinite6 : CoeffClass := .finite ⟨6, by decide⟩
def g2aFinite8 : CoeffClass := .finite ⟨8, by decide⟩
def g2aFinite12 : CoeffClass := .finite ⟨12, by decide⟩
def g2aFinite24 : CoeffClass := .finite ⟨24, by decide⟩
def g2aFinite30 : CoeffClass := .finite ⟨30, by decide⟩

def g2aNoProductFacts : ProductFacts :=
  { leftConstantPolynomial := false
    rightConstantPolynomial := false
    rightKnownZeroRows := none
    leftSupportUpper := none
    rightSupportUpper := none }

def g2aLeftSupportFacts : ProductFacts :=
  { g2aNoProductFacts with leftSupportUpper := some 2 }

def g2aLeftConstantFacts : ProductFacts :=
  { g2aNoProductFacts with leftConstantPolynomial := true }

def g2aRightConstantFacts : ProductFacts :=
  { g2aNoProductFacts with rightConstantPolynomial := true }

def g2aZeroRowFacts : ProductFacts :=
  { g2aNoProductFacts with rightKnownZeroRows := some 1 }

def g2aInvalidSupportFacts : ProductFacts :=
  { g2aNoProductFacts with leftSupportUpper := some 5 }

def g2aInvalidZeroRowFacts : ProductFacts :=
  { g2aNoProductFacts with rightKnownZeroRows := some 4 }

theorem g2a_bound_replay :
    addKnown g2aFinite2 g2aFinite3 = g2aFinite5 ∧
      maxKnown g2aFinite2 g2aFinite3 = g2aFinite3 ∧
      scaleMagnitude 4 g2aFinite2 = g2aFinite8 ∧
      scaleValue g2aFinite2 g2aFinite3 = g2aFinite6 ∧
      productWithFactor 7 .exactZero g2aFinite3 = .exactZero ∧
      productWithFactor 7 .exactZero .large = .exactZero ∧
      productWithFactor 7 .large .exactZero = .exactZero ∧
      productWithFactor 7 .large g2aFinite3 = .large ∧
      productFactor 1 1 1 1 4 g2aNoProductFacts = some 4 ∧
      productFactor 1 1 2 3 4 g2aNoProductFacts = some 4 ∧
      productFactor 2 3 1 1 4 g2aNoProductFacts = some 4 ∧
      productFactor 2 3 3 4 4 g2aNoProductFacts = some 12 ∧
      productFactor 1 1 2 3 4 g2aLeftSupportFacts = some 2 ∧
      productFactor 2 3 3 4 4 g2aLeftConstantFacts = some 3 ∧
      productFactor 2 3 3 4 4 g2aRightConstantFacts = some 3 ∧
      productFactor 2 3 3 4 4 g2aZeroRowFacts = some 8 ∧
      productFactor 2 3 3 4 4 g2aInvalidSupportFacts = none ∧
      productFactor 2 3 3 4 4 g2aInvalidZeroRowFacts = none ∧
      productFactor 2 2 3 4 4 g2aNoProductFacts = none ∧
      productFactor 2 3 3 4 0 g2aNoProductFacts = none ∧
      tensorFactor 4 g2aNoProductFacts = 4 ∧
      tensorFactor 4 g2aLeftConstantFacts = 1 ∧
      tensorFactor 4 g2aRightConstantFacts = 1 ∧
      productNonempty g2aFinite2 [g2aFinite3] = g2aFinite6 ∧
      productNonempty g2aFinite2 [g2aFinite3, g2aFinite5] = g2aFinite30 ∧
      scaleMagnitude 4 (productNonempty g2aFinite2 [g2aFinite3]) = g2aFinite24 := by
  decide

#print axioms g2_four_role_product_kernel
#print axioms operatorProductContribution_natAbs_le
#print axioms g2_replay_composition
#print axioms addKnown_sound
#print axioms maxKnown_sound
#print axioms productWithFactor_sound
#print axioms scaleMagnitude_sound
#print axioms scaleValue_sound
#print axioms productNonempty_sound
#print axioms productWithFacts_sound
#print axioms tensorWithFacts_sound
#print axioms g2a_bound_replay

end Mxx.Certificate.OperationalNoise.EventReplay
