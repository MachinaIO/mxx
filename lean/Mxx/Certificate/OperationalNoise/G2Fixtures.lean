import Mxx.Certificate.OperationalNoise.Core

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

#print axioms g2_four_role_product_kernel
#print axioms operatorProductContribution_natAbs_le

end Mxx.Certificate.OperationalNoise.EventReplay
