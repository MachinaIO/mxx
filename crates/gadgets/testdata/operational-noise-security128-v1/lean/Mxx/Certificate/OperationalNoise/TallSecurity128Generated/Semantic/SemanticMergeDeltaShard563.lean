import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge93986
def owner : Owner := ⟨.program ⟨257⟩, ⟨27975⟩⟩
def mergeEvent : Nat := 93986
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27974⟩⟩] } }
def leftRaw : List Term := Proof.Events367.exact93977RawTerms
def rightRaw : List Term := Proof.Events366.exact93913RawTerms
def group : MergeGroup := .operator 93977 93913
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93977) (leftOrdinal := 0)
    (rightResult := 93913) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27974⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27974⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93986

namespace LeftMerge94000
def owner : Owner := ⟨.program ⟨257⟩, ⟨26902⟩⟩
def mergeEvent : Nat := 94000
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26899⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90620RawTerms
def rightRaw : List Term := Proof.Events367.exact93994RawTerms
def group : MergeGroup := .operator 90620 93994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90620) (leftOrdinal := 0)
    (rightResult := 93994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨26899⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26899⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94000

namespace LeftMerge94079
def owner : Owner := ⟨.program ⟨257⟩, ⟨26215⟩⟩
def mergeEvent : Nat := 94079
def frameStart : Nat := 94049
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13056⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events367.exact94075RawTerms
def rightRaw : List Term := Proof.Events367.exact94072RawTerms
def group : MergeGroup := .operator 94075 94072
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94075) (leftOrdinal := 0)
    (rightResult := 94072) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13056⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26214⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13056⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94079

namespace LeftMerge94109
def owner : Owner := ⟨.program ⟨257⟩, ⟨27708⟩⟩
def mergeEvent : Nat := 94109
def frameStart : Nat := 94049
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13056⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events367.exact94105RawTerms
def rightRaw : List Term := Proof.Events367.exact94103RawTerms
def group : MergeGroup := .operator 94105 94103
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94105) (leftOrdinal := 0)
    (rightResult := 94103) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13056⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13056⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94109

namespace LeftMerge94132
def owner : Owner := ⟨.program ⟨257⟩, ⟨9546⟩⟩
def mergeEvent : Nat := 94132
def frameStart : Nat := 94049
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }
def leftRaw : List Term := Proof.Events367.exact94128RawTerms
def rightRaw : List Term := Proof.Events367.exact94125RawTerms
def group : MergeGroup := .operator 94128 94125
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94128) (leftOrdinal := 0)
    (rightResult := 94125) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9544⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94132

namespace LeftMerge94141
def owner : Owner := ⟨.program ⟨257⟩, ⟨27977⟩⟩
def mergeEvent : Nat := 94141
def frameStart : Nat := 94049
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27974⟩⟩] } }
def leftRaw : List Term := Proof.Events367.exact94137RawTerms
def rightRaw : List Term := Proof.Events367.exact94094RawTerms
def group : MergeGroup := .operator 94137 94094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94137) (leftOrdinal := 0)
    (rightResult := 94094) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27974⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27974⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94141

namespace LeftMerge94142
def owner : Owner := ⟨.program ⟨257⟩, ⟨27977⟩⟩
def mergeEvent : Nat := 94142
def frameStart : Nat := 94049
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13056⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27974⟩⟩] } }
def leftRaw : List Term := Proof.Events367.exact94137RawTerms
def rightRaw : List Term := Proof.Events367.exact94094RawTerms
def group : MergeGroup := .operator 94137 94094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94137) (leftOrdinal := 1)
    (rightResult := 94094) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13056⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27974⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13056⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27974⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94142

namespace LeftMerge94144
def owner : Owner := ⟨.program ⟨257⟩, ⟨27977⟩⟩
def mergeEvent : Nat := 94144
def frameStart : Nat := 94049
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13056⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27439⟩⟩] } }
def rhsRaw : List Term := Proof.Events367.exact94091RawTerms
def group : MergeGroup := .relation 94143
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 94143) (rhsResult := 94091)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13056⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27974⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27974⟩⟩) ⟨27439⟩ 94091) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27439⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13056⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], [⟨.program ⟨257⟩, ⟨27439⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94144

namespace LeftMerge94152
def owner : Owner := ⟨.program ⟨257⟩, ⟨26450⟩⟩
def mergeEvent : Nat := 94152
def frameStart : Nat := 94049
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26448⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events367.exact94105RawTerms
def rightRaw : List Term := Proof.Events367.exact94148RawTerms
def group : MergeGroup := .operator 94105 94148
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94105) (leftOrdinal := 0)
    (rightResult := 94148) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26448⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26448⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94152

namespace LeftMerge94169
def owner : Owner := ⟨.program ⟨257⟩, ⟨26902⟩⟩
def mergeEvent : Nat := 94169
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }
def rhsRaw : List Term := Proof.Events367.exact94166RawTerms
def group : MergeGroup := .relation 94168
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 94168) (rhsResult := 94166)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26899⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 94167 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26899⟩⟩]⟩) (none) 94166) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94169

namespace LeftMerge94170
def owner : Owner := ⟨.program ⟨257⟩, ⟨26902⟩⟩
def mergeEvent : Nat := 94170
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27974⟩⟩] } }
def rhsRaw : List Term := Proof.Events367.exact94166RawTerms
def group : MergeGroup := .relation 94168
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 94168) (rhsResult := 94166)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26899⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 94167 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26899⟩⟩]⟩) (none) 94166) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27974⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27974⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94170

namespace LeftMerge94171
def owner : Owner := ⟨.program ⟨257⟩, ⟨26902⟩⟩
def mergeEvent : Nat := 94171
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13056⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27439⟩⟩] } }
def rhsRaw : List Term := Proof.Events367.exact94166RawTerms
def group : MergeGroup := .relation 94168
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 94168) (rhsResult := 94166)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26899⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 94167 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26899⟩⟩]⟩) (none) 94166) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13056⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27439⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13056⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], [⟨.program ⟨257⟩, ⟨27439⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94171

namespace LeftMerge94172
def owner : Owner := ⟨.program ⟨257⟩, ⟨26902⟩⟩
def mergeEvent : Nat := 94172
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨26448⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events367.exact94166RawTerms
def group : MergeGroup := .relation 94168
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 94168) (rhsResult := 94166)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26899⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 94167 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26899⟩⟩]⟩) (none) 94166) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26448⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨26448⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94172

namespace LeftMerge94177
def owner : Owner := ⟨.program ⟨257⟩, ⟨27976⟩⟩
def mergeEvent : Nat := 94177
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13056⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27439⟩⟩] } }
def leftRaw : List Term := Proof.Events367.exact94173RawTerms
def rightRaw : List Term := Proof.Events367.exact93987RawTerms
def group : MergeGroup := .operator 94173 93987
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94173) (leftOrdinal := 2)
    (rightResult := 93987) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13056⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27439⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13056⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27439⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13056⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], [⟨.program ⟨257⟩, ⟨27439⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94177

namespace LeftMerge94178
def owner : Owner := ⟨.program ⟨257⟩, ⟨27976⟩⟩
def mergeEvent : Nat := 94178
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27974⟩⟩] } }
def leftRaw : List Term := Proof.Events367.exact94173RawTerms
def rightRaw : List Term := Proof.Events367.exact93987RawTerms
def group : MergeGroup := .operator 94173 93987
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94173) (leftOrdinal := 1)
    (rightResult := 93987) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27974⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27974⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27974⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94178

namespace LeftMerge94186
def owner : Owner := ⟨.program ⟨257⟩, ⟨28416⟩⟩
def mergeEvent : Nat := 94186
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28414⟩⟩] } }
def leftRaw : List Term := Proof.Events367.exact94180RawTerms
def rightRaw : List Term := Proof.Events366.exact93903RawTerms
def group : MergeGroup := .operator 94180 93903
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94180) (leftOrdinal := 0)
    (rightResult := 93903) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28414⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28414⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94186

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
