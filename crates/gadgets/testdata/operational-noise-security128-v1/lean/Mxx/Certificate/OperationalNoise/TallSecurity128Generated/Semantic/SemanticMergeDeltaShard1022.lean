import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge167110
def owner : Owner := ⟨.program ⟨257⟩, ⟨27964⟩⟩
def mergeEvent : Nat := 167110
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27433⟩⟩] } }
def rhsRaw : List Term := Proof.Events652.exact167035RawTerms
def group : MergeGroup := .relation 167109
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 167109) (rhsResult := 167035)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27963⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27963⟩⟩) ⟨27433⟩ 167035) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27433⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], [⟨.program ⟨257⟩, ⟨27433⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge167110

namespace LeftMerge167111
def owner : Owner := ⟨.program ⟨257⟩, ⟨27964⟩⟩
def mergeEvent : Nat := 167111
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27963⟩⟩] } }
def leftRaw : List Term := Proof.Events652.exact167102RawTerms
def rightRaw : List Term := Proof.Events652.exact167038RawTerms
def group : MergeGroup := .operator 167102 167038
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167102) (leftOrdinal := 0)
    (rightResult := 167038) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27963⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27963⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge167111

namespace LeftMerge167125
def owner : Owner := ⟨.program ⟨257⟩, ⟨26892⟩⟩
def mergeEvent : Nat := 167125
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26889⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events652.exact167119RawTerms
def group : MergeGroup := .operator 163745 167119
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 167119) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨26889⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26889⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge167125

namespace LeftMerge167204
def owner : Owner := ⟨.program ⟨257⟩, ⟨26191⟩⟩
def mergeEvent : Nat := 167204
def frameStart : Nat := 167174
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events653.exact167200RawTerms
def rightRaw : List Term := Proof.Events653.exact167197RawTerms
def group : MergeGroup := .operator 167200 167197
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167200) (leftOrdinal := 0)
    (rightResult := 167197) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13041⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26190⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge167204

namespace LeftMerge167234
def owner : Owner := ⟨.program ⟨257⟩, ⟨27704⟩⟩
def mergeEvent : Nat := 167234
def frameStart : Nat := 167174
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events653.exact167230RawTerms
def rightRaw : List Term := Proof.Events653.exact167228RawTerms
def group : MergeGroup := .operator 167230 167228
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167230) (leftOrdinal := 0)
    (rightResult := 167228) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge167234

namespace LeftMerge167257
def owner : Owner := ⟨.program ⟨257⟩, ⟨9546⟩⟩
def mergeEvent : Nat := 167257
def frameStart : Nat := 167174
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }
def leftRaw : List Term := Proof.Events653.exact167253RawTerms
def rightRaw : List Term := Proof.Events653.exact167250RawTerms
def group : MergeGroup := .operator 167253 167250
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167253) (leftOrdinal := 0)
    (rightResult := 167250) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9544⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge167257

namespace LeftMerge167266
def owner : Owner := ⟨.program ⟨257⟩, ⟨27966⟩⟩
def mergeEvent : Nat := 167266
def frameStart : Nat := 167174
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27963⟩⟩] } }
def leftRaw : List Term := Proof.Events653.exact167262RawTerms
def rightRaw : List Term := Proof.Events653.exact167219RawTerms
def group : MergeGroup := .operator 167262 167219
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167262) (leftOrdinal := 0)
    (rightResult := 167219) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27963⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27963⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge167266

namespace LeftMerge167267
def owner : Owner := ⟨.program ⟨257⟩, ⟨27966⟩⟩
def mergeEvent : Nat := 167267
def frameStart : Nat := 167174
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27963⟩⟩] } }
def leftRaw : List Term := Proof.Events653.exact167262RawTerms
def rightRaw : List Term := Proof.Events653.exact167219RawTerms
def group : MergeGroup := .operator 167262 167219
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167262) (leftOrdinal := 1)
    (rightResult := 167219) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27963⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27963⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge167267

namespace LeftMerge167269
def owner : Owner := ⟨.program ⟨257⟩, ⟨27966⟩⟩
def mergeEvent : Nat := 167269
def frameStart : Nat := 167174
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27433⟩⟩] } }
def rhsRaw : List Term := Proof.Events653.exact167216RawTerms
def group : MergeGroup := .relation 167268
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 167268) (rhsResult := 167216)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27963⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27963⟩⟩) ⟨27433⟩ 167216) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27433⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], [⟨.program ⟨257⟩, ⟨27433⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge167269

namespace LeftMerge167277
def owner : Owner := ⟨.program ⟨257⟩, ⟨26442⟩⟩
def mergeEvent : Nat := 167277
def frameStart : Nat := 167174
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26440⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events653.exact167230RawTerms
def rightRaw : List Term := Proof.Events653.exact167273RawTerms
def group : MergeGroup := .operator 167230 167273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167230) (leftOrdinal := 0)
    (rightResult := 167273) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26440⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26440⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge167277

namespace LeftMerge167294
def owner : Owner := ⟨.program ⟨257⟩, ⟨26892⟩⟩
def mergeEvent : Nat := 167294
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }
def rhsRaw : List Term := Proof.Events653.exact167291RawTerms
def group : MergeGroup := .relation 167293
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 167293) (rhsResult := 167291)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26889⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 167292 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26889⟩⟩]⟩) (none) 167291) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge167294

namespace LeftMerge167295
def owner : Owner := ⟨.program ⟨257⟩, ⟨26892⟩⟩
def mergeEvent : Nat := 167295
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27963⟩⟩] } }
def rhsRaw : List Term := Proof.Events653.exact167291RawTerms
def group : MergeGroup := .relation 167293
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 167293) (rhsResult := 167291)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26889⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 167292 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26889⟩⟩]⟩) (none) 167291) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27963⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27963⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge167295

namespace LeftMerge167296
def owner : Owner := ⟨.program ⟨257⟩, ⟨26892⟩⟩
def mergeEvent : Nat := 167296
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27433⟩⟩] } }
def rhsRaw : List Term := Proof.Events653.exact167291RawTerms
def group : MergeGroup := .relation 167293
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 167293) (rhsResult := 167291)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26889⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 167292 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26889⟩⟩]⟩) (none) 167291) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27433⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], [⟨.program ⟨257⟩, ⟨27433⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge167296

namespace LeftMerge167297
def owner : Owner := ⟨.program ⟨257⟩, ⟨26892⟩⟩
def mergeEvent : Nat := 167297
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨26440⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events653.exact167291RawTerms
def group : MergeGroup := .relation 167293
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 167293) (rhsResult := 167291)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26889⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 167292 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26889⟩⟩]⟩) (none) 167291) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26440⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨26440⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge167297

namespace LeftMerge167302
def owner : Owner := ⟨.program ⟨257⟩, ⟨27965⟩⟩
def mergeEvent : Nat := 167302
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27433⟩⟩] } }
def leftRaw : List Term := Proof.Events653.exact167298RawTerms
def rightRaw : List Term := Proof.Events652.exact167112RawTerms
def group : MergeGroup := .operator 167298 167112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167298) (leftOrdinal := 2)
    (rightResult := 167112) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27433⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27433⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13041⟩⟩, ⟨.program ⟨257⟩, ⟨26190⟩⟩], [⟨.program ⟨257⟩, ⟨27433⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge167302

namespace LeftMerge167303
def owner : Owner := ⟨.program ⟨257⟩, ⟨27965⟩⟩
def mergeEvent : Nat := 167303
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27963⟩⟩] } }
def leftRaw : List Term := Proof.Events653.exact167298RawTerms
def rightRaw : List Term := Proof.Events652.exact167112RawTerms
def group : MergeGroup := .operator 167298 167112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167298) (leftOrdinal := 1)
    (rightResult := 167112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27963⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27963⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27963⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge167303

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
