import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge254875
def owner : Owner := ⟨.program ⟨257⟩, ⟨26802⟩⟩
def mergeEvent : Nat := 254875
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26799⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events995.exact254869RawTerms
def group : MergeGroup := .operator 251495 254869
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 254869) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨26799⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26799⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254875

namespace LeftMerge254954
def owner : Owner := ⟨.program ⟨257⟩, ⟨25975⟩⟩
def mergeEvent : Nat := 254954
def frameStart : Nat := 254924
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12906⟩⟩, ⟨.program ⟨257⟩, ⟨25974⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events995.exact254950RawTerms
def rightRaw : List Term := Proof.Events995.exact254947RawTerms
def group : MergeGroup := .operator 254950 254947
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 254950) (leftOrdinal := 0)
    (rightResult := 254947) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12906⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25974⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12906⟩⟩, ⟨.program ⟨257⟩, ⟨25974⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254954

namespace LeftMerge254984
def owner : Owner := ⟨.program ⟨257⟩, ⟨27668⟩⟩
def mergeEvent : Nat := 254984
def frameStart : Nat := 254924
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12906⟩⟩, ⟨.program ⟨257⟩, ⟨25974⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events996.exact254980RawTerms
def rightRaw : List Term := Proof.Events996.exact254978RawTerms
def group : MergeGroup := .operator 254980 254978
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 254980) (leftOrdinal := 0)
    (rightResult := 254978) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12906⟩⟩, ⟨.program ⟨257⟩, ⟨25974⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12906⟩⟩, ⟨.program ⟨257⟩, ⟨25974⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254984

namespace LeftMerge255007
def owner : Owner := ⟨.program ⟨257⟩, ⟨9546⟩⟩
def mergeEvent : Nat := 255007
def frameStart : Nat := 254924
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }
def leftRaw : List Term := Proof.Events996.exact255003RawTerms
def rightRaw : List Term := Proof.Events996.exact255000RawTerms
def group : MergeGroup := .operator 255003 255000
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255003) (leftOrdinal := 0)
    (rightResult := 255000) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9544⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255007

namespace LeftMerge255016
def owner : Owner := ⟨.program ⟨257⟩, ⟨27867⟩⟩
def mergeEvent : Nat := 255016
def frameStart : Nat := 254924
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27864⟩⟩] } }
def leftRaw : List Term := Proof.Events996.exact255012RawTerms
def rightRaw : List Term := Proof.Events995.exact254969RawTerms
def group : MergeGroup := .operator 255012 254969
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255012) (leftOrdinal := 0)
    (rightResult := 254969) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27864⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27864⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255016

namespace LeftMerge255017
def owner : Owner := ⟨.program ⟨257⟩, ⟨27867⟩⟩
def mergeEvent : Nat := 255017
def frameStart : Nat := 254924
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12906⟩⟩, ⟨.program ⟨257⟩, ⟨25974⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27864⟩⟩] } }
def leftRaw : List Term := Proof.Events996.exact255012RawTerms
def rightRaw : List Term := Proof.Events995.exact254969RawTerms
def group : MergeGroup := .operator 255012 254969
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255012) (leftOrdinal := 1)
    (rightResult := 254969) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12906⟩⟩, ⟨.program ⟨257⟩, ⟨25974⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27864⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12906⟩⟩, ⟨.program ⟨257⟩, ⟨25974⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27864⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge255017

namespace LeftMerge255019
def owner : Owner := ⟨.program ⟨257⟩, ⟨27867⟩⟩
def mergeEvent : Nat := 255019
def frameStart : Nat := 254924
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12906⟩⟩, ⟨.program ⟨257⟩, ⟨25974⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27379⟩⟩] } }
def rhsRaw : List Term := Proof.Events995.exact254966RawTerms
def group : MergeGroup := .relation 255018
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 255018) (rhsResult := 254966)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12906⟩⟩, ⟨.program ⟨257⟩, ⟨25974⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27864⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27864⟩⟩) ⟨27379⟩ 254966) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27379⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12906⟩⟩, ⟨.program ⟨257⟩, ⟨25974⟩⟩], [⟨.program ⟨257⟩, ⟨27379⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge255019

namespace LeftMerge255027
def owner : Owner := ⟨.program ⟨257⟩, ⟨26370⟩⟩
def mergeEvent : Nat := 255027
def frameStart : Nat := 254924
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26368⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events996.exact254980RawTerms
def rightRaw : List Term := Proof.Events996.exact255023RawTerms
def group : MergeGroup := .operator 254980 255023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 254980) (leftOrdinal := 0)
    (rightResult := 255023) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26368⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26368⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255027

namespace LeftMerge255044
def owner : Owner := ⟨.program ⟨257⟩, ⟨26802⟩⟩
def mergeEvent : Nat := 255044
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }
def rhsRaw : List Term := Proof.Events996.exact255041RawTerms
def group : MergeGroup := .relation 255043
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 255043) (rhsResult := 255041)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26799⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 255042 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26799⟩⟩]⟩) (none) 255041) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255044

namespace LeftMerge255045
def owner : Owner := ⟨.program ⟨257⟩, ⟨26802⟩⟩
def mergeEvent : Nat := 255045
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27864⟩⟩] } }
def rhsRaw : List Term := Proof.Events996.exact255041RawTerms
def group : MergeGroup := .relation 255043
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 255043) (rhsResult := 255041)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26799⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 255042 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26799⟩⟩]⟩) (none) 255041) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27864⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27864⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge255045

namespace LeftMerge255046
def owner : Owner := ⟨.program ⟨257⟩, ⟨26802⟩⟩
def mergeEvent : Nat := 255046
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12906⟩⟩, ⟨.program ⟨257⟩, ⟨25974⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27379⟩⟩] } }
def rhsRaw : List Term := Proof.Events996.exact255041RawTerms
def group : MergeGroup := .relation 255043
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 255043) (rhsResult := 255041)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26799⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 255042 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26799⟩⟩]⟩) (none) 255041) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12906⟩⟩, ⟨.program ⟨257⟩, ⟨25974⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27379⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12906⟩⟩, ⟨.program ⟨257⟩, ⟨25974⟩⟩], [⟨.program ⟨257⟩, ⟨27379⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255046

namespace LeftMerge255047
def owner : Owner := ⟨.program ⟨257⟩, ⟨26802⟩⟩
def mergeEvent : Nat := 255047
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨26368⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events996.exact255041RawTerms
def group : MergeGroup := .relation 255043
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 255043) (rhsResult := 255041)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26799⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 255042 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26799⟩⟩]⟩) (none) 255041) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26368⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨26368⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge255047

namespace LeftMerge255052
def owner : Owner := ⟨.program ⟨257⟩, ⟨27866⟩⟩
def mergeEvent : Nat := 255052
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12906⟩⟩, ⟨.program ⟨257⟩, ⟨25974⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27379⟩⟩] } }
def leftRaw : List Term := Proof.Events996.exact255048RawTerms
def rightRaw : List Term := Proof.Events995.exact254862RawTerms
def group : MergeGroup := .operator 255048 254862
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255048) (leftOrdinal := 2)
    (rightResult := 254862) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12906⟩⟩, ⟨.program ⟨257⟩, ⟨25974⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27379⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12906⟩⟩, ⟨.program ⟨257⟩, ⟨25974⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27379⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12906⟩⟩, ⟨.program ⟨257⟩, ⟨25974⟩⟩], [⟨.program ⟨257⟩, ⟨27379⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge255052

namespace LeftMerge255053
def owner : Owner := ⟨.program ⟨257⟩, ⟨27866⟩⟩
def mergeEvent : Nat := 255053
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27864⟩⟩] } }
def leftRaw : List Term := Proof.Events996.exact255048RawTerms
def rightRaw : List Term := Proof.Events995.exact254862RawTerms
def group : MergeGroup := .operator 255048 254862
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255048) (leftOrdinal := 1)
    (rightResult := 254862) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27864⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27864⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27864⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255053

namespace LeftMerge255061
def owner : Owner := ⟨.program ⟨257⟩, ⟨28166⟩⟩
def mergeEvent : Nat := 255061
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28164⟩⟩] } }
def leftRaw : List Term := Proof.Events996.exact255055RawTerms
def rightRaw : List Term := Proof.Events995.exact254778RawTerms
def group : MergeGroup := .operator 255055 254778
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255055) (leftOrdinal := 0)
    (rightResult := 254778) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28164⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28164⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255061

namespace LeftMerge255062
def owner : Owner := ⟨.program ⟨257⟩, ⟨28166⟩⟩
def mergeEvent : Nat := 255062
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨26368⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28164⟩⟩] } }
def leftRaw : List Term := Proof.Events996.exact255055RawTerms
def rightRaw : List Term := Proof.Events995.exact254778RawTerms
def group : MergeGroup := .operator 255055 254778
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255055) (leftOrdinal := 1)
    (rightResult := 254778) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨26368⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28164⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨26368⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28164⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge255062

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
