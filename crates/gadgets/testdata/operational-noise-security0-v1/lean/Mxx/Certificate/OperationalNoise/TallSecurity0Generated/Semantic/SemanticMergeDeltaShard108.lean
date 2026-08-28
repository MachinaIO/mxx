import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge19646
def owner : Owner := ⟨.program ⟨214⟩, ⟨27479⟩⟩
def mergeEvent : Nat := 19646
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩] } }
def leftRaw : List Term := Proof.Events049.exact12759RawTerms
def rightRaw : List Term := Proof.Events076.exact19640RawTerms
def group : MergeGroup := .operator 12759 19640
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12759) (leftOrdinal := 1)
    (rightResult := 19640) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27477⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15718⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19646

namespace LeftMerge19648
def owner : Owner := ⟨.program ⟨214⟩, ⟨27479⟩⟩
def mergeEvent : Nat := 19648
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24047⟩⟩] } }
def rhsRaw : List Term := Proof.Events076.exact19637RawTerms
def group : MergeGroup := .relation 19647
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 19647) (rhsResult := 19637)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15718⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27477⟩⟩) ⟨24047⟩ 19637) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24047⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15718⟩⟩], [⟨.program ⟨214⟩, ⟨24047⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19648

namespace LeftMerge19649
def owner : Owner := ⟨.program ⟨214⟩, ⟨27479⟩⟩
def mergeEvent : Nat := 19649
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩] } }
def leftRaw : List Term := Proof.Events049.exact12759RawTerms
def rightRaw : List Term := Proof.Events076.exact19640RawTerms
def group : MergeGroup := .operator 12759 19640
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12759) (leftOrdinal := 0)
    (rightResult := 19640) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27477⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19649

namespace LeftMerge19663
def owner : Owner := ⟨.program ⟨214⟩, ⟨21059⟩⟩
def mergeEvent : Nat := 19663
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21056⟩⟩] } }
def leftRaw : List Term := Proof.Events025.exact6561RawTerms
def rightRaw : List Term := Proof.Events076.exact19657RawTerms
def group : MergeGroup := .operator 6561 19657
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6561) (leftOrdinal := 0)
    (rightResult := 19657) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21056⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21056⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19663

namespace LeftMerge19784
def owner : Owner := ⟨.program ⟨214⟩, ⟨15795⟩⟩
def mergeEvent : Nat := 19784
def frameStart : Nat := 19718
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events077.exact19780RawTerms
def rightRaw : List Term := Proof.Events077.exact19778RawTerms
def group : MergeGroup := .operator 19780 19778
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19780) (leftOrdinal := 0)
    (rightResult := 19778) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15718⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15718⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19784

namespace LeftMerge19796
def owner : Owner := ⟨.program ⟨214⟩, ⟨27478⟩⟩
def mergeEvent : Nat := 19796
def frameStart : Nat := 19718
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩] } }
def leftRaw : List Term := Proof.Events077.exact19792RawTerms
def rightRaw : List Term := Proof.Events077.exact19769RawTerms
def group : MergeGroup := .operator 19792 19769
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19792) (leftOrdinal := 1)
    (rightResult := 19769) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27477⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15718⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19796

namespace LeftMerge19798
def owner : Owner := ⟨.program ⟨214⟩, ⟨27478⟩⟩
def mergeEvent : Nat := 19798
def frameStart : Nat := 19718
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24047⟩⟩] } }
def rhsRaw : List Term := Proof.Events077.exact19766RawTerms
def group : MergeGroup := .relation 19797
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 19797) (rhsResult := 19766)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15718⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27477⟩⟩) ⟨24047⟩ 19766) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24047⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15718⟩⟩], [⟨.program ⟨214⟩, ⟨24047⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19798

namespace LeftMerge19799
def owner : Owner := ⟨.program ⟨214⟩, ⟨27478⟩⟩
def mergeEvent : Nat := 19799
def frameStart : Nat := 19718
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩] } }
def leftRaw : List Term := Proof.Events077.exact19792RawTerms
def rightRaw : List Term := Proof.Events077.exact19769RawTerms
def group : MergeGroup := .operator 19792 19769
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19792) (leftOrdinal := 0)
    (rightResult := 19769) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27477⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19799

namespace LeftMerge19807
def owner : Owner := ⟨.program ⟨214⟩, ⟨17456⟩⟩
def mergeEvent : Nat := 19807
def frameStart : Nat := 19718
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events077.exact19780RawTerms
def rightRaw : List Term := Proof.Events077.exact19803RawTerms
def group : MergeGroup := .operator 19780 19803
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19780) (leftOrdinal := 0)
    (rightResult := 19803) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17454⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19807

namespace LeftMerge19824
def owner : Owner := ⟨.program ⟨214⟩, ⟨21059⟩⟩
def mergeEvent : Nat := 19824
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩] } }
def rhsRaw : List Term := Proof.Events077.exact19821RawTerms
def group : MergeGroup := .relation 19823
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 19823) (rhsResult := 19821)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21056⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 19822 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21056⟩⟩]⟩) (none) 19821) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19824

namespace LeftMerge19825
def owner : Owner := ⟨.program ⟨214⟩, ⟨21059⟩⟩
def mergeEvent : Nat := 19825
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24047⟩⟩] } }
def rhsRaw : List Term := Proof.Events077.exact19821RawTerms
def group : MergeGroup := .relation 19823
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 19823) (rhsResult := 19821)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21056⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 19822 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21056⟩⟩]⟩) (none) 19821) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24047⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15718⟩⟩], [⟨.program ⟨214⟩, ⟨24047⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19825

namespace LeftMerge19826
def owner : Owner := ⟨.program ⟨214⟩, ⟨21059⟩⟩
def mergeEvent : Nat := 19826
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩] } }
def rhsRaw : List Term := Proof.Events077.exact19821RawTerms
def group : MergeGroup := .relation 19823
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 19823) (rhsResult := 19821)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21056⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 19822 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21056⟩⟩]⟩) (none) 19821) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19826

namespace LeftMerge19827
def owner : Owner := ⟨.program ⟨214⟩, ⟨21059⟩⟩
def mergeEvent : Nat := 19827
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events077.exact19821RawTerms
def group : MergeGroup := .relation 19823
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 19823) (rhsResult := 19821)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21056⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 19822 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21056⟩⟩]⟩) (none) 19821) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19827

namespace LeftMerge19832
def owner : Owner := ⟨.program ⟨214⟩, ⟨27480⟩⟩
def mergeEvent : Nat := 19832
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24047⟩⟩] } }
def leftRaw : List Term := Proof.Events077.exact19828RawTerms
def rightRaw : List Term := Proof.Events076.exact19650RawTerms
def group : MergeGroup := .operator 19828 19650
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19828) (leftOrdinal := 2)
    (rightResult := 19650) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24047⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24047⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15718⟩⟩], [⟨.program ⟨214⟩, ⟨24047⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19832

namespace LeftMerge19833
def owner : Owner := ⟨.program ⟨214⟩, ⟨27480⟩⟩
def mergeEvent : Nat := 19833
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩] } }
def leftRaw : List Term := Proof.Events077.exact19828RawTerms
def rightRaw : List Term := Proof.Events076.exact19650RawTerms
def group : MergeGroup := .operator 19828 19650
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19828) (leftOrdinal := 0)
    (rightResult := 19650) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19833

namespace LeftMerge19841
def owner : Owner := ⟨.program ⟨214⟩, ⟨27481⟩⟩
def mergeEvent : Nat := 19841
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩] } }
def leftRaw : List Term := Proof.Events077.exact19835RawTerms
def rightRaw : List Term := Proof.Events022.exact5759RawTerms
def group : MergeGroup := .operator 19835 5759
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19835) (leftOrdinal := 0)
    (rightResult := 5759) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6647⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19841

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
