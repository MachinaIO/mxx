import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge98528
def owner : Owner := ⟨.program ⟨214⟩, ⟨26132⟩⟩
def mergeEvent : Nat := 98528
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26130⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98523RawTerms
def rightRaw : List Term := Proof.Events384.exact98361RawTerms
def group : MergeGroup := .operator 98523 98361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98523) (leftOrdinal := 1)
    (rightResult := 98361) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26130⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26130⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26130⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98528

namespace LeftMerge98536
def owner : Owner := ⟨.program ⟨214⟩, ⟨28050⟩⟩
def mergeEvent : Nat := 98536
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28048⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98530RawTerms
def rightRaw : List Term := Proof.Events383.exact98277RawTerms
def group : MergeGroup := .operator 98530 98277
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98530) (leftOrdinal := 0)
    (rightResult := 98277) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28048⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98536

namespace LeftMerge98537
def owner : Owner := ⟨.program ⟨214⟩, ⟨28050⟩⟩
def mergeEvent : Nat := 98537
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28048⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98530RawTerms
def rightRaw : List Term := Proof.Events383.exact98277RawTerms
def group : MergeGroup := .operator 98530 98277
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98530) (leftOrdinal := 1)
    (rightResult := 98277) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28048⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98537

namespace LeftMerge98539
def owner : Owner := ⟨.program ⟨214⟩, ⟨28050⟩⟩
def mergeEvent : Nat := 98539
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24216⟩⟩] } }
def rhsRaw : List Term := Proof.Events383.exact98274RawTerms
def group : MergeGroup := .relation 98538
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98538) (rhsResult := 98274)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28048⟩⟩) ⟨24216⟩ 98274) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24216⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨24216⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98539

namespace LeftMerge98553
def owner : Owner := ⟨.program ⟨214⟩, ⟨21536⟩⟩
def mergeEvent : Nat := 98553
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21533⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94462RawTerms
def rightRaw : List Term := Proof.Events384.exact98547RawTerms
def group : MergeGroup := .operator 94462 98547
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94462) (leftOrdinal := 0)
    (rightResult := 98547) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21533⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21533⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98553

namespace LeftMerge98650
def owner : Owner := ⟨.program ⟨214⟩, ⟨16128⟩⟩
def mergeEvent : Nat := 98650
def frameStart : Nat := 98596
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events385.exact98646RawTerms
def rightRaw : List Term := Proof.Events385.exact98644RawTerms
def group : MergeGroup := .operator 98646 98644
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98646) (leftOrdinal := 0)
    (rightResult := 98644) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98650

namespace LeftMerge98662
def owner : Owner := ⟨.program ⟨214⟩, ⟨28049⟩⟩
def mergeEvent : Nat := 98662
def frameStart : Nat := 98596
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28048⟩⟩] } }
def leftRaw : List Term := Proof.Events385.exact98658RawTerms
def rightRaw : List Term := Proof.Events385.exact98635RawTerms
def group : MergeGroup := .operator 98658 98635
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98658) (leftOrdinal := 0)
    (rightResult := 98635) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28048⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98662

namespace LeftMerge98663
def owner : Owner := ⟨.program ⟨214⟩, ⟨28049⟩⟩
def mergeEvent : Nat := 98663
def frameStart : Nat := 98596
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28048⟩⟩] } }
def leftRaw : List Term := Proof.Events385.exact98658RawTerms
def rightRaw : List Term := Proof.Events385.exact98635RawTerms
def group : MergeGroup := .operator 98658 98635
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98658) (leftOrdinal := 1)
    (rightResult := 98635) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28048⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98663

namespace LeftMerge98665
def owner : Owner := ⟨.program ⟨214⟩, ⟨28049⟩⟩
def mergeEvent : Nat := 98665
def frameStart : Nat := 98596
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24216⟩⟩] } }
def rhsRaw : List Term := Proof.Events385.exact98632RawTerms
def group : MergeGroup := .relation 98664
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98664) (rhsResult := 98632)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28048⟩⟩) ⟨24216⟩ 98632) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24216⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨24216⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98665

namespace LeftMerge98673
def owner : Owner := ⟨.program ⟨214⟩, ⟨16099⟩⟩
def mergeEvent : Nat := 98673
def frameStart : Nat := 98596
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16098⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events385.exact98646RawTerms
def rightRaw : List Term := Proof.Events385.exact98669RawTerms
def group : MergeGroup := .operator 98646 98669
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98646) (leftOrdinal := 0)
    (rightResult := 98669) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16098⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98673

namespace LeftMerge98690
def owner : Owner := ⟨.program ⟨214⟩, ⟨21536⟩⟩
def mergeEvent : Nat := 98690
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }
def rhsRaw : List Term := Proof.Events385.exact98687RawTerms
def group : MergeGroup := .relation 98689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98689) (rhsResult := 98687)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21533⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21533⟩⟩]⟩) (none) 98687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98690

namespace LeftMerge98691
def owner : Owner := ⟨.program ⟨214⟩, ⟨21536⟩⟩
def mergeEvent : Nat := 98691
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28048⟩⟩] } }
def rhsRaw : List Term := Proof.Events385.exact98687RawTerms
def group : MergeGroup := .relation 98689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98689) (rhsResult := 98687)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21533⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21533⟩⟩]⟩) (none) 98687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28048⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98691

namespace LeftMerge98692
def owner : Owner := ⟨.program ⟨214⟩, ⟨21536⟩⟩
def mergeEvent : Nat := 98692
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24216⟩⟩] } }
def rhsRaw : List Term := Proof.Events385.exact98687RawTerms
def group : MergeGroup := .relation 98689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98689) (rhsResult := 98687)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21533⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21533⟩⟩]⟩) (none) 98687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24216⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨24216⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98692

namespace LeftMerge98693
def owner : Owner := ⟨.program ⟨214⟩, ⟨21536⟩⟩
def mergeEvent : Nat := 98693
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events385.exact98687RawTerms
def group : MergeGroup := .relation 98689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98689) (rhsResult := 98687)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21533⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21533⟩⟩]⟩) (none) 98687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16098⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98693

namespace LeftMerge98698
def owner : Owner := ⟨.program ⟨214⟩, ⟨28051⟩⟩
def mergeEvent : Nat := 98698
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28048⟩⟩] } }
def leftRaw : List Term := Proof.Events385.exact98694RawTerms
def rightRaw : List Term := Proof.Events384.exact98540RawTerms
def group : MergeGroup := .operator 98694 98540
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98694) (leftOrdinal := 0)
    (rightResult := 98540) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28048⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28048⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98698

namespace LeftMerge98699
def owner : Owner := ⟨.program ⟨214⟩, ⟨28051⟩⟩
def mergeEvent : Nat := 98699
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24216⟩⟩] } }
def leftRaw : List Term := Proof.Events385.exact98694RawTerms
def rightRaw : List Term := Proof.Events384.exact98540RawTerms
def group : MergeGroup := .operator 98694 98540
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98694) (leftOrdinal := 2)
    (rightResult := 98540) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24216⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24216⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨24216⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98699

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
