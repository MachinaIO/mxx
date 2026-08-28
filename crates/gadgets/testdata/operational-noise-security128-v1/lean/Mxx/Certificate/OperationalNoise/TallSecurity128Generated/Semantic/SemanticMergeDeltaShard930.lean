import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge152470
def owner : Owner := ⟨.program ⟨257⟩, ⟨12940⟩⟩
def mergeEvent : Nat := 152470
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }
def leftRaw : List Term := Proof.Events595.exact152461RawTerms
def rightRaw : List Term := Proof.Events080.exact20617RawTerms
def group : MergeGroup := .operator 152461 20617
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 152461) (leftOrdinal := 0)
    (rightResult := 20617) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9544⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge152470

namespace LeftMerge152475
def owner : Owner := ⟨.program ⟨257⟩, ⟨26029⟩⟩
def mergeEvent : Nat := 152475
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12936⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }
def leftRaw : List Term := Proof.Events595.exact152471RawTerms
def rightRaw : List Term := Proof.Events595.exact152441RawTerms
def group : MergeGroup := .operator 152471 152441
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 152471) (leftOrdinal := 1)
    (rightResult := 152441) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12936⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12936⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12936⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge152475

namespace LeftMerge152483
def owner : Owner := ⟨.program ⟨257⟩, ⟨27887⟩⟩
def mergeEvent : Nat := 152483
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12936⟩⟩, ⟨.program ⟨257⟩, ⟨26022⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27886⟩⟩] } }
def leftRaw : List Term := Proof.Events595.exact152477RawTerms
def rightRaw : List Term := Proof.Events595.exact152413RawTerms
def group : MergeGroup := .operator 152477 152413
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 152477) (leftOrdinal := 1)
    (rightResult := 152413) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12936⟩⟩, ⟨.program ⟨257⟩, ⟨26022⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27886⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12936⟩⟩, ⟨.program ⟨257⟩, ⟨26022⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27886⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge152483

namespace LeftMerge152485
def owner : Owner := ⟨.program ⟨257⟩, ⟨27887⟩⟩
def mergeEvent : Nat := 152485
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12936⟩⟩, ⟨.program ⟨257⟩, ⟨26022⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27391⟩⟩] } }
def rhsRaw : List Term := Proof.Events595.exact152410RawTerms
def group : MergeGroup := .relation 152484
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 152484) (rhsResult := 152410)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12936⟩⟩, ⟨.program ⟨257⟩, ⟨26022⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27886⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27886⟩⟩) ⟨27391⟩ 152410) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27391⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12936⟩⟩, ⟨.program ⟨257⟩, ⟨26022⟩⟩], [⟨.program ⟨257⟩, ⟨27391⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge152485

namespace LeftMerge152486
def owner : Owner := ⟨.program ⟨257⟩, ⟨27887⟩⟩
def mergeEvent : Nat := 152486
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27886⟩⟩] } }
def leftRaw : List Term := Proof.Events595.exact152477RawTerms
def rightRaw : List Term := Proof.Events595.exact152413RawTerms
def group : MergeGroup := .operator 152477 152413
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 152477) (leftOrdinal := 0)
    (rightResult := 152413) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27886⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27886⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge152486

namespace LeftMerge152500
def owner : Owner := ⟨.program ⟨257⟩, ⟨26822⟩⟩
def mergeEvent : Nat := 152500
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26819⟩⟩] } }
def leftRaw : List Term := Proof.Events582.exact149120RawTerms
def rightRaw : List Term := Proof.Events595.exact152494RawTerms
def group : MergeGroup := .operator 149120 152494
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149120) (leftOrdinal := 0)
    (rightResult := 152494) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨26819⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26819⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge152500

namespace LeftMerge152579
def owner : Owner := ⟨.program ⟨257⟩, ⟨26023⟩⟩
def mergeEvent : Nat := 152579
def frameStart : Nat := 152549
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12936⟩⟩, ⟨.program ⟨257⟩, ⟨26022⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events595.exact152575RawTerms
def rightRaw : List Term := Proof.Events595.exact152572RawTerms
def group : MergeGroup := .operator 152575 152572
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 152575) (leftOrdinal := 0)
    (rightResult := 152572) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12936⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26022⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12936⟩⟩, ⟨.program ⟨257⟩, ⟨26022⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge152579

namespace LeftMerge152609
def owner : Owner := ⟨.program ⟨257⟩, ⟨27676⟩⟩
def mergeEvent : Nat := 152609
def frameStart : Nat := 152549
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12936⟩⟩, ⟨.program ⟨257⟩, ⟨26022⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events596.exact152605RawTerms
def rightRaw : List Term := Proof.Events596.exact152603RawTerms
def group : MergeGroup := .operator 152605 152603
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 152605) (leftOrdinal := 0)
    (rightResult := 152603) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12936⟩⟩, ⟨.program ⟨257⟩, ⟨26022⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12936⟩⟩, ⟨.program ⟨257⟩, ⟨26022⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge152609

namespace LeftMerge152632
def owner : Owner := ⟨.program ⟨257⟩, ⟨9546⟩⟩
def mergeEvent : Nat := 152632
def frameStart : Nat := 152549
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }
def leftRaw : List Term := Proof.Events596.exact152628RawTerms
def rightRaw : List Term := Proof.Events596.exact152625RawTerms
def group : MergeGroup := .operator 152628 152625
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 152628) (leftOrdinal := 0)
    (rightResult := 152625) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9544⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge152632

namespace LeftMerge152641
def owner : Owner := ⟨.program ⟨257⟩, ⟨27889⟩⟩
def mergeEvent : Nat := 152641
def frameStart : Nat := 152549
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27886⟩⟩] } }
def leftRaw : List Term := Proof.Events596.exact152637RawTerms
def rightRaw : List Term := Proof.Events596.exact152594RawTerms
def group : MergeGroup := .operator 152637 152594
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 152637) (leftOrdinal := 0)
    (rightResult := 152594) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27886⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27886⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge152641

namespace LeftMerge152642
def owner : Owner := ⟨.program ⟨257⟩, ⟨27889⟩⟩
def mergeEvent : Nat := 152642
def frameStart : Nat := 152549
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12936⟩⟩, ⟨.program ⟨257⟩, ⟨26022⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27886⟩⟩] } }
def leftRaw : List Term := Proof.Events596.exact152637RawTerms
def rightRaw : List Term := Proof.Events596.exact152594RawTerms
def group : MergeGroup := .operator 152637 152594
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 152637) (leftOrdinal := 1)
    (rightResult := 152594) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12936⟩⟩, ⟨.program ⟨257⟩, ⟨26022⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27886⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12936⟩⟩, ⟨.program ⟨257⟩, ⟨26022⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27886⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge152642

namespace LeftMerge152644
def owner : Owner := ⟨.program ⟨257⟩, ⟨27889⟩⟩
def mergeEvent : Nat := 152644
def frameStart : Nat := 152549
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12936⟩⟩, ⟨.program ⟨257⟩, ⟨26022⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27391⟩⟩] } }
def rhsRaw : List Term := Proof.Events596.exact152591RawTerms
def group : MergeGroup := .relation 152643
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 152643) (rhsResult := 152591)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12936⟩⟩, ⟨.program ⟨257⟩, ⟨26022⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27886⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27886⟩⟩) ⟨27391⟩ 152591) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27391⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12936⟩⟩, ⟨.program ⟨257⟩, ⟨26022⟩⟩], [⟨.program ⟨257⟩, ⟨27391⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge152644

namespace LeftMerge152652
def owner : Owner := ⟨.program ⟨257⟩, ⟨26386⟩⟩
def mergeEvent : Nat := 152652
def frameStart : Nat := 152549
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26384⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events596.exact152605RawTerms
def rightRaw : List Term := Proof.Events596.exact152648RawTerms
def group : MergeGroup := .operator 152605 152648
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 152605) (leftOrdinal := 0)
    (rightResult := 152648) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26384⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26384⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge152652

namespace LeftMerge152669
def owner : Owner := ⟨.program ⟨257⟩, ⟨26822⟩⟩
def mergeEvent : Nat := 152669
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }
def rhsRaw : List Term := Proof.Events596.exact152666RawTerms
def group : MergeGroup := .relation 152668
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 152668) (rhsResult := 152666)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26819⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 152667 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26819⟩⟩]⟩) (none) 152666) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge152669

namespace LeftMerge152670
def owner : Owner := ⟨.program ⟨257⟩, ⟨26822⟩⟩
def mergeEvent : Nat := 152670
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27886⟩⟩] } }
def rhsRaw : List Term := Proof.Events596.exact152666RawTerms
def group : MergeGroup := .relation 152668
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 152668) (rhsResult := 152666)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26819⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 152667 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26819⟩⟩]⟩) (none) 152666) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27886⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27886⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge152670

namespace LeftMerge152671
def owner : Owner := ⟨.program ⟨257⟩, ⟨26822⟩⟩
def mergeEvent : Nat := 152671
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12936⟩⟩, ⟨.program ⟨257⟩, ⟨26022⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27391⟩⟩] } }
def rhsRaw : List Term := Proof.Events596.exact152666RawTerms
def group : MergeGroup := .relation 152668
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 152668) (rhsResult := 152666)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26819⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 152667 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26819⟩⟩]⟩) (none) 152666) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12936⟩⟩, ⟨.program ⟨257⟩, ⟨26022⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27391⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12936⟩⟩, ⟨.program ⟨257⟩, ⟨26022⟩⟩], [⟨.program ⟨257⟩, ⟨27391⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge152671

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
