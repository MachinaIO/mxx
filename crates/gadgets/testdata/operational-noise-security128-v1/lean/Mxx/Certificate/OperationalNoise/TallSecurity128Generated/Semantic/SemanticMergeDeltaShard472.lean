import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge79552
def owner : Owner := ⟨.program ⟨257⟩, ⟨27987⟩⟩
def mergeEvent : Nat := 79552
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27445⟩⟩] } }
def leftRaw : List Term := Proof.Events310.exact79548RawTerms
def rightRaw : List Term := Proof.Events310.exact79362RawTerms
def group : MergeGroup := .operator 79548 79362
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79548) (leftOrdinal := 2)
    (rightResult := 79362) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27445⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27445⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨27445⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge79552

namespace LeftMerge79553
def owner : Owner := ⟨.program ⟨257⟩, ⟨27987⟩⟩
def mergeEvent : Nat := 79553
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27985⟩⟩] } }
def leftRaw : List Term := Proof.Events310.exact79548RawTerms
def rightRaw : List Term := Proof.Events310.exact79362RawTerms
def group : MergeGroup := .operator 79548 79362
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79548) (leftOrdinal := 1)
    (rightResult := 79362) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27985⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27985⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27985⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79553

namespace LeftMerge79561
def owner : Owner := ⟨.program ⟨257⟩, ⟨28441⟩⟩
def mergeEvent : Nat := 79561
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩] } }
def leftRaw : List Term := Proof.Events310.exact79555RawTerms
def rightRaw : List Term := Proof.Events309.exact79278RawTerms
def group : MergeGroup := .operator 79555 79278
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79555) (leftOrdinal := 0)
    (rightResult := 79278) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28439⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79561

namespace LeftMerge79562
def owner : Owner := ⟨.program ⟨257⟩, ⟨28441⟩⟩
def mergeEvent : Nat := 79562
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩] } }
def leftRaw : List Term := Proof.Events310.exact79555RawTerms
def rightRaw : List Term := Proof.Events309.exact79278RawTerms
def group : MergeGroup := .operator 79555 79278
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79555) (leftOrdinal := 1)
    (rightResult := 79278) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28439⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge79562

namespace LeftMerge79564
def owner : Owner := ⟨.program ⟨257⟩, ⟨28441⟩⟩
def mergeEvent : Nat := 79564
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27615⟩⟩] } }
def rhsRaw : List Term := Proof.Events309.exact79275RawTerms
def group : MergeGroup := .relation 79563
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 79563) (rhsResult := 79275)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28439⟩⟩) ⟨27615⟩ 79275) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27615⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27615⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge79564

namespace LeftMerge79578
def owner : Owner := ⟨.program ⟨257⟩, ⟨27279⟩⟩
def mergeEvent : Nat := 79578
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27276⟩⟩] } }
def leftRaw : List Term := Proof.Events296.exact75995RawTerms
def rightRaw : List Term := Proof.Events310.exact79572RawTerms
def group : MergeGroup := .operator 75995 79572
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75995) (leftOrdinal := 0)
    (rightResult := 79572) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27276⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27276⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79578

namespace LeftMerge79699
def owner : Owner := ⟨.program ⟨257⟩, ⟨27792⟩⟩
def mergeEvent : Nat := 79699
def frameStart : Nat := 79633
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26456⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79695RawTerms
def rightRaw : List Term := Proof.Events311.exact79693RawTerms
def group : MergeGroup := .operator 79695 79693
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79695) (leftOrdinal := 0)
    (rightResult := 79693) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26456⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79699

namespace LeftMerge79711
def owner : Owner := ⟨.program ⟨257⟩, ⟨28440⟩⟩
def mergeEvent : Nat := 79711
def frameStart : Nat := 79633
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79707RawTerms
def rightRaw : List Term := Proof.Events311.exact79684RawTerms
def group : MergeGroup := .operator 79707 79684
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79707) (leftOrdinal := 0)
    (rightResult := 79684) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28439⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79711

namespace LeftMerge79712
def owner : Owner := ⟨.program ⟨257⟩, ⟨28440⟩⟩
def mergeEvent : Nat := 79712
def frameStart : Nat := 79633
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26456⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79707RawTerms
def rightRaw : List Term := Proof.Events311.exact79684RawTerms
def group : MergeGroup := .operator 79707 79684
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79707) (leftOrdinal := 1)
    (rightResult := 79684) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26456⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28439⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge79712

namespace LeftMerge79714
def owner : Owner := ⟨.program ⟨257⟩, ⟨28440⟩⟩
def mergeEvent : Nat := 79714
def frameStart : Nat := 79633
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26456⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27615⟩⟩] } }
def rhsRaw : List Term := Proof.Events311.exact79681RawTerms
def group : MergeGroup := .relation 79713
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 79713) (rhsResult := 79681)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28439⟩⟩) ⟨27615⟩ 79681) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27615⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27615⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge79714

namespace LeftMerge79722
def owner : Owner := ⟨.program ⟨257⟩, ⟨26698⟩⟩
def mergeEvent : Nat := 79722
def frameStart : Nat := 79633
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26697⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79695RawTerms
def rightRaw : List Term := Proof.Events311.exact79718RawTerms
def group : MergeGroup := .operator 79695 79718
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79695) (leftOrdinal := 0)
    (rightResult := 79718) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26697⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79722

namespace LeftMerge79739
def owner : Owner := ⟨.program ⟨257⟩, ⟨27279⟩⟩
def mergeEvent : Nat := 79739
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩] } }
def rhsRaw : List Term := Proof.Events311.exact79736RawTerms
def group : MergeGroup := .relation 79738
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 79738) (rhsResult := 79736)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27276⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 79737 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27276⟩⟩]⟩) (none) 79736) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79739

namespace LeftMerge79740
def owner : Owner := ⟨.program ⟨257⟩, ⟨27279⟩⟩
def mergeEvent : Nat := 79740
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩] } }
def rhsRaw : List Term := Proof.Events311.exact79736RawTerms
def group : MergeGroup := .relation 79738
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 79738) (rhsResult := 79736)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27276⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 79737 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27276⟩⟩]⟩) (none) 79736) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge79740

namespace LeftMerge79741
def owner : Owner := ⟨.program ⟨257⟩, ⟨27279⟩⟩
def mergeEvent : Nat := 79741
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27615⟩⟩] } }
def rhsRaw : List Term := Proof.Events311.exact79736RawTerms
def group : MergeGroup := .relation 79738
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 79738) (rhsResult := 79736)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27276⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 79737 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27276⟩⟩]⟩) (none) 79736) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26456⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27615⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27615⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79741

namespace LeftMerge79742
def owner : Owner := ⟨.program ⟨257⟩, ⟨27279⟩⟩
def mergeEvent : Nat := 79742
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events311.exact79736RawTerms
def group : MergeGroup := .relation 79738
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 79738) (rhsResult := 79736)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27276⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 79737 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27276⟩⟩]⟩) (none) 79736) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26697⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge79742

namespace LeftMerge79747
def owner : Owner := ⟨.program ⟨257⟩, ⟨28442⟩⟩
def mergeEvent : Nat := 79747
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79743RawTerms
def rightRaw : List Term := Proof.Events310.exact79565RawTerms
def group : MergeGroup := .operator 79743 79565
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79743) (leftOrdinal := 0)
    (rightResult := 79565) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79747

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
