import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge204664
def owner : Owner := ⟨.program ⟨257⟩, ⟨29875⟩⟩
def mergeEvent : Nat := 204664
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31013⟩⟩] } }
def rhsRaw : List Term := Proof.Events799.exact204660RawTerms
def group : MergeGroup := .relation 204662
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 204662) (rhsResult := 204660)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29872⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 204661 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29872⟩⟩]⟩) (none) 204660) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31013⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31013⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204664

namespace LeftMerge204665
def owner : Owner := ⟨.program ⟨257⟩, ⟨29875⟩⟩
def mergeEvent : Nat := 204665
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30258⟩⟩] } }
def rhsRaw : List Term := Proof.Events799.exact204660RawTerms
def group : MergeGroup := .relation 204662
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 204662) (rhsResult := 204660)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29872⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 204661 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29872⟩⟩]⟩) (none) 204660) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30258⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29104⟩⟩], [⟨.program ⟨257⟩, ⟨30258⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge204665

namespace LeftMerge204666
def owner : Owner := ⟨.program ⟨257⟩, ⟨29875⟩⟩
def mergeEvent : Nat := 204666
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events799.exact204660RawTerms
def group : MergeGroup := .relation 204662
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 204662) (rhsResult := 204660)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29872⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 204661 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29872⟩⟩]⟩) (none) 204660) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29328⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204666

namespace LeftMerge204671
def owner : Owner := ⟨.program ⟨257⟩, ⟨31016⟩⟩
def mergeEvent : Nat := 204671
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31013⟩⟩] } }
def leftRaw : List Term := Proof.Events799.exact204667RawTerms
def rightRaw : List Term := Proof.Events798.exact204489RawTerms
def group : MergeGroup := .operator 204667 204489
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 204667) (leftOrdinal := 0)
    (rightResult := 204489) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31013⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31013⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31013⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge204671

namespace LeftMerge204672
def owner : Owner := ⟨.program ⟨257⟩, ⟨31016⟩⟩
def mergeEvent : Nat := 204672
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30258⟩⟩] } }
def leftRaw : List Term := Proof.Events799.exact204667RawTerms
def rightRaw : List Term := Proof.Events798.exact204489RawTerms
def group : MergeGroup := .operator 204667 204489
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 204667) (leftOrdinal := 2)
    (rightResult := 204489) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30258⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30258⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29104⟩⟩], [⟨.program ⟨257⟩, ⟨30258⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204672

namespace LeftMerge204680
def owner : Owner := ⟨.program ⟨257⟩, ⟨31017⟩⟩
def mergeEvent : Nat := 204680
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩] } }
def leftRaw : List Term := Proof.Events799.exact204674RawTerms
def rightRaw : List Term := Proof.Events061.exact15662RawTerms
def group : MergeGroup := .operator 204674 15662
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 204674) (leftOrdinal := 0)
    (rightResult := 15662) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7219⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7167⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge204680

namespace LeftMerge204681
def owner : Owner := ⟨.program ⟨257⟩, ⟨31017⟩⟩
def mergeEvent : Nat := 204681
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩] } }
def leftRaw : List Term := Proof.Events799.exact204674RawTerms
def rightRaw : List Term := Proof.Events061.exact15662RawTerms
def group : MergeGroup := .operator 204674 15662
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 204674) (leftOrdinal := 1)
    (rightResult := 15662) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7167⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204681

namespace LeftMerge204683
def owner : Owner := ⟨.program ⟨257⟩, ⟨31017⟩⟩
def mergeEvent : Nat := 204683
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15655RawTerms
def group : MergeGroup := .relation 204682
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 204682) (rhsResult := 15655)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7167⟩⟩) ⟨7049⟩ 15655) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204683

namespace LeftMerge204697
def owner : Owner := ⟨.program ⟨257⟩, ⟨28335⟩⟩
def mergeEvent : Nat := 204697
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28333⟩⟩] } }
def leftRaw : List Term := Proof.Events767.exact196555RawTerms
def rightRaw : List Term := Proof.Events799.exact204691RawTerms
def group : MergeGroup := .operator 196555 204691
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 196555) (leftOrdinal := 0)
    (rightResult := 204691) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28333⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28333⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge204697

namespace LeftMerge204698
def owner : Owner := ⟨.program ⟨257⟩, ⟨28335⟩⟩
def mergeEvent : Nat := 204698
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28333⟩⟩] } }
def leftRaw : List Term := Proof.Events767.exact196555RawTerms
def rightRaw : List Term := Proof.Events799.exact204691RawTerms
def group : MergeGroup := .operator 196555 204691
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 196555) (leftOrdinal := 1)
    (rightResult := 204691) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28333⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28333⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204698

namespace LeftMerge204700
def owner : Owner := ⟨.program ⟨257⟩, ⟨28335⟩⟩
def mergeEvent : Nat := 204700
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27578⟩⟩] } }
def rhsRaw : List Term := Proof.Events799.exact204688RawTerms
def group : MergeGroup := .relation 204699
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 204699) (rhsResult := 204688)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28333⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28333⟩⟩) ⟨27578⟩ 204688) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27578⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨27578⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204700

namespace LeftMerge204714
def owner : Owner := ⟨.program ⟨257⟩, ⟨27195⟩⟩
def mergeEvent : Nat := 204714
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27192⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events799.exact204708RawTerms
def group : MergeGroup := .operator 192995 204708
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 204708) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27192⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27192⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge204714

namespace LeftMerge204835
def owner : Owner := ⟨.program ⟨257⟩, ⟨27776⟩⟩
def mergeEvent : Nat := 204835
def frameStart : Nat := 204769
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events800.exact204831RawTerms
def rightRaw : List Term := Proof.Events800.exact204829RawTerms
def group : MergeGroup := .operator 204831 204829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 204831) (leftOrdinal := 0)
    (rightResult := 204829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26424⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge204835

namespace LeftMerge204847
def owner : Owner := ⟨.program ⟨257⟩, ⟨28334⟩⟩
def mergeEvent : Nat := 204847
def frameStart : Nat := 204769
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28333⟩⟩] } }
def leftRaw : List Term := Proof.Events800.exact204843RawTerms
def rightRaw : List Term := Proof.Events800.exact204820RawTerms
def group : MergeGroup := .operator 204843 204820
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 204843) (leftOrdinal := 0)
    (rightResult := 204820) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28333⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28333⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge204847

namespace LeftMerge204848
def owner : Owner := ⟨.program ⟨257⟩, ⟨28334⟩⟩
def mergeEvent : Nat := 204848
def frameStart : Nat := 204769
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28333⟩⟩] } }
def leftRaw : List Term := Proof.Events800.exact204843RawTerms
def rightRaw : List Term := Proof.Events800.exact204820RawTerms
def group : MergeGroup := .operator 204843 204820
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 204843) (leftOrdinal := 1)
    (rightResult := 204820) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28333⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28333⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204848

namespace LeftMerge204850
def owner : Owner := ⟨.program ⟨257⟩, ⟨28334⟩⟩
def mergeEvent : Nat := 204850
def frameStart : Nat := 204769
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27578⟩⟩] } }
def rhsRaw : List Term := Proof.Events800.exact204817RawTerms
def group : MergeGroup := .relation 204849
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 204849) (rhsResult := 204817)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28333⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28333⟩⟩) ⟨27578⟩ 204817) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27578⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨27578⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204850

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
