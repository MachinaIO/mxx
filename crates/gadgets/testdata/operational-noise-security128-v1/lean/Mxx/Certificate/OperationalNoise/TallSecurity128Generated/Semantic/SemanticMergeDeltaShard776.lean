import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge128563
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128563
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨29247⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }
def rhsRaw : List Term := Proof.Events467.exact119750RawTerms
def group : MergeGroup := .relation 128562
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 128562) (rhsResult := 119750)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨29247⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71113⟩⟩) ⟨68806⟩ 119750) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨29247⟩⟩], [⟨.program ⟨257⟩, ⟨68806⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128563

namespace LeftMerge128564
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128564
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events502.exact128530RawTerms
def rightRaw : List Term := Proof.Events467.exact119753RawTerms
def group : MergeGroup := .operator 128530 119753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128530) (leftOrdinal := 10)
    (rightResult := 119753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128564

namespace LeftMerge128565
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128565
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨26567⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events502.exact128530RawTerms
def rightRaw : List Term := Proof.Events467.exact119753RawTerms
def group : MergeGroup := .operator 128530 119753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128530) (leftOrdinal := 21)
    (rightResult := 119753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨26567⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨26567⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128565

namespace LeftMerge128567
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128567
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨26567⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }
def rhsRaw : List Term := Proof.Events467.exact119750RawTerms
def group : MergeGroup := .relation 128566
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 128566) (rhsResult := 119750)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨26567⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71113⟩⟩) ⟨68806⟩ 119750) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨26567⟩⟩], [⟨.program ⟨257⟩, ⟨68806⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128567

namespace LeftMerge128568
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128568
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events502.exact128530RawTerms
def rightRaw : List Term := Proof.Events467.exact119753RawTerms
def group : MergeGroup := .operator 128530 119753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128530) (leftOrdinal := 9)
    (rightResult := 119753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128568

namespace LeftMerge128569
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128569
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨66321⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events502.exact128530RawTerms
def rightRaw : List Term := Proof.Events467.exact119753RawTerms
def group : MergeGroup := .operator 128530 119753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128530) (leftOrdinal := 35)
    (rightResult := 119753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨66321⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨66321⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128569

namespace LeftMerge128571
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128571
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨66321⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }
def rhsRaw : List Term := Proof.Events467.exact119750RawTerms
def group : MergeGroup := .relation 128570
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 128570) (rhsResult := 119750)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨66321⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71113⟩⟩) ⟨68806⟩ 119750) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨66321⟩⟩], [⟨.program ⟨257⟩, ⟨68806⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128571

namespace LeftMerge128572
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128572
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events502.exact128530RawTerms
def rightRaw : List Term := Proof.Events467.exact119753RawTerms
def group : MergeGroup := .operator 128530 119753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128530) (leftOrdinal := 8)
    (rightResult := 119753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128572

namespace LeftMerge128573
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128573
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨63005⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events502.exact128530RawTerms
def rightRaw : List Term := Proof.Events467.exact119753RawTerms
def group : MergeGroup := .operator 128530 119753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128530) (leftOrdinal := 34)
    (rightResult := 119753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨63005⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨63005⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128573

namespace LeftMerge128575
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128575
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨63005⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }
def rhsRaw : List Term := Proof.Events467.exact119750RawTerms
def group : MergeGroup := .relation 128574
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 128574) (rhsResult := 119750)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨63005⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71113⟩⟩) ⟨68806⟩ 119750) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨63005⟩⟩], [⟨.program ⟨257⟩, ⟨68806⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128575

namespace LeftMerge128576
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128576
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events502.exact128530RawTerms
def rightRaw : List Term := Proof.Events467.exact119753RawTerms
def group : MergeGroup := .operator 128530 119753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128530) (leftOrdinal := 7)
    (rightResult := 119753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128576

namespace LeftMerge128577
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128577
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨60025⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events502.exact128530RawTerms
def rightRaw : List Term := Proof.Events467.exact119753RawTerms
def group : MergeGroup := .operator 128530 119753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128530) (leftOrdinal := 33)
    (rightResult := 119753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨60025⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨60025⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128577

namespace LeftMerge128579
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128579
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨60025⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }
def rhsRaw : List Term := Proof.Events467.exact119750RawTerms
def group : MergeGroup := .relation 128578
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 128578) (rhsResult := 119750)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨60025⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71113⟩⟩) ⟨68806⟩ 119750) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨60025⟩⟩], [⟨.program ⟨257⟩, ⟨68806⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128579

namespace LeftMerge128580
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128580
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events502.exact128530RawTerms
def rightRaw : List Term := Proof.Events467.exact119753RawTerms
def group : MergeGroup := .operator 128530 119753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128530) (leftOrdinal := 6)
    (rightResult := 119753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128580

namespace LeftMerge128581
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128581
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨57045⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events502.exact128530RawTerms
def rightRaw : List Term := Proof.Events467.exact119753RawTerms
def group : MergeGroup := .operator 128530 119753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128530) (leftOrdinal := 32)
    (rightResult := 119753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨57045⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨57045⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128581

namespace LeftMerge128583
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128583
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨57045⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }
def rhsRaw : List Term := Proof.Events467.exact119750RawTerms
def group : MergeGroup := .relation 128582
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 128582) (rhsResult := 119750)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨57045⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71113⟩⟩) ⟨68806⟩ 119750) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨57045⟩⟩], [⟨.program ⟨257⟩, ⟨68806⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128583

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
