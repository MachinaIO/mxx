import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge164404
def owner : Owner := ⟨.program ⟨257⟩, ⟨45952⟩⟩
def mergeEvent : Nat := 164404
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46493⟩⟩] } }
def rhsRaw : List Term := Proof.Events642.exact164399RawTerms
def group : MergeGroup := .relation 164401
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164401) (rhsResult := 164399)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45949⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 164400 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45949⟩⟩]⟩) (none) 164399) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46493⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], [⟨.program ⟨257⟩, ⟨46493⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164404

namespace LeftMerge164405
def owner : Owner := ⟨.program ⟨257⟩, ⟨45952⟩⟩
def mergeEvent : Nat := 164405
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events642.exact164399RawTerms
def group : MergeGroup := .relation 164401
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164401) (rhsResult := 164399)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45949⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 164400 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45949⟩⟩]⟩) (none) 164399) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164405

namespace LeftMerge164410
def owner : Owner := ⟨.program ⟨257⟩, ⟨47025⟩⟩
def mergeEvent : Nat := 164410
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46493⟩⟩] } }
def leftRaw : List Term := Proof.Events642.exact164406RawTerms
def rightRaw : List Term := Proof.Events641.exact164220RawTerms
def group : MergeGroup := .operator 164406 164220
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164406) (leftOrdinal := 2)
    (rightResult := 164220) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46493⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46493⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], [⟨.program ⟨257⟩, ⟨46493⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164410

namespace LeftMerge164411
def owner : Owner := ⟨.program ⟨257⟩, ⟨47025⟩⟩
def mergeEvent : Nat := 164411
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47023⟩⟩] } }
def leftRaw : List Term := Proof.Events642.exact164406RawTerms
def rightRaw : List Term := Proof.Events641.exact164220RawTerms
def group : MergeGroup := .operator 164406 164220
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164406) (leftOrdinal := 1)
    (rightResult := 164220) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47023⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47023⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47023⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164411

namespace LeftMerge164419
def owner : Owner := ⟨.program ⟨257⟩, ⟨47451⟩⟩
def mergeEvent : Nat := 164419
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47449⟩⟩] } }
def leftRaw : List Term := Proof.Events642.exact164413RawTerms
def rightRaw : List Term := Proof.Events641.exact164136RawTerms
def group : MergeGroup := .operator 164413 164136
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164413) (leftOrdinal := 0)
    (rightResult := 164136) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47449⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47449⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164419

namespace LeftMerge164420
def owner : Owner := ⟨.program ⟨257⟩, ⟨47451⟩⟩
def mergeEvent : Nat := 164420
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47449⟩⟩] } }
def leftRaw : List Term := Proof.Events642.exact164413RawTerms
def rightRaw : List Term := Proof.Events641.exact164136RawTerms
def group : MergeGroup := .operator 164413 164136
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164413) (leftOrdinal := 1)
    (rightResult := 164136) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47449⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47449⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164420

namespace LeftMerge164422
def owner : Owner := ⟨.program ⟨257⟩, ⟨47451⟩⟩
def mergeEvent : Nat := 164422
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46657⟩⟩] } }
def rhsRaw : List Term := Proof.Events641.exact164133RawTerms
def group : MergeGroup := .relation 164421
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164421) (rhsResult := 164133)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47449⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47449⟩⟩) ⟨46657⟩ 164133) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46657⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45500⟩⟩], [⟨.program ⟨257⟩, ⟨46657⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164422

namespace LeftMerge164436
def owner : Owner := ⟨.program ⟨257⟩, ⟨46299⟩⟩
def mergeEvent : Nat := 164436
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46296⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events642.exact164430RawTerms
def group : MergeGroup := .operator 163745 164430
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 164430) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46296⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46296⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164436

namespace LeftMerge164557
def owner : Owner := ⟨.program ⟨257⟩, ⟨46844⟩⟩
def mergeEvent : Nat := 164557
def frameStart : Nat := 164491
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events642.exact164553RawTerms
def rightRaw : List Term := Proof.Events642.exact164551RawTerms
def group : MergeGroup := .operator 164553 164551
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164553) (leftOrdinal := 0)
    (rightResult := 164551) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45500⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164557

namespace LeftMerge164569
def owner : Owner := ⟨.program ⟨257⟩, ⟨47450⟩⟩
def mergeEvent : Nat := 164569
def frameStart : Nat := 164491
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47449⟩⟩] } }
def leftRaw : List Term := Proof.Events642.exact164565RawTerms
def rightRaw : List Term := Proof.Events642.exact164542RawTerms
def group : MergeGroup := .operator 164565 164542
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164565) (leftOrdinal := 0)
    (rightResult := 164542) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47449⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47449⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164569

namespace LeftMerge164570
def owner : Owner := ⟨.program ⟨257⟩, ⟨47450⟩⟩
def mergeEvent : Nat := 164570
def frameStart : Nat := 164491
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47449⟩⟩] } }
def leftRaw : List Term := Proof.Events642.exact164565RawTerms
def rightRaw : List Term := Proof.Events642.exact164542RawTerms
def group : MergeGroup := .operator 164565 164542
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164565) (leftOrdinal := 1)
    (rightResult := 164542) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47449⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47449⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164570

namespace LeftMerge164572
def owner : Owner := ⟨.program ⟨257⟩, ⟨47450⟩⟩
def mergeEvent : Nat := 164572
def frameStart : Nat := 164491
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46657⟩⟩] } }
def rhsRaw : List Term := Proof.Events642.exact164539RawTerms
def group : MergeGroup := .relation 164571
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164571) (rhsResult := 164539)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47449⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47449⟩⟩) ⟨46657⟩ 164539) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46657⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45500⟩⟩], [⟨.program ⟨257⟩, ⟨46657⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164572

namespace LeftMerge164580
def owner : Owner := ⟨.program ⟨257⟩, ⟨45736⟩⟩
def mergeEvent : Nat := 164580
def frameStart : Nat := 164491
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45735⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events642.exact164553RawTerms
def rightRaw : List Term := Proof.Events642.exact164576RawTerms
def group : MergeGroup := .operator 164553 164576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164553) (leftOrdinal := 0)
    (rightResult := 164576) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45735⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45735⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164580

namespace LeftMerge164597
def owner : Owner := ⟨.program ⟨257⟩, ⟨46299⟩⟩
def mergeEvent : Nat := 164597
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }
def rhsRaw : List Term := Proof.Events642.exact164594RawTerms
def group : MergeGroup := .relation 164596
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164596) (rhsResult := 164594)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46296⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 164595 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46296⟩⟩]⟩) (none) 164594) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164597

namespace LeftMerge164598
def owner : Owner := ⟨.program ⟨257⟩, ⟨46299⟩⟩
def mergeEvent : Nat := 164598
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47449⟩⟩] } }
def rhsRaw : List Term := Proof.Events642.exact164594RawTerms
def group : MergeGroup := .relation 164596
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164596) (rhsResult := 164594)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46296⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 164595 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46296⟩⟩]⟩) (none) 164594) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47449⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47449⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164598

namespace LeftMerge164599
def owner : Owner := ⟨.program ⟨257⟩, ⟨46299⟩⟩
def mergeEvent : Nat := 164599
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46657⟩⟩] } }
def rhsRaw : List Term := Proof.Events642.exact164594RawTerms
def group : MergeGroup := .relation 164596
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164596) (rhsResult := 164594)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46296⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 164595 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46296⟩⟩]⟩) (none) 164594) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46657⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45500⟩⟩], [⟨.program ⟨257⟩, ⟨46657⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164599

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
