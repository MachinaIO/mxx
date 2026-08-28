import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge287471
def owner : Owner := ⟨.program ⟨257⟩, ⟨32332⟩⟩
def mergeEvent : Nat := 287471
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32329⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280745RawTerms
def rightRaw : List Term := Proof.Events1122.exact287465RawTerms
def group : MergeGroup := .operator 280745 287465
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280745) (leftOrdinal := 0)
    (rightResult := 287465) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge287471

namespace LeftMerge287550
def owner : Owner := ⟨.program ⟨257⟩, ⟨31324⟩⟩
def mergeEvent : Nat := 287550
def frameStart : Nat := 287520
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1123.exact287546RawTerms
def rightRaw : List Term := Proof.Events1123.exact287543RawTerms
def group : MergeGroup := .operator 287546 287543
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 287546) (leftOrdinal := 0)
    (rightResult := 287543) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31323⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24218⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge287550

namespace LeftMerge287580
def owner : Owner := ⟨.program ⟨257⟩, ⟨33204⟩⟩
def mergeEvent : Nat := 287580
def frameStart : Nat := 287520
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1123.exact287576RawTerms
def rightRaw : List Term := Proof.Events1123.exact287574RawTerms
def group : MergeGroup := .operator 287576 287574
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 287576) (leftOrdinal := 0)
    (rightResult := 287574) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge287580

namespace LeftMerge287601
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def mergeEvent : Nat := 287601
def frameStart : Nat := 287520
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }
def leftRaw : List Term := Proof.Events1123.exact287597RawTerms
def rightRaw : List Term := Proof.Events1123.exact287594RawTerms
def group : MergeGroup := .operator 287597 287594
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 287597) (leftOrdinal := 0)
    (rightResult := 287594) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9577⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge287601

namespace LeftMerge287610
def owner : Owner := ⟨.program ⟨257⟩, ⟨33396⟩⟩
def mergeEvent : Nat := 287610
def frameStart : Nat := 287520
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩] } }
def leftRaw : List Term := Proof.Events1123.exact287606RawTerms
def rightRaw : List Term := Proof.Events1123.exact287565RawTerms
def group : MergeGroup := .operator 287606 287565
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 287606) (leftOrdinal := 0)
    (rightResult := 287565) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33393⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge287610

namespace LeftMerge287611
def owner : Owner := ⟨.program ⟨257⟩, ⟨33396⟩⟩
def mergeEvent : Nat := 287611
def frameStart : Nat := 287520
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩] } }
def leftRaw : List Term := Proof.Events1123.exact287606RawTerms
def rightRaw : List Term := Proof.Events1123.exact287565RawTerms
def group : MergeGroup := .operator 287606 287565
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 287606) (leftOrdinal := 1)
    (rightResult := 287565) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33393⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge287611

namespace LeftMerge287613
def owner : Owner := ⟨.program ⟨257⟩, ⟨33396⟩⟩
def mergeEvent : Nat := 287613
def frameStart : Nat := 287520
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32913⟩⟩] } }
def rhsRaw : List Term := Proof.Events1123.exact287562RawTerms
def group : MergeGroup := .relation 287612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 287612) (rhsResult := 287562)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33393⟩⟩) ⟨32913⟩ 287562) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32913⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], [⟨.program ⟨257⟩, ⟨32913⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge287613

namespace LeftMerge287621
def owner : Owner := ⟨.program ⟨257⟩, ⟨31782⟩⟩
def mergeEvent : Nat := 287621
def frameStart : Nat := 287520
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1123.exact287576RawTerms
def rightRaw : List Term := Proof.Events1123.exact287617RawTerms
def group : MergeGroup := .operator 287576 287617
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 287576) (leftOrdinal := 0)
    (rightResult := 287617) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31780⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge287621

namespace LeftMerge287638
def owner : Owner := ⟨.program ⟨257⟩, ⟨32332⟩⟩
def mergeEvent : Nat := 287638
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }
def rhsRaw : List Term := Proof.Events1123.exact287635RawTerms
def group : MergeGroup := .relation 287637
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 287637) (rhsResult := 287635)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32329⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 287636 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32329⟩⟩]⟩) (none) 287635) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge287638

namespace LeftMerge287639
def owner : Owner := ⟨.program ⟨257⟩, ⟨32332⟩⟩
def mergeEvent : Nat := 287639
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩] } }
def rhsRaw : List Term := Proof.Events1123.exact287635RawTerms
def group : MergeGroup := .relation 287637
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 287637) (rhsResult := 287635)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32329⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 287636 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32329⟩⟩]⟩) (none) 287635) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge287639

namespace LeftMerge287640
def owner : Owner := ⟨.program ⟨257⟩, ⟨32332⟩⟩
def mergeEvent : Nat := 287640
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32913⟩⟩] } }
def rhsRaw : List Term := Proof.Events1123.exact287635RawTerms
def group : MergeGroup := .relation 287637
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 287637) (rhsResult := 287635)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32329⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 287636 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32329⟩⟩]⟩) (none) 287635) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32913⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], [⟨.program ⟨257⟩, ⟨32913⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge287640

namespace LeftMerge287641
def owner : Owner := ⟨.program ⟨257⟩, ⟨32332⟩⟩
def mergeEvent : Nat := 287641
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1123.exact287635RawTerms
def group : MergeGroup := .relation 287637
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 287637) (rhsResult := 287635)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32329⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 287636 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32329⟩⟩]⟩) (none) 287635) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge287641

namespace LeftMerge287646
def owner : Owner := ⟨.program ⟨257⟩, ⟨33395⟩⟩
def mergeEvent : Nat := 287646
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32913⟩⟩] } }
def leftRaw : List Term := Proof.Events1123.exact287642RawTerms
def rightRaw : List Term := Proof.Events1122.exact287458RawTerms
def group : MergeGroup := .operator 287642 287458
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 287642) (leftOrdinal := 2)
    (rightResult := 287458) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32913⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32913⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], [⟨.program ⟨257⟩, ⟨32913⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge287646

namespace LeftMerge287647
def owner : Owner := ⟨.program ⟨257⟩, ⟨33395⟩⟩
def mergeEvent : Nat := 287647
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩] } }
def leftRaw : List Term := Proof.Events1123.exact287642RawTerms
def rightRaw : List Term := Proof.Events1122.exact287458RawTerms
def group : MergeGroup := .operator 287642 287458
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 287642) (leftOrdinal := 1)
    (rightResult := 287458) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge287647

namespace LeftMerge287655
def owner : Owner := ⟨.program ⟨257⟩, ⟨33708⟩⟩
def mergeEvent : Nat := 287655
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33706⟩⟩] } }
def leftRaw : List Term := Proof.Events1123.exact287649RawTerms
def rightRaw : List Term := Proof.Events1122.exact287374RawTerms
def group : MergeGroup := .operator 287649 287374
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 287649) (leftOrdinal := 0)
    (rightResult := 287374) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33706⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33706⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge287655

namespace LeftMerge287656
def owner : Owner := ⟨.program ⟨257⟩, ⟨33708⟩⟩
def mergeEvent : Nat := 287656
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33706⟩⟩] } }
def leftRaw : List Term := Proof.Events1123.exact287649RawTerms
def rightRaw : List Term := Proof.Events1122.exact287374RawTerms
def group : MergeGroup := .operator 287649 287374
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 287649) (leftOrdinal := 1)
    (rightResult := 287374) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33706⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33706⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge287656

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
