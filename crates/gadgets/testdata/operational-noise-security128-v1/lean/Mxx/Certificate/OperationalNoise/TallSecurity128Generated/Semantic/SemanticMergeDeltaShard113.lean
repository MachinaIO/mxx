import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge22509
def owner : Owner := ⟨.program ⟨257⟩, ⟨61272⟩⟩
def mergeEvent : Nat := 22509
def frameStart : Nat := 22443
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59758⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events087.exact22505RawTerms
def rightRaw : List Term := Proof.Events087.exact22503RawTerms
def group : MergeGroup := .operator 22505 22503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22505) (leftOrdinal := 0)
    (rightResult := 22503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59758⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59758⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22509

namespace LeftMerge22521
def owner : Owner := ⟨.program ⟨257⟩, ⟨61623⟩⟩
def mergeEvent : Nat := 22521
def frameStart : Nat := 22443
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59758⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61622⟩⟩] } }
def leftRaw : List Term := Proof.Events087.exact22517RawTerms
def rightRaw : List Term := Proof.Events087.exact22494RawTerms
def group : MergeGroup := .operator 22517 22494
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22517) (leftOrdinal := 1)
    (rightResult := 22494) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59758⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61622⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59758⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61622⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22521

namespace LeftMerge22523
def owner : Owner := ⟨.program ⟨257⟩, ⟨61623⟩⟩
def mergeEvent : Nat := 22523
def frameStart : Nat := 22443
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59758⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61023⟩⟩] } }
def rhsRaw : List Term := Proof.Events087.exact22491RawTerms
def group : MergeGroup := .relation 22522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 22522) (rhsResult := 22491)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨59758⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61622⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61622⟩⟩) ⟨61023⟩ 22491) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61023⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨59758⟩⟩], [⟨.program ⟨257⟩, ⟨61023⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22523

namespace LeftMerge22524
def owner : Owner := ⟨.program ⟨257⟩, ⟨61623⟩⟩
def mergeEvent : Nat := 22524
def frameStart : Nat := 22443
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61622⟩⟩] } }
def leftRaw : List Term := Proof.Events087.exact22517RawTerms
def rightRaw : List Term := Proof.Events087.exact22494RawTerms
def group : MergeGroup := .operator 22517 22494
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22517) (leftOrdinal := 0)
    (rightResult := 22494) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61622⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61622⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22524

namespace LeftMerge22532
def owner : Owner := ⟨.program ⟨257⟩, ⟨59937⟩⟩
def mergeEvent : Nat := 22532
def frameStart : Nat := 22443
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59935⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events087.exact22505RawTerms
def rightRaw : List Term := Proof.Events088.exact22528RawTerms
def group : MergeGroup := .operator 22505 22528
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22505) (leftOrdinal := 0)
    (rightResult := 22528) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59935⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22532

namespace LeftMerge22549
def owner : Owner := ⟨.program ⟨257⟩, ⟨60525⟩⟩
def mergeEvent : Nat := 22549
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59758⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61023⟩⟩] } }
def rhsRaw : List Term := Proof.Events088.exact22546RawTerms
def group : MergeGroup := .relation 22548
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 22548) (rhsResult := 22546)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60522⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 22547 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60522⟩⟩]⟩) (none) 22546) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59758⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61023⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59758⟩⟩], [⟨.program ⟨257⟩, ⟨61023⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22549

namespace LeftMerge22550
def owner : Owner := ⟨.program ⟨257⟩, ⟨60525⟩⟩
def mergeEvent : Nat := 22550
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61622⟩⟩] } }
def rhsRaw : List Term := Proof.Events088.exact22546RawTerms
def group : MergeGroup := .relation 22548
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 22548) (rhsResult := 22546)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60522⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 22547 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60522⟩⟩]⟩) (none) 22546) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61622⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61622⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22550

namespace LeftMerge22551
def owner : Owner := ⟨.program ⟨257⟩, ⟨60525⟩⟩
def mergeEvent : Nat := 22551
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59935⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events088.exact22546RawTerms
def group : MergeGroup := .relation 22548
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 22548) (rhsResult := 22546)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60522⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 22547 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60522⟩⟩]⟩) (none) 22546) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59935⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22551

namespace LeftMerge22552
def owner : Owner := ⟨.program ⟨257⟩, ⟨60525⟩⟩
def mergeEvent : Nat := 22552
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }
def rhsRaw : List Term := Proof.Events088.exact22546RawTerms
def group : MergeGroup := .relation 22548
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 22548) (rhsResult := 22546)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60522⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 22547 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60522⟩⟩]⟩) (none) 22546) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22552

namespace LeftMerge22557
def owner : Owner := ⟨.program ⟨257⟩, ⟨61625⟩⟩
def mergeEvent : Nat := 22557
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59758⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61023⟩⟩] } }
def leftRaw : List Term := Proof.Events088.exact22553RawTerms
def rightRaw : List Term := Proof.Events087.exact22375RawTerms
def group : MergeGroup := .operator 22553 22375
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22553) (leftOrdinal := 2)
    (rightResult := 22375) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59758⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61023⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59758⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61023⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59758⟩⟩], [⟨.program ⟨257⟩, ⟨61023⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22557

namespace LeftMerge22558
def owner : Owner := ⟨.program ⟨257⟩, ⟨61625⟩⟩
def mergeEvent : Nat := 22558
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61622⟩⟩] } }
def leftRaw : List Term := Proof.Events088.exact22553RawTerms
def rightRaw : List Term := Proof.Events087.exact22375RawTerms
def group : MergeGroup := .operator 22553 22375
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22553) (leftOrdinal := 0)
    (rightResult := 22375) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61622⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61622⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61622⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22558

namespace LeftMerge22587
def owner : Owner := ⟨.program ⟨257⟩, ⟨24907⟩⟩
def mergeEvent : Nat := 22587
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24906⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events001.exact304RawTerms
def rightRaw : List Term := Proof.Events066.exact17057RawTerms
def group : MergeGroup := .operator 304 17057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304) (leftOrdinal := 0)
    (rightResult := 17057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24906⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24906⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22587

namespace LeftMerge22595
def owner : Owner := ⟨.program ⟨257⟩, ⟨7591⟩⟩
def mergeEvent : Nat := 22595
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }
def leftRaw : List Term := Proof.Events066.exact16922RawTerms
def rightRaw : List Term := Proof.Events088.exact22591RawTerms
def group : MergeGroup := .operator 16922 22591
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16922) (leftOrdinal := 0)
    (rightResult := 22591) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22595

namespace LeftMerge22612
def owner : Owner := ⟨.program ⟨257⟩, ⟨56274⟩⟩
def mergeEvent : Nat := 22612
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24906⟩⟩, ⟨.program ⟨257⟩, ⟨56271⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events088.exact22606RawTerms
def rightRaw : List Term := Proof.Events001.exact307RawTerms
def group : MergeGroup := .operator 22606 307
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22606) (leftOrdinal := 1)
    (rightResult := 307) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24906⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56271⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24906⟩⟩, ⟨.program ⟨257⟩, ⟨56271⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22612

namespace LeftMerge22613
def owner : Owner := ⟨.program ⟨257⟩, ⟨56274⟩⟩
def mergeEvent : Nat := 22613
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56271⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }
def leftRaw : List Term := Proof.Events088.exact22606RawTerms
def rightRaw : List Term := Proof.Events001.exact307RawTerms
def group : MergeGroup := .operator 22606 307
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22606) (leftOrdinal := 0)
    (rightResult := 307) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56271⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56271⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22613

namespace LeftMerge22628
def owner : Owner := ⟨.program ⟨257⟩, ⟨56275⟩⟩
def mergeEvent : Nat := 22628
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56271⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events001.exact307RawTerms
def rightRaw : List Term := Proof.Events066.exact17057RawTerms
def group : MergeGroup := .operator 307 17057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 307) (leftOrdinal := 0)
    (rightResult := 17057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56271⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56271⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22628

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
