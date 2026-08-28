import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge298324
def owner : Owner := ⟨.program ⟨257⟩, ⟨27648⟩⟩
def mergeEvent : Nat := 298324
def frameStart : Nat := 298276
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1165.exact298320RawTerms
def rightRaw : List Term := Proof.Events1165.exact298318RawTerms
def group : MergeGroup := .operator 298320 298318
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298320) (leftOrdinal := 0)
    (rightResult := 298318) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298324

namespace LeftMerge298347
def owner : Owner := ⟨.program ⟨257⟩, ⟨9546⟩⟩
def mergeEvent : Nat := 298347
def frameStart : Nat := 298276
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }
def leftRaw : List Term := Proof.Events1165.exact298343RawTerms
def rightRaw : List Term := Proof.Events1165.exact298340RawTerms
def group : MergeGroup := .operator 298343 298340
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298343) (leftOrdinal := 0)
    (rightResult := 298340) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9544⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298347

namespace LeftMerge298356
def owner : Owner := ⟨.program ⟨257⟩, ⟨27812⟩⟩
def mergeEvent : Nat := 298356
def frameStart : Nat := 298276
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27809⟩⟩] } }
def leftRaw : List Term := Proof.Events1165.exact298352RawTerms
def rightRaw : List Term := Proof.Events1165.exact298309RawTerms
def group : MergeGroup := .operator 298352 298309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298352) (leftOrdinal := 0)
    (rightResult := 298309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27809⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27809⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298356

namespace LeftMerge298357
def owner : Owner := ⟨.program ⟨257⟩, ⟨27812⟩⟩
def mergeEvent : Nat := 298357
def frameStart : Nat := 298276
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27809⟩⟩] } }
def leftRaw : List Term := Proof.Events1165.exact298352RawTerms
def rightRaw : List Term := Proof.Events1165.exact298309RawTerms
def group : MergeGroup := .operator 298352 298309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298352) (leftOrdinal := 1)
    (rightResult := 298309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27809⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27809⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge298357

namespace LeftMerge298359
def owner : Owner := ⟨.program ⟨257⟩, ⟨27812⟩⟩
def mergeEvent : Nat := 298359
def frameStart : Nat := 298276
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27349⟩⟩] } }
def rhsRaw : List Term := Proof.Events1165.exact298306RawTerms
def group : MergeGroup := .relation 298358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 298358) (rhsResult := 298306)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27809⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27809⟩⟩) ⟨27349⟩ 298306) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27349⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], [⟨.program ⟨257⟩, ⟨27349⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge298359

namespace LeftMerge298367
def owner : Owner := ⟨.program ⟨257⟩, ⟨26330⟩⟩
def mergeEvent : Nat := 298367
def frameStart : Nat := 298276
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26328⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1165.exact298320RawTerms
def rightRaw : List Term := Proof.Events1165.exact298363RawTerms
def group : MergeGroup := .operator 298320 298363
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298320) (leftOrdinal := 0)
    (rightResult := 298363) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26328⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26328⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298367

namespace LeftMerge298384
def owner : Owner := ⟨.program ⟨257⟩, ⟨26752⟩⟩
def mergeEvent : Nat := 298384
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }
def rhsRaw : List Term := Proof.Events1165.exact298381RawTerms
def group : MergeGroup := .relation 298383
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 298383) (rhsResult := 298381)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26749⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 298382 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26749⟩⟩]⟩) (none) 298381) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298384

namespace LeftMerge298385
def owner : Owner := ⟨.program ⟨257⟩, ⟨26752⟩⟩
def mergeEvent : Nat := 298385
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27809⟩⟩] } }
def rhsRaw : List Term := Proof.Events1165.exact298381RawTerms
def group : MergeGroup := .relation 298383
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 298383) (rhsResult := 298381)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26749⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 298382 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26749⟩⟩]⟩) (none) 298381) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27809⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27809⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge298385

namespace LeftMerge298386
def owner : Owner := ⟨.program ⟨257⟩, ⟨26752⟩⟩
def mergeEvent : Nat := 298386
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27349⟩⟩] } }
def rhsRaw : List Term := Proof.Events1165.exact298381RawTerms
def group : MergeGroup := .relation 298383
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 298383) (rhsResult := 298381)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26749⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 298382 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26749⟩⟩]⟩) (none) 298381) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27349⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], [⟨.program ⟨257⟩, ⟨27349⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298386

namespace LeftMerge298387
def owner : Owner := ⟨.program ⟨257⟩, ⟨26752⟩⟩
def mergeEvent : Nat := 298387
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨26328⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1165.exact298381RawTerms
def group : MergeGroup := .relation 298383
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 298383) (rhsResult := 298381)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26749⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 298382 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26749⟩⟩]⟩) (none) 298381) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26328⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨26328⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge298387

namespace LeftMerge298392
def owner : Owner := ⟨.program ⟨257⟩, ⟨27811⟩⟩
def mergeEvent : Nat := 298392
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27349⟩⟩] } }
def leftRaw : List Term := Proof.Events1165.exact298388RawTerms
def rightRaw : List Term := Proof.Events1164.exact298226RawTerms
def group : MergeGroup := .operator 298388 298226
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298388) (leftOrdinal := 2)
    (rightResult := 298226) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27349⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27349⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], [⟨.program ⟨257⟩, ⟨27349⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge298392

namespace LeftMerge298393
def owner : Owner := ⟨.program ⟨257⟩, ⟨27811⟩⟩
def mergeEvent : Nat := 298393
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27809⟩⟩] } }
def leftRaw : List Term := Proof.Events1165.exact298388RawTerms
def rightRaw : List Term := Proof.Events1164.exact298226RawTerms
def group : MergeGroup := .operator 298388 298226
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298388) (leftOrdinal := 1)
    (rightResult := 298226) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27809⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27809⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27809⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298393

namespace LeftMerge298401
def owner : Owner := ⟨.program ⟨257⟩, ⟨28041⟩⟩
def mergeEvent : Nat := 298401
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28039⟩⟩] } }
def leftRaw : List Term := Proof.Events1165.exact298395RawTerms
def rightRaw : List Term := Proof.Events1164.exact298142RawTerms
def group : MergeGroup := .operator 298395 298142
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298395) (leftOrdinal := 0)
    (rightResult := 298142) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28039⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28039⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298401

namespace LeftMerge298402
def owner : Owner := ⟨.program ⟨257⟩, ⟨28041⟩⟩
def mergeEvent : Nat := 298402
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨26328⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28039⟩⟩] } }
def leftRaw : List Term := Proof.Events1165.exact298395RawTerms
def rightRaw : List Term := Proof.Events1164.exact298142RawTerms
def group : MergeGroup := .operator 298395 298142
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298395) (leftOrdinal := 1)
    (rightResult := 298142) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨26328⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28039⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨26328⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28039⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge298402

namespace LeftMerge298404
def owner : Owner := ⟨.program ⟨257⟩, ⟨28041⟩⟩
def mergeEvent : Nat := 298404
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨26328⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27471⟩⟩] } }
def rhsRaw : List Term := Proof.Events1164.exact298139RawTerms
def group : MergeGroup := .relation 298403
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 298403) (rhsResult := 298139)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨26328⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28039⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28039⟩⟩) ⟨27471⟩ 298139) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27471⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨26328⟩⟩], [⟨.program ⟨257⟩, ⟨27471⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge298404

namespace LeftMerge298418
def owner : Owner := ⟨.program ⟨257⟩, ⟨26959⟩⟩
def mergeEvent : Nat := 298418
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26956⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1165.exact298412RawTerms
def group : MergeGroup := .operator 295195 298412
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 298412) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨26956⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26956⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298418

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
