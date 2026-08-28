import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge123392
def owner : Owner := ⟨.program ⟨257⟩, ⟨27878⟩⟩
def mergeEvent : Nat := 123392
def frameStart : Nat := 123299
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12921⟩⟩, ⟨.program ⟨257⟩, ⟨25998⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27875⟩⟩] } }
def leftRaw : List Term := Proof.Events481.exact123387RawTerms
def rightRaw : List Term := Proof.Events481.exact123344RawTerms
def group : MergeGroup := .operator 123387 123344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 123387) (leftOrdinal := 1)
    (rightResult := 123344) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12921⟩⟩, ⟨.program ⟨257⟩, ⟨25998⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27875⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12921⟩⟩, ⟨.program ⟨257⟩, ⟨25998⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27875⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge123392

namespace LeftMerge123394
def owner : Owner := ⟨.program ⟨257⟩, ⟨27878⟩⟩
def mergeEvent : Nat := 123394
def frameStart : Nat := 123299
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12921⟩⟩, ⟨.program ⟨257⟩, ⟨25998⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27385⟩⟩] } }
def rhsRaw : List Term := Proof.Events481.exact123341RawTerms
def group : MergeGroup := .relation 123393
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 123393) (rhsResult := 123341)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12921⟩⟩, ⟨.program ⟨257⟩, ⟨25998⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27875⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27875⟩⟩) ⟨27385⟩ 123341) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27385⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12921⟩⟩, ⟨.program ⟨257⟩, ⟨25998⟩⟩], [⟨.program ⟨257⟩, ⟨27385⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge123394

namespace LeftMerge123402
def owner : Owner := ⟨.program ⟨257⟩, ⟨26378⟩⟩
def mergeEvent : Nat := 123402
def frameStart : Nat := 123299
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26376⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events481.exact123355RawTerms
def rightRaw : List Term := Proof.Events482.exact123398RawTerms
def group : MergeGroup := .operator 123355 123398
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 123355) (leftOrdinal := 0)
    (rightResult := 123398) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26376⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26376⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge123402

namespace LeftMerge123419
def owner : Owner := ⟨.program ⟨257⟩, ⟨26812⟩⟩
def mergeEvent : Nat := 123419
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }
def rhsRaw : List Term := Proof.Events482.exact123416RawTerms
def group : MergeGroup := .relation 123418
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 123418) (rhsResult := 123416)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26809⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 123417 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26809⟩⟩]⟩) (none) 123416) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge123419

namespace LeftMerge123420
def owner : Owner := ⟨.program ⟨257⟩, ⟨26812⟩⟩
def mergeEvent : Nat := 123420
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27875⟩⟩] } }
def rhsRaw : List Term := Proof.Events482.exact123416RawTerms
def group : MergeGroup := .relation 123418
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 123418) (rhsResult := 123416)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26809⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 123417 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26809⟩⟩]⟩) (none) 123416) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27875⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27875⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge123420

namespace LeftMerge123421
def owner : Owner := ⟨.program ⟨257⟩, ⟨26812⟩⟩
def mergeEvent : Nat := 123421
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12921⟩⟩, ⟨.program ⟨257⟩, ⟨25998⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27385⟩⟩] } }
def rhsRaw : List Term := Proof.Events482.exact123416RawTerms
def group : MergeGroup := .relation 123418
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 123418) (rhsResult := 123416)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26809⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 123417 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26809⟩⟩]⟩) (none) 123416) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12921⟩⟩, ⟨.program ⟨257⟩, ⟨25998⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27385⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12921⟩⟩, ⟨.program ⟨257⟩, ⟨25998⟩⟩], [⟨.program ⟨257⟩, ⟨27385⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge123421

namespace LeftMerge123422
def owner : Owner := ⟨.program ⟨257⟩, ⟨26812⟩⟩
def mergeEvent : Nat := 123422
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨26376⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events482.exact123416RawTerms
def group : MergeGroup := .relation 123418
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 123418) (rhsResult := 123416)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26809⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 123417 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26809⟩⟩]⟩) (none) 123416) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26376⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨26376⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge123422

namespace LeftMerge123427
def owner : Owner := ⟨.program ⟨257⟩, ⟨27877⟩⟩
def mergeEvent : Nat := 123427
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12921⟩⟩, ⟨.program ⟨257⟩, ⟨25998⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27385⟩⟩] } }
def leftRaw : List Term := Proof.Events482.exact123423RawTerms
def rightRaw : List Term := Proof.Events481.exact123237RawTerms
def group : MergeGroup := .operator 123423 123237
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 123423) (leftOrdinal := 2)
    (rightResult := 123237) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12921⟩⟩, ⟨.program ⟨257⟩, ⟨25998⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27385⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12921⟩⟩, ⟨.program ⟨257⟩, ⟨25998⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27385⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12921⟩⟩, ⟨.program ⟨257⟩, ⟨25998⟩⟩], [⟨.program ⟨257⟩, ⟨27385⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge123427

namespace LeftMerge123428
def owner : Owner := ⟨.program ⟨257⟩, ⟨27877⟩⟩
def mergeEvent : Nat := 123428
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27875⟩⟩] } }
def leftRaw : List Term := Proof.Events482.exact123423RawTerms
def rightRaw : List Term := Proof.Events481.exact123237RawTerms
def group : MergeGroup := .operator 123423 123237
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 123423) (leftOrdinal := 1)
    (rightResult := 123237) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27875⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27875⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27875⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge123428

namespace LeftMerge123436
def owner : Owner := ⟨.program ⟨257⟩, ⟨28191⟩⟩
def mergeEvent : Nat := 123436
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28189⟩⟩] } }
def leftRaw : List Term := Proof.Events482.exact123430RawTerms
def rightRaw : List Term := Proof.Events481.exact123153RawTerms
def group : MergeGroup := .operator 123430 123153
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 123430) (leftOrdinal := 0)
    (rightResult := 123153) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28189⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge123436

namespace LeftMerge123437
def owner : Owner := ⟨.program ⟨257⟩, ⟨28191⟩⟩
def mergeEvent : Nat := 123437
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨26376⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28189⟩⟩] } }
def leftRaw : List Term := Proof.Events482.exact123430RawTerms
def rightRaw : List Term := Proof.Events481.exact123153RawTerms
def group : MergeGroup := .operator 123430 123153
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 123430) (leftOrdinal := 1)
    (rightResult := 123153) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨26376⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28189⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨26376⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28189⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge123437

namespace LeftMerge123439
def owner : Owner := ⟨.program ⟨257⟩, ⟨28191⟩⟩
def mergeEvent : Nat := 123439
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨26376⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27525⟩⟩] } }
def rhsRaw : List Term := Proof.Events481.exact123150RawTerms
def group : MergeGroup := .relation 123438
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 123438) (rhsResult := 123150)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨26376⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28189⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28189⟩⟩) ⟨27525⟩ 123150) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27525⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨26376⟩⟩], [⟨.program ⟨257⟩, ⟨27525⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge123439

namespace LeftMerge123453
def owner : Owner := ⟨.program ⟨257⟩, ⟨27079⟩⟩
def mergeEvent : Nat := 123453
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27076⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119870RawTerms
def rightRaw : List Term := Proof.Events482.exact123447RawTerms
def group : MergeGroup := .operator 119870 123447
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119870) (leftOrdinal := 0)
    (rightResult := 123447) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27076⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27076⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge123453

namespace LeftMerge123574
def owner : Owner := ⟨.program ⟨257⟩, ⟨27752⟩⟩
def mergeEvent : Nat := 123574
def frameStart : Nat := 123508
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26376⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events482.exact123570RawTerms
def rightRaw : List Term := Proof.Events482.exact123568RawTerms
def group : MergeGroup := .operator 123570 123568
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 123570) (leftOrdinal := 0)
    (rightResult := 123568) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26376⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26376⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge123574

namespace LeftMerge123586
def owner : Owner := ⟨.program ⟨257⟩, ⟨28190⟩⟩
def mergeEvent : Nat := 123586
def frameStart : Nat := 123508
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28189⟩⟩] } }
def leftRaw : List Term := Proof.Events482.exact123582RawTerms
def rightRaw : List Term := Proof.Events482.exact123559RawTerms
def group : MergeGroup := .operator 123582 123559
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 123582) (leftOrdinal := 0)
    (rightResult := 123559) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28189⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge123586

namespace LeftMerge123587
def owner : Owner := ⟨.program ⟨257⟩, ⟨28190⟩⟩
def mergeEvent : Nat := 123587
def frameStart : Nat := 123508
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26376⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28189⟩⟩] } }
def leftRaw : List Term := Proof.Events482.exact123582RawTerms
def rightRaw : List Term := Proof.Events482.exact123559RawTerms
def group : MergeGroup := .operator 123582 123559
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 123582) (leftOrdinal := 1)
    (rightResult := 123559) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26376⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28189⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26376⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28189⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge123587

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
