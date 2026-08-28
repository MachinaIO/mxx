import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge50269
def owner : Owner := ⟨.program ⟨257⟩, ⟨28010⟩⟩
def mergeEvent : Nat := 50269
def frameStart : Nat := 50174
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27457⟩⟩] } }
def rhsRaw : List Term := Proof.Events196.exact50216RawTerms
def group : MergeGroup := .relation 50268
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50268) (rhsResult := 50216)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28007⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28007⟩⟩) ⟨27457⟩ 50216) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27457⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], [⟨.program ⟨257⟩, ⟨27457⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50269

namespace LeftMerge50277
def owner : Owner := ⟨.program ⟨257⟩, ⟨26474⟩⟩
def mergeEvent : Nat := 50277
def frameStart : Nat := 50174
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26472⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events196.exact50230RawTerms
def rightRaw : List Term := Proof.Events196.exact50273RawTerms
def group : MergeGroup := .operator 50230 50273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50230) (leftOrdinal := 0)
    (rightResult := 50273) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26472⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50277

namespace LeftMerge50294
def owner : Owner := ⟨.program ⟨257⟩, ⟨26932⟩⟩
def mergeEvent : Nat := 50294
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }
def rhsRaw : List Term := Proof.Events196.exact50291RawTerms
def group : MergeGroup := .relation 50293
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50293) (rhsResult := 50291)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26929⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 50292 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26929⟩⟩]⟩) (none) 50291) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50294

namespace LeftMerge50295
def owner : Owner := ⟨.program ⟨257⟩, ⟨26932⟩⟩
def mergeEvent : Nat := 50295
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨28007⟩⟩] } }
def rhsRaw : List Term := Proof.Events196.exact50291RawTerms
def group : MergeGroup := .relation 50293
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50293) (rhsResult := 50291)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26929⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 50292 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26929⟩⟩]⟩) (none) 50291) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨28007⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨28007⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50295

namespace LeftMerge50296
def owner : Owner := ⟨.program ⟨257⟩, ⟨26932⟩⟩
def mergeEvent : Nat := 50296
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27457⟩⟩] } }
def rhsRaw : List Term := Proof.Events196.exact50291RawTerms
def group : MergeGroup := .relation 50293
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50293) (rhsResult := 50291)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26929⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 50292 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26929⟩⟩]⟩) (none) 50291) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27457⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], [⟨.program ⟨257⟩, ⟨27457⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50296

namespace LeftMerge50297
def owner : Owner := ⟨.program ⟨257⟩, ⟨26932⟩⟩
def mergeEvent : Nat := 50297
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26472⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events196.exact50291RawTerms
def group : MergeGroup := .relation 50293
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50293) (rhsResult := 50291)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26929⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 50292 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26929⟩⟩]⟩) (none) 50291) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26472⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50297

namespace LeftMerge50302
def owner : Owner := ⟨.program ⟨257⟩, ⟨28009⟩⟩
def mergeEvent : Nat := 50302
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27457⟩⟩] } }
def leftRaw : List Term := Proof.Events196.exact50298RawTerms
def rightRaw : List Term := Proof.Events195.exact50112RawTerms
def group : MergeGroup := .operator 50298 50112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50298) (leftOrdinal := 2)
    (rightResult := 50112) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27457⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27457⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], [⟨.program ⟨257⟩, ⟨27457⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50302

namespace LeftMerge50303
def owner : Owner := ⟨.program ⟨257⟩, ⟨28009⟩⟩
def mergeEvent : Nat := 50303
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨28007⟩⟩] } }
def leftRaw : List Term := Proof.Events196.exact50298RawTerms
def rightRaw : List Term := Proof.Events195.exact50112RawTerms
def group : MergeGroup := .operator 50298 50112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50298) (leftOrdinal := 1)
    (rightResult := 50112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨28007⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨28007⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨28007⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50303

namespace LeftMerge50311
def owner : Owner := ⟨.program ⟨257⟩, ⟨28491⟩⟩
def mergeEvent : Nat := 50311
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28489⟩⟩] } }
def leftRaw : List Term := Proof.Events196.exact50305RawTerms
def rightRaw : List Term := Proof.Events195.exact50028RawTerms
def group : MergeGroup := .operator 50305 50028
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50305) (leftOrdinal := 0)
    (rightResult := 50028) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28489⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28489⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50311

namespace LeftMerge50312
def owner : Owner := ⟨.program ⟨257⟩, ⟨28491⟩⟩
def mergeEvent : Nat := 50312
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26472⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28489⟩⟩] } }
def leftRaw : List Term := Proof.Events196.exact50305RawTerms
def rightRaw : List Term := Proof.Events195.exact50028RawTerms
def group : MergeGroup := .operator 50305 50028
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50305) (leftOrdinal := 1)
    (rightResult := 50028) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26472⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28489⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28489⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50312

namespace LeftMerge50314
def owner : Owner := ⟨.program ⟨257⟩, ⟨28491⟩⟩
def mergeEvent : Nat := 50314
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26472⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27633⟩⟩] } }
def rhsRaw : List Term := Proof.Events195.exact50025RawTerms
def group : MergeGroup := .relation 50313
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50313) (rhsResult := 50025)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28489⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28489⟩⟩) ⟨27633⟩ 50025) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27633⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨27633⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50314

namespace LeftMerge50328
def owner : Owner := ⟨.program ⟨257⟩, ⟨27319⟩⟩
def mergeEvent : Nat := 50328
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27316⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46745RawTerms
def rightRaw : List Term := Proof.Events196.exact50322RawTerms
def group : MergeGroup := .operator 46745 50322
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46745) (leftOrdinal := 0)
    (rightResult := 50322) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27316⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27316⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50328

namespace LeftMerge50449
def owner : Owner := ⟨.program ⟨257⟩, ⟨27800⟩⟩
def mergeEvent : Nat := 50449
def frameStart : Nat := 50383
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26472⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events197.exact50445RawTerms
def rightRaw : List Term := Proof.Events197.exact50443RawTerms
def group : MergeGroup := .operator 50445 50443
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50445) (leftOrdinal := 0)
    (rightResult := 50443) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26472⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50449

namespace LeftMerge50461
def owner : Owner := ⟨.program ⟨257⟩, ⟨28490⟩⟩
def mergeEvent : Nat := 50461
def frameStart : Nat := 50383
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28489⟩⟩] } }
def leftRaw : List Term := Proof.Events197.exact50457RawTerms
def rightRaw : List Term := Proof.Events197.exact50434RawTerms
def group : MergeGroup := .operator 50457 50434
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50457) (leftOrdinal := 0)
    (rightResult := 50434) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28489⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28489⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50461

namespace LeftMerge50462
def owner : Owner := ⟨.program ⟨257⟩, ⟨28490⟩⟩
def mergeEvent : Nat := 50462
def frameStart : Nat := 50383
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26472⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28489⟩⟩] } }
def leftRaw : List Term := Proof.Events197.exact50457RawTerms
def rightRaw : List Term := Proof.Events197.exact50434RawTerms
def group : MergeGroup := .operator 50457 50434
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50457) (leftOrdinal := 1)
    (rightResult := 50434) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26472⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28489⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28489⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50462

namespace LeftMerge50464
def owner : Owner := ⟨.program ⟨257⟩, ⟨28490⟩⟩
def mergeEvent : Nat := 50464
def frameStart : Nat := 50383
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26472⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27633⟩⟩] } }
def rhsRaw : List Term := Proof.Events196.exact50431RawTerms
def group : MergeGroup := .relation 50463
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50463) (rhsResult := 50431)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28489⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28489⟩⟩) ⟨27633⟩ 50431) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27633⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨27633⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50464

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
