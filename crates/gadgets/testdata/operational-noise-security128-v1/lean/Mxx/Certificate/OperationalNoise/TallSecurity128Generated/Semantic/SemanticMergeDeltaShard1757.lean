import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge284253
def owner : Owner := ⟨.program ⟨257⟩, ⟨27856⟩⟩
def mergeEvent : Nat := 284253
def frameStart : Nat := 284160
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27373⟩⟩] } }
def rhsRaw : List Term := Proof.Events1110.exact284202RawTerms
def group : MergeGroup := .relation 284252
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 284252) (rhsResult := 284202)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27853⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27853⟩⟩) ⟨27373⟩ 284202) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27373⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], [⟨.program ⟨257⟩, ⟨27373⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge284253

namespace LeftMerge284261
def owner : Owner := ⟨.program ⟨257⟩, ⟨26362⟩⟩
def mergeEvent : Nat := 284261
def frameStart : Nat := 284160
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26360⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1110.exact284216RawTerms
def rightRaw : List Term := Proof.Events1110.exact284257RawTerms
def group : MergeGroup := .operator 284216 284257
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 284216) (leftOrdinal := 0)
    (rightResult := 284257) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26360⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge284261

namespace LeftMerge284278
def owner : Owner := ⟨.program ⟨257⟩, ⟨26792⟩⟩
def mergeEvent : Nat := 284278
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }
def rhsRaw : List Term := Proof.Events1110.exact284275RawTerms
def group : MergeGroup := .relation 284277
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 284277) (rhsResult := 284275)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26789⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 284276 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26789⟩⟩]⟩) (none) 284275) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge284278

namespace LeftMerge284279
def owner : Owner := ⟨.program ⟨257⟩, ⟨26792⟩⟩
def mergeEvent : Nat := 284279
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27853⟩⟩] } }
def rhsRaw : List Term := Proof.Events1110.exact284275RawTerms
def group : MergeGroup := .relation 284277
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 284277) (rhsResult := 284275)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26789⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 284276 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26789⟩⟩]⟩) (none) 284275) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27853⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27853⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge284279

namespace LeftMerge284280
def owner : Owner := ⟨.program ⟨257⟩, ⟨26792⟩⟩
def mergeEvent : Nat := 284280
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27373⟩⟩] } }
def rhsRaw : List Term := Proof.Events1110.exact284275RawTerms
def group : MergeGroup := .relation 284277
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 284277) (rhsResult := 284275)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26789⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 284276 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26789⟩⟩]⟩) (none) 284275) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27373⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], [⟨.program ⟨257⟩, ⟨27373⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge284280

namespace LeftMerge284281
def owner : Owner := ⟨.program ⟨257⟩, ⟨26792⟩⟩
def mergeEvent : Nat := 284281
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26360⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1110.exact284275RawTerms
def group : MergeGroup := .relation 284277
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 284277) (rhsResult := 284275)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26789⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 284276 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26789⟩⟩]⟩) (none) 284275) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26360⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge284281

namespace LeftMerge284286
def owner : Owner := ⟨.program ⟨257⟩, ⟨27855⟩⟩
def mergeEvent : Nat := 284286
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27373⟩⟩] } }
def leftRaw : List Term := Proof.Events1110.exact284282RawTerms
def rightRaw : List Term := Proof.Events1109.exact284098RawTerms
def group : MergeGroup := .operator 284282 284098
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 284282) (leftOrdinal := 2)
    (rightResult := 284098) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27373⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27373⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], [⟨.program ⟨257⟩, ⟨27373⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge284286

namespace LeftMerge284287
def owner : Owner := ⟨.program ⟨257⟩, ⟨27855⟩⟩
def mergeEvent : Nat := 284287
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27853⟩⟩] } }
def leftRaw : List Term := Proof.Events1110.exact284282RawTerms
def rightRaw : List Term := Proof.Events1109.exact284098RawTerms
def group : MergeGroup := .operator 284282 284098
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 284282) (leftOrdinal := 1)
    (rightResult := 284098) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27853⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27853⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27853⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge284287

namespace LeftMerge284295
def owner : Owner := ⟨.program ⟨257⟩, ⟨28141⟩⟩
def mergeEvent : Nat := 284295
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩] } }
def leftRaw : List Term := Proof.Events1110.exact284289RawTerms
def rightRaw : List Term := Proof.Events1109.exact284014RawTerms
def group : MergeGroup := .operator 284289 284014
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 284289) (leftOrdinal := 0)
    (rightResult := 284014) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28139⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge284295

namespace LeftMerge284296
def owner : Owner := ⟨.program ⟨257⟩, ⟨28141⟩⟩
def mergeEvent : Nat := 284296
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26360⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩] } }
def leftRaw : List Term := Proof.Events1110.exact284289RawTerms
def rightRaw : List Term := Proof.Events1109.exact284014RawTerms
def group : MergeGroup := .operator 284289 284014
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 284289) (leftOrdinal := 1)
    (rightResult := 284014) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26360⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28139⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge284296

namespace LeftMerge284298
def owner : Owner := ⟨.program ⟨257⟩, ⟨28141⟩⟩
def mergeEvent : Nat := 284298
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26360⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27507⟩⟩] } }
def rhsRaw : List Term := Proof.Events1109.exact284011RawTerms
def group : MergeGroup := .relation 284297
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 284297) (rhsResult := 284011)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28139⟩⟩) ⟨27507⟩ 284011) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27507⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨27507⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge284298

namespace LeftMerge284312
def owner : Owner := ⟨.program ⟨257⟩, ⟨27039⟩⟩
def mergeEvent : Nat := 284312
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27036⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280745RawTerms
def rightRaw : List Term := Proof.Events1110.exact284306RawTerms
def group : MergeGroup := .operator 280745 284306
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280745) (leftOrdinal := 0)
    (rightResult := 284306) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27036⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27036⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge284312

namespace LeftMerge284433
def owner : Owner := ⟨.program ⟨257⟩, ⟨27744⟩⟩
def mergeEvent : Nat := 284433
def frameStart : Nat := 284367
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26360⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1111.exact284429RawTerms
def rightRaw : List Term := Proof.Events1111.exact284427RawTerms
def group : MergeGroup := .operator 284429 284427
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 284429) (leftOrdinal := 0)
    (rightResult := 284427) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26360⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge284433

namespace LeftMerge284445
def owner : Owner := ⟨.program ⟨257⟩, ⟨28140⟩⟩
def mergeEvent : Nat := 284445
def frameStart : Nat := 284367
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩] } }
def leftRaw : List Term := Proof.Events1111.exact284441RawTerms
def rightRaw : List Term := Proof.Events1111.exact284418RawTerms
def group : MergeGroup := .operator 284441 284418
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 284441) (leftOrdinal := 0)
    (rightResult := 284418) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28139⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge284445

namespace LeftMerge284446
def owner : Owner := ⟨.program ⟨257⟩, ⟨28140⟩⟩
def mergeEvent : Nat := 284446
def frameStart : Nat := 284367
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26360⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩] } }
def leftRaw : List Term := Proof.Events1111.exact284441RawTerms
def rightRaw : List Term := Proof.Events1111.exact284418RawTerms
def group : MergeGroup := .operator 284441 284418
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 284441) (leftOrdinal := 1)
    (rightResult := 284418) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26360⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28139⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge284446

namespace LeftMerge284448
def owner : Owner := ⟨.program ⟨257⟩, ⟨28140⟩⟩
def mergeEvent : Nat := 284448
def frameStart : Nat := 284367
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26360⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27507⟩⟩] } }
def rhsRaw : List Term := Proof.Events1110.exact284415RawTerms
def group : MergeGroup := .relation 284447
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 284447) (rhsResult := 284415)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28139⟩⟩) ⟨27507⟩ 284415) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27507⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨27507⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge284448

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
