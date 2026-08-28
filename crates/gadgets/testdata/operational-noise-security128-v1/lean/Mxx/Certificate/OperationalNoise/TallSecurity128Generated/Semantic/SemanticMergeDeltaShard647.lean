import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge107179
def owner : Owner := ⟨.program ⟨257⟩, ⟨37882⟩⟩
def mergeEvent : Nat := 107179
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37879⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105245RawTerms
def rightRaw : List Term := Proof.Events418.exact107173RawTerms
def group : MergeGroup := .operator 105245 107173
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105245) (leftOrdinal := 0)
    (rightResult := 107173) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨37879⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37879⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge107179

namespace LeftMerge107258
def owner : Owner := ⟨.program ⟨257⟩, ⟨37139⟩⟩
def mergeEvent : Nat := 107258
def frameStart : Nat := 107228
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events418.exact107254RawTerms
def rightRaw : List Term := Proof.Events418.exact107251RawTerms
def group : MergeGroup := .operator 107254 107251
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 107254) (leftOrdinal := 0)
    (rightResult := 107251) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13896⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37138⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge107258

namespace LeftMerge107288
def owner : Owner := ⟨.program ⟨257⟩, ⟨38712⟩⟩
def mergeEvent : Nat := 107288
def frameStart : Nat := 107228
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events419.exact107284RawTerms
def rightRaw : List Term := Proof.Events419.exact107282RawTerms
def group : MergeGroup := .operator 107284 107282
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 107284) (leftOrdinal := 0)
    (rightResult := 107282) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge107288

namespace LeftMerge107311
def owner : Owner := ⟨.program ⟨257⟩, ⟨9555⟩⟩
def mergeEvent : Nat := 107311
def frameStart : Nat := 107228
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }
def leftRaw : List Term := Proof.Events419.exact107307RawTerms
def rightRaw : List Term := Proof.Events419.exact107304RawTerms
def group : MergeGroup := .operator 107307 107304
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 107307) (leftOrdinal := 0)
    (rightResult := 107304) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9553⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge107311

namespace LeftMerge107320
def owner : Owner := ⟨.program ⟨257⟩, ⟨38953⟩⟩
def mergeEvent : Nat := 107320
def frameStart : Nat := 107228
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38950⟩⟩] } }
def leftRaw : List Term := Proof.Events419.exact107316RawTerms
def rightRaw : List Term := Proof.Events419.exact107273RawTerms
def group : MergeGroup := .operator 107316 107273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 107316) (leftOrdinal := 0)
    (rightResult := 107273) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38950⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38950⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge107320

namespace LeftMerge107321
def owner : Owner := ⟨.program ⟨257⟩, ⟨38953⟩⟩
def mergeEvent : Nat := 107321
def frameStart : Nat := 107228
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38950⟩⟩] } }
def leftRaw : List Term := Proof.Events419.exact107316RawTerms
def rightRaw : List Term := Proof.Events419.exact107273RawTerms
def group : MergeGroup := .operator 107316 107273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 107316) (leftOrdinal := 1)
    (rightResult := 107273) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38950⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38950⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge107321

namespace LeftMerge107323
def owner : Owner := ⟨.program ⟨257⟩, ⟨38953⟩⟩
def mergeEvent : Nat := 107323
def frameStart : Nat := 107228
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38435⟩⟩] } }
def rhsRaw : List Term := Proof.Events419.exact107270RawTerms
def group : MergeGroup := .relation 107322
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 107322) (rhsResult := 107270)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38950⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38950⟩⟩) ⟨38435⟩ 107270) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38435⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], [⟨.program ⟨257⟩, ⟨38435⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge107323

namespace LeftMerge107331
def owner : Owner := ⟨.program ⟨257⟩, ⟨37438⟩⟩
def mergeEvent : Nat := 107331
def frameStart : Nat := 107228
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events419.exact107284RawTerms
def rightRaw : List Term := Proof.Events419.exact107327RawTerms
def group : MergeGroup := .operator 107284 107327
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 107284) (leftOrdinal := 0)
    (rightResult := 107327) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37436⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge107331

namespace LeftMerge107348
def owner : Owner := ⟨.program ⟨257⟩, ⟨37882⟩⟩
def mergeEvent : Nat := 107348
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }
def rhsRaw : List Term := Proof.Events419.exact107345RawTerms
def group : MergeGroup := .relation 107347
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 107347) (rhsResult := 107345)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37879⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 107346 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37879⟩⟩]⟩) (none) 107345) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge107348

namespace LeftMerge107349
def owner : Owner := ⟨.program ⟨257⟩, ⟨37882⟩⟩
def mergeEvent : Nat := 107349
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38950⟩⟩] } }
def rhsRaw : List Term := Proof.Events419.exact107345RawTerms
def group : MergeGroup := .relation 107347
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 107347) (rhsResult := 107345)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37879⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 107346 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37879⟩⟩]⟩) (none) 107345) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38950⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38950⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge107349

namespace LeftMerge107350
def owner : Owner := ⟨.program ⟨257⟩, ⟨37882⟩⟩
def mergeEvent : Nat := 107350
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38435⟩⟩] } }
def rhsRaw : List Term := Proof.Events419.exact107345RawTerms
def group : MergeGroup := .relation 107347
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 107347) (rhsResult := 107345)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37879⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 107346 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37879⟩⟩]⟩) (none) 107345) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38435⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], [⟨.program ⟨257⟩, ⟨38435⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge107350

namespace LeftMerge107351
def owner : Owner := ⟨.program ⟨257⟩, ⟨37882⟩⟩
def mergeEvent : Nat := 107351
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events419.exact107345RawTerms
def group : MergeGroup := .relation 107347
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 107347) (rhsResult := 107345)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37879⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 107346 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37879⟩⟩]⟩) (none) 107345) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge107351

namespace LeftMerge107356
def owner : Owner := ⟨.program ⟨257⟩, ⟨38952⟩⟩
def mergeEvent : Nat := 107356
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38435⟩⟩] } }
def leftRaw : List Term := Proof.Events419.exact107352RawTerms
def rightRaw : List Term := Proof.Events418.exact107166RawTerms
def group : MergeGroup := .operator 107352 107166
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 107352) (leftOrdinal := 2)
    (rightResult := 107166) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38435⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38435⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], [⟨.program ⟨257⟩, ⟨38435⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge107356

namespace LeftMerge107357
def owner : Owner := ⟨.program ⟨257⟩, ⟨38952⟩⟩
def mergeEvent : Nat := 107357
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38950⟩⟩] } }
def leftRaw : List Term := Proof.Events419.exact107352RawTerms
def rightRaw : List Term := Proof.Events418.exact107166RawTerms
def group : MergeGroup := .operator 107352 107166
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 107352) (leftOrdinal := 1)
    (rightResult := 107166) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38950⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38950⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38950⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge107357

namespace LeftMerge107365
def owner : Owner := ⟨.program ⟨257⟩, ⟨39336⟩⟩
def mergeEvent : Nat := 107365
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39334⟩⟩] } }
def leftRaw : List Term := Proof.Events419.exact107359RawTerms
def rightRaw : List Term := Proof.Events418.exact107082RawTerms
def group : MergeGroup := .operator 107359 107082
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 107359) (leftOrdinal := 0)
    (rightResult := 107082) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39334⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39334⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge107365

namespace LeftMerge107366
def owner : Owner := ⟨.program ⟨257⟩, ⟨39336⟩⟩
def mergeEvent : Nat := 107366
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39334⟩⟩] } }
def leftRaw : List Term := Proof.Events419.exact107359RawTerms
def rightRaw : List Term := Proof.Events418.exact107082RawTerms
def group : MergeGroup := .operator 107359 107082
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 107359) (leftOrdinal := 1)
    (rightResult := 107082) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39334⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39334⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge107366

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
