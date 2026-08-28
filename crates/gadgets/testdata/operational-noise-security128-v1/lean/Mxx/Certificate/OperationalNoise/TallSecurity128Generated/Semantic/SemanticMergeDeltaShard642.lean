import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge106356
def owner : Owner := ⟨.program ⟨257⟩, ⟨44313⟩⟩
def mergeEvent : Nat := 106356
def frameStart : Nat := 106264
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩] } }
def leftRaw : List Term := Proof.Events415.exact106352RawTerms
def rightRaw : List Term := Proof.Events415.exact106309RawTerms
def group : MergeGroup := .operator 106352 106309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 106352) (leftOrdinal := 0)
    (rightResult := 106309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44310⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge106356

namespace LeftMerge106357
def owner : Owner := ⟨.program ⟨257⟩, ⟨44313⟩⟩
def mergeEvent : Nat := 106357
def frameStart : Nat := 106264
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩] } }
def leftRaw : List Term := Proof.Events415.exact106352RawTerms
def rightRaw : List Term := Proof.Events415.exact106309RawTerms
def group : MergeGroup := .operator 106352 106309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 106352) (leftOrdinal := 1)
    (rightResult := 106309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44310⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge106357

namespace LeftMerge106359
def owner : Owner := ⟨.program ⟨257⟩, ⟨44313⟩⟩
def mergeEvent : Nat := 106359
def frameStart : Nat := 106264
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43795⟩⟩] } }
def rhsRaw : List Term := Proof.Events415.exact106306RawTerms
def group : MergeGroup := .relation 106358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 106358) (rhsResult := 106306)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44310⟩⟩) ⟨43795⟩ 106306) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43795⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], [⟨.program ⟨257⟩, ⟨43795⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge106359

namespace LeftMerge106367
def owner : Owner := ⟨.program ⟨257⟩, ⟨42798⟩⟩
def mergeEvent : Nat := 106367
def frameStart : Nat := 106264
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events415.exact106320RawTerms
def rightRaw : List Term := Proof.Events415.exact106363RawTerms
def group : MergeGroup := .operator 106320 106363
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 106320) (leftOrdinal := 0)
    (rightResult := 106363) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42796⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge106367

namespace LeftMerge106384
def owner : Owner := ⟨.program ⟨257⟩, ⟨43242⟩⟩
def mergeEvent : Nat := 106384
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }
def rhsRaw : List Term := Proof.Events415.exact106381RawTerms
def group : MergeGroup := .relation 106383
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 106383) (rhsResult := 106381)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43239⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 106382 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43239⟩⟩]⟩) (none) 106381) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge106384

namespace LeftMerge106385
def owner : Owner := ⟨.program ⟨257⟩, ⟨43242⟩⟩
def mergeEvent : Nat := 106385
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩] } }
def rhsRaw : List Term := Proof.Events415.exact106381RawTerms
def group : MergeGroup := .relation 106383
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 106383) (rhsResult := 106381)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43239⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 106382 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43239⟩⟩]⟩) (none) 106381) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge106385

namespace LeftMerge106386
def owner : Owner := ⟨.program ⟨257⟩, ⟨43242⟩⟩
def mergeEvent : Nat := 106386
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43795⟩⟩] } }
def rhsRaw : List Term := Proof.Events415.exact106381RawTerms
def group : MergeGroup := .relation 106383
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 106383) (rhsResult := 106381)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43239⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 106382 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43239⟩⟩]⟩) (none) 106381) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43795⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], [⟨.program ⟨257⟩, ⟨43795⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge106386

namespace LeftMerge106387
def owner : Owner := ⟨.program ⟨257⟩, ⟨43242⟩⟩
def mergeEvent : Nat := 106387
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨42796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events415.exact106381RawTerms
def group : MergeGroup := .relation 106383
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 106383) (rhsResult := 106381)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43239⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 106382 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43239⟩⟩]⟩) (none) 106381) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge106387

namespace LeftMerge106392
def owner : Owner := ⟨.program ⟨257⟩, ⟨44312⟩⟩
def mergeEvent : Nat := 106392
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43795⟩⟩] } }
def leftRaw : List Term := Proof.Events415.exact106388RawTerms
def rightRaw : List Term := Proof.Events414.exact106202RawTerms
def group : MergeGroup := .operator 106388 106202
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 106388) (leftOrdinal := 2)
    (rightResult := 106202) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43795⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43795⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], [⟨.program ⟨257⟩, ⟨43795⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge106392

namespace LeftMerge106393
def owner : Owner := ⟨.program ⟨257⟩, ⟨44312⟩⟩
def mergeEvent : Nat := 106393
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩] } }
def leftRaw : List Term := Proof.Events415.exact106388RawTerms
def rightRaw : List Term := Proof.Events414.exact106202RawTerms
def group : MergeGroup := .operator 106388 106202
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 106388) (leftOrdinal := 1)
    (rightResult := 106202) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge106393

namespace LeftMerge106401
def owner : Owner := ⟨.program ⟨257⟩, ⟨44696⟩⟩
def mergeEvent : Nat := 106401
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩] } }
def leftRaw : List Term := Proof.Events415.exact106395RawTerms
def rightRaw : List Term := Proof.Events414.exact106118RawTerms
def group : MergeGroup := .operator 106395 106118
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 106395) (leftOrdinal := 0)
    (rightResult := 106118) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44694⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge106401

namespace LeftMerge106402
def owner : Owner := ⟨.program ⟨257⟩, ⟨44696⟩⟩
def mergeEvent : Nat := 106402
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨42796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩] } }
def leftRaw : List Term := Proof.Events415.exact106395RawTerms
def rightRaw : List Term := Proof.Events414.exact106118RawTerms
def group : MergeGroup := .operator 106395 106118
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 106395) (leftOrdinal := 1)
    (rightResult := 106118) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨42796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44694⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge106402

namespace LeftMerge106404
def owner : Owner := ⟨.program ⟨257⟩, ⟨44696⟩⟩
def mergeEvent : Nat := 106404
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨42796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43950⟩⟩] } }
def rhsRaw : List Term := Proof.Events414.exact106115RawTerms
def group : MergeGroup := .relation 106403
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 106403) (rhsResult := 106115)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44694⟩⟩) ⟨43950⟩ 106115) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43950⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨43950⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge106404

namespace LeftMerge106418
def owner : Owner := ⟨.program ⟨257⟩, ⟨43559⟩⟩
def mergeEvent : Nat := 106418
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43556⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105245RawTerms
def rightRaw : List Term := Proof.Events415.exact106412RawTerms
def group : MergeGroup := .operator 105245 106412
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105245) (leftOrdinal := 0)
    (rightResult := 106412) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43556⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge106418

namespace LeftMerge106539
def owner : Owner := ⟨.program ⟨257⟩, ⟨44152⟩⟩
def mergeEvent : Nat := 106539
def frameStart : Nat := 106473
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events416.exact106535RawTerms
def rightRaw : List Term := Proof.Events416.exact106533RawTerms
def group : MergeGroup := .operator 106535 106533
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 106535) (leftOrdinal := 0)
    (rightResult := 106533) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42796⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge106539

namespace LeftMerge106551
def owner : Owner := ⟨.program ⟨257⟩, ⟨44695⟩⟩
def mergeEvent : Nat := 106551
def frameStart : Nat := 106473
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩] } }
def leftRaw : List Term := Proof.Events416.exact106547RawTerms
def rightRaw : List Term := Proof.Events416.exact106524RawTerms
def group : MergeGroup := .operator 106547 106524
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 106547) (leftOrdinal := 0)
    (rightResult := 106524) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44694⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge106551

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
