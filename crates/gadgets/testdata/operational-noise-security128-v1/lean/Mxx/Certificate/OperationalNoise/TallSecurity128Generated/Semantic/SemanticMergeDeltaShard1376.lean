import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge223190
def owner : Owner := ⟨.program ⟨257⟩, ⟨42457⟩⟩
def mergeEvent : Nat := 223190
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }
def leftRaw : List Term := Proof.Events871.exact223186RawTerms
def rightRaw : List Term := Proof.Events871.exact223156RawTerms
def group : MergeGroup := .operator 223186 223156
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223186) (leftOrdinal := 1)
    (rightResult := 223156) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223190

namespace LeftMerge223198
def owner : Owner := ⟨.program ⟨257⟩, ⟨44289⟩⟩
def mergeEvent : Nat := 223198
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44288⟩⟩] } }
def leftRaw : List Term := Proof.Events871.exact223192RawTerms
def rightRaw : List Term := Proof.Events871.exact223128RawTerms
def group : MergeGroup := .operator 223192 223128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223192) (leftOrdinal := 1)
    (rightResult := 223128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44288⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44288⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223198

namespace LeftMerge223200
def owner : Owner := ⟨.program ⟨257⟩, ⟨44289⟩⟩
def mergeEvent : Nat := 223200
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43783⟩⟩] } }
def rhsRaw : List Term := Proof.Events871.exact223125RawTerms
def group : MergeGroup := .relation 223199
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 223199) (rhsResult := 223125)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44288⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44288⟩⟩) ⟨43783⟩ 223125) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43783⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], [⟨.program ⟨257⟩, ⟨43783⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223200

namespace LeftMerge223201
def owner : Owner := ⟨.program ⟨257⟩, ⟨44289⟩⟩
def mergeEvent : Nat := 223201
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44288⟩⟩] } }
def leftRaw : List Term := Proof.Events871.exact223192RawTerms
def rightRaw : List Term := Proof.Events871.exact223128RawTerms
def group : MergeGroup := .operator 223192 223128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223192) (leftOrdinal := 0)
    (rightResult := 223128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44288⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44288⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223201

namespace LeftMerge223215
def owner : Owner := ⟨.program ⟨257⟩, ⟨43222⟩⟩
def mergeEvent : Nat := 223215
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43219⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events871.exact223209RawTerms
def group : MergeGroup := .operator 222245 223209
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 223209) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43219⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43219⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223215

namespace LeftMerge223294
def owner : Owner := ⟨.program ⟨257⟩, ⟨42451⟩⟩
def mergeEvent : Nat := 223294
def frameStart : Nat := 223264
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events872.exact223290RawTerms
def rightRaw : List Term := Proof.Events872.exact223287RawTerms
def group : MergeGroup := .operator 223290 223287
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223290) (leftOrdinal := 0)
    (rightResult := 223287) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14466⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42450⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223294

namespace LeftMerge223324
def owner : Owner := ⟨.program ⟨257⟩, ⟨44064⟩⟩
def mergeEvent : Nat := 223324
def frameStart : Nat := 223264
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events872.exact223320RawTerms
def rightRaw : List Term := Proof.Events872.exact223318RawTerms
def group : MergeGroup := .operator 223320 223318
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223320) (leftOrdinal := 0)
    (rightResult := 223318) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223324

namespace LeftMerge223347
def owner : Owner := ⟨.program ⟨257⟩, ⟨9561⟩⟩
def mergeEvent : Nat := 223347
def frameStart : Nat := 223264
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }
def leftRaw : List Term := Proof.Events872.exact223343RawTerms
def rightRaw : List Term := Proof.Events872.exact223340RawTerms
def group : MergeGroup := .operator 223343 223340
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223343) (leftOrdinal := 0)
    (rightResult := 223340) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9559⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223347

namespace LeftMerge223356
def owner : Owner := ⟨.program ⟨257⟩, ⟨44291⟩⟩
def mergeEvent : Nat := 223356
def frameStart : Nat := 223264
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44288⟩⟩] } }
def leftRaw : List Term := Proof.Events872.exact223352RawTerms
def rightRaw : List Term := Proof.Events872.exact223309RawTerms
def group : MergeGroup := .operator 223352 223309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223352) (leftOrdinal := 0)
    (rightResult := 223309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44288⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44288⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223356

namespace LeftMerge223357
def owner : Owner := ⟨.program ⟨257⟩, ⟨44291⟩⟩
def mergeEvent : Nat := 223357
def frameStart : Nat := 223264
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44288⟩⟩] } }
def leftRaw : List Term := Proof.Events872.exact223352RawTerms
def rightRaw : List Term := Proof.Events872.exact223309RawTerms
def group : MergeGroup := .operator 223352 223309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223352) (leftOrdinal := 1)
    (rightResult := 223309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44288⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44288⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223357

namespace LeftMerge223359
def owner : Owner := ⟨.program ⟨257⟩, ⟨44291⟩⟩
def mergeEvent : Nat := 223359
def frameStart : Nat := 223264
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43783⟩⟩] } }
def rhsRaw : List Term := Proof.Events872.exact223306RawTerms
def group : MergeGroup := .relation 223358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 223358) (rhsResult := 223306)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44288⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44288⟩⟩) ⟨43783⟩ 223306) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43783⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], [⟨.program ⟨257⟩, ⟨43783⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223359

namespace LeftMerge223367
def owner : Owner := ⟨.program ⟨257⟩, ⟨42782⟩⟩
def mergeEvent : Nat := 223367
def frameStart : Nat := 223264
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events872.exact223320RawTerms
def rightRaw : List Term := Proof.Events872.exact223363RawTerms
def group : MergeGroup := .operator 223320 223363
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223320) (leftOrdinal := 0)
    (rightResult := 223363) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42780⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223367

namespace LeftMerge223384
def owner : Owner := ⟨.program ⟨257⟩, ⟨43222⟩⟩
def mergeEvent : Nat := 223384
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }
def rhsRaw : List Term := Proof.Events872.exact223381RawTerms
def group : MergeGroup := .relation 223383
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 223383) (rhsResult := 223381)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43219⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 223382 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43219⟩⟩]⟩) (none) 223381) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223384

namespace LeftMerge223385
def owner : Owner := ⟨.program ⟨257⟩, ⟨43222⟩⟩
def mergeEvent : Nat := 223385
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44288⟩⟩] } }
def rhsRaw : List Term := Proof.Events872.exact223381RawTerms
def group : MergeGroup := .relation 223383
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 223383) (rhsResult := 223381)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43219⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 223382 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43219⟩⟩]⟩) (none) 223381) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44288⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44288⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223385

namespace LeftMerge223386
def owner : Owner := ⟨.program ⟨257⟩, ⟨43222⟩⟩
def mergeEvent : Nat := 223386
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43783⟩⟩] } }
def rhsRaw : List Term := Proof.Events872.exact223381RawTerms
def group : MergeGroup := .relation 223383
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 223383) (rhsResult := 223381)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43219⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 223382 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43219⟩⟩]⟩) (none) 223381) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43783⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], [⟨.program ⟨257⟩, ⟨43783⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223386

namespace LeftMerge223387
def owner : Owner := ⟨.program ⟨257⟩, ⟨43222⟩⟩
def mergeEvent : Nat := 223387
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨42780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events872.exact223381RawTerms
def group : MergeGroup := .relation 223383
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 223383) (rhsResult := 223381)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43219⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 223382 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43219⟩⟩]⟩) (none) 223381) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨42780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223387

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
