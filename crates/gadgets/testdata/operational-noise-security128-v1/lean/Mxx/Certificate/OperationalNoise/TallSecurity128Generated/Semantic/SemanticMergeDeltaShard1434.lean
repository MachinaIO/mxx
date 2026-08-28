import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge233250
def owner : Owner := ⟨.program ⟨257⟩, ⟨41959⟩⟩
def mergeEvent : Nat := 233250
def frameStart : Nat := 233171
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41958⟩⟩] } }
def leftRaw : List Term := Proof.Events911.exact233245RawTerms
def rightRaw : List Term := Proof.Events911.exact233222RawTerms
def group : MergeGroup := .operator 233245 233222
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 233245) (leftOrdinal := 1)
    (rightResult := 233222) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41958⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41958⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge233250

namespace LeftMerge233252
def owner : Owner := ⟨.program ⟨257⟩, ⟨41959⟩⟩
def mergeEvent : Nat := 233252
def frameStart : Nat := 233171
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41251⟩⟩] } }
def rhsRaw : List Term := Proof.Events911.exact233219RawTerms
def group : MergeGroup := .relation 233251
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 233251) (rhsResult := 233219)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41958⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41958⟩⟩) ⟨41251⟩ 233219) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41251⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40100⟩⟩], [⟨.program ⟨257⟩, ⟨41251⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge233252

namespace LeftMerge233260
def owner : Owner := ⟨.program ⟨257⟩, ⟨40311⟩⟩
def mergeEvent : Nat := 233260
def frameStart : Nat := 233171
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40309⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events911.exact233233RawTerms
def rightRaw : List Term := Proof.Events911.exact233256RawTerms
def group : MergeGroup := .operator 233233 233256
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 233233) (leftOrdinal := 0)
    (rightResult := 233256) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40309⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40309⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge233260

namespace LeftMerge233277
def owner : Owner := ⟨.program ⟨257⟩, ⟨40835⟩⟩
def mergeEvent : Nat := 233277
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩] } }
def rhsRaw : List Term := Proof.Events911.exact233274RawTerms
def group : MergeGroup := .relation 233276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 233276) (rhsResult := 233274)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40832⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 233275 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40832⟩⟩]⟩) (none) 233274) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge233277

namespace LeftMerge233278
def owner : Owner := ⟨.program ⟨257⟩, ⟨40835⟩⟩
def mergeEvent : Nat := 233278
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41958⟩⟩] } }
def rhsRaw : List Term := Proof.Events911.exact233274RawTerms
def group : MergeGroup := .relation 233276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 233276) (rhsResult := 233274)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40832⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 233275 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40832⟩⟩]⟩) (none) 233274) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41958⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41958⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge233278

namespace LeftMerge233279
def owner : Owner := ⟨.program ⟨257⟩, ⟨40835⟩⟩
def mergeEvent : Nat := 233279
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41251⟩⟩] } }
def rhsRaw : List Term := Proof.Events911.exact233274RawTerms
def group : MergeGroup := .relation 233276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 233276) (rhsResult := 233274)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40832⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 233275 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40832⟩⟩]⟩) (none) 233274) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41251⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40100⟩⟩], [⟨.program ⟨257⟩, ⟨41251⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge233279

namespace LeftMerge233280
def owner : Owner := ⟨.program ⟨257⟩, ⟨40835⟩⟩
def mergeEvent : Nat := 233280
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40309⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events911.exact233274RawTerms
def group : MergeGroup := .relation 233276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 233276) (rhsResult := 233274)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40832⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 233275 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40832⟩⟩]⟩) (none) 233274) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40309⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40309⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge233280

namespace LeftMerge233285
def owner : Owner := ⟨.program ⟨257⟩, ⟨41961⟩⟩
def mergeEvent : Nat := 233285
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41958⟩⟩] } }
def leftRaw : List Term := Proof.Events911.exact233281RawTerms
def rightRaw : List Term := Proof.Events910.exact233103RawTerms
def group : MergeGroup := .operator 233281 233103
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 233281) (leftOrdinal := 0)
    (rightResult := 233103) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41958⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41958⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41958⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge233285

namespace LeftMerge233286
def owner : Owner := ⟨.program ⟨257⟩, ⟨41961⟩⟩
def mergeEvent : Nat := 233286
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41251⟩⟩] } }
def leftRaw : List Term := Proof.Events911.exact233281RawTerms
def rightRaw : List Term := Proof.Events910.exact233103RawTerms
def group : MergeGroup := .operator 233281 233103
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 233281) (leftOrdinal := 2)
    (rightResult := 233103) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41251⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41251⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40100⟩⟩], [⟨.program ⟨257⟩, ⟨41251⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge233286

namespace LeftMerge233294
def owner : Owner := ⟨.program ⟨257⟩, ⟨41962⟩⟩
def mergeEvent : Nat := 233294
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩] } }
def leftRaw : List Term := Proof.Events911.exact233288RawTerms
def rightRaw : List Term := Proof.Events060.exact15602RawTerms
def group : MergeGroup := .operator 233288 15602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 233288) (leftOrdinal := 0)
    (rightResult := 15602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7159⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge233294

namespace LeftMerge233295
def owner : Owner := ⟨.program ⟨257⟩, ⟨41962⟩⟩
def mergeEvent : Nat := 233295
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40309⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩] } }
def leftRaw : List Term := Proof.Events911.exact233288RawTerms
def rightRaw : List Term := Proof.Events060.exact15602RawTerms
def group : MergeGroup := .operator 233288 15602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 233288) (leftOrdinal := 1)
    (rightResult := 15602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40309⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7159⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40309⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge233295

namespace LeftMerge233297
def owner : Owner := ⟨.program ⟨257⟩, ⟨41962⟩⟩
def mergeEvent : Nat := 233297
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40309⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events060.exact15595RawTerms
def group : MergeGroup := .relation 233296
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 233296) (rhsResult := 15595)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40309⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7159⟩⟩) ⟨7045⟩ 15595) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40309⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge233297

namespace LeftMerge233311
def owner : Owner := ⟨.program ⟨257⟩, ⟨39280⟩⟩
def mergeEvent : Nat := 233311
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39278⟩⟩] } }
def leftRaw : List Term := Proof.Events876.exact224359RawTerms
def rightRaw : List Term := Proof.Events911.exact233305RawTerms
def group : MergeGroup := .operator 224359 233305
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 224359) (leftOrdinal := 0)
    (rightResult := 233305) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39278⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39278⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge233311

namespace LeftMerge233312
def owner : Owner := ⟨.program ⟨257⟩, ⟨39280⟩⟩
def mergeEvent : Nat := 233312
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨37420⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39278⟩⟩] } }
def leftRaw : List Term := Proof.Events876.exact224359RawTerms
def rightRaw : List Term := Proof.Events911.exact233305RawTerms
def group : MergeGroup := .operator 224359 233305
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 224359) (leftOrdinal := 1)
    (rightResult := 233305) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨37420⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39278⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨37420⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39278⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge233312

namespace LeftMerge233314
def owner : Owner := ⟨.program ⟨257⟩, ⟨39280⟩⟩
def mergeEvent : Nat := 233314
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨37420⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38571⟩⟩] } }
def rhsRaw : List Term := Proof.Events911.exact233302RawTerms
def group : MergeGroup := .relation 233313
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 233313) (rhsResult := 233302)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨37420⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39278⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39278⟩⟩) ⟨38571⟩ 233302) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38571⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨37420⟩⟩], [⟨.program ⟨257⟩, ⟨38571⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge233314

namespace LeftMerge233328
def owner : Owner := ⟨.program ⟨257⟩, ⟨38155⟩⟩
def mergeEvent : Nat := 233328
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38152⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events911.exact233322RawTerms
def group : MergeGroup := .operator 222245 233322
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 233322) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38152⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38152⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge233328

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
