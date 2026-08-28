import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge115378
def owner : Owner := ⟨.program ⟨257⟩, ⟨68383⟩⟩
def mergeEvent : Nat := 115378
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def rhsRaw : List Term := Proof.Events450.exact115358RawTerms
def group : MergeGroup := .relation 115360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115360) (rhsResult := 115358)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 115359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩) (none) 115358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115378

namespace LeftMerge115379
def owner : Owner := ⟨.program ⟨257⟩, ⟨68383⟩⟩
def mergeEvent : Nat := 115379
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def rhsRaw : List Term := Proof.Events450.exact115358RawTerms
def group : MergeGroup := .relation 115360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115360) (rhsResult := 115358)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 115359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩) (none) 115358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115379

namespace LeftMerge115380
def owner : Owner := ⟨.program ⟨257⟩, ⟨68383⟩⟩
def mergeEvent : Nat := 115380
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨48376⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events450.exact115358RawTerms
def group : MergeGroup := .relation 115360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115360) (rhsResult := 115358)
    (sourceTermOrdinal := 30) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 115359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩) (none) 115358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48376⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨48376⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115380

namespace LeftMerge115381
def owner : Owner := ⟨.program ⟨257⟩, ⟨68383⟩⟩
def mergeEvent : Nat := 115381
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨45696⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events450.exact115358RawTerms
def group : MergeGroup := .relation 115360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115360) (rhsResult := 115358)
    (sourceTermOrdinal := 29) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 115359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩) (none) 115358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45696⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨45696⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115381

namespace LeftMerge115382
def owner : Owner := ⟨.program ⟨257⟩, ⟨68383⟩⟩
def mergeEvent : Nat := 115382
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨43012⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events450.exact115358RawTerms
def group : MergeGroup := .relation 115360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115360) (rhsResult := 115358)
    (sourceTermOrdinal := 28) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 115359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩) (none) 115358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨43012⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨43012⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115382

namespace LeftMerge115383
def owner : Owner := ⟨.program ⟨257⟩, ⟨68383⟩⟩
def mergeEvent : Nat := 115383
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40332⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events450.exact115358RawTerms
def group : MergeGroup := .relation 115360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115360) (rhsResult := 115358)
    (sourceTermOrdinal := 27) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 115359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩) (none) 115358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40332⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40332⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115383

namespace LeftMerge115384
def owner : Owner := ⟨.program ⟨257⟩, ⟨68383⟩⟩
def mergeEvent : Nat := 115384
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37656⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events450.exact115358RawTerms
def group : MergeGroup := .relation 115360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115360) (rhsResult := 115358)
    (sourceTermOrdinal := 26) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 115359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩) (none) 115358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37656⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37656⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115384

namespace LeftMerge115385
def owner : Owner := ⟨.program ⟨257⟩, ⟨68383⟩⟩
def mergeEvent : Nat := 115385
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨34976⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events450.exact115358RawTerms
def group : MergeGroup := .relation 115360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115360) (rhsResult := 115358)
    (sourceTermOrdinal := 25) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 115359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩) (none) 115358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34976⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨34976⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115385

namespace LeftMerge115386
def owner : Owner := ⟨.program ⟨257⟩, ⟨68383⟩⟩
def mergeEvent : Nat := 115386
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29312⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events450.exact115358RawTerms
def group : MergeGroup := .relation 115360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115360) (rhsResult := 115358)
    (sourceTermOrdinal := 23) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 115359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩) (none) 115358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29312⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29312⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115386

namespace LeftMerge115387
def owner : Owner := ⟨.program ⟨257⟩, ⟨68383⟩⟩
def mergeEvent : Nat := 115387
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events450.exact115358RawTerms
def group : MergeGroup := .relation 115360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115360) (rhsResult := 115358)
    (sourceTermOrdinal := 22) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 115359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩) (none) 115358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26632⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115387

namespace LeftMerge115388
def owner : Owner := ⟨.program ⟨257⟩, ⟨68383⟩⟩
def mergeEvent : Nat := 115388
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨66671⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events450.exact115358RawTerms
def group : MergeGroup := .relation 115360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115360) (rhsResult := 115358)
    (sourceTermOrdinal := 36) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 115359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩) (none) 115358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66671⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨66671⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115388

namespace LeftMerge115389
def owner : Owner := ⟨.program ⟨257⟩, ⟨68383⟩⟩
def mergeEvent : Nat := 115389
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨63100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events450.exact115358RawTerms
def group : MergeGroup := .relation 115360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115360) (rhsResult := 115358)
    (sourceTermOrdinal := 35) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 115359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩) (none) 115358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨63100⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115389

namespace LeftMerge115390
def owner : Owner := ⟨.program ⟨257⟩, ⟨68383⟩⟩
def mergeEvent : Nat := 115390
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨60120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events450.exact115358RawTerms
def group : MergeGroup := .relation 115360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115360) (rhsResult := 115358)
    (sourceTermOrdinal := 34) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 115359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩) (none) 115358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨60120⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115390

namespace LeftMerge115391
def owner : Owner := ⟨.program ⟨257⟩, ⟨68383⟩⟩
def mergeEvent : Nat := 115391
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨57140⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events450.exact115358RawTerms
def group : MergeGroup := .relation 115360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115360) (rhsResult := 115358)
    (sourceTermOrdinal := 33) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 115359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩) (none) 115358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57140⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨57140⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115391

namespace LeftMerge115392
def owner : Owner := ⟨.program ⟨257⟩, ⟨68383⟩⟩
def mergeEvent : Nat := 115392
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨54160⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events450.exact115358RawTerms
def group : MergeGroup := .relation 115360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115360) (rhsResult := 115358)
    (sourceTermOrdinal := 32) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 115359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩) (none) 115358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54160⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨54160⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115392

namespace LeftMerge115393
def owner : Owner := ⟨.program ⟨257⟩, ⟨68383⟩⟩
def mergeEvent : Nat := 115393
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨51180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events450.exact115358RawTerms
def group : MergeGroup := .relation 115360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115360) (rhsResult := 115358)
    (sourceTermOrdinal := 31) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 115359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩) (none) 115358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨51180⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115393

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
