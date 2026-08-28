import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge232370
def owner : Owner := ⟨.program ⟨257⟩, ⟨68363⟩⟩
def mergeEvent : Nat := 232370
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def rhsRaw : List Term := Proof.Events907.exact232358RawTerms
def group : MergeGroup := .relation 232360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232360) (rhsResult := 232358)
    (sourceTermOrdinal := 9) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 232359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩) (none) 232358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232370

namespace LeftMerge232371
def owner : Owner := ⟨.program ⟨257⟩, ⟨68363⟩⟩
def mergeEvent : Nat := 232371
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def rhsRaw : List Term := Proof.Events907.exact232358RawTerms
def group : MergeGroup := .relation 232360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232360) (rhsResult := 232358)
    (sourceTermOrdinal := 8) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 232359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩) (none) 232358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232371

namespace LeftMerge232372
def owner : Owner := ⟨.program ⟨257⟩, ⟨68363⟩⟩
def mergeEvent : Nat := 232372
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def rhsRaw : List Term := Proof.Events907.exact232358RawTerms
def group : MergeGroup := .relation 232360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232360) (rhsResult := 232358)
    (sourceTermOrdinal := 7) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 232359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩) (none) 232358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232372

namespace LeftMerge232373
def owner : Owner := ⟨.program ⟨257⟩, ⟨68363⟩⟩
def mergeEvent : Nat := 232373
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def rhsRaw : List Term := Proof.Events907.exact232358RawTerms
def group : MergeGroup := .relation 232360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232360) (rhsResult := 232358)
    (sourceTermOrdinal := 6) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 232359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩) (none) 232358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232373

namespace LeftMerge232374
def owner : Owner := ⟨.program ⟨257⟩, ⟨68363⟩⟩
def mergeEvent : Nat := 232374
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def rhsRaw : List Term := Proof.Events907.exact232358RawTerms
def group : MergeGroup := .relation 232360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232360) (rhsResult := 232358)
    (sourceTermOrdinal := 5) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 232359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩) (none) 232358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232374

namespace LeftMerge232375
def owner : Owner := ⟨.program ⟨257⟩, ⟨68363⟩⟩
def mergeEvent : Nat := 232375
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def rhsRaw : List Term := Proof.Events907.exact232358RawTerms
def group : MergeGroup := .relation 232360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232360) (rhsResult := 232358)
    (sourceTermOrdinal := 4) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 232359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩) (none) 232358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232375

namespace LeftMerge232376
def owner : Owner := ⟨.program ⟨257⟩, ⟨68363⟩⟩
def mergeEvent : Nat := 232376
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def rhsRaw : List Term := Proof.Events907.exact232358RawTerms
def group : MergeGroup := .relation 232360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232360) (rhsResult := 232358)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 232359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩) (none) 232358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232376

namespace LeftMerge232377
def owner : Owner := ⟨.program ⟨257⟩, ⟨68363⟩⟩
def mergeEvent : Nat := 232377
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def rhsRaw : List Term := Proof.Events907.exact232358RawTerms
def group : MergeGroup := .relation 232360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232360) (rhsResult := 232358)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 232359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩) (none) 232358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232377

namespace LeftMerge232378
def owner : Owner := ⟨.program ⟨257⟩, ⟨68363⟩⟩
def mergeEvent : Nat := 232378
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def rhsRaw : List Term := Proof.Events907.exact232358RawTerms
def group : MergeGroup := .relation 232360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232360) (rhsResult := 232358)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 232359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩) (none) 232358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232378

namespace LeftMerge232379
def owner : Owner := ⟨.program ⟨257⟩, ⟨68363⟩⟩
def mergeEvent : Nat := 232379
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def rhsRaw : List Term := Proof.Events907.exact232358RawTerms
def group : MergeGroup := .relation 232360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232360) (rhsResult := 232358)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 232359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩) (none) 232358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232379

namespace LeftMerge232380
def owner : Owner := ⟨.program ⟨257⟩, ⟨68363⟩⟩
def mergeEvent : Nat := 232380
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨48350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }
def rhsRaw : List Term := Proof.Events907.exact232358RawTerms
def group : MergeGroup := .relation 232360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232360) (rhsResult := 232358)
    (sourceTermOrdinal := 30) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 232359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩) (none) 232358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨48350⟩⟩], [⟨.program ⟨257⟩, ⟨68824⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232380

namespace LeftMerge232381
def owner : Owner := ⟨.program ⟨257⟩, ⟨68363⟩⟩
def mergeEvent : Nat := 232381
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }
def rhsRaw : List Term := Proof.Events907.exact232358RawTerms
def group : MergeGroup := .relation 232360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232360) (rhsResult := 232358)
    (sourceTermOrdinal := 29) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 232359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩) (none) 232358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45670⟩⟩], [⟨.program ⟨257⟩, ⟨68824⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232381

namespace LeftMerge232382
def owner : Owner := ⟨.program ⟨257⟩, ⟨68363⟩⟩
def mergeEvent : Nat := 232382
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨42986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }
def rhsRaw : List Term := Proof.Events907.exact232358RawTerms
def group : MergeGroup := .relation 232360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232360) (rhsResult := 232358)
    (sourceTermOrdinal := 28) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 232359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩) (none) 232358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨42986⟩⟩], [⟨.program ⟨257⟩, ⟨68824⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232382

namespace LeftMerge232383
def owner : Owner := ⟨.program ⟨257⟩, ⟨68363⟩⟩
def mergeEvent : Nat := 232383
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }
def rhsRaw : List Term := Proof.Events907.exact232358RawTerms
def group : MergeGroup := .relation 232360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232360) (rhsResult := 232358)
    (sourceTermOrdinal := 27) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 232359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩) (none) 232358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40306⟩⟩], [⟨.program ⟨257⟩, ⟨68824⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232383

namespace LeftMerge232384
def owner : Owner := ⟨.program ⟨257⟩, ⟨68363⟩⟩
def mergeEvent : Nat := 232384
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨37630⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }
def rhsRaw : List Term := Proof.Events907.exact232358RawTerms
def group : MergeGroup := .relation 232360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232360) (rhsResult := 232358)
    (sourceTermOrdinal := 26) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 232359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩) (none) 232358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37630⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨37630⟩⟩], [⟨.program ⟨257⟩, ⟨68824⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232384

namespace LeftMerge232385
def owner : Owner := ⟨.program ⟨257⟩, ⟨68363⟩⟩
def mergeEvent : Nat := 232385
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨34950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }
def rhsRaw : List Term := Proof.Events907.exact232358RawTerms
def group : MergeGroup := .relation 232360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232360) (rhsResult := 232358)
    (sourceTermOrdinal := 25) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 232359 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68360⟩⟩]⟩) (none) 232358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨34950⟩⟩], [⟨.program ⟨257⟩, ⟨68824⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232385

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
