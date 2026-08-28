import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge301252
def owner : Owner := ⟨.program ⟨257⟩, ⟨31223⟩⟩
def mergeEvent : Nat := 301252
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }
def leftRaw : List Term := Proof.Events1176.exact301248RawTerms
def rightRaw : List Term := Proof.Events1176.exact301218RawTerms
def group : MergeGroup := .operator 301248 301218
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301248) (leftOrdinal := 1)
    (rightResult := 301218) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301252

namespace LeftMerge301260
def owner : Owner := ⟨.program ⟨257⟩, ⟨33350⟩⟩
def mergeEvent : Nat := 301260
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33349⟩⟩] } }
def leftRaw : List Term := Proof.Events1176.exact301254RawTerms
def rightRaw : List Term := Proof.Events1176.exact301190RawTerms
def group : MergeGroup := .operator 301254 301190
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301254) (leftOrdinal := 1)
    (rightResult := 301190) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33349⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33349⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301260

namespace LeftMerge301262
def owner : Owner := ⟨.program ⟨257⟩, ⟨33350⟩⟩
def mergeEvent : Nat := 301262
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32889⟩⟩] } }
def rhsRaw : List Term := Proof.Events1176.exact301187RawTerms
def group : MergeGroup := .relation 301261
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 301261) (rhsResult := 301187)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33349⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33349⟩⟩) ⟨32889⟩ 301187) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32889⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], [⟨.program ⟨257⟩, ⟨32889⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301262

namespace LeftMerge301263
def owner : Owner := ⟨.program ⟨257⟩, ⟨33350⟩⟩
def mergeEvent : Nat := 301263
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33349⟩⟩] } }
def leftRaw : List Term := Proof.Events1176.exact301254RawTerms
def rightRaw : List Term := Proof.Events1176.exact301190RawTerms
def group : MergeGroup := .operator 301254 301190
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301254) (leftOrdinal := 0)
    (rightResult := 301190) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33349⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33349⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301263

namespace LeftMerge301277
def owner : Owner := ⟨.program ⟨257⟩, ⟨32292⟩⟩
def mergeEvent : Nat := 301277
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32289⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1176.exact301271RawTerms
def group : MergeGroup := .operator 295195 301271
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 301271) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32289⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32289⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301277

namespace LeftMerge301332
def owner : Owner := ⟨.program ⟨257⟩, ⟨31216⟩⟩
def mergeEvent : Nat := 301332
def frameStart : Nat := 301314
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1177.exact301328RawTerms
def rightRaw : List Term := Proof.Events1177.exact301325RawTerms
def group : MergeGroup := .operator 301328 301325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301328) (leftOrdinal := 0)
    (rightResult := 301325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31215⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24170⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301332

namespace LeftMerge301362
def owner : Owner := ⟨.program ⟨257⟩, ⟨33188⟩⟩
def mergeEvent : Nat := 301362
def frameStart : Nat := 301314
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1177.exact301358RawTerms
def rightRaw : List Term := Proof.Events1177.exact301356RawTerms
def group : MergeGroup := .operator 301358 301356
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301358) (leftOrdinal := 0)
    (rightResult := 301356) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301362

namespace LeftMerge301385
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def mergeEvent : Nat := 301385
def frameStart : Nat := 301314
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }
def leftRaw : List Term := Proof.Events1177.exact301381RawTerms
def rightRaw : List Term := Proof.Events1177.exact301378RawTerms
def group : MergeGroup := .operator 301381 301378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301381) (leftOrdinal := 0)
    (rightResult := 301378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9577⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301385

namespace LeftMerge301394
def owner : Owner := ⟨.program ⟨257⟩, ⟨33352⟩⟩
def mergeEvent : Nat := 301394
def frameStart : Nat := 301314
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33349⟩⟩] } }
def leftRaw : List Term := Proof.Events1177.exact301390RawTerms
def rightRaw : List Term := Proof.Events1177.exact301347RawTerms
def group : MergeGroup := .operator 301390 301347
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301390) (leftOrdinal := 0)
    (rightResult := 301347) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33349⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33349⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301394

namespace LeftMerge301395
def owner : Owner := ⟨.program ⟨257⟩, ⟨33352⟩⟩
def mergeEvent : Nat := 301395
def frameStart : Nat := 301314
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33349⟩⟩] } }
def leftRaw : List Term := Proof.Events1177.exact301390RawTerms
def rightRaw : List Term := Proof.Events1177.exact301347RawTerms
def group : MergeGroup := .operator 301390 301347
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301390) (leftOrdinal := 1)
    (rightResult := 301347) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33349⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33349⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301395

namespace LeftMerge301397
def owner : Owner := ⟨.program ⟨257⟩, ⟨33352⟩⟩
def mergeEvent : Nat := 301397
def frameStart : Nat := 301314
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32889⟩⟩] } }
def rhsRaw : List Term := Proof.Events1177.exact301344RawTerms
def group : MergeGroup := .relation 301396
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 301396) (rhsResult := 301344)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33349⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33349⟩⟩) ⟨32889⟩ 301344) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32889⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], [⟨.program ⟨257⟩, ⟨32889⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301397

namespace LeftMerge301405
def owner : Owner := ⟨.program ⟨257⟩, ⟨31750⟩⟩
def mergeEvent : Nat := 301405
def frameStart : Nat := 301314
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1177.exact301358RawTerms
def rightRaw : List Term := Proof.Events1177.exact301401RawTerms
def group : MergeGroup := .operator 301358 301401
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301358) (leftOrdinal := 0)
    (rightResult := 301401) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31748⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301405

namespace LeftMerge301422
def owner : Owner := ⟨.program ⟨257⟩, ⟨32292⟩⟩
def mergeEvent : Nat := 301422
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }
def rhsRaw : List Term := Proof.Events1177.exact301419RawTerms
def group : MergeGroup := .relation 301421
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 301421) (rhsResult := 301419)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32289⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 301420 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32289⟩⟩]⟩) (none) 301419) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301422

namespace LeftMerge301423
def owner : Owner := ⟨.program ⟨257⟩, ⟨32292⟩⟩
def mergeEvent : Nat := 301423
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33349⟩⟩] } }
def rhsRaw : List Term := Proof.Events1177.exact301419RawTerms
def group : MergeGroup := .relation 301421
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 301421) (rhsResult := 301419)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32289⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 301420 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32289⟩⟩]⟩) (none) 301419) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33349⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33349⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301423

namespace LeftMerge301424
def owner : Owner := ⟨.program ⟨257⟩, ⟨32292⟩⟩
def mergeEvent : Nat := 301424
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32889⟩⟩] } }
def rhsRaw : List Term := Proof.Events1177.exact301419RawTerms
def group : MergeGroup := .relation 301421
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 301421) (rhsResult := 301419)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32289⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 301420 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32289⟩⟩]⟩) (none) 301419) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32889⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], [⟨.program ⟨257⟩, ⟨32889⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301424

namespace LeftMerge301425
def owner : Owner := ⟨.program ⟨257⟩, ⟨32292⟩⟩
def mergeEvent : Nat := 301425
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1177.exact301419RawTerms
def group : MergeGroup := .relation 301421
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 301421) (rhsResult := 301419)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32289⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 301420 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32289⟩⟩]⟩) (none) 301419) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301425

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
