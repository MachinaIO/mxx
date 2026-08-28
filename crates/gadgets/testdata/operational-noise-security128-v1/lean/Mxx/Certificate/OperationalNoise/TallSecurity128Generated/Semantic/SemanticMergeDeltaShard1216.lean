import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge198278
def owner : Owner := ⟨.program ⟨257⟩, ⟨56567⟩⟩
def mergeEvent : Nat := 198278
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }
def leftRaw : List Term := Proof.Events774.exact198274RawTerms
def rightRaw : List Term := Proof.Events774.exact198244RawTerms
def group : MergeGroup := .operator 198274 198244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198274) (leftOrdinal := 1)
    (rightResult := 198244) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198278

namespace LeftMerge198286
def owner : Owner := ⟨.program ⟨257⟩, ⟨58502⟩⟩
def mergeEvent : Nat := 198286
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58501⟩⟩] } }
def leftRaw : List Term := Proof.Events774.exact198280RawTerms
def rightRaw : List Term := Proof.Events774.exact198216RawTerms
def group : MergeGroup := .operator 198280 198216
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198280) (leftOrdinal := 1)
    (rightResult := 198216) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge198286

namespace LeftMerge198288
def owner : Owner := ⟨.program ⟨257⟩, ⟨58502⟩⟩
def mergeEvent : Nat := 198288
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57981⟩⟩] } }
def rhsRaw : List Term := Proof.Events774.exact198213RawTerms
def group : MergeGroup := .relation 198287
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 198287) (rhsResult := 198213)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58501⟩⟩) ⟨57981⟩ 198213) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57981⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], [⟨.program ⟨257⟩, ⟨57981⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge198288

namespace LeftMerge198289
def owner : Owner := ⟨.program ⟨257⟩, ⟨58502⟩⟩
def mergeEvent : Nat := 198289
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58501⟩⟩] } }
def leftRaw : List Term := Proof.Events774.exact198280RawTerms
def rightRaw : List Term := Proof.Events774.exact198216RawTerms
def group : MergeGroup := .operator 198280 198216
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198280) (leftOrdinal := 0)
    (rightResult := 198216) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198289

namespace LeftMerge198303
def owner : Owner := ⟨.program ⟨257⟩, ⟨57432⟩⟩
def mergeEvent : Nat := 198303
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57429⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events774.exact198297RawTerms
def group : MergeGroup := .operator 192995 198297
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 198297) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57429⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57429⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198303

namespace LeftMerge198382
def owner : Owner := ⟨.program ⟨257⟩, ⟨56560⟩⟩
def mergeEvent : Nat := 198382
def frameStart : Nat := 198352
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events774.exact198378RawTerms
def rightRaw : List Term := Proof.Events774.exact198375RawTerms
def group : MergeGroup := .operator 198378 198375
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198378) (leftOrdinal := 0)
    (rightResult := 198375) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56559⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25034⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198382

namespace LeftMerge198412
def owner : Owner := ⟨.program ⟨257⟩, ⟨58256⟩⟩
def mergeEvent : Nat := 198412
def frameStart : Nat := 198352
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events775.exact198408RawTerms
def rightRaw : List Term := Proof.Events775.exact198406RawTerms
def group : MergeGroup := .operator 198408 198406
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198408) (leftOrdinal := 0)
    (rightResult := 198406) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198412

namespace LeftMerge198435
def owner : Owner := ⟨.program ⟨257⟩, ⟨9534⟩⟩
def mergeEvent : Nat := 198435
def frameStart : Nat := 198352
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }
def leftRaw : List Term := Proof.Events775.exact198431RawTerms
def rightRaw : List Term := Proof.Events775.exact198428RawTerms
def group : MergeGroup := .operator 198431 198428
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198431) (leftOrdinal := 0)
    (rightResult := 198428) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9532⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198435

namespace LeftMerge198444
def owner : Owner := ⟨.program ⟨257⟩, ⟨58504⟩⟩
def mergeEvent : Nat := 198444
def frameStart : Nat := 198352
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58501⟩⟩] } }
def leftRaw : List Term := Proof.Events775.exact198440RawTerms
def rightRaw : List Term := Proof.Events774.exact198397RawTerms
def group : MergeGroup := .operator 198440 198397
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198440) (leftOrdinal := 0)
    (rightResult := 198397) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58501⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198444

namespace LeftMerge198445
def owner : Owner := ⟨.program ⟨257⟩, ⟨58504⟩⟩
def mergeEvent : Nat := 198445
def frameStart : Nat := 198352
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58501⟩⟩] } }
def leftRaw : List Term := Proof.Events775.exact198440RawTerms
def rightRaw : List Term := Proof.Events774.exact198397RawTerms
def group : MergeGroup := .operator 198440 198397
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198440) (leftOrdinal := 1)
    (rightResult := 198397) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge198445

namespace LeftMerge198447
def owner : Owner := ⟨.program ⟨257⟩, ⟨58504⟩⟩
def mergeEvent : Nat := 198447
def frameStart : Nat := 198352
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57981⟩⟩] } }
def rhsRaw : List Term := Proof.Events774.exact198394RawTerms
def group : MergeGroup := .relation 198446
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 198446) (rhsResult := 198394)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58501⟩⟩) ⟨57981⟩ 198394) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57981⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], [⟨.program ⟨257⟩, ⟨57981⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge198447

namespace LeftMerge198455
def owner : Owner := ⟨.program ⟨257⟩, ⟨56866⟩⟩
def mergeEvent : Nat := 198455
def frameStart : Nat := 198352
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events775.exact198408RawTerms
def rightRaw : List Term := Proof.Events775.exact198451RawTerms
def group : MergeGroup := .operator 198408 198451
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198408) (leftOrdinal := 0)
    (rightResult := 198451) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56864⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198455

namespace LeftMerge198472
def owner : Owner := ⟨.program ⟨257⟩, ⟨57432⟩⟩
def mergeEvent : Nat := 198472
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }
def rhsRaw : List Term := Proof.Events775.exact198469RawTerms
def group : MergeGroup := .relation 198471
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 198471) (rhsResult := 198469)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 198470 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57429⟩⟩]⟩) (none) 198469) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198472

namespace LeftMerge198473
def owner : Owner := ⟨.program ⟨257⟩, ⟨57432⟩⟩
def mergeEvent : Nat := 198473
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58501⟩⟩] } }
def rhsRaw : List Term := Proof.Events775.exact198469RawTerms
def group : MergeGroup := .relation 198471
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 198471) (rhsResult := 198469)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 198470 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57429⟩⟩]⟩) (none) 198469) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58501⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge198473

namespace LeftMerge198474
def owner : Owner := ⟨.program ⟨257⟩, ⟨57432⟩⟩
def mergeEvent : Nat := 198474
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57981⟩⟩] } }
def rhsRaw : List Term := Proof.Events775.exact198469RawTerms
def group : MergeGroup := .relation 198471
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 198471) (rhsResult := 198469)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 198470 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57429⟩⟩]⟩) (none) 198469) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57981⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], [⟨.program ⟨257⟩, ⟨57981⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198474

namespace LeftMerge198475
def owner : Owner := ⟨.program ⟨257⟩, ⟨57432⟩⟩
def mergeEvent : Nat := 198475
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨56864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events775.exact198469RawTerms
def group : MergeGroup := .relation 198471
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 198471) (rhsResult := 198469)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 198470 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57429⟩⟩]⟩) (none) 198469) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨56864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge198475

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
