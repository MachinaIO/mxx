import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge176264
def owner : Owner := ⟨.program ⟨257⟩, ⟨60775⟩⟩
def mergeEvent : Nat := 176264
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events688.exact176258RawTerms
def group : MergeGroup := .relation 176260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 176260) (rhsResult := 176258)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60772⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 176259 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60772⟩⟩]⟩) (none) 176258) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60181⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176264

namespace LeftMerge176269
def owner : Owner := ⟨.program ⟨257⟩, ⟨62012⟩⟩
def mergeEvent : Nat := 176269
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62009⟩⟩] } }
def leftRaw : List Term := Proof.Events688.exact176265RawTerms
def rightRaw : List Term := Proof.Events687.exact176087RawTerms
def group : MergeGroup := .operator 176265 176087
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 176265) (leftOrdinal := 0)
    (rightResult := 176087) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62009⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62009⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62009⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176269

namespace LeftMerge176270
def owner : Owner := ⟨.program ⟨257⟩, ⟨62012⟩⟩
def mergeEvent : Nat := 176270
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨59860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61136⟩⟩] } }
def leftRaw : List Term := Proof.Events688.exact176265RawTerms
def rightRaw : List Term := Proof.Events687.exact176087RawTerms
def group : MergeGroup := .operator 176265 176087
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 176265) (leftOrdinal := 2)
    (rightResult := 176087) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨59860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61136⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨59860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61136⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨59860⟩⟩], [⟨.program ⟨257⟩, ⟨61136⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176270

namespace LeftMerge176278
def owner : Owner := ⟨.program ⟨257⟩, ⟨62013⟩⟩
def mergeEvent : Nat := 176278
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩] } }
def leftRaw : List Term := Proof.Events688.exact176272RawTerms
def rightRaw : List Term := Proof.Events061.exact15742RawTerms
def group : MergeGroup := .operator 176272 15742
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 176272) (leftOrdinal := 0)
    (rightResult := 15742) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7211⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7103⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176278

namespace LeftMerge176279
def owner : Owner := ⟨.program ⟨257⟩, ⟨62013⟩⟩
def mergeEvent : Nat := 176279
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩] } }
def leftRaw : List Term := Proof.Events688.exact176272RawTerms
def rightRaw : List Term := Proof.Events061.exact15742RawTerms
def group : MergeGroup := .operator 176272 15742
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 176272) (leftOrdinal := 1)
    (rightResult := 15742) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7103⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176279

namespace LeftMerge176281
def owner : Owner := ⟨.program ⟨257⟩, ⟨62013⟩⟩
def mergeEvent : Nat := 176281
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15735RawTerms
def group : MergeGroup := .relation 176280
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 176280) (rhsResult := 15735)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7103⟩⟩) ⟨7017⟩ 15735) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176281

namespace LeftMerge176295
def owner : Owner := ⟨.program ⟨257⟩, ⟨59031⟩⟩
def mergeEvent : Nat := 176295
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59029⟩⟩] } }
def leftRaw : List Term := Proof.Events661.exact169233RawTerms
def rightRaw : List Term := Proof.Events688.exact176289RawTerms
def group : MergeGroup := .operator 169233 176289
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169233) (leftOrdinal := 0)
    (rightResult := 176289) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59029⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59029⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176295

namespace LeftMerge176296
def owner : Owner := ⟨.program ⟨257⟩, ⟨59031⟩⟩
def mergeEvent : Nat := 176296
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59029⟩⟩] } }
def leftRaw : List Term := Proof.Events661.exact169233RawTerms
def rightRaw : List Term := Proof.Events688.exact176289RawTerms
def group : MergeGroup := .operator 169233 176289
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169233) (leftOrdinal := 1)
    (rightResult := 176289) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59029⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨56880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59029⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176296

namespace LeftMerge176298
def owner : Owner := ⟨.program ⟨257⟩, ⟨59031⟩⟩
def mergeEvent : Nat := 176298
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58156⟩⟩] } }
def rhsRaw : List Term := Proof.Events688.exact176286RawTerms
def group : MergeGroup := .relation 176297
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 176297) (rhsResult := 176286)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨56880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59029⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨59029⟩⟩) ⟨58156⟩ 176286) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58156⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨56880⟩⟩], [⟨.program ⟨257⟩, ⟨58156⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176298

namespace LeftMerge176312
def owner : Owner := ⟨.program ⟨257⟩, ⟨57795⟩⟩
def mergeEvent : Nat := 176312
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57792⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events688.exact176306RawTerms
def group : MergeGroup := .operator 163745 176306
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 176306) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57792⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57792⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176312

namespace LeftMerge176433
def owner : Owner := ⟨.program ⟨257⟩, ⟨58344⟩⟩
def mergeEvent : Nat := 176433
def frameStart : Nat := 176367
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events689.exact176429RawTerms
def rightRaw : List Term := Proof.Events689.exact176427RawTerms
def group : MergeGroup := .operator 176429 176427
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 176429) (leftOrdinal := 0)
    (rightResult := 176427) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176433

namespace LeftMerge176445
def owner : Owner := ⟨.program ⟨257⟩, ⟨59030⟩⟩
def mergeEvent : Nat := 176445
def frameStart : Nat := 176367
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59029⟩⟩] } }
def leftRaw : List Term := Proof.Events689.exact176441RawTerms
def rightRaw : List Term := Proof.Events689.exact176418RawTerms
def group : MergeGroup := .operator 176441 176418
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 176441) (leftOrdinal := 0)
    (rightResult := 176418) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59029⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59029⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176445

namespace LeftMerge176446
def owner : Owner := ⟨.program ⟨257⟩, ⟨59030⟩⟩
def mergeEvent : Nat := 176446
def frameStart : Nat := 176367
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59029⟩⟩] } }
def leftRaw : List Term := Proof.Events689.exact176441RawTerms
def rightRaw : List Term := Proof.Events689.exact176418RawTerms
def group : MergeGroup := .operator 176441 176418
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 176441) (leftOrdinal := 1)
    (rightResult := 176418) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59029⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59029⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176446

namespace LeftMerge176448
def owner : Owner := ⟨.program ⟨257⟩, ⟨59030⟩⟩
def mergeEvent : Nat := 176448
def frameStart : Nat := 176367
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58156⟩⟩] } }
def rhsRaw : List Term := Proof.Events689.exact176415RawTerms
def group : MergeGroup := .relation 176447
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 176447) (rhsResult := 176415)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨56880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59029⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨59029⟩⟩) ⟨58156⟩ 176415) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58156⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨56880⟩⟩], [⟨.program ⟨257⟩, ⟨58156⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176448

namespace LeftMerge176456
def owner : Owner := ⟨.program ⟨257⟩, ⟨57204⟩⟩
def mergeEvent : Nat := 176456
def frameStart : Nat := 176367
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57201⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events689.exact176429RawTerms
def rightRaw : List Term := Proof.Events689.exact176452RawTerms
def group : MergeGroup := .operator 176429 176452
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 176429) (leftOrdinal := 0)
    (rightResult := 176452) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57201⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨57201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176456

namespace LeftMerge176473
def owner : Owner := ⟨.program ⟨257⟩, ⟨57795⟩⟩
def mergeEvent : Nat := 176473
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7209⟩⟩] } }
def rhsRaw : List Term := Proof.Events689.exact176470RawTerms
def group : MergeGroup := .relation 176472
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 176472) (rhsResult := 176470)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57792⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 176471 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57792⟩⟩]⟩) (none) 176470) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7209⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176473

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
