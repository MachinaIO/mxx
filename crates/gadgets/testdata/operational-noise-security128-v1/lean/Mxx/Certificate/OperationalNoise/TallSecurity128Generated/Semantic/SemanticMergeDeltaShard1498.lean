import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge243345
def owner : Owner := ⟨.program ⟨257⟩, ⟨51719⟩⟩
def mergeEvent : Nat := 243345
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51716⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236870RawTerms
def rightRaw : List Term := Proof.Events950.exact243339RawTerms
def group : MergeGroup := .operator 236870 243339
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236870) (leftOrdinal := 0)
    (rightResult := 243339) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51716⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51716⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243345

namespace LeftMerge243466
def owner : Owner := ⟨.program ⟨257⟩, ⟨52360⟩⟩
def mergeEvent : Nat := 243466
def frameStart : Nat := 243400
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events951.exact243462RawTerms
def rightRaw : List Term := Proof.Events951.exact243460RawTerms
def group : MergeGroup := .operator 243462 243460
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243462) (leftOrdinal := 0)
    (rightResult := 243460) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243466

namespace LeftMerge243478
def owner : Owner := ⟨.program ⟨257⟩, ⟨52891⟩⟩
def mergeEvent : Nat := 243478
def frameStart : Nat := 243400
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52890⟩⟩] } }
def leftRaw : List Term := Proof.Events951.exact243474RawTerms
def rightRaw : List Term := Proof.Events950.exact243451RawTerms
def group : MergeGroup := .operator 243474 243451
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243474) (leftOrdinal := 0)
    (rightResult := 243451) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52890⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52890⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243478

namespace LeftMerge243479
def owner : Owner := ⟨.program ⟨257⟩, ⟨52891⟩⟩
def mergeEvent : Nat := 243479
def frameStart : Nat := 243400
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52890⟩⟩] } }
def leftRaw : List Term := Proof.Events951.exact243474RawTerms
def rightRaw : List Term := Proof.Events950.exact243451RawTerms
def group : MergeGroup := .operator 243474 243451
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243474) (leftOrdinal := 1)
    (rightResult := 243451) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52890⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52890⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243479

namespace LeftMerge243481
def owner : Owner := ⟨.program ⟨257⟩, ⟨52891⟩⟩
def mergeEvent : Nat := 243481
def frameStart : Nat := 243400
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52143⟩⟩] } }
def rhsRaw : List Term := Proof.Events950.exact243448RawTerms
def group : MergeGroup := .relation 243480
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243480) (rhsResult := 243448)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨50872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52890⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52890⟩⟩) ⟨52143⟩ 243448) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52143⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨50872⟩⟩], [⟨.program ⟨257⟩, ⟨52143⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243481

namespace LeftMerge243489
def owner : Owner := ⟨.program ⟨257⟩, ⟨51125⟩⟩
def mergeEvent : Nat := 243489
def frameStart : Nat := 243400
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51123⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events951.exact243462RawTerms
def rightRaw : List Term := Proof.Events951.exact243485RawTerms
def group : MergeGroup := .operator 243462 243485
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243462) (leftOrdinal := 0)
    (rightResult := 243485) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51123⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨51123⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243489

namespace LeftMerge243506
def owner : Owner := ⟨.program ⟨257⟩, ⟨51719⟩⟩
def mergeEvent : Nat := 243506
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }
def rhsRaw : List Term := Proof.Events951.exact243503RawTerms
def group : MergeGroup := .relation 243505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243505) (rhsResult := 243503)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51716⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 243504 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51716⟩⟩]⟩) (none) 243503) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243506

namespace LeftMerge243507
def owner : Owner := ⟨.program ⟨257⟩, ⟨51719⟩⟩
def mergeEvent : Nat := 243507
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52890⟩⟩] } }
def rhsRaw : List Term := Proof.Events951.exact243503RawTerms
def group : MergeGroup := .relation 243505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243505) (rhsResult := 243503)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51716⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 243504 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51716⟩⟩]⟩) (none) 243503) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52890⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52890⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243507

namespace LeftMerge243508
def owner : Owner := ⟨.program ⟨257⟩, ⟨51719⟩⟩
def mergeEvent : Nat := 243508
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52143⟩⟩] } }
def rhsRaw : List Term := Proof.Events951.exact243503RawTerms
def group : MergeGroup := .relation 243505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243505) (rhsResult := 243503)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51716⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 243504 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51716⟩⟩]⟩) (none) 243503) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52143⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨50872⟩⟩], [⟨.program ⟨257⟩, ⟨52143⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243508

namespace LeftMerge243509
def owner : Owner := ⟨.program ⟨257⟩, ⟨51719⟩⟩
def mergeEvent : Nat := 243509
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨51123⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events951.exact243503RawTerms
def group : MergeGroup := .relation 243505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243505) (rhsResult := 243503)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51716⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 243504 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51716⟩⟩]⟩) (none) 243503) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51123⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨51123⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243509

namespace LeftMerge243514
def owner : Owner := ⟨.program ⟨257⟩, ⟨52893⟩⟩
def mergeEvent : Nat := 243514
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52890⟩⟩] } }
def leftRaw : List Term := Proof.Events951.exact243510RawTerms
def rightRaw : List Term := Proof.Events950.exact243332RawTerms
def group : MergeGroup := .operator 243510 243332
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243510) (leftOrdinal := 0)
    (rightResult := 243332) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52890⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52890⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52890⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243514

namespace LeftMerge243515
def owner : Owner := ⟨.program ⟨257⟩, ⟨52893⟩⟩
def mergeEvent : Nat := 243515
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52143⟩⟩] } }
def leftRaw : List Term := Proof.Events951.exact243510RawTerms
def rightRaw : List Term := Proof.Events950.exact243332RawTerms
def group : MergeGroup := .operator 243510 243332
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243510) (leftOrdinal := 2)
    (rightResult := 243332) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52143⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52143⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨50872⟩⟩], [⟨.program ⟨257⟩, ⟨52143⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243515

namespace LeftMerge243541
def owner : Owner := ⟨.program ⟨257⟩, ⟨24267⟩⟩
def mergeEvent : Nat := 243541
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24266⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events045.exact11636RawTerms
def rightRaw : List Term := Proof.Events924.exact236778RawTerms
def group : MergeGroup := .operator 11636 236778
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11636) (leftOrdinal := 0)
    (rightResult := 236778) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24266⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24266⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243541

namespace LeftMerge243546
def owner : Owner := ⟨.program ⟨257⟩, ⟨8385⟩⟩
def mergeEvent : Nat := 243546
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }
def leftRaw : List Term := Proof.Events924.exact236648RawTerms
def rightRaw : List Term := Proof.Events094.exact24094RawTerms
def group : MergeGroup := .operator 236648 24094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236648) (leftOrdinal := 0)
    (rightResult := 24094) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243546

namespace LeftMerge243563
def owner : Owner := ⟨.program ⟨257⟩, ⟨31434⟩⟩
def mergeEvent : Nat := 243563
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events951.exact243557RawTerms
def rightRaw : List Term := Proof.Events045.exact11639RawTerms
def group : MergeGroup := .operator 243557 11639
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243557) (leftOrdinal := 1)
    (rightResult := 11639) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24266⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243563

namespace LeftMerge243564
def owner : Owner := ⟨.program ⟨257⟩, ⟨31434⟩⟩
def mergeEvent : Nat := 243564
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }
def leftRaw : List Term := Proof.Events951.exact243557RawTerms
def rightRaw : List Term := Proof.Events045.exact11639RawTerms
def group : MergeGroup := .operator 243557 11639
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243557) (leftOrdinal := 0)
    (rightResult := 11639) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243564

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
