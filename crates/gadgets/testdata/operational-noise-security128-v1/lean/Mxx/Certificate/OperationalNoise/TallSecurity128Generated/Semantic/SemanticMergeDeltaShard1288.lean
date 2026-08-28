import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge209259
def owner : Owner := ⟨.program ⟨257⟩, ⟨41991⟩⟩
def mergeEvent : Nat := 209259
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩] } }
def leftRaw : List Term := Proof.Events817.exact209252RawTerms
def rightRaw : List Term := Proof.Events816.exact208975RawTerms
def group : MergeGroup := .operator 209252 208975
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209252) (leftOrdinal := 1)
    (rightResult := 208975) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41989⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge209259

namespace LeftMerge209261
def owner : Owner := ⟨.program ⟨257⟩, ⟨41991⟩⟩
def mergeEvent : Nat := 209261
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41261⟩⟩] } }
def rhsRaw : List Term := Proof.Events816.exact208972RawTerms
def group : MergeGroup := .relation 209260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 209260) (rhsResult := 208972)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41989⟩⟩) ⟨41261⟩ 208972) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41261⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40108⟩⟩], [⟨.program ⟨257⟩, ⟨41261⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge209261

namespace LeftMerge209275
def owner : Owner := ⟨.program ⟨257⟩, ⟨40859⟩⟩
def mergeEvent : Nat := 209275
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40856⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207620RawTerms
def rightRaw : List Term := Proof.Events817.exact209269RawTerms
def group : MergeGroup := .operator 207620 209269
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207620) (leftOrdinal := 0)
    (rightResult := 209269) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40856⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40856⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209275

namespace LeftMerge209396
def owner : Owner := ⟨.program ⟨257⟩, ⟨41468⟩⟩
def mergeEvent : Nat := 209396
def frameStart : Nat := 209330
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events817.exact209392RawTerms
def rightRaw : List Term := Proof.Events817.exact209390RawTerms
def group : MergeGroup := .operator 209392 209390
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209392) (leftOrdinal := 0)
    (rightResult := 209390) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40108⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209396

namespace LeftMerge209408
def owner : Owner := ⟨.program ⟨257⟩, ⟨41990⟩⟩
def mergeEvent : Nat := 209408
def frameStart : Nat := 209330
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩] } }
def leftRaw : List Term := Proof.Events817.exact209404RawTerms
def rightRaw : List Term := Proof.Events817.exact209381RawTerms
def group : MergeGroup := .operator 209404 209381
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209404) (leftOrdinal := 0)
    (rightResult := 209381) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41989⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209408

namespace LeftMerge209409
def owner : Owner := ⟨.program ⟨257⟩, ⟨41990⟩⟩
def mergeEvent : Nat := 209409
def frameStart : Nat := 209330
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩] } }
def leftRaw : List Term := Proof.Events817.exact209404RawTerms
def rightRaw : List Term := Proof.Events817.exact209381RawTerms
def group : MergeGroup := .operator 209404 209381
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209404) (leftOrdinal := 1)
    (rightResult := 209381) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41989⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge209409

namespace LeftMerge209411
def owner : Owner := ⟨.program ⟨257⟩, ⟨41990⟩⟩
def mergeEvent : Nat := 209411
def frameStart : Nat := 209330
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41261⟩⟩] } }
def rhsRaw : List Term := Proof.Events817.exact209378RawTerms
def group : MergeGroup := .relation 209410
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 209410) (rhsResult := 209378)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41989⟩⟩) ⟨41261⟩ 209378) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41261⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40108⟩⟩], [⟨.program ⟨257⟩, ⟨41261⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge209411

namespace LeftMerge209419
def owner : Owner := ⟨.program ⟨257⟩, ⟨40320⟩⟩
def mergeEvent : Nat := 209419
def frameStart : Nat := 209330
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events817.exact209392RawTerms
def rightRaw : List Term := Proof.Events818.exact209415RawTerms
def group : MergeGroup := .operator 209392 209415
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209392) (leftOrdinal := 0)
    (rightResult := 209415) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209419

namespace LeftMerge209436
def owner : Owner := ⟨.program ⟨257⟩, ⟨40859⟩⟩
def mergeEvent : Nat := 209436
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }
def rhsRaw : List Term := Proof.Events818.exact209433RawTerms
def group : MergeGroup := .relation 209435
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 209435) (rhsResult := 209433)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 209434 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40856⟩⟩]⟩) (none) 209433) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209436

namespace LeftMerge209437
def owner : Owner := ⟨.program ⟨257⟩, ⟨40859⟩⟩
def mergeEvent : Nat := 209437
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩] } }
def rhsRaw : List Term := Proof.Events818.exact209433RawTerms
def group : MergeGroup := .relation 209435
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 209435) (rhsResult := 209433)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 209434 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40856⟩⟩]⟩) (none) 209433) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge209437

namespace LeftMerge209438
def owner : Owner := ⟨.program ⟨257⟩, ⟨40859⟩⟩
def mergeEvent : Nat := 209438
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41261⟩⟩] } }
def rhsRaw : List Term := Proof.Events818.exact209433RawTerms
def group : MergeGroup := .relation 209435
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 209435) (rhsResult := 209433)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 209434 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40856⟩⟩]⟩) (none) 209433) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41261⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40108⟩⟩], [⟨.program ⟨257⟩, ⟨41261⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209438

namespace LeftMerge209439
def owner : Owner := ⟨.program ⟨257⟩, ⟨40859⟩⟩
def mergeEvent : Nat := 209439
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events818.exact209433RawTerms
def group : MergeGroup := .relation 209435
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 209435) (rhsResult := 209433)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 209434 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40856⟩⟩]⟩) (none) 209433) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge209439

namespace LeftMerge209444
def owner : Owner := ⟨.program ⟨257⟩, ⟨41992⟩⟩
def mergeEvent : Nat := 209444
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩] } }
def leftRaw : List Term := Proof.Events818.exact209440RawTerms
def rightRaw : List Term := Proof.Events817.exact209262RawTerms
def group : MergeGroup := .operator 209440 209262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209440) (leftOrdinal := 0)
    (rightResult := 209262) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209444

namespace LeftMerge209445
def owner : Owner := ⟨.program ⟨257⟩, ⟨41992⟩⟩
def mergeEvent : Nat := 209445
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41261⟩⟩] } }
def leftRaw : List Term := Proof.Events818.exact209440RawTerms
def rightRaw : List Term := Proof.Events817.exact209262RawTerms
def group : MergeGroup := .operator 209440 209262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209440) (leftOrdinal := 2)
    (rightResult := 209262) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41261⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41261⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40108⟩⟩], [⟨.program ⟨257⟩, ⟨41261⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge209445

namespace LeftMerge209471
def owner : Owner := ⟨.program ⟨257⟩, ⟨37117⟩⟩
def mergeEvent : Nat := 209471
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37114⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events038.exact9910RawTerms
def rightRaw : List Term := Proof.Events810.exact207528RawTerms
def group : MergeGroup := .operator 9910 207528
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9910) (leftOrdinal := 0)
    (rightResult := 207528) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37114⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37114⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209471

namespace LeftMerge209476
def owner : Owner := ⟨.program ⟨257⟩, ⟨8587⟩⟩
def mergeEvent : Nat := 209476
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } }
def leftRaw : List Term := Proof.Events810.exact207398RawTerms
def rightRaw : List Term := Proof.Events074.exact19084RawTerms
def group : MergeGroup := .operator 207398 19084
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207398) (leftOrdinal := 0)
    (rightResult := 19084) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209476

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
