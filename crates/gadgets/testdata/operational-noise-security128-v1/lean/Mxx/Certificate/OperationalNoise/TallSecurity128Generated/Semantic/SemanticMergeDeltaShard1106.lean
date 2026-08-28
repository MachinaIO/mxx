import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge180290
def owner : Owner := ⟨.program ⟨257⟩, ⟨38973⟩⟩
def mergeEvent : Nat := 180290
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38972⟩⟩] } }
def leftRaw : List Term := Proof.Events704.exact180281RawTerms
def rightRaw : List Term := Proof.Events703.exact180217RawTerms
def group : MergeGroup := .operator 180281 180217
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 180281) (leftOrdinal := 0)
    (rightResult := 180217) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38972⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38972⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge180290

namespace LeftMerge180304
def owner : Owner := ⟨.program ⟨257⟩, ⟨37902⟩⟩
def mergeEvent : Nat := 180304
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37899⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events704.exact180298RawTerms
def group : MergeGroup := .operator 178370 180298
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 180298) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨37899⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37899⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge180304

namespace LeftMerge180383
def owner : Owner := ⟨.program ⟨257⟩, ⟨37187⟩⟩
def mergeEvent : Nat := 180383
def frameStart : Nat := 180353
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13926⟩⟩, ⟨.program ⟨257⟩, ⟨37186⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events704.exact180379RawTerms
def rightRaw : List Term := Proof.Events704.exact180376RawTerms
def group : MergeGroup := .operator 180379 180376
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 180379) (leftOrdinal := 0)
    (rightResult := 180376) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13926⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37186⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13926⟩⟩, ⟨.program ⟨257⟩, ⟨37186⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge180383

namespace LeftMerge180413
def owner : Owner := ⟨.program ⟨257⟩, ⟨38720⟩⟩
def mergeEvent : Nat := 180413
def frameStart : Nat := 180353
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13926⟩⟩, ⟨.program ⟨257⟩, ⟨37186⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events704.exact180409RawTerms
def rightRaw : List Term := Proof.Events704.exact180407RawTerms
def group : MergeGroup := .operator 180409 180407
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 180409) (leftOrdinal := 0)
    (rightResult := 180407) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13926⟩⟩, ⟨.program ⟨257⟩, ⟨37186⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13926⟩⟩, ⟨.program ⟨257⟩, ⟨37186⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge180413

namespace LeftMerge180436
def owner : Owner := ⟨.program ⟨257⟩, ⟨9555⟩⟩
def mergeEvent : Nat := 180436
def frameStart : Nat := 180353
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }
def leftRaw : List Term := Proof.Events704.exact180432RawTerms
def rightRaw : List Term := Proof.Events704.exact180429RawTerms
def group : MergeGroup := .operator 180432 180429
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 180432) (leftOrdinal := 0)
    (rightResult := 180429) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9553⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge180436

namespace LeftMerge180445
def owner : Owner := ⟨.program ⟨257⟩, ⟨38975⟩⟩
def mergeEvent : Nat := 180445
def frameStart : Nat := 180353
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38972⟩⟩] } }
def leftRaw : List Term := Proof.Events704.exact180441RawTerms
def rightRaw : List Term := Proof.Events704.exact180398RawTerms
def group : MergeGroup := .operator 180441 180398
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 180441) (leftOrdinal := 0)
    (rightResult := 180398) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38972⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38972⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge180445

namespace LeftMerge180446
def owner : Owner := ⟨.program ⟨257⟩, ⟨38975⟩⟩
def mergeEvent : Nat := 180446
def frameStart : Nat := 180353
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13926⟩⟩, ⟨.program ⟨257⟩, ⟨37186⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38972⟩⟩] } }
def leftRaw : List Term := Proof.Events704.exact180441RawTerms
def rightRaw : List Term := Proof.Events704.exact180398RawTerms
def group : MergeGroup := .operator 180441 180398
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 180441) (leftOrdinal := 1)
    (rightResult := 180398) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13926⟩⟩, ⟨.program ⟨257⟩, ⟨37186⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38972⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13926⟩⟩, ⟨.program ⟨257⟩, ⟨37186⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38972⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge180446

namespace LeftMerge180448
def owner : Owner := ⟨.program ⟨257⟩, ⟨38975⟩⟩
def mergeEvent : Nat := 180448
def frameStart : Nat := 180353
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13926⟩⟩, ⟨.program ⟨257⟩, ⟨37186⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38447⟩⟩] } }
def rhsRaw : List Term := Proof.Events704.exact180395RawTerms
def group : MergeGroup := .relation 180447
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 180447) (rhsResult := 180395)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13926⟩⟩, ⟨.program ⟨257⟩, ⟨37186⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38972⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38972⟩⟩) ⟨38447⟩ 180395) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38447⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13926⟩⟩, ⟨.program ⟨257⟩, ⟨37186⟩⟩], [⟨.program ⟨257⟩, ⟨38447⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge180448

namespace LeftMerge180456
def owner : Owner := ⟨.program ⟨257⟩, ⟨37454⟩⟩
def mergeEvent : Nat := 180456
def frameStart : Nat := 180353
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events704.exact180409RawTerms
def rightRaw : List Term := Proof.Events704.exact180452RawTerms
def group : MergeGroup := .operator 180409 180452
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 180409) (leftOrdinal := 0)
    (rightResult := 180452) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37452⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37452⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge180456

namespace LeftMerge180473
def owner : Owner := ⟨.program ⟨257⟩, ⟨37902⟩⟩
def mergeEvent : Nat := 180473
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }
def rhsRaw : List Term := Proof.Events704.exact180470RawTerms
def group : MergeGroup := .relation 180472
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 180472) (rhsResult := 180470)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37899⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 180471 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37899⟩⟩]⟩) (none) 180470) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge180473

namespace LeftMerge180474
def owner : Owner := ⟨.program ⟨257⟩, ⟨37902⟩⟩
def mergeEvent : Nat := 180474
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38972⟩⟩] } }
def rhsRaw : List Term := Proof.Events704.exact180470RawTerms
def group : MergeGroup := .relation 180472
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 180472) (rhsResult := 180470)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37899⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 180471 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37899⟩⟩]⟩) (none) 180470) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38972⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38972⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge180474

namespace LeftMerge180475
def owner : Owner := ⟨.program ⟨257⟩, ⟨37902⟩⟩
def mergeEvent : Nat := 180475
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13926⟩⟩, ⟨.program ⟨257⟩, ⟨37186⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38447⟩⟩] } }
def rhsRaw : List Term := Proof.Events704.exact180470RawTerms
def group : MergeGroup := .relation 180472
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 180472) (rhsResult := 180470)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37899⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 180471 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37899⟩⟩]⟩) (none) 180470) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13926⟩⟩, ⟨.program ⟨257⟩, ⟨37186⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38447⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13926⟩⟩, ⟨.program ⟨257⟩, ⟨37186⟩⟩], [⟨.program ⟨257⟩, ⟨38447⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge180475

namespace LeftMerge180476
def owner : Owner := ⟨.program ⟨257⟩, ⟨37902⟩⟩
def mergeEvent : Nat := 180476
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events704.exact180470RawTerms
def group : MergeGroup := .relation 180472
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 180472) (rhsResult := 180470)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37899⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 180471 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37899⟩⟩]⟩) (none) 180470) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37452⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge180476

namespace LeftMerge180481
def owner : Owner := ⟨.program ⟨257⟩, ⟨38974⟩⟩
def mergeEvent : Nat := 180481
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13926⟩⟩, ⟨.program ⟨257⟩, ⟨37186⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38447⟩⟩] } }
def leftRaw : List Term := Proof.Events704.exact180477RawTerms
def rightRaw : List Term := Proof.Events704.exact180291RawTerms
def group : MergeGroup := .operator 180477 180291
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 180477) (leftOrdinal := 2)
    (rightResult := 180291) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13926⟩⟩, ⟨.program ⟨257⟩, ⟨37186⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38447⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13926⟩⟩, ⟨.program ⟨257⟩, ⟨37186⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38447⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13926⟩⟩, ⟨.program ⟨257⟩, ⟨37186⟩⟩], [⟨.program ⟨257⟩, ⟨38447⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge180481

namespace LeftMerge180482
def owner : Owner := ⟨.program ⟨257⟩, ⟨38974⟩⟩
def mergeEvent : Nat := 180482
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38972⟩⟩] } }
def leftRaw : List Term := Proof.Events704.exact180477RawTerms
def rightRaw : List Term := Proof.Events704.exact180291RawTerms
def group : MergeGroup := .operator 180477 180291
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 180477) (leftOrdinal := 1)
    (rightResult := 180291) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38972⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38972⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38972⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge180482

namespace LeftMerge180490
def owner : Owner := ⟨.program ⟨257⟩, ⟨39386⟩⟩
def mergeEvent : Nat := 180490
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39384⟩⟩] } }
def leftRaw : List Term := Proof.Events705.exact180484RawTerms
def rightRaw : List Term := Proof.Events703.exact180207RawTerms
def group : MergeGroup := .operator 180484 180207
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 180484) (leftOrdinal := 0)
    (rightResult := 180207) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39384⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39384⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge180490

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
