import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge141358
def owner : Owner := ⟨.program ⟨257⟩, ⟨33200⟩⟩
def mergeEvent : Nat := 141358
def frameStart : Nat := 141298
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24206⟩⟩, ⟨.program ⟨257⟩, ⟨31296⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events552.exact141354RawTerms
def rightRaw : List Term := Proof.Events552.exact141352RawTerms
def group : MergeGroup := .operator 141354 141352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141354) (leftOrdinal := 0)
    (rightResult := 141352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24206⟩⟩, ⟨.program ⟨257⟩, ⟨31296⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24206⟩⟩, ⟨.program ⟨257⟩, ⟨31296⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141358

namespace LeftMerge141381
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def mergeEvent : Nat := 141381
def frameStart : Nat := 141298
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }
def leftRaw : List Term := Proof.Events552.exact141377RawTerms
def rightRaw : List Term := Proof.Events552.exact141374RawTerms
def group : MergeGroup := .operator 141377 141374
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141377) (leftOrdinal := 0)
    (rightResult := 141374) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9577⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141381

namespace LeftMerge141390
def owner : Owner := ⟨.program ⟨257⟩, ⟨33385⟩⟩
def mergeEvent : Nat := 141390
def frameStart : Nat := 141298
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33382⟩⟩] } }
def leftRaw : List Term := Proof.Events552.exact141386RawTerms
def rightRaw : List Term := Proof.Events552.exact141343RawTerms
def group : MergeGroup := .operator 141386 141343
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141386) (leftOrdinal := 0)
    (rightResult := 141343) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33382⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33382⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141390

namespace LeftMerge141391
def owner : Owner := ⟨.program ⟨257⟩, ⟨33385⟩⟩
def mergeEvent : Nat := 141391
def frameStart : Nat := 141298
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24206⟩⟩, ⟨.program ⟨257⟩, ⟨31296⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33382⟩⟩] } }
def leftRaw : List Term := Proof.Events552.exact141386RawTerms
def rightRaw : List Term := Proof.Events552.exact141343RawTerms
def group : MergeGroup := .operator 141386 141343
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141386) (leftOrdinal := 1)
    (rightResult := 141343) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24206⟩⟩, ⟨.program ⟨257⟩, ⟨31296⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33382⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24206⟩⟩, ⟨.program ⟨257⟩, ⟨31296⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33382⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141391

namespace LeftMerge141393
def owner : Owner := ⟨.program ⟨257⟩, ⟨33385⟩⟩
def mergeEvent : Nat := 141393
def frameStart : Nat := 141298
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24206⟩⟩, ⟨.program ⟨257⟩, ⟨31296⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32907⟩⟩] } }
def rhsRaw : List Term := Proof.Events552.exact141340RawTerms
def group : MergeGroup := .relation 141392
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 141392) (rhsResult := 141340)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24206⟩⟩, ⟨.program ⟨257⟩, ⟨31296⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33382⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33382⟩⟩) ⟨32907⟩ 141340) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32907⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24206⟩⟩, ⟨.program ⟨257⟩, ⟨31296⟩⟩], [⟨.program ⟨257⟩, ⟨32907⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141393

namespace LeftMerge141401
def owner : Owner := ⟨.program ⟨257⟩, ⟨31774⟩⟩
def mergeEvent : Nat := 141401
def frameStart : Nat := 141298
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events552.exact141354RawTerms
def rightRaw : List Term := Proof.Events552.exact141397RawTerms
def group : MergeGroup := .operator 141354 141397
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141354) (leftOrdinal := 0)
    (rightResult := 141397) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31772⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141401

namespace LeftMerge141418
def owner : Owner := ⟨.program ⟨257⟩, ⟨32322⟩⟩
def mergeEvent : Nat := 141418
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }
def rhsRaw : List Term := Proof.Events552.exact141415RawTerms
def group : MergeGroup := .relation 141417
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 141417) (rhsResult := 141415)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32319⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 141416 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32319⟩⟩]⟩) (none) 141415) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141418

namespace LeftMerge141419
def owner : Owner := ⟨.program ⟨257⟩, ⟨32322⟩⟩
def mergeEvent : Nat := 141419
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33382⟩⟩] } }
def rhsRaw : List Term := Proof.Events552.exact141415RawTerms
def group : MergeGroup := .relation 141417
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 141417) (rhsResult := 141415)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32319⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 141416 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32319⟩⟩]⟩) (none) 141415) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33382⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33382⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141419

namespace LeftMerge141420
def owner : Owner := ⟨.program ⟨257⟩, ⟨32322⟩⟩
def mergeEvent : Nat := 141420
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24206⟩⟩, ⟨.program ⟨257⟩, ⟨31296⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32907⟩⟩] } }
def rhsRaw : List Term := Proof.Events552.exact141415RawTerms
def group : MergeGroup := .relation 141417
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 141417) (rhsResult := 141415)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32319⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 141416 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32319⟩⟩]⟩) (none) 141415) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24206⟩⟩, ⟨.program ⟨257⟩, ⟨31296⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32907⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24206⟩⟩, ⟨.program ⟨257⟩, ⟨31296⟩⟩], [⟨.program ⟨257⟩, ⟨32907⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141420

namespace LeftMerge141421
def owner : Owner := ⟨.program ⟨257⟩, ⟨32322⟩⟩
def mergeEvent : Nat := 141421
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events552.exact141415RawTerms
def group : MergeGroup := .relation 141417
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 141417) (rhsResult := 141415)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32319⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 141416 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32319⟩⟩]⟩) (none) 141415) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141421

namespace LeftMerge141426
def owner : Owner := ⟨.program ⟨257⟩, ⟨33384⟩⟩
def mergeEvent : Nat := 141426
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24206⟩⟩, ⟨.program ⟨257⟩, ⟨31296⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32907⟩⟩] } }
def leftRaw : List Term := Proof.Events552.exact141422RawTerms
def rightRaw : List Term := Proof.Events551.exact141236RawTerms
def group : MergeGroup := .operator 141422 141236
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141422) (leftOrdinal := 2)
    (rightResult := 141236) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24206⟩⟩, ⟨.program ⟨257⟩, ⟨31296⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32907⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24206⟩⟩, ⟨.program ⟨257⟩, ⟨31296⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32907⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24206⟩⟩, ⟨.program ⟨257⟩, ⟨31296⟩⟩], [⟨.program ⟨257⟩, ⟨32907⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141426

namespace LeftMerge141427
def owner : Owner := ⟨.program ⟨257⟩, ⟨33384⟩⟩
def mergeEvent : Nat := 141427
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33382⟩⟩] } }
def leftRaw : List Term := Proof.Events552.exact141422RawTerms
def rightRaw : List Term := Proof.Events551.exact141236RawTerms
def group : MergeGroup := .operator 141422 141236
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141422) (leftOrdinal := 1)
    (rightResult := 141236) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33382⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33382⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33382⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141427

namespace LeftMerge141435
def owner : Owner := ⟨.program ⟨257⟩, ⟨33677⟩⟩
def mergeEvent : Nat := 141435
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33675⟩⟩] } }
def leftRaw : List Term := Proof.Events552.exact141429RawTerms
def rightRaw : List Term := Proof.Events551.exact141152RawTerms
def group : MergeGroup := .operator 141429 141152
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141429) (leftOrdinal := 0)
    (rightResult := 141152) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33675⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33675⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141435

namespace LeftMerge141436
def owner : Owner := ⟨.program ⟨257⟩, ⟨33677⟩⟩
def mergeEvent : Nat := 141436
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33675⟩⟩] } }
def leftRaw : List Term := Proof.Events552.exact141429RawTerms
def rightRaw : List Term := Proof.Events551.exact141152RawTerms
def group : MergeGroup := .operator 141429 141152
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141429) (leftOrdinal := 1)
    (rightResult := 141152) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33675⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33675⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141436

namespace LeftMerge141438
def owner : Owner := ⟨.program ⟨257⟩, ⟨33677⟩⟩
def mergeEvent : Nat := 141438
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33038⟩⟩] } }
def rhsRaw : List Term := Proof.Events551.exact141149RawTerms
def group : MergeGroup := .relation 141437
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 141437) (rhsResult := 141149)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33675⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33675⟩⟩) ⟨33038⟩ 141149) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33038⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31772⟩⟩], [⟨.program ⟨257⟩, ⟨33038⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141438

namespace LeftMerge141452
def owner : Owner := ⟨.program ⟨257⟩, ⟨32559⟩⟩
def mergeEvent : Nat := 141452
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32556⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134495RawTerms
def rightRaw : List Term := Proof.Events552.exact141446RawTerms
def group : MergeGroup := .operator 134495 141446
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134495) (leftOrdinal := 0)
    (rightResult := 141446) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32556⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141452

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
