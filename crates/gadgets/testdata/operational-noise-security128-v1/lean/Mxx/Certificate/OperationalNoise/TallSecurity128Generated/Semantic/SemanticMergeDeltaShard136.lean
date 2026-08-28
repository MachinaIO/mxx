import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge27372
def owner : Owner := ⟨.program ⟨257⟩, ⟨69053⟩⟩
def mergeEvent : Nat := 27372
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events106.exact27360RawTerms
def rightRaw : List Term := Proof.Events106.exact27358RawTerms
def group : MergeGroup := .operator 27360 27358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27360) (leftOrdinal := 0)
    (rightResult := 27358) (rightOrdinal := 17) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65993⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65993⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27372

namespace LeftMerge27373
def owner : Owner := ⟨.program ⟨257⟩, ⟨69053⟩⟩
def mergeEvent : Nat := 27373
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62915⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events106.exact27360RawTerms
def rightRaw : List Term := Proof.Events106.exact27358RawTerms
def group : MergeGroup := .operator 27360 27358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27360) (leftOrdinal := 0)
    (rightResult := 27358) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62915⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27373

namespace LeftMerge27374
def owner : Owner := ⟨.program ⟨257⟩, ⟨69053⟩⟩
def mergeEvent : Nat := 27374
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59935⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events106.exact27360RawTerms
def rightRaw : List Term := Proof.Events106.exact27358RawTerms
def group : MergeGroup := .operator 27360 27358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27360) (leftOrdinal := 0)
    (rightResult := 27358) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59935⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27374

namespace LeftMerge27375
def owner : Owner := ⟨.program ⟨257⟩, ⟨69053⟩⟩
def mergeEvent : Nat := 27375
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56955⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events106.exact27360RawTerms
def rightRaw : List Term := Proof.Events106.exact27358RawTerms
def group : MergeGroup := .operator 27360 27358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27360) (leftOrdinal := 0)
    (rightResult := 27358) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56955⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27375

namespace LeftMerge27376
def owner : Owner := ⟨.program ⟨257⟩, ⟨69053⟩⟩
def mergeEvent : Nat := 27376
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53975⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events106.exact27360RawTerms
def rightRaw : List Term := Proof.Events106.exact27358RawTerms
def group : MergeGroup := .operator 27360 27358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27360) (leftOrdinal := 0)
    (rightResult := 27358) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53975⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27376

namespace LeftMerge27377
def owner : Owner := ⟨.program ⟨257⟩, ⟨69053⟩⟩
def mergeEvent : Nat := 27377
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50995⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events106.exact27360RawTerms
def rightRaw : List Term := Proof.Events106.exact27358RawTerms
def group : MergeGroup := .operator 27360 27358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27360) (leftOrdinal := 0)
    (rightResult := 27358) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50995⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27377

namespace LeftMerge27378
def owner : Owner := ⟨.program ⟨257⟩, ⟨69053⟩⟩
def mergeEvent : Nat := 27378
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31940⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events106.exact27360RawTerms
def rightRaw : List Term := Proof.Events106.exact27358RawTerms
def group : MergeGroup := .operator 27360 27358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27360) (leftOrdinal := 0)
    (rightResult := 27358) (rightOrdinal := 5) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31940⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27378

namespace LeftMerge27379
def owner : Owner := ⟨.program ⟨257⟩, ⟨69053⟩⟩
def mergeEvent : Nat := 27379
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21920⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events106.exact27360RawTerms
def rightRaw : List Term := Proof.Events106.exact27358RawTerms
def group : MergeGroup := .operator 27360 27358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27360) (leftOrdinal := 0)
    (rightResult := 27358) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21920⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27379

namespace LeftMerge27380
def owner : Owner := ⟨.program ⟨257⟩, ⟨69053⟩⟩
def mergeEvent : Nat := 27380
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18700⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events106.exact27360RawTerms
def rightRaw : List Term := Proof.Events106.exact27358RawTerms
def group : MergeGroup := .operator 27360 27358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27360) (leftOrdinal := 0)
    (rightResult := 27358) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18700⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27380

namespace LeftMerge27381
def owner : Owner := ⟨.program ⟨257⟩, ⟨69053⟩⟩
def mergeEvent : Nat := 27381
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15895⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events106.exact27360RawTerms
def rightRaw : List Term := Proof.Events106.exact27358RawTerms
def group : MergeGroup := .operator 27360 27358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27360) (leftOrdinal := 0)
    (rightResult := 27358) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15895⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27381

namespace LeftMerge27512
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27512
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48249⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 29)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48249⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48249⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27512

namespace LeftMerge27514
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27514
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48249⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }
def rhsRaw : List Term := Proof.Events106.exact27346RawTerms
def group : MergeGroup := .relation 27513
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27513) (rhsResult := 27346)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48249⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 27346) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48249⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27514

namespace LeftMerge27515
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27515
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 17)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27515

namespace LeftMerge27516
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27516
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45569⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 28)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45569⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45569⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27516

namespace LeftMerge27518
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27518
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45569⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }
def rhsRaw : List Term := Proof.Events106.exact27346RawTerms
def group : MergeGroup := .relation 27517
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27517) (rhsResult := 27346)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45569⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 27346) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45569⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27518

namespace LeftMerge27519
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27519
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 16)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27519

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
