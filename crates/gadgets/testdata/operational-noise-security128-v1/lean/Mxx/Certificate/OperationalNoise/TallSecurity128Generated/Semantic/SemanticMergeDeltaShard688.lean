import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge115273
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115273
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 9)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115273

namespace LeftMerge115274
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115274
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 8)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115274

namespace LeftMerge115275
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115275
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 7)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115275

namespace LeftMerge115276
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115276
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 6)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115276

namespace LeftMerge115277
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115277
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 5)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115277

namespace LeftMerge115278
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115278
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 4)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115278

namespace LeftMerge115279
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115279
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 3)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115279

namespace LeftMerge115280
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115280
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 2)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115280

namespace LeftMerge115281
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115281
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 1)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115281

namespace LeftMerge115282
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115282
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 0)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge115282

namespace LeftMerge115283
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115283
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48376⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 29)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48376⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48376⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115283

namespace LeftMerge115285
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115285
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48376⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events449.exact115099RawTerms
def group : MergeGroup := .relation 115284
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115284) (rhsResult := 115099)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48376⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71267⟩⟩) ⟨68836⟩ 115099) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48376⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115285

namespace LeftMerge115286
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115286
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45696⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 28)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45696⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45696⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115286

namespace LeftMerge115288
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115288
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45696⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events449.exact115099RawTerms
def group : MergeGroup := .relation 115287
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115287) (rhsResult := 115099)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45696⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71267⟩⟩) ⟨68836⟩ 115099) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45696⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115288

namespace LeftMerge115289
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115289
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨43012⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 27)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨43012⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨43012⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115289

namespace LeftMerge115291
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115291
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨43012⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events449.exact115099RawTerms
def group : MergeGroup := .relation 115290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115290) (rhsResult := 115099)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨43012⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71267⟩⟩) ⟨68836⟩ 115099) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨43012⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115291

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
