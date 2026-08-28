import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge55439
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55439
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 10)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55439

namespace LeftMerge55440
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55440
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26723⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 21)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26723⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26723⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55440

namespace LeftMerge55442
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55442
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26723⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def rhsRaw : List Term := Proof.Events182.exact46625RawTerms
def group : MergeGroup := .relation 55441
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55441) (rhsResult := 46625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26723⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26723⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55442

namespace LeftMerge55443
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55443
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 9)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55443

namespace LeftMerge55444
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55444
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 35)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55444

namespace LeftMerge55446
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55446
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def rhsRaw : List Term := Proof.Events182.exact46625RawTerms
def group : MergeGroup := .relation 55445
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55445) (rhsResult := 46625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67161⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55446

namespace LeftMerge55447
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55447
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 8)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55447

namespace LeftMerge55448
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55448
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 34)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55448

namespace LeftMerge55450
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55450
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def rhsRaw : List Term := Proof.Events182.exact46625RawTerms
def group : MergeGroup := .relation 55449
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55449) (rhsResult := 46625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55450

namespace LeftMerge55451
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55451
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 7)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55451

namespace LeftMerge55452
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55452
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 33)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55452

namespace LeftMerge55454
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55454
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def rhsRaw : List Term := Proof.Events182.exact46625RawTerms
def group : MergeGroup := .relation 55453
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55453) (rhsResult := 46625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55454

namespace LeftMerge55455
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55455
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 6)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55455

namespace LeftMerge55456
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55456
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 32)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55456

namespace LeftMerge55458
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55458
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def rhsRaw : List Term := Proof.Events182.exact46625RawTerms
def group : MergeGroup := .relation 55457
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55457) (rhsResult := 46625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55458

namespace LeftMerge55459
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55459
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 5)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55459

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
