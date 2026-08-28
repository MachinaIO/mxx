import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge41586
def owner : Owner := ⟨.program ⟨214⟩, ⟨26002⟩⟩
def mergeEvent : Nat := 41586
def frameStart : Nat := 41494
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25999⟩⟩] } }
def leftRaw : List Term := Proof.Events162.exact41582RawTerms
def rightRaw : List Term := Proof.Events162.exact41539RawTerms
def group : MergeGroup := .operator 41582 41539
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41582) (leftOrdinal := 0)
    (rightResult := 41539) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25999⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25999⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41586

namespace LeftMerge41587
def owner : Owner := ⟨.program ⟨214⟩, ⟨26002⟩⟩
def mergeEvent : Nat := 41587
def frameStart : Nat := 41494
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11393⟩⟩, ⟨.program ⟨214⟩, ⟨14008⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25999⟩⟩] } }
def leftRaw : List Term := Proof.Events162.exact41582RawTerms
def rightRaw : List Term := Proof.Events162.exact41539RawTerms
def group : MergeGroup := .operator 41582 41539
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41582) (leftOrdinal := 1)
    (rightResult := 41539) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11393⟩⟩, ⟨.program ⟨214⟩, ⟨14008⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25999⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11393⟩⟩, ⟨.program ⟨214⟩, ⟨14008⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25999⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge41587

namespace LeftMerge41589
def owner : Owner := ⟨.program ⟨214⟩, ⟨26002⟩⟩
def mergeEvent : Nat := 41589
def frameStart : Nat := 41494
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11393⟩⟩, ⟨.program ⟨214⟩, ⟨14008⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23546⟩⟩] } }
def rhsRaw : List Term := Proof.Events162.exact41536RawTerms
def group : MergeGroup := .relation 41588
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 41588) (rhsResult := 41536)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11393⟩⟩, ⟨.program ⟨214⟩, ⟨14008⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25999⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25999⟩⟩) ⟨23546⟩ 41536) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23546⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11393⟩⟩, ⟨.program ⟨214⟩, ⟨14008⟩⟩], [⟨.program ⟨214⟩, ⟨23546⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge41589

namespace LeftMerge41597
def owner : Owner := ⟨.program ⟨214⟩, ⟨15831⟩⟩
def mergeEvent : Nat := 41597
def frameStart : Nat := 41494
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15829⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events162.exact41550RawTerms
def rightRaw : List Term := Proof.Events162.exact41593RawTerms
def group : MergeGroup := .operator 41550 41593
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41550) (leftOrdinal := 0)
    (rightResult := 41593) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15829⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41597

namespace LeftMerge41614
def owner : Owner := ⟨.program ⟨214⟩, ⟨19467⟩⟩
def mergeEvent : Nat := 41614
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩] } }
def rhsRaw : List Term := Proof.Events162.exact41611RawTerms
def group : MergeGroup := .relation 41613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 41613) (rhsResult := 41611)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19464⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 41612 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19464⟩⟩]⟩) (none) 41611) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41614

namespace LeftMerge41615
def owner : Owner := ⟨.program ⟨214⟩, ⟨19467⟩⟩
def mergeEvent : Nat := 41615
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25999⟩⟩] } }
def rhsRaw : List Term := Proof.Events162.exact41611RawTerms
def group : MergeGroup := .relation 41613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 41613) (rhsResult := 41611)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19464⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 41612 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19464⟩⟩]⟩) (none) 41611) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25999⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25999⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge41615

namespace LeftMerge41616
def owner : Owner := ⟨.program ⟨214⟩, ⟨19467⟩⟩
def mergeEvent : Nat := 41616
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11393⟩⟩, ⟨.program ⟨214⟩, ⟨14008⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23546⟩⟩] } }
def rhsRaw : List Term := Proof.Events162.exact41611RawTerms
def group : MergeGroup := .relation 41613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 41613) (rhsResult := 41611)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19464⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 41612 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19464⟩⟩]⟩) (none) 41611) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11393⟩⟩, ⟨.program ⟨214⟩, ⟨14008⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23546⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11393⟩⟩, ⟨.program ⟨214⟩, ⟨14008⟩⟩], [⟨.program ⟨214⟩, ⟨23546⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41616

namespace LeftMerge41617
def owner : Owner := ⟨.program ⟨214⟩, ⟨19467⟩⟩
def mergeEvent : Nat := 41617
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15829⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events162.exact41611RawTerms
def group : MergeGroup := .relation 41613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 41613) (rhsResult := 41611)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19464⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 41612 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19464⟩⟩]⟩) (none) 41611) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15829⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge41617

namespace LeftMerge41622
def owner : Owner := ⟨.program ⟨214⟩, ⟨26001⟩⟩
def mergeEvent : Nat := 41622
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11393⟩⟩, ⟨.program ⟨214⟩, ⟨14008⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23546⟩⟩] } }
def leftRaw : List Term := Proof.Events162.exact41618RawTerms
def rightRaw : List Term := Proof.Events161.exact41432RawTerms
def group : MergeGroup := .operator 41618 41432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41618) (leftOrdinal := 2)
    (rightResult := 41432) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11393⟩⟩, ⟨.program ⟨214⟩, ⟨14008⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23546⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11393⟩⟩, ⟨.program ⟨214⟩, ⟨14008⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23546⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11393⟩⟩, ⟨.program ⟨214⟩, ⟨14008⟩⟩], [⟨.program ⟨214⟩, ⟨23546⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge41622

namespace LeftMerge41623
def owner : Owner := ⟨.program ⟨214⟩, ⟨26001⟩⟩
def mergeEvent : Nat := 41623
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25999⟩⟩] } }
def leftRaw : List Term := Proof.Events162.exact41618RawTerms
def rightRaw : List Term := Proof.Events161.exact41432RawTerms
def group : MergeGroup := .operator 41618 41432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41618) (leftOrdinal := 1)
    (rightResult := 41432) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25999⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25999⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25999⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41623

namespace LeftMerge41631
def owner : Owner := ⟨.program ⟨214⟩, ⟨27677⟩⟩
def mergeEvent : Nat := 41631
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27675⟩⟩] } }
def leftRaw : List Term := Proof.Events162.exact41625RawTerms
def rightRaw : List Term := Proof.Events161.exact41348RawTerms
def group : MergeGroup := .operator 41625 41348
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41625) (leftOrdinal := 0)
    (rightResult := 41348) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27675⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27675⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41631

namespace LeftMerge41632
def owner : Owner := ⟨.program ⟨214⟩, ⟨27677⟩⟩
def mergeEvent : Nat := 41632
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15829⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27675⟩⟩] } }
def leftRaw : List Term := Proof.Events162.exact41625RawTerms
def rightRaw : List Term := Proof.Events161.exact41348RawTerms
def group : MergeGroup := .operator 41625 41348
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41625) (leftOrdinal := 1)
    (rightResult := 41348) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15829⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27675⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27675⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge41632

namespace LeftMerge41634
def owner : Owner := ⟨.program ⟨214⟩, ⟨27677⟩⟩
def mergeEvent : Nat := 41634
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15829⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24105⟩⟩] } }
def rhsRaw : List Term := Proof.Events161.exact41345RawTerms
def group : MergeGroup := .relation 41633
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 41633) (rhsResult := 41345)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27675⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27675⟩⟩) ⟨24105⟩ 41345) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24105⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨24105⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge41634

namespace LeftMerge41648
def owner : Owner := ⟨.program ⟨214⟩, ⟨21267⟩⟩
def mergeEvent : Nat := 41648
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21264⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events162.exact41642RawTerms
def group : MergeGroup := .operator 36137 41642
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 41642) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21264⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21264⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41648

namespace LeftMerge41769
def owner : Owner := ⟨.program ⟨214⟩, ⟨15906⟩⟩
def mergeEvent : Nat := 41769
def frameStart : Nat := 41703
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15829⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events163.exact41765RawTerms
def rightRaw : List Term := Proof.Events163.exact41763RawTerms
def group : MergeGroup := .operator 41765 41763
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41765) (leftOrdinal := 0)
    (rightResult := 41763) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15829⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41769

namespace LeftMerge41781
def owner : Owner := ⟨.program ⟨214⟩, ⟨27676⟩⟩
def mergeEvent : Nat := 41781
def frameStart : Nat := 41703
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27675⟩⟩] } }
def leftRaw : List Term := Proof.Events163.exact41777RawTerms
def rightRaw : List Term := Proof.Events163.exact41754RawTerms
def group : MergeGroup := .operator 41777 41754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41777) (leftOrdinal := 0)
    (rightResult := 41754) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27675⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27675⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41781

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
