import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge63492
def owner : Owner := ⟨.program ⟨214⟩, ⟨21191⟩⟩
def mergeEvent : Nat := 63492
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15825⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24101⟩⟩] } }
def rhsRaw : List Term := Proof.Events247.exact63487RawTerms
def group : MergeGroup := .relation 63489
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63489) (rhsResult := 63487)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21188⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 63488 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21188⟩⟩]⟩) (none) 63487) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15825⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24101⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨24101⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63492

namespace LeftMerge63493
def owner : Owner := ⟨.program ⟨214⟩, ⟨21191⟩⟩
def mergeEvent : Nat := 63493
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17225⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events247.exact63487RawTerms
def group : MergeGroup := .relation 63489
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63489) (rhsResult := 63487)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21188⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 63488 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21188⟩⟩]⟩) (none) 63487) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17225⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17225⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63493

namespace LeftMerge63498
def owner : Owner := ⟨.program ⟨214⟩, ⟨27658⟩⟩
def mergeEvent : Nat := 63498
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27655⟩⟩] } }
def leftRaw : List Term := Proof.Events248.exact63494RawTerms
def rightRaw : List Term := Proof.Events247.exact63316RawTerms
def group : MergeGroup := .operator 63494 63316
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63494) (leftOrdinal := 0)
    (rightResult := 63316) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27655⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27655⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27655⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63498

namespace LeftMerge63499
def owner : Owner := ⟨.program ⟨214⟩, ⟨27658⟩⟩
def mergeEvent : Nat := 63499
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15825⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24101⟩⟩] } }
def leftRaw : List Term := Proof.Events248.exact63494RawTerms
def rightRaw : List Term := Proof.Events247.exact63316RawTerms
def group : MergeGroup := .operator 63494 63316
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63494) (leftOrdinal := 2)
    (rightResult := 63316) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15825⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24101⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15825⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24101⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨24101⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63499

namespace LeftMerge63507
def owner : Owner := ⟨.program ⟨214⟩, ⟨27659⟩⟩
def mergeEvent : Nat := 63507
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩] } }
def leftRaw : List Term := Proof.Events248.exact63501RawTerms
def rightRaw : List Term := Proof.Events022.exact5739RawTerms
def group : MergeGroup := .operator 63501 5739
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63501) (leftOrdinal := 0)
    (rightResult := 5739) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6643⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63507

namespace LeftMerge63508
def owner : Owner := ⟨.program ⟨214⟩, ⟨27659⟩⟩
def mergeEvent : Nat := 63508
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17225⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩] } }
def leftRaw : List Term := Proof.Events248.exact63501RawTerms
def rightRaw : List Term := Proof.Events022.exact5739RawTerms
def group : MergeGroup := .operator 63501 5739
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63501) (leftOrdinal := 1)
    (rightResult := 5739) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17225⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6643⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17225⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63508

namespace LeftMerge63510
def owner : Owner := ⟨.program ⟨214⟩, ⟨27659⟩⟩
def mergeEvent : Nat := 63510
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17225⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events022.exact5732RawTerms
def group : MergeGroup := .relation 63509
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63509) (rhsResult := 5732)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17225⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6643⟩⟩) ⟨6593⟩ 5732) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17225⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63510

namespace LeftMerge63524
def owner : Owner := ⟨.program ⟨214⟩, ⟨27440⟩⟩
def mergeEvent : Nat := 63524
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27438⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56732RawTerms
def rightRaw : List Term := Proof.Events248.exact63518RawTerms
def group : MergeGroup := .operator 56732 63518
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56732) (leftOrdinal := 0)
    (rightResult := 63518) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27438⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27438⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63524

namespace LeftMerge63525
def owner : Owner := ⟨.program ⟨214⟩, ⟨27440⟩⟩
def mergeEvent : Nat := 63525
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15706⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27438⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56732RawTerms
def rightRaw : List Term := Proof.Events248.exact63518RawTerms
def group : MergeGroup := .operator 56732 63518
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56732) (leftOrdinal := 1)
    (rightResult := 63518) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15706⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27438⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15706⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27438⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63525

namespace LeftMerge63527
def owner : Owner := ⟨.program ⟨214⟩, ⟨27440⟩⟩
def mergeEvent : Nat := 63527
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15706⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24038⟩⟩] } }
def rhsRaw : List Term := Proof.Events248.exact63515RawTerms
def group : MergeGroup := .relation 63526
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63526) (rhsResult := 63515)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15706⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27438⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27438⟩⟩) ⟨24038⟩ 63515) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24038⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15706⟩⟩], [⟨.program ⟨214⟩, ⟨24038⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63527

namespace LeftMerge63541
def owner : Owner := ⟨.program ⟨214⟩, ⟨21047⟩⟩
def mergeEvent : Nat := 63541
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21044⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events248.exact63535RawTerms
def group : MergeGroup := .operator 50762 63535
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 63535) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21044⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21044⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63541

namespace LeftMerge63662
def owner : Owner := ⟨.program ⟨214⟩, ⟨15783⟩⟩
def mergeEvent : Nat := 63662
def frameStart : Nat := 63596
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15706⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events248.exact63658RawTerms
def rightRaw : List Term := Proof.Events248.exact63656RawTerms
def group : MergeGroup := .operator 63658 63656
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63658) (leftOrdinal := 0)
    (rightResult := 63656) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15706⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15706⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63662

namespace LeftMerge63674
def owner : Owner := ⟨.program ⟨214⟩, ⟨27439⟩⟩
def mergeEvent : Nat := 63674
def frameStart : Nat := 63596
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27438⟩⟩] } }
def leftRaw : List Term := Proof.Events248.exact63670RawTerms
def rightRaw : List Term := Proof.Events248.exact63647RawTerms
def group : MergeGroup := .operator 63670 63647
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63670) (leftOrdinal := 0)
    (rightResult := 63647) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27438⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27438⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63674

namespace LeftMerge63675
def owner : Owner := ⟨.program ⟨214⟩, ⟨27439⟩⟩
def mergeEvent : Nat := 63675
def frameStart : Nat := 63596
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15706⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27438⟩⟩] } }
def leftRaw : List Term := Proof.Events248.exact63670RawTerms
def rightRaw : List Term := Proof.Events248.exact63647RawTerms
def group : MergeGroup := .operator 63670 63647
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63670) (leftOrdinal := 1)
    (rightResult := 63647) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15706⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27438⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15706⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27438⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63675

namespace LeftMerge63677
def owner : Owner := ⟨.program ⟨214⟩, ⟨27439⟩⟩
def mergeEvent : Nat := 63677
def frameStart : Nat := 63596
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15706⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24038⟩⟩] } }
def rhsRaw : List Term := Proof.Events248.exact63644RawTerms
def group : MergeGroup := .relation 63676
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63676) (rhsResult := 63644)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15706⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27438⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27438⟩⟩) ⟨24038⟩ 63644) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24038⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15706⟩⟩], [⟨.program ⟨214⟩, ⟨24038⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63677

namespace LeftMerge63685
def owner : Owner := ⟨.program ⟨214⟩, ⟨17444⟩⟩
def mergeEvent : Nat := 63685
def frameStart : Nat := 63596
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17442⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events248.exact63658RawTerms
def rightRaw : List Term := Proof.Events248.exact63681RawTerms
def group : MergeGroup := .operator 63658 63681
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63658) (leftOrdinal := 0)
    (rightResult := 63681) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17442⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17442⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63685

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
