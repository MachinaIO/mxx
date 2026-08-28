import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge63286
def owner : Owner := ⟨.program ⟨214⟩, ⟨27875⟩⟩
def mergeEvent : Nat := 63286
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27872⟩⟩] } }
def leftRaw : List Term := Proof.Events247.exact63282RawTerms
def rightRaw : List Term := Proof.Events246.exact63104RawTerms
def group : MergeGroup := .operator 63282 63104
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63282) (leftOrdinal := 0)
    (rightResult := 63104) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27872⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27872⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27872⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63286

namespace LeftMerge63287
def owner : Owner := ⟨.program ⟨214⟩, ⟨27875⟩⟩
def mergeEvent : Nat := 63287
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15944⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24164⟩⟩] } }
def leftRaw : List Term := Proof.Events247.exact63282RawTerms
def rightRaw : List Term := Proof.Events246.exact63104RawTerms
def group : MergeGroup := .operator 63282 63104
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63282) (leftOrdinal := 2)
    (rightResult := 63104) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15944⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24164⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15944⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24164⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15944⟩⟩], [⟨.program ⟨214⟩, ⟨24164⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63287

namespace LeftMerge63295
def owner : Owner := ⟨.program ⟨214⟩, ⟨27876⟩⟩
def mergeEvent : Nat := 63295
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩] } }
def leftRaw : List Term := Proof.Events247.exact63289RawTerms
def rightRaw : List Term := Proof.Events022.exact5719RawTerms
def group : MergeGroup := .operator 63289 5719
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63289) (leftOrdinal := 0)
    (rightResult := 5719) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6722⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6641⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63295

namespace LeftMerge63296
def owner : Owner := ⟨.program ⟨214⟩, ⟨27876⟩⟩
def mergeEvent : Nat := 63296
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17169⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩] } }
def leftRaw : List Term := Proof.Events247.exact63289RawTerms
def rightRaw : List Term := Proof.Events022.exact5719RawTerms
def group : MergeGroup := .operator 63289 5719
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63289) (leftOrdinal := 1)
    (rightResult := 5719) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17169⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6641⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17169⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63296

namespace LeftMerge63298
def owner : Owner := ⟨.program ⟨214⟩, ⟨27876⟩⟩
def mergeEvent : Nat := 63298
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17169⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events022.exact5712RawTerms
def group : MergeGroup := .relation 63297
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63297) (rhsResult := 5712)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17169⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6641⟩⟩) ⟨6592⟩ 5712) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17169⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63298

namespace LeftMerge63312
def owner : Owner := ⟨.program ⟨214⟩, ⟨27657⟩⟩
def mergeEvent : Nat := 63312
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27655⟩⟩] } }
def leftRaw : List Term := Proof.Events219.exact56250RawTerms
def rightRaw : List Term := Proof.Events247.exact63306RawTerms
def group : MergeGroup := .operator 56250 63306
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56250) (leftOrdinal := 0)
    (rightResult := 63306) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27655⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27655⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63312

namespace LeftMerge63313
def owner : Owner := ⟨.program ⟨214⟩, ⟨27657⟩⟩
def mergeEvent : Nat := 63313
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15825⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27655⟩⟩] } }
def leftRaw : List Term := Proof.Events219.exact56250RawTerms
def rightRaw : List Term := Proof.Events247.exact63306RawTerms
def group : MergeGroup := .operator 56250 63306
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56250) (leftOrdinal := 1)
    (rightResult := 63306) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15825⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27655⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27655⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63313

namespace LeftMerge63315
def owner : Owner := ⟨.program ⟨214⟩, ⟨27657⟩⟩
def mergeEvent : Nat := 63315
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15825⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24101⟩⟩] } }
def rhsRaw : List Term := Proof.Events247.exact63303RawTerms
def group : MergeGroup := .relation 63314
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63314) (rhsResult := 63303)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27655⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27655⟩⟩) ⟨24101⟩ 63303) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24101⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨24101⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63315

namespace LeftMerge63329
def owner : Owner := ⟨.program ⟨214⟩, ⟨21191⟩⟩
def mergeEvent : Nat := 63329
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21188⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events247.exact63323RawTerms
def group : MergeGroup := .operator 50762 63323
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 63323) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21188⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21188⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63329

namespace LeftMerge63450
def owner : Owner := ⟨.program ⟨214⟩, ⟨15902⟩⟩
def mergeEvent : Nat := 63450
def frameStart : Nat := 63384
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15825⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events247.exact63446RawTerms
def rightRaw : List Term := Proof.Events247.exact63444RawTerms
def group : MergeGroup := .operator 63446 63444
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63446) (leftOrdinal := 0)
    (rightResult := 63444) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15825⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63450

namespace LeftMerge63462
def owner : Owner := ⟨.program ⟨214⟩, ⟨27656⟩⟩
def mergeEvent : Nat := 63462
def frameStart : Nat := 63384
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27655⟩⟩] } }
def leftRaw : List Term := Proof.Events247.exact63458RawTerms
def rightRaw : List Term := Proof.Events247.exact63435RawTerms
def group : MergeGroup := .operator 63458 63435
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63458) (leftOrdinal := 0)
    (rightResult := 63435) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27655⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27655⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63462

namespace LeftMerge63463
def owner : Owner := ⟨.program ⟨214⟩, ⟨27656⟩⟩
def mergeEvent : Nat := 63463
def frameStart : Nat := 63384
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15825⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27655⟩⟩] } }
def leftRaw : List Term := Proof.Events247.exact63458RawTerms
def rightRaw : List Term := Proof.Events247.exact63435RawTerms
def group : MergeGroup := .operator 63458 63435
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63458) (leftOrdinal := 1)
    (rightResult := 63435) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15825⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27655⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27655⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63463

namespace LeftMerge63465
def owner : Owner := ⟨.program ⟨214⟩, ⟨27656⟩⟩
def mergeEvent : Nat := 63465
def frameStart : Nat := 63384
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15825⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24101⟩⟩] } }
def rhsRaw : List Term := Proof.Events247.exact63432RawTerms
def group : MergeGroup := .relation 63464
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63464) (rhsResult := 63432)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27655⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27655⟩⟩) ⟨24101⟩ 63432) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24101⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨24101⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63465

namespace LeftMerge63473
def owner : Owner := ⟨.program ⟨214⟩, ⟨17227⟩⟩
def mergeEvent : Nat := 63473
def frameStart : Nat := 63384
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17225⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events247.exact63446RawTerms
def rightRaw : List Term := Proof.Events247.exact63469RawTerms
def group : MergeGroup := .operator 63446 63469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63446) (leftOrdinal := 0)
    (rightResult := 63469) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17225⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17225⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63473

namespace LeftMerge63490
def owner : Owner := ⟨.program ⟨214⟩, ⟨21191⟩⟩
def mergeEvent : Nat := 63490
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩] } }
def rhsRaw : List Term := Proof.Events247.exact63487RawTerms
def group : MergeGroup := .relation 63489
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63489) (rhsResult := 63487)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21188⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 63488 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21188⟩⟩]⟩) (none) 63487) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63490

namespace LeftMerge63491
def owner : Owner := ⟨.program ⟨214⟩, ⟨21191⟩⟩
def mergeEvent : Nat := 63491
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27655⟩⟩] } }
def rhsRaw : List Term := Proof.Events247.exact63487RawTerms
def group : MergeGroup := .relation 63489
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63489) (rhsResult := 63487)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21188⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 63488 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21188⟩⟩]⟩) (none) 63487) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27655⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27655⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63491

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
