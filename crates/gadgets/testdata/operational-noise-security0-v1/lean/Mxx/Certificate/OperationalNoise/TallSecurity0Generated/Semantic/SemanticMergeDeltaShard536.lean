import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge87668
def owner : Owner := ⟨.program ⟨214⟩, ⟨9509⟩⟩
def mergeEvent : Nat := 87668
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩] } }
def leftRaw : List Term := Proof.Events342.exact87659RawTerms
def rightRaw : List Term := Proof.Events056.exact14518RawTerms
def group : MergeGroup := .operator 87659 14518
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87659) (leftOrdinal := 0)
    (rightResult := 14518) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7834⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87668

namespace LeftMerge87673
def owner : Owner := ⟨.program ⟨214⟩, ⟨10683⟩⟩
def mergeEvent : Nat := 87673
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9505⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } }
def leftRaw : List Term := Proof.Events342.exact87669RawTerms
def rightRaw : List Term := Proof.Events342.exact87639RawTerms
def group : MergeGroup := .operator 87669 87639
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87669) (leftOrdinal := 1)
    (rightResult := 87639) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9505⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9505⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9505⟩⟩], [⟨.program ⟨214⟩, ⟨6773⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87673

namespace LeftMerge87681
def owner : Owner := ⟨.program ⟨214⟩, ⟨24989⟩⟩
def mergeEvent : Nat := 87681
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩] } }
def leftRaw : List Term := Proof.Events342.exact87675RawTerms
def rightRaw : List Term := Proof.Events342.exact87611RawTerms
def group : MergeGroup := .operator 87675 87611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87675) (leftOrdinal := 1)
    (rightResult := 87611) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24988⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87681

namespace LeftMerge87683
def owner : Owner := ⟨.program ⟨214⟩, ⟨24989⟩⟩
def mergeEvent : Nat := 87683
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22996⟩⟩] } }
def rhsRaw : List Term := Proof.Events342.exact87608RawTerms
def group : MergeGroup := .relation 87682
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87682) (rhsResult := 87608)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨24988⟩⟩) ⟨22996⟩ 87608) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22996⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], [⟨.program ⟨214⟩, ⟨22996⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87683

namespace LeftMerge87684
def owner : Owner := ⟨.program ⟨214⟩, ⟨24989⟩⟩
def mergeEvent : Nat := 87684
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩] } }
def leftRaw : List Term := Proof.Events342.exact87675RawTerms
def rightRaw : List Term := Proof.Events342.exact87611RawTerms
def group : MergeGroup := .operator 87675 87611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87675) (leftOrdinal := 0)
    (rightResult := 87611) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24988⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87684

namespace LeftMerge87698
def owner : Owner := ⟨.program ⟨214⟩, ⟨19099⟩⟩
def mergeEvent : Nat := 87698
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19096⟩⟩] } }
def leftRaw : List Term := Proof.Events312.exact80012RawTerms
def rightRaw : List Term := Proof.Events342.exact87692RawTerms
def group : MergeGroup := .operator 80012 87692
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80012) (leftOrdinal := 0)
    (rightResult := 87692) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19096⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19096⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87698

namespace LeftMerge87777
def owner : Owner := ⟨.program ⟨214⟩, ⟨10677⟩⟩
def mergeEvent : Nat := 87777
def frameStart : Nat := 87747
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events342.exact87773RawTerms
def rightRaw : List Term := Proof.Events342.exact87770RawTerms
def group : MergeGroup := .operator 87773 87770
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87773) (leftOrdinal := 0)
    (rightResult := 87770) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9505⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10676⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87777

namespace LeftMerge87807
def owner : Owner := ⟨.program ⟨214⟩, ⟨10774⟩⟩
def mergeEvent : Nat := 87807
def frameStart : Nat := 87747
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events342.exact87803RawTerms
def rightRaw : List Term := Proof.Events342.exact87801RawTerms
def group : MergeGroup := .operator 87803 87801
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87803) (leftOrdinal := 0)
    (rightResult := 87801) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87807

namespace LeftMerge87828
def owner : Owner := ⟨.program ⟨214⟩, ⟨7836⟩⟩
def mergeEvent : Nat := 87828
def frameStart : Nat := 87747
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩] } }
def leftRaw : List Term := Proof.Events343.exact87824RawTerms
def rightRaw : List Term := Proof.Events343.exact87821RawTerms
def group : MergeGroup := .operator 87824 87821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87824) (leftOrdinal := 0)
    (rightResult := 87821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7834⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87828

namespace LeftMerge87837
def owner : Owner := ⟨.program ⟨214⟩, ⟨24991⟩⟩
def mergeEvent : Nat := 87837
def frameStart : Nat := 87747
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩] } }
def leftRaw : List Term := Proof.Events343.exact87833RawTerms
def rightRaw : List Term := Proof.Events342.exact87792RawTerms
def group : MergeGroup := .operator 87833 87792
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87833) (leftOrdinal := 0)
    (rightResult := 87792) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24988⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87837

namespace LeftMerge87838
def owner : Owner := ⟨.program ⟨214⟩, ⟨24991⟩⟩
def mergeEvent : Nat := 87838
def frameStart : Nat := 87747
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩] } }
def leftRaw : List Term := Proof.Events343.exact87833RawTerms
def rightRaw : List Term := Proof.Events342.exact87792RawTerms
def group : MergeGroup := .operator 87833 87792
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87833) (leftOrdinal := 1)
    (rightResult := 87792) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24988⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87838

namespace LeftMerge87840
def owner : Owner := ⟨.program ⟨214⟩, ⟨24991⟩⟩
def mergeEvent : Nat := 87840
def frameStart : Nat := 87747
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22996⟩⟩] } }
def rhsRaw : List Term := Proof.Events342.exact87789RawTerms
def group : MergeGroup := .relation 87839
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87839) (rhsResult := 87789)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨24988⟩⟩) ⟨22996⟩ 87789) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22996⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], [⟨.program ⟨214⟩, ⟨22996⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87840

namespace LeftMerge87848
def owner : Owner := ⟨.program ⟨214⟩, ⟨14955⟩⟩
def mergeEvent : Nat := 87848
def frameStart : Nat := 87747
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14953⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events342.exact87803RawTerms
def rightRaw : List Term := Proof.Events343.exact87844RawTerms
def group : MergeGroup := .operator 87803 87844
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87803) (leftOrdinal := 0)
    (rightResult := 87844) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14953⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87848

namespace LeftMerge87865
def owner : Owner := ⟨.program ⟨214⟩, ⟨19099⟩⟩
def mergeEvent : Nat := 87865
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }
def rhsRaw : List Term := Proof.Events343.exact87862RawTerms
def group : MergeGroup := .relation 87864
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87864) (rhsResult := 87862)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19096⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 87863 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19096⟩⟩]⟩) (none) 87862) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87865

namespace LeftMerge87866
def owner : Owner := ⟨.program ⟨214⟩, ⟨19099⟩⟩
def mergeEvent : Nat := 87866
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩] } }
def rhsRaw : List Term := Proof.Events343.exact87862RawTerms
def group : MergeGroup := .relation 87864
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87864) (rhsResult := 87862)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19096⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 87863 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19096⟩⟩]⟩) (none) 87862) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87866

namespace LeftMerge87867
def owner : Owner := ⟨.program ⟨214⟩, ⟨19099⟩⟩
def mergeEvent : Nat := 87867
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22996⟩⟩] } }
def rhsRaw : List Term := Proof.Events343.exact87862RawTerms
def group : MergeGroup := .relation 87864
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87864) (rhsResult := 87862)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19096⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 87863 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19096⟩⟩]⟩) (none) 87862) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22996⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], [⟨.program ⟨214⟩, ⟨22996⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87867

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
