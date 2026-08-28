import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge259670
def owner : Owner := ⟨.program ⟨257⟩, ⟨15361⟩⟩
def mergeEvent : Nat := 259670
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } }
def leftRaw : List Term := Proof.Events1014.exact259666RawTerms
def rightRaw : List Term := Proof.Events1014.exact259636RawTerms
def group : MergeGroup := .operator 259666 259636
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259666) (leftOrdinal := 1)
    (rightResult := 259636) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12306⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259670

namespace LeftMerge259678
def owner : Owner := ⟨.program ⟨257⟩, ⟨17305⟩⟩
def mergeEvent : Nat := 259678
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩] } }
def leftRaw : List Term := Proof.Events1014.exact259672RawTerms
def rightRaw : List Term := Proof.Events1014.exact259608RawTerms
def group : MergeGroup := .operator 259672 259608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259672) (leftOrdinal := 1)
    (rightResult := 259608) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17304⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge259678

namespace LeftMerge259680
def owner : Owner := ⟨.program ⟨257⟩, ⟨17305⟩⟩
def mergeEvent : Nat := 259680
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16819⟩⟩] } }
def rhsRaw : List Term := Proof.Events1014.exact259605RawTerms
def group : MergeGroup := .relation 259679
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 259679) (rhsResult := 259605)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17304⟩⟩) ⟨16819⟩ 259605) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16819⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], [⟨.program ⟨257⟩, ⟨16819⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge259680

namespace LeftMerge259681
def owner : Owner := ⟨.program ⟨257⟩, ⟨17305⟩⟩
def mergeEvent : Nat := 259681
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩] } }
def leftRaw : List Term := Proof.Events1014.exact259672RawTerms
def rightRaw : List Term := Proof.Events1014.exact259608RawTerms
def group : MergeGroup := .operator 259672 259608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259672) (leftOrdinal := 0)
    (rightResult := 259608) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17304⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259681

namespace LeftMerge259695
def owner : Owner := ⟨.program ⟨257⟩, ⟨16242⟩⟩
def mergeEvent : Nat := 259695
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16239⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events1014.exact259689RawTerms
def group : MergeGroup := .operator 251495 259689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 259689) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16239⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16239⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259695

namespace LeftMerge259774
def owner : Owner := ⟨.program ⟨257⟩, ⟨15355⟩⟩
def mergeEvent : Nat := 259774
def frameStart : Nat := 259744
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1014.exact259770RawTerms
def rightRaw : List Term := Proof.Events1014.exact259767RawTerms
def group : MergeGroup := .operator 259770 259767
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259770) (leftOrdinal := 0)
    (rightResult := 259767) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12306⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15354⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259774

namespace LeftMerge259804
def owner : Owner := ⟨.program ⟨257⟩, ⟨17108⟩⟩
def mergeEvent : Nat := 259804
def frameStart : Nat := 259744
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1014.exact259800RawTerms
def rightRaw : List Term := Proof.Events1014.exact259798RawTerms
def group : MergeGroup := .operator 259800 259798
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259800) (leftOrdinal := 0)
    (rightResult := 259798) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259804

namespace LeftMerge259827
def owner : Owner := ⟨.program ⟨257⟩, ⟨9570⟩⟩
def mergeEvent : Nat := 259827
def frameStart : Nat := 259744
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }
def leftRaw : List Term := Proof.Events1014.exact259823RawTerms
def rightRaw : List Term := Proof.Events1014.exact259820RawTerms
def group : MergeGroup := .operator 259823 259820
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259823) (leftOrdinal := 0)
    (rightResult := 259820) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9568⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259827

namespace LeftMerge259836
def owner : Owner := ⟨.program ⟨257⟩, ⟨17307⟩⟩
def mergeEvent : Nat := 259836
def frameStart : Nat := 259744
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩] } }
def leftRaw : List Term := Proof.Events1014.exact259832RawTerms
def rightRaw : List Term := Proof.Events1014.exact259789RawTerms
def group : MergeGroup := .operator 259832 259789
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259832) (leftOrdinal := 0)
    (rightResult := 259789) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17304⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259836

namespace LeftMerge259837
def owner : Owner := ⟨.program ⟨257⟩, ⟨17307⟩⟩
def mergeEvent : Nat := 259837
def frameStart : Nat := 259744
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩] } }
def leftRaw : List Term := Proof.Events1014.exact259832RawTerms
def rightRaw : List Term := Proof.Events1014.exact259789RawTerms
def group : MergeGroup := .operator 259832 259789
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259832) (leftOrdinal := 1)
    (rightResult := 259789) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17304⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge259837

namespace LeftMerge259839
def owner : Owner := ⟨.program ⟨257⟩, ⟨17307⟩⟩
def mergeEvent : Nat := 259839
def frameStart : Nat := 259744
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16819⟩⟩] } }
def rhsRaw : List Term := Proof.Events1014.exact259786RawTerms
def group : MergeGroup := .relation 259838
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 259838) (rhsResult := 259786)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17304⟩⟩) ⟨16819⟩ 259786) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16819⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], [⟨.program ⟨257⟩, ⟨16819⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge259839

namespace LeftMerge259847
def owner : Owner := ⟨.program ⟨257⟩, ⟨15750⟩⟩
def mergeEvent : Nat := 259847
def frameStart : Nat := 259744
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1014.exact259800RawTerms
def rightRaw : List Term := Proof.Events1015.exact259843RawTerms
def group : MergeGroup := .operator 259800 259843
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259800) (leftOrdinal := 0)
    (rightResult := 259843) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15748⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259847

namespace LeftMerge259864
def owner : Owner := ⟨.program ⟨257⟩, ⟨16242⟩⟩
def mergeEvent : Nat := 259864
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }
def rhsRaw : List Term := Proof.Events1015.exact259861RawTerms
def group : MergeGroup := .relation 259863
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 259863) (rhsResult := 259861)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16239⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 259862 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16239⟩⟩]⟩) (none) 259861) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259864

namespace LeftMerge259865
def owner : Owner := ⟨.program ⟨257⟩, ⟨16242⟩⟩
def mergeEvent : Nat := 259865
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩] } }
def rhsRaw : List Term := Proof.Events1015.exact259861RawTerms
def group : MergeGroup := .relation 259863
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 259863) (rhsResult := 259861)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16239⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 259862 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16239⟩⟩]⟩) (none) 259861) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge259865

namespace LeftMerge259866
def owner : Owner := ⟨.program ⟨257⟩, ⟨16242⟩⟩
def mergeEvent : Nat := 259866
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16819⟩⟩] } }
def rhsRaw : List Term := Proof.Events1015.exact259861RawTerms
def group : MergeGroup := .relation 259863
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 259863) (rhsResult := 259861)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16239⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 259862 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16239⟩⟩]⟩) (none) 259861) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16819⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], [⟨.program ⟨257⟩, ⟨16819⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259866

namespace LeftMerge259867
def owner : Owner := ⟨.program ⟨257⟩, ⟨16242⟩⟩
def mergeEvent : Nat := 259867
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1015.exact259861RawTerms
def group : MergeGroup := .relation 259863
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 259863) (rhsResult := 259861)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16239⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 259862 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16239⟩⟩]⟩) (none) 259861) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge259867

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
