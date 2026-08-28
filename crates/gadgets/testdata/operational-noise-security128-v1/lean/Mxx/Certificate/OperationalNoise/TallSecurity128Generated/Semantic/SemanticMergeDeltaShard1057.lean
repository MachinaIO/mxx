import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge173810
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173810
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63157⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 34)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63157⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173810

namespace LeftMerge173812
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173812
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63157⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }
def rhsRaw : List Term := Proof.Events678.exact173599RawTerms
def group : MergeGroup := .relation 173811
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173811) (rhsResult := 173599)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173812

namespace LeftMerge173813
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173813
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60177⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 33)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60177⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173813

namespace LeftMerge173815
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173815
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60177⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }
def rhsRaw : List Term := Proof.Events678.exact173599RawTerms
def group : MergeGroup := .relation 173814
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173814) (rhsResult := 173599)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173815

namespace LeftMerge173816
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173816
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57197⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 32)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57197⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173816

namespace LeftMerge173818
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173818
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57197⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }
def rhsRaw : List Term := Proof.Events678.exact173599RawTerms
def group : MergeGroup := .relation 173817
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173817) (rhsResult := 173599)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173818

namespace LeftMerge173819
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173819
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54217⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 31)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54217⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173819

namespace LeftMerge173821
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173821
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54217⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }
def rhsRaw : List Term := Proof.Events678.exact173599RawTerms
def group : MergeGroup := .relation 173820
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173820) (rhsResult := 173599)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173821

namespace LeftMerge173822
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173822
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51237⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 30)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51237⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173822

namespace LeftMerge173824
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173824
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51237⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }
def rhsRaw : List Term := Proof.Events678.exact173599RawTerms
def group : MergeGroup := .relation 173823
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173823) (rhsResult := 173599)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173824

namespace LeftMerge173825
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173825
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32182⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 23)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32182⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173825

namespace LeftMerge173827
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173827
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32182⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }
def rhsRaw : List Term := Proof.Events678.exact173599RawTerms
def group : MergeGroup := .relation 173826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173826) (rhsResult := 173599)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173827

namespace LeftMerge173828
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173828
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22162⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 20)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22162⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173828

namespace LeftMerge173830
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173830
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22162⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }
def rhsRaw : List Term := Proof.Events678.exact173599RawTerms
def group : MergeGroup := .relation 173829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173829) (rhsResult := 173599)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173830

namespace LeftMerge173831
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173831
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18942⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 19)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18942⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173831

namespace LeftMerge173833
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173833
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18942⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }
def rhsRaw : List Term := Proof.Events678.exact173599RawTerms
def group : MergeGroup := .relation 173832
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173832) (rhsResult := 173599)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173833

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
