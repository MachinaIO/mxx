import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge6744
def owner : Owner := ⟨.program ⟨214⟩, ⟨25780⟩⟩
def mergeEvent : Nat := 6744
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23424⟩⟩] } }
def leftRaw : List Term := Proof.Events026.exact6740RawTerms
def rightRaw : List Term := Proof.Events025.exact6539RawTerms
def group : MergeGroup := .operator 6740 6539
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6740) (leftOrdinal := 2)
    (rightResult := 6539) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23424⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23424⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], [⟨.program ⟨214⟩, ⟨23424⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge6744

namespace LeftMerge6745
def owner : Owner := ⟨.program ⟨214⟩, ⟨25780⟩⟩
def mergeEvent : Nat := 6745
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25778⟩⟩] } }
def leftRaw : List Term := Proof.Events026.exact6740RawTerms
def rightRaw : List Term := Proof.Events025.exact6539RawTerms
def group : MergeGroup := .operator 6740 6539
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6740) (leftOrdinal := 1)
    (rightResult := 6539) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25778⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25778⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25778⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6745

namespace LeftMerge6753
def owner : Owner := ⟨.program ⟨214⟩, ⟨30207⟩⟩
def mergeEvent : Nat := 6753
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30205⟩⟩] } }
def leftRaw : List Term := Proof.Events026.exact6747RawTerms
def rightRaw : List Term := Proof.Events025.exact6429RawTerms
def group : MergeGroup := .operator 6747 6429
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6747) (leftOrdinal := 1)
    (rightResult := 6429) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30205⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30205⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge6753

namespace LeftMerge6755
def owner : Owner := ⟨.program ⟨214⟩, ⟨30207⟩⟩
def mergeEvent : Nat := 6755
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24804⟩⟩] } }
def rhsRaw : List Term := Proof.Events025.exact6426RawTerms
def group : MergeGroup := .relation 6754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 6754) (rhsResult := 6426)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30205⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨30205⟩⟩) ⟨24804⟩ 6426) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24804⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨24804⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge6755

namespace LeftMerge6756
def owner : Owner := ⟨.program ⟨214⟩, ⟨30207⟩⟩
def mergeEvent : Nat := 6756
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30205⟩⟩] } }
def leftRaw : List Term := Proof.Events026.exact6747RawTerms
def rightRaw : List Term := Proof.Events025.exact6429RawTerms
def group : MergeGroup := .operator 6747 6429
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6747) (leftOrdinal := 0)
    (rightResult := 6429) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30205⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30205⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6756

namespace LeftMerge6770
def owner : Owner := ⟨.program ⟨214⟩, ⟨22859⟩⟩
def mergeEvent : Nat := 6770
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22856⟩⟩] } }
def leftRaw : List Term := Proof.Events025.exact6561RawTerms
def rightRaw : List Term := Proof.Events026.exact6764RawTerms
def group : MergeGroup := .operator 6561 6764
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6561) (leftOrdinal := 0)
    (rightResult := 6764) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22856⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22856⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6770

namespace LeftMerge6891
def owner : Owner := ⟨.program ⟨214⟩, ⟨17069⟩⟩
def mergeEvent : Nat := 6891
def frameStart : Nat := 6825
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events026.exact6887RawTerms
def rightRaw : List Term := Proof.Events026.exact6885RawTerms
def group : MergeGroup := .operator 6887 6885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6887) (leftOrdinal := 0)
    (rightResult := 6885) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6891

namespace LeftMerge6903
def owner : Owner := ⟨.program ⟨214⟩, ⟨30206⟩⟩
def mergeEvent : Nat := 6903
def frameStart : Nat := 6825
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30205⟩⟩] } }
def leftRaw : List Term := Proof.Events026.exact6899RawTerms
def rightRaw : List Term := Proof.Events026.exact6876RawTerms
def group : MergeGroup := .operator 6899 6876
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6899) (leftOrdinal := 1)
    (rightResult := 6876) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30205⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30205⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge6903

namespace LeftMerge6905
def owner : Owner := ⟨.program ⟨214⟩, ⟨30206⟩⟩
def mergeEvent : Nat := 6905
def frameStart : Nat := 6825
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24804⟩⟩] } }
def rhsRaw : List Term := Proof.Events026.exact6873RawTerms
def group : MergeGroup := .relation 6904
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 6904) (rhsResult := 6873)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30205⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨30205⟩⟩) ⟨24804⟩ 6873) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24804⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨24804⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge6905

namespace LeftMerge6906
def owner : Owner := ⟨.program ⟨214⟩, ⟨30206⟩⟩
def mergeEvent : Nat := 6906
def frameStart : Nat := 6825
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30205⟩⟩] } }
def leftRaw : List Term := Proof.Events026.exact6899RawTerms
def rightRaw : List Term := Proof.Events026.exact6876RawTerms
def group : MergeGroup := .operator 6899 6876
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6899) (leftOrdinal := 0)
    (rightResult := 6876) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30205⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30205⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6906

namespace LeftMerge6914
def owner : Owner := ⟨.program ⟨214⟩, ⟨18183⟩⟩
def mergeEvent : Nat := 6914
def frameStart : Nat := 6825
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events026.exact6887RawTerms
def rightRaw : List Term := Proof.Events026.exact6910RawTerms
def group : MergeGroup := .operator 6887 6910
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6887) (leftOrdinal := 0)
    (rightResult := 6910) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18182⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6914

namespace LeftMerge6931
def owner : Owner := ⟨.program ⟨214⟩, ⟨22859⟩⟩
def mergeEvent : Nat := 6931
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24804⟩⟩] } }
def rhsRaw : List Term := Proof.Events027.exact6928RawTerms
def group : MergeGroup := .relation 6930
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 6930) (rhsResult := 6928)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 6929 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22856⟩⟩]⟩) (none) 6928) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24804⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨24804⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6931

namespace LeftMerge6932
def owner : Owner := ⟨.program ⟨214⟩, ⟨22859⟩⟩
def mergeEvent : Nat := 6932
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30205⟩⟩] } }
def rhsRaw : List Term := Proof.Events027.exact6928RawTerms
def group : MergeGroup := .relation 6930
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 6930) (rhsResult := 6928)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 6929 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22856⟩⟩]⟩) (none) 6928) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30205⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30205⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge6932

namespace LeftMerge6933
def owner : Owner := ⟨.program ⟨214⟩, ⟨22859⟩⟩
def mergeEvent : Nat := 6933
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events027.exact6928RawTerms
def group : MergeGroup := .relation 6930
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 6930) (rhsResult := 6928)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 6929 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22856⟩⟩]⟩) (none) 6928) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge6933

namespace LeftMerge6934
def owner : Owner := ⟨.program ⟨214⟩, ⟨22859⟩⟩
def mergeEvent : Nat := 6934
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩] } }
def rhsRaw : List Term := Proof.Events027.exact6928RawTerms
def group : MergeGroup := .relation 6930
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 6930) (rhsResult := 6928)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 6929 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22856⟩⟩]⟩) (none) 6928) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6934

namespace LeftMerge6939
def owner : Owner := ⟨.program ⟨214⟩, ⟨30208⟩⟩
def mergeEvent : Nat := 6939
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24804⟩⟩] } }
def leftRaw : List Term := Proof.Events027.exact6935RawTerms
def rightRaw : List Term := Proof.Events026.exact6757RawTerms
def group : MergeGroup := .operator 6935 6757
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6935) (leftOrdinal := 2)
    (rightResult := 6757) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24804⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24804⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨24804⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge6939

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
