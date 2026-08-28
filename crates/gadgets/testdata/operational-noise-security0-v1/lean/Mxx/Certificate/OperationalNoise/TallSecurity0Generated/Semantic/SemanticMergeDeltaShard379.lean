import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge62677
def owner : Owner := ⟨.program ⟨214⟩, ⟨28308⟩⟩
def mergeEvent : Nat := 62677
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28306⟩⟩] } }
def leftRaw : List Term := Proof.Events214.exact54804RawTerms
def rightRaw : List Term := Proof.Events244.exact62670RawTerms
def group : MergeGroup := .operator 54804 62670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54804) (leftOrdinal := 1)
    (rightResult := 62670) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28306⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28306⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62677

namespace LeftMerge62679
def owner : Owner := ⟨.program ⟨214⟩, ⟨28308⟩⟩
def mergeEvent : Nat := 62679
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24290⟩⟩] } }
def rhsRaw : List Term := Proof.Events244.exact62667RawTerms
def group : MergeGroup := .relation 62678
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62678) (rhsResult := 62667)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28306⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28306⟩⟩) ⟨24290⟩ 62667) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24290⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨24290⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62679

namespace LeftMerge62693
def owner : Owner := ⟨.program ⟨214⟩, ⟨21623⟩⟩
def mergeEvent : Nat := 62693
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21620⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events244.exact62687RawTerms
def group : MergeGroup := .operator 50762 62687
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 62687) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21620⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21620⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62693

namespace LeftMerge62814
def owner : Owner := ⟨.program ⟨214⟩, ⟨16224⟩⟩
def mergeEvent : Nat := 62814
def frameStart : Nat := 62748
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events245.exact62810RawTerms
def rightRaw : List Term := Proof.Events245.exact62808RawTerms
def group : MergeGroup := .operator 62810 62808
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62810) (leftOrdinal := 0)
    (rightResult := 62808) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62814

namespace LeftMerge62826
def owner : Owner := ⟨.program ⟨214⟩, ⟨28307⟩⟩
def mergeEvent : Nat := 62826
def frameStart : Nat := 62748
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28306⟩⟩] } }
def leftRaw : List Term := Proof.Events245.exact62822RawTerms
def rightRaw : List Term := Proof.Events245.exact62799RawTerms
def group : MergeGroup := .operator 62822 62799
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62822) (leftOrdinal := 0)
    (rightResult := 62799) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28306⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28306⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62826

namespace LeftMerge62827
def owner : Owner := ⟨.program ⟨214⟩, ⟨28307⟩⟩
def mergeEvent : Nat := 62827
def frameStart : Nat := 62748
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28306⟩⟩] } }
def leftRaw : List Term := Proof.Events245.exact62822RawTerms
def rightRaw : List Term := Proof.Events245.exact62799RawTerms
def group : MergeGroup := .operator 62822 62799
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62822) (leftOrdinal := 1)
    (rightResult := 62799) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28306⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28306⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62827

namespace LeftMerge62829
def owner : Owner := ⟨.program ⟨214⟩, ⟨28307⟩⟩
def mergeEvent : Nat := 62829
def frameStart : Nat := 62748
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24290⟩⟩] } }
def rhsRaw : List Term := Proof.Events245.exact62796RawTerms
def group : MergeGroup := .relation 62828
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62828) (rhsResult := 62796)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28306⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28306⟩⟩) ⟨24290⟩ 62796) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24290⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨24290⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62829

namespace LeftMerge62837
def owner : Owner := ⟨.program ⟨214⟩, ⟨17668⟩⟩
def mergeEvent : Nat := 62837
def frameStart : Nat := 62748
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17666⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events245.exact62810RawTerms
def rightRaw : List Term := Proof.Events245.exact62833RawTerms
def group : MergeGroup := .operator 62810 62833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62810) (leftOrdinal := 0)
    (rightResult := 62833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17666⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17666⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62837

namespace LeftMerge62854
def owner : Owner := ⟨.program ⟨214⟩, ⟨21623⟩⟩
def mergeEvent : Nat := 62854
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6726⟩⟩] } }
def rhsRaw : List Term := Proof.Events245.exact62851RawTerms
def group : MergeGroup := .relation 62853
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62853) (rhsResult := 62851)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21620⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62852 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21620⟩⟩]⟩) (none) 62851) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6726⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62854

namespace LeftMerge62855
def owner : Owner := ⟨.program ⟨214⟩, ⟨21623⟩⟩
def mergeEvent : Nat := 62855
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28306⟩⟩] } }
def rhsRaw : List Term := Proof.Events245.exact62851RawTerms
def group : MergeGroup := .relation 62853
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62853) (rhsResult := 62851)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21620⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62852 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21620⟩⟩]⟩) (none) 62851) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28306⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28306⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62855

namespace LeftMerge62856
def owner : Owner := ⟨.program ⟨214⟩, ⟨21623⟩⟩
def mergeEvent : Nat := 62856
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24290⟩⟩] } }
def rhsRaw : List Term := Proof.Events245.exact62851RawTerms
def group : MergeGroup := .relation 62853
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62853) (rhsResult := 62851)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21620⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62852 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21620⟩⟩]⟩) (none) 62851) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24290⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨24290⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62856

namespace LeftMerge62857
def owner : Owner := ⟨.program ⟨214⟩, ⟨21623⟩⟩
def mergeEvent : Nat := 62857
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17666⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events245.exact62851RawTerms
def group : MergeGroup := .relation 62853
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62853) (rhsResult := 62851)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21620⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62852 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21620⟩⟩]⟩) (none) 62851) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17666⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17666⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62857

namespace LeftMerge62862
def owner : Owner := ⟨.program ⟨214⟩, ⟨28309⟩⟩
def mergeEvent : Nat := 62862
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28306⟩⟩] } }
def leftRaw : List Term := Proof.Events245.exact62858RawTerms
def rightRaw : List Term := Proof.Events244.exact62680RawTerms
def group : MergeGroup := .operator 62858 62680
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62858) (leftOrdinal := 0)
    (rightResult := 62680) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28306⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28306⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28306⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62862

namespace LeftMerge62863
def owner : Owner := ⟨.program ⟨214⟩, ⟨28309⟩⟩
def mergeEvent : Nat := 62863
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24290⟩⟩] } }
def leftRaw : List Term := Proof.Events245.exact62858RawTerms
def rightRaw : List Term := Proof.Events244.exact62680RawTerms
def group : MergeGroup := .operator 62858 62680
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62858) (leftOrdinal := 2)
    (rightResult := 62680) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24290⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24290⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨24290⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62863

namespace LeftMerge62871
def owner : Owner := ⟨.program ⟨214⟩, ⟨28310⟩⟩
def mergeEvent : Nat := 62871
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩] } }
def leftRaw : List Term := Proof.Events245.exact62865RawTerms
def rightRaw : List Term := Proof.Events022.exact5679RawTerms
def group : MergeGroup := .operator 62865 5679
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62865) (leftOrdinal := 0)
    (rightResult := 5679) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6726⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62871

namespace LeftMerge62872
def owner : Owner := ⟨.program ⟨214⟩, ⟨28310⟩⟩
def mergeEvent : Nat := 62872
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17666⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩] } }
def leftRaw : List Term := Proof.Events245.exact62865RawTerms
def rightRaw : List Term := Proof.Events022.exact5679RawTerms
def group : MergeGroup := .operator 62865 5679
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62865) (leftOrdinal := 1)
    (rightResult := 5679) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17666⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17666⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62872

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
