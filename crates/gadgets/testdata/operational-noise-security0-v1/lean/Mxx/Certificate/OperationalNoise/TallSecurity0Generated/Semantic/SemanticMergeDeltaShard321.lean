import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge52679
def owner : Owner := ⟨.program ⟨214⟩, ⟨25456⟩⟩
def mergeEvent : Nat := 52679
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩] } }
def leftRaw : List Term := Proof.Events205.exact52673RawTerms
def rightRaw : List Term := Proof.Events205.exact52609RawTerms
def group : MergeGroup := .operator 52673 52609
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52673) (leftOrdinal := 1)
    (rightResult := 52609) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25455⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52679

namespace LeftMerge52681
def owner : Owner := ⟨.program ⟨214⟩, ⟨25456⟩⟩
def mergeEvent : Nat := 52681
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23250⟩⟩] } }
def rhsRaw : List Term := Proof.Events205.exact52606RawTerms
def group : MergeGroup := .relation 52680
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52680) (rhsResult := 52606)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25455⟩⟩) ⟨23250⟩ 52606) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23250⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨23250⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52681

namespace LeftMerge52682
def owner : Owner := ⟨.program ⟨214⟩, ⟨25456⟩⟩
def mergeEvent : Nat := 52682
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩] } }
def leftRaw : List Term := Proof.Events205.exact52673RawTerms
def rightRaw : List Term := Proof.Events205.exact52609RawTerms
def group : MergeGroup := .operator 52673 52609
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52673) (leftOrdinal := 0)
    (rightResult := 52609) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25455⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52682

namespace LeftMerge52696
def owner : Owner := ⟨.program ⟨214⟩, ⟨19967⟩⟩
def mergeEvent : Nat := 52696
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19964⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events205.exact52690RawTerms
def group : MergeGroup := .operator 50762 52690
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 52690) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19964⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19964⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52696

namespace LeftMerge52775
def owner : Owner := ⟨.program ⟨214⟩, ⟨12575⟩⟩
def mergeEvent : Nat := 52775
def frameStart : Nat := 52745
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events206.exact52771RawTerms
def rightRaw : List Term := Proof.Events206.exact52768RawTerms
def group : MergeGroup := .operator 52771 52768
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52771) (leftOrdinal := 0)
    (rightResult := 52768) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9930⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12574⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52775

namespace LeftMerge52805
def owner : Owner := ⟨.program ⟨214⟩, ⟨12668⟩⟩
def mergeEvent : Nat := 52805
def frameStart : Nat := 52745
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events206.exact52801RawTerms
def rightRaw : List Term := Proof.Events206.exact52799RawTerms
def group : MergeGroup := .operator 52801 52799
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52801) (leftOrdinal := 0)
    (rightResult := 52799) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52805

namespace LeftMerge52828
def owner : Owner := ⟨.program ⟨214⟩, ⟨7872⟩⟩
def mergeEvent : Nat := 52828
def frameStart : Nat := 52745
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩] } }
def leftRaw : List Term := Proof.Events206.exact52824RawTerms
def rightRaw : List Term := Proof.Events206.exact52821RawTerms
def group : MergeGroup := .operator 52824 52821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52824) (leftOrdinal := 0)
    (rightResult := 52821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7870⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52828

namespace LeftMerge52837
def owner : Owner := ⟨.program ⟨214⟩, ⟨25458⟩⟩
def mergeEvent : Nat := 52837
def frameStart : Nat := 52745
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩] } }
def leftRaw : List Term := Proof.Events206.exact52833RawTerms
def rightRaw : List Term := Proof.Events206.exact52790RawTerms
def group : MergeGroup := .operator 52833 52790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52833) (leftOrdinal := 0)
    (rightResult := 52790) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25455⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52837

namespace LeftMerge52838
def owner : Owner := ⟨.program ⟨214⟩, ⟨25458⟩⟩
def mergeEvent : Nat := 52838
def frameStart : Nat := 52745
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩] } }
def leftRaw : List Term := Proof.Events206.exact52833RawTerms
def rightRaw : List Term := Proof.Events206.exact52790RawTerms
def group : MergeGroup := .operator 52833 52790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52833) (leftOrdinal := 1)
    (rightResult := 52790) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25455⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52838

namespace LeftMerge52840
def owner : Owner := ⟨.program ⟨214⟩, ⟨25458⟩⟩
def mergeEvent : Nat := 52840
def frameStart : Nat := 52745
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23250⟩⟩] } }
def rhsRaw : List Term := Proof.Events206.exact52787RawTerms
def group : MergeGroup := .relation 52839
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52839) (rhsResult := 52787)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25455⟩⟩) ⟨23250⟩ 52787) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23250⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨23250⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52840

namespace LeftMerge52848
def owner : Owner := ⟨.program ⟨214⟩, ⟨16555⟩⟩
def mergeEvent : Nat := 52848
def frameStart : Nat := 52745
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16553⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events206.exact52801RawTerms
def rightRaw : List Term := Proof.Events206.exact52844RawTerms
def group : MergeGroup := .operator 52801 52844
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52801) (leftOrdinal := 0)
    (rightResult := 52844) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16553⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16553⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52848

namespace LeftMerge52865
def owner : Owner := ⟨.program ⟨214⟩, ⟨19967⟩⟩
def mergeEvent : Nat := 52865
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩] } }
def rhsRaw : List Term := Proof.Events206.exact52862RawTerms
def group : MergeGroup := .relation 52864
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52864) (rhsResult := 52862)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19964⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 52863 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19964⟩⟩]⟩) (none) 52862) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52865

namespace LeftMerge52866
def owner : Owner := ⟨.program ⟨214⟩, ⟨19967⟩⟩
def mergeEvent : Nat := 52866
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩] } }
def rhsRaw : List Term := Proof.Events206.exact52862RawTerms
def group : MergeGroup := .relation 52864
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52864) (rhsResult := 52862)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19964⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 52863 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19964⟩⟩]⟩) (none) 52862) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52866

namespace LeftMerge52867
def owner : Owner := ⟨.program ⟨214⟩, ⟨19967⟩⟩
def mergeEvent : Nat := 52867
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23250⟩⟩] } }
def rhsRaw : List Term := Proof.Events206.exact52862RawTerms
def group : MergeGroup := .relation 52864
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52864) (rhsResult := 52862)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19964⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 52863 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19964⟩⟩]⟩) (none) 52862) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23250⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨23250⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52867

namespace LeftMerge52868
def owner : Owner := ⟨.program ⟨214⟩, ⟨19967⟩⟩
def mergeEvent : Nat := 52868
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16553⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events206.exact52862RawTerms
def group : MergeGroup := .relation 52864
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52864) (rhsResult := 52862)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19964⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 52863 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19964⟩⟩]⟩) (none) 52862) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16553⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16553⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52868

namespace LeftMerge52873
def owner : Owner := ⟨.program ⟨214⟩, ⟨25457⟩⟩
def mergeEvent : Nat := 52873
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23250⟩⟩] } }
def leftRaw : List Term := Proof.Events206.exact52869RawTerms
def rightRaw : List Term := Proof.Events205.exact52683RawTerms
def group : MergeGroup := .operator 52869 52683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52869) (leftOrdinal := 2)
    (rightResult := 52683) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23250⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23250⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨23250⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52873

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
