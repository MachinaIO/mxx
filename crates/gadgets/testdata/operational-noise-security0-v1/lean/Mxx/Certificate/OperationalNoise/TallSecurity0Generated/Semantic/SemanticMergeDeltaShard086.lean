import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge16770
def owner : Owner := ⟨.program ⟨214⟩, ⟨18665⟩⟩
def mergeEvent : Nat := 16770
def frameStart : Nat := 16225
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events065.exact16752RawTerms
def rightRaw : List Term := Proof.Events065.exact16750RawTerms
def group : MergeGroup := .operator 16752 16750
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16752) (leftOrdinal := 0)
    (rightResult := 16750) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16770

namespace LeftMerge16771
def owner : Owner := ⟨.program ⟨214⟩, ⟨18665⟩⟩
def mergeEvent : Nat := 16771
def frameStart : Nat := 16225
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events065.exact16752RawTerms
def rightRaw : List Term := Proof.Events065.exact16750RawTerms
def group : MergeGroup := .operator 16752 16750
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16752) (leftOrdinal := 0)
    (rightResult := 16750) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16771

namespace LeftMerge16772
def owner : Owner := ⟨.program ⟨214⟩, ⟨18665⟩⟩
def mergeEvent : Nat := 16772
def frameStart : Nat := 16225
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events065.exact16752RawTerms
def rightRaw : List Term := Proof.Events065.exact16750RawTerms
def group : MergeGroup := .operator 16752 16750
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16752) (leftOrdinal := 0)
    (rightResult := 16750) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16772

namespace LeftMerge16773
def owner : Owner := ⟨.program ⟨214⟩, ⟨18665⟩⟩
def mergeEvent : Nat := 16773
def frameStart : Nat := 16225
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events065.exact16752RawTerms
def rightRaw : List Term := Proof.Events065.exact16750RawTerms
def group : MergeGroup := .operator 16752 16750
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16752) (leftOrdinal := 0)
    (rightResult := 16750) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16773

namespace LeftMerge16904
def owner : Owner := ⟨.program ⟨214⟩, ⟨18694⟩⟩
def mergeEvent : Nat := 16904
def frameStart : Nat := 16225
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events066.exact16900RawTerms
def rightRaw : List Term := Proof.Events065.exact16741RawTerms
def group : MergeGroup := .operator 16900 16741
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16900) (leftOrdinal := 33)
    (rightResult := 16741) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge16904

namespace LeftMerge16906
def owner : Owner := ⟨.program ⟨214⟩, ⟨18694⟩⟩
def mergeEvent : Nat := 16906
def frameStart : Nat := 16225
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }
def rhsRaw : List Term := Proof.Events065.exact16738RawTerms
def group : MergeGroup := .relation 16905
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 16905) (rhsResult := 16738)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨18182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨18182⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge16906

namespace LeftMerge16907
def owner : Owner := ⟨.program ⟨214⟩, ⟨18694⟩⟩
def mergeEvent : Nat := 16907
def frameStart : Nat := 16225
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events066.exact16900RawTerms
def rightRaw : List Term := Proof.Events065.exact16741RawTerms
def group : MergeGroup := .operator 16900 16741
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16900) (leftOrdinal := 17)
    (rightResult := 16741) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16907

namespace LeftMerge16908
def owner : Owner := ⟨.program ⟨214⟩, ⟨18694⟩⟩
def mergeEvent : Nat := 16908
def frameStart : Nat := 16225
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17097⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events066.exact16900RawTerms
def rightRaw : List Term := Proof.Events065.exact16741RawTerms
def group : MergeGroup := .operator 16900 16741
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16900) (leftOrdinal := 29)
    (rightResult := 16741) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17097⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17097⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge16908

namespace LeftMerge16910
def owner : Owner := ⟨.program ⟨214⟩, ⟨18694⟩⟩
def mergeEvent : Nat := 16910
def frameStart : Nat := 16225
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17097⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }
def rhsRaw : List Term := Proof.Events065.exact16738RawTerms
def group : MergeGroup := .relation 16909
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 16909) (rhsResult := 16738)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨17097⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨17097⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge16910

namespace LeftMerge16911
def owner : Owner := ⟨.program ⟨214⟩, ⟨18694⟩⟩
def mergeEvent : Nat := 16911
def frameStart : Nat := 16225
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events066.exact16900RawTerms
def rightRaw : List Term := Proof.Events065.exact16741RawTerms
def group : MergeGroup := .operator 16900 16741
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16900) (leftOrdinal := 16)
    (rightResult := 16741) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6741⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16911

namespace LeftMerge16912
def owner : Owner := ⟨.program ⟨214⟩, ⟨18694⟩⟩
def mergeEvent : Nat := 16912
def frameStart : Nat := 16225
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16810⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events066.exact16900RawTerms
def rightRaw : List Term := Proof.Events065.exact16741RawTerms
def group : MergeGroup := .operator 16900 16741
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16900) (leftOrdinal := 28)
    (rightResult := 16741) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16810⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge16912

namespace LeftMerge16914
def owner : Owner := ⟨.program ⟨214⟩, ⟨18694⟩⟩
def mergeEvent : Nat := 16914
def frameStart : Nat := 16225
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16810⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }
def rhsRaw : List Term := Proof.Events065.exact16738RawTerms
def group : MergeGroup := .relation 16913
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 16913) (rhsResult := 16738)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge16914

namespace LeftMerge16915
def owner : Owner := ⟨.program ⟨214⟩, ⟨18694⟩⟩
def mergeEvent : Nat := 16915
def frameStart : Nat := 16225
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events066.exact16900RawTerms
def rightRaw : List Term := Proof.Events065.exact16741RawTerms
def group : MergeGroup := .operator 16900 16741
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16900) (leftOrdinal := 15)
    (rightResult := 16741) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16915

namespace LeftMerge16916
def owner : Owner := ⟨.program ⟨214⟩, ⟨18694⟩⟩
def mergeEvent : Nat := 16916
def frameStart : Nat := 16225
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events066.exact16900RawTerms
def rightRaw : List Term := Proof.Events065.exact16741RawTerms
def group : MergeGroup := .operator 16900 16741
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16900) (leftOrdinal := 27)
    (rightResult := 16741) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge16916

namespace LeftMerge16918
def owner : Owner := ⟨.program ⟨214⟩, ⟨18694⟩⟩
def mergeEvent : Nat := 16918
def frameStart : Nat := 16225
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }
def rhsRaw : List Term := Proof.Events065.exact16738RawTerms
def group : MergeGroup := .relation 16917
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 16917) (rhsResult := 16738)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge16918

namespace LeftMerge16919
def owner : Owner := ⟨.program ⟨214⟩, ⟨18694⟩⟩
def mergeEvent : Nat := 16919
def frameStart : Nat := 16225
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events066.exact16900RawTerms
def rightRaw : List Term := Proof.Events065.exact16741RawTerms
def group : MergeGroup := .operator 16900 16741
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16900) (leftOrdinal := 14)
    (rightResult := 16741) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16919

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
