import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge11746
def owner : Owner := ⟨.program ⟨214⟩, ⟨19547⟩⟩
def mergeEvent : Nat := 11746
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11485⟩⟩, ⟨.program ⟨214⟩, ⟨14243⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23592⟩⟩] } }
def rhsRaw : List Term := Proof.Events045.exact11743RawTerms
def group : MergeGroup := .relation 11745
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 11745) (rhsResult := 11743)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19544⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 11744 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19544⟩⟩]⟩) (none) 11743) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11485⟩⟩, ⟨.program ⟨214⟩, ⟨14243⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23592⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11485⟩⟩, ⟨.program ⟨214⟩, ⟨14243⟩⟩], [⟨.program ⟨214⟩, ⟨23592⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11746

namespace LeftMerge11747
def owner : Owner := ⟨.program ⟨214⟩, ⟨19547⟩⟩
def mergeEvent : Nat := 11747
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26086⟩⟩] } }
def rhsRaw : List Term := Proof.Events045.exact11743RawTerms
def group : MergeGroup := .relation 11745
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 11745) (rhsResult := 11743)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19544⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 11744 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19544⟩⟩]⟩) (none) 11743) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26086⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26086⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge11747

namespace LeftMerge11748
def owner : Owner := ⟨.program ⟨214⟩, ⟨19547⟩⟩
def mergeEvent : Nat := 11748
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15956⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events045.exact11743RawTerms
def group : MergeGroup := .relation 11745
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 11745) (rhsResult := 11743)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19544⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 11744 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19544⟩⟩]⟩) (none) 11743) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15956⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15956⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge11748

namespace LeftMerge11749
def owner : Owner := ⟨.program ⟨214⟩, ⟨19547⟩⟩
def mergeEvent : Nat := 11749
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩] } }
def rhsRaw : List Term := Proof.Events045.exact11743RawTerms
def group : MergeGroup := .relation 11745
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 11745) (rhsResult := 11743)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19544⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 11744 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19544⟩⟩]⟩) (none) 11743) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11749

namespace LeftMerge11754
def owner : Owner := ⟨.program ⟨214⟩, ⟨26088⟩⟩
def mergeEvent : Nat := 11754
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11485⟩⟩, ⟨.program ⟨214⟩, ⟨14243⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23592⟩⟩] } }
def leftRaw : List Term := Proof.Events045.exact11750RawTerms
def rightRaw : List Term := Proof.Events045.exact11564RawTerms
def group : MergeGroup := .operator 11750 11564
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11750) (leftOrdinal := 2)
    (rightResult := 11564) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11485⟩⟩, ⟨.program ⟨214⟩, ⟨14243⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23592⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11485⟩⟩, ⟨.program ⟨214⟩, ⟨14243⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23592⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11485⟩⟩, ⟨.program ⟨214⟩, ⟨14243⟩⟩], [⟨.program ⟨214⟩, ⟨23592⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge11754

namespace LeftMerge11755
def owner : Owner := ⟨.program ⟨214⟩, ⟨26088⟩⟩
def mergeEvent : Nat := 11755
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26086⟩⟩] } }
def leftRaw : List Term := Proof.Events045.exact11750RawTerms
def rightRaw : List Term := Proof.Events045.exact11564RawTerms
def group : MergeGroup := .operator 11750 11564
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11750) (leftOrdinal := 1)
    (rightResult := 11564) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26086⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26086⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26086⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11755

namespace LeftMerge11763
def owner : Owner := ⟨.program ⟨214⟩, ⟨27920⟩⟩
def mergeEvent : Nat := 11763
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15956⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27918⟩⟩] } }
def leftRaw : List Term := Proof.Events045.exact11757RawTerms
def rightRaw : List Term := Proof.Events044.exact11461RawTerms
def group : MergeGroup := .operator 11757 11461
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11757) (leftOrdinal := 1)
    (rightResult := 11461) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15956⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27918⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15956⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27918⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge11763

namespace LeftMerge11765
def owner : Owner := ⟨.program ⟨214⟩, ⟨27920⟩⟩
def mergeEvent : Nat := 11765
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15956⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24174⟩⟩] } }
def rhsRaw : List Term := Proof.Events044.exact11458RawTerms
def group : MergeGroup := .relation 11764
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 11764) (rhsResult := 11458)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15956⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27918⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27918⟩⟩) ⟨24174⟩ 11458) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24174⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15956⟩⟩], [⟨.program ⟨214⟩, ⟨24174⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge11765

namespace LeftMerge11766
def owner : Owner := ⟨.program ⟨214⟩, ⟨27920⟩⟩
def mergeEvent : Nat := 11766
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27918⟩⟩] } }
def leftRaw : List Term := Proof.Events045.exact11757RawTerms
def rightRaw : List Term := Proof.Events044.exact11461RawTerms
def group : MergeGroup := .operator 11757 11461
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11757) (leftOrdinal := 0)
    (rightResult := 11461) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27918⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27918⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11766

namespace LeftMerge11780
def owner : Owner := ⟨.program ⟨214⟩, ⟨21419⟩⟩
def mergeEvent : Nat := 11780
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21416⟩⟩] } }
def leftRaw : List Term := Proof.Events025.exact6561RawTerms
def rightRaw : List Term := Proof.Events045.exact11774RawTerms
def group : MergeGroup := .operator 6561 11774
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6561) (leftOrdinal := 0)
    (rightResult := 11774) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21416⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21416⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11780

namespace LeftMerge11901
def owner : Owner := ⟨.program ⟨214⟩, ⟨16033⟩⟩
def mergeEvent : Nat := 11901
def frameStart : Nat := 11835
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15956⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events046.exact11897RawTerms
def rightRaw : List Term := Proof.Events046.exact11895RawTerms
def group : MergeGroup := .operator 11897 11895
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11897) (leftOrdinal := 0)
    (rightResult := 11895) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15956⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15956⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11901

namespace LeftMerge11913
def owner : Owner := ⟨.program ⟨214⟩, ⟨27919⟩⟩
def mergeEvent : Nat := 11913
def frameStart : Nat := 11835
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15956⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27918⟩⟩] } }
def leftRaw : List Term := Proof.Events046.exact11909RawTerms
def rightRaw : List Term := Proof.Events046.exact11886RawTerms
def group : MergeGroup := .operator 11909 11886
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11909) (leftOrdinal := 1)
    (rightResult := 11886) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15956⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27918⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15956⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27918⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge11913

namespace LeftMerge11915
def owner : Owner := ⟨.program ⟨214⟩, ⟨27919⟩⟩
def mergeEvent : Nat := 11915
def frameStart : Nat := 11835
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15956⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24174⟩⟩] } }
def rhsRaw : List Term := Proof.Events046.exact11883RawTerms
def group : MergeGroup := .relation 11914
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 11914) (rhsResult := 11883)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15956⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27918⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27918⟩⟩) ⟨24174⟩ 11883) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24174⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15956⟩⟩], [⟨.program ⟨214⟩, ⟨24174⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge11915

namespace LeftMerge11916
def owner : Owner := ⟨.program ⟨214⟩, ⟨27919⟩⟩
def mergeEvent : Nat := 11916
def frameStart : Nat := 11835
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27918⟩⟩] } }
def leftRaw : List Term := Proof.Events046.exact11909RawTerms
def rightRaw : List Term := Proof.Events046.exact11886RawTerms
def group : MergeGroup := .operator 11909 11886
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11909) (leftOrdinal := 0)
    (rightResult := 11886) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27918⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27918⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11916

namespace LeftMerge11924
def owner : Owner := ⟨.program ⟨214⟩, ⟨15999⟩⟩
def mergeEvent : Nat := 11924
def frameStart : Nat := 11835
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events046.exact11897RawTerms
def rightRaw : List Term := Proof.Events046.exact11920RawTerms
def group : MergeGroup := .operator 11897 11920
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11897) (leftOrdinal := 0)
    (rightResult := 11920) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11924

namespace LeftMerge11941
def owner : Owner := ⟨.program ⟨214⟩, ⟨21419⟩⟩
def mergeEvent : Nat := 11941
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15956⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24174⟩⟩] } }
def rhsRaw : List Term := Proof.Events046.exact11938RawTerms
def group : MergeGroup := .relation 11940
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 11940) (rhsResult := 11938)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21416⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 11939 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21416⟩⟩]⟩) (none) 11938) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15956⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24174⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15956⟩⟩], [⟨.program ⟨214⟩, ⟨24174⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11941

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
