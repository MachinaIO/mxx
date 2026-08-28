import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge27767
def owner : Owner := ⟨.program ⟨214⟩, ⟨25851⟩⟩
def mergeEvent : Nat := 27767
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩] } }
def leftRaw : List Term := Proof.Events108.exact27761RawTerms
def rightRaw : List Term := Proof.Events108.exact27697RawTerms
def group : MergeGroup := .operator 27761 27697
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27761) (leftOrdinal := 1)
    (rightResult := 27697) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25850⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27767

namespace LeftMerge27769
def owner : Owner := ⟨.program ⟨214⟩, ⟨25851⟩⟩
def mergeEvent : Nat := 27769
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23464⟩⟩] } }
def rhsRaw : List Term := Proof.Events108.exact27694RawTerms
def group : MergeGroup := .relation 27768
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27768) (rhsResult := 27694)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25850⟩⟩) ⟨23464⟩ 27694) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23464⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨23464⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27769

namespace LeftMerge27770
def owner : Owner := ⟨.program ⟨214⟩, ⟨25851⟩⟩
def mergeEvent : Nat := 27770
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩] } }
def leftRaw : List Term := Proof.Events108.exact27761RawTerms
def rightRaw : List Term := Proof.Events108.exact27697RawTerms
def group : MergeGroup := .operator 27761 27697
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27761) (leftOrdinal := 0)
    (rightResult := 27697) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25850⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27770

namespace LeftMerge27784
def owner : Owner := ⟨.program ⟨214⟩, ⟨19327⟩⟩
def mergeEvent : Nat := 27784
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19324⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21512RawTerms
def rightRaw : List Term := Proof.Events108.exact27778RawTerms
def group : MergeGroup := .operator 21512 27778
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21512) (leftOrdinal := 0)
    (rightResult := 27778) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19324⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27784

namespace LeftMerge27863
def owner : Owner := ⟨.program ⟨214⟩, ⟨13584⟩⟩
def mergeEvent : Nat := 27863
def frameStart : Nat := 27833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events108.exact27859RawTerms
def rightRaw : List Term := Proof.Events108.exact27856RawTerms
def group : MergeGroup := .operator 27859 27856
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27859) (leftOrdinal := 0)
    (rightResult := 27856) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13583⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11229⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27863

namespace LeftMerge27893
def owner : Owner := ⟨.program ⟨214⟩, ⟨13677⟩⟩
def mergeEvent : Nat := 27893
def frameStart : Nat := 27833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events108.exact27889RawTerms
def rightRaw : List Term := Proof.Events108.exact27887RawTerms
def group : MergeGroup := .operator 27889 27887
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27889) (leftOrdinal := 0)
    (rightResult := 27887) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27893

namespace LeftMerge27916
def owner : Owner := ⟨.program ⟨214⟩, ⟨7845⟩⟩
def mergeEvent : Nat := 27916
def frameStart : Nat := 27833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩] } }
def leftRaw : List Term := Proof.Events109.exact27912RawTerms
def rightRaw : List Term := Proof.Events109.exact27909RawTerms
def group : MergeGroup := .operator 27912 27909
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27912) (leftOrdinal := 0)
    (rightResult := 27909) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7843⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27916

namespace LeftMerge27925
def owner : Owner := ⟨.program ⟨214⟩, ⟨25853⟩⟩
def mergeEvent : Nat := 27925
def frameStart : Nat := 27833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩] } }
def leftRaw : List Term := Proof.Events109.exact27921RawTerms
def rightRaw : List Term := Proof.Events108.exact27878RawTerms
def group : MergeGroup := .operator 27921 27878
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27921) (leftOrdinal := 0)
    (rightResult := 27878) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25850⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27925

namespace LeftMerge27926
def owner : Owner := ⟨.program ⟨214⟩, ⟨25853⟩⟩
def mergeEvent : Nat := 27926
def frameStart : Nat := 27833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩] } }
def leftRaw : List Term := Proof.Events109.exact27921RawTerms
def rightRaw : List Term := Proof.Events108.exact27878RawTerms
def group : MergeGroup := .operator 27921 27878
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27921) (leftOrdinal := 1)
    (rightResult := 27878) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25850⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27926

namespace LeftMerge27928
def owner : Owner := ⟨.program ⟨214⟩, ⟨25853⟩⟩
def mergeEvent : Nat := 27928
def frameStart : Nat := 27833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23464⟩⟩] } }
def rhsRaw : List Term := Proof.Events108.exact27875RawTerms
def group : MergeGroup := .relation 27927
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27927) (rhsResult := 27875)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25850⟩⟩) ⟨23464⟩ 27875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23464⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨23464⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27928

namespace LeftMerge27936
def owner : Owner := ⟨.program ⟨214⟩, ⟨15597⟩⟩
def mergeEvent : Nat := 27936
def frameStart : Nat := 27833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15595⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events108.exact27889RawTerms
def rightRaw : List Term := Proof.Events109.exact27932RawTerms
def group : MergeGroup := .operator 27889 27932
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27889) (leftOrdinal := 0)
    (rightResult := 27932) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15595⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27936

namespace LeftMerge27953
def owner : Owner := ⟨.program ⟨214⟩, ⟨19327⟩⟩
def mergeEvent : Nat := 27953
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6694⟩⟩] } }
def rhsRaw : List Term := Proof.Events109.exact27950RawTerms
def group : MergeGroup := .relation 27952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27952) (rhsResult := 27950)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27951 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩) (none) 27950) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6694⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27953

namespace LeftMerge27954
def owner : Owner := ⟨.program ⟨214⟩, ⟨19327⟩⟩
def mergeEvent : Nat := 27954
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩] } }
def rhsRaw : List Term := Proof.Events109.exact27950RawTerms
def group : MergeGroup := .relation 27952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27952) (rhsResult := 27950)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27951 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩) (none) 27950) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27954

namespace LeftMerge27955
def owner : Owner := ⟨.program ⟨214⟩, ⟨19327⟩⟩
def mergeEvent : Nat := 27955
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23464⟩⟩] } }
def rhsRaw : List Term := Proof.Events109.exact27950RawTerms
def group : MergeGroup := .relation 27952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27952) (rhsResult := 27950)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27951 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩) (none) 27950) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23464⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨23464⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27955

namespace LeftMerge27956
def owner : Owner := ⟨.program ⟨214⟩, ⟨19327⟩⟩
def mergeEvent : Nat := 27956
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15595⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events109.exact27950RawTerms
def group : MergeGroup := .relation 27952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27952) (rhsResult := 27950)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27951 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩) (none) 27950) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15595⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27956

namespace LeftMerge27961
def owner : Owner := ⟨.program ⟨214⟩, ⟨25852⟩⟩
def mergeEvent : Nat := 27961
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23464⟩⟩] } }
def leftRaw : List Term := Proof.Events109.exact27957RawTerms
def rightRaw : List Term := Proof.Events108.exact27771RawTerms
def group : MergeGroup := .operator 27957 27771
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27957) (leftOrdinal := 2)
    (rightResult := 27771) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23464⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23464⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨23464⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27961

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
