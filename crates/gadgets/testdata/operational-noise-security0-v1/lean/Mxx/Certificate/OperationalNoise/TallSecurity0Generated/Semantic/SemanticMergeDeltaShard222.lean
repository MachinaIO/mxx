import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge36794
def owner : Owner := ⟨.program ⟨214⟩, ⟨20187⟩⟩
def mergeEvent : Nat := 36794
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }
def rhsRaw : List Term := Proof.Events143.exact36791RawTerms
def group : MergeGroup := .relation 36793
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36793) (rhsResult := 36791)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20184⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 36792 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20184⟩⟩]⟩) (none) 36791) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36794

namespace LeftMerge36795
def owner : Owner := ⟨.program ⟨214⟩, ⟨20187⟩⟩
def mergeEvent : Nat := 36795
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25691⟩⟩] } }
def rhsRaw : List Term := Proof.Events143.exact36791RawTerms
def group : MergeGroup := .relation 36793
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36793) (rhsResult := 36791)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20184⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 36792 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20184⟩⟩]⟩) (none) 36791) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25691⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25691⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36795

namespace LeftMerge36796
def owner : Owner := ⟨.program ⟨214⟩, ⟨20187⟩⟩
def mergeEvent : Nat := 36796
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23378⟩⟩] } }
def rhsRaw : List Term := Proof.Events143.exact36791RawTerms
def group : MergeGroup := .relation 36793
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36793) (rhsResult := 36791)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20184⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 36792 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20184⟩⟩]⟩) (none) 36791) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23378⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], [⟨.program ⟨214⟩, ⟨23378⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36796

namespace LeftMerge36797
def owner : Owner := ⟨.program ⟨214⟩, ⟨20187⟩⟩
def mergeEvent : Nat := 36797
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events143.exact36791RawTerms
def group : MergeGroup := .relation 36793
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36793) (rhsResult := 36791)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20184⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 36792 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20184⟩⟩]⟩) (none) 36791) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36797

namespace LeftMerge36802
def owner : Owner := ⟨.program ⟨214⟩, ⟨25693⟩⟩
def mergeEvent : Nat := 36802
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23378⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36798RawTerms
def rightRaw : List Term := Proof.Events143.exact36612RawTerms
def group : MergeGroup := .operator 36798 36612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36798) (leftOrdinal := 2)
    (rightResult := 36612) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23378⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23378⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], [⟨.program ⟨214⟩, ⟨23378⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36802

namespace LeftMerge36803
def owner : Owner := ⟨.program ⟨214⟩, ⟨25693⟩⟩
def mergeEvent : Nat := 36803
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25691⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36798RawTerms
def rightRaw : List Term := Proof.Events143.exact36612RawTerms
def group : MergeGroup := .operator 36798 36612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36798) (leftOrdinal := 1)
    (rightResult := 36612) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25691⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25691⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25691⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36803

namespace LeftMerge36811
def owner : Owner := ⟨.program ⟨214⟩, ⟨29847⟩⟩
def mergeEvent : Nat := 36811
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29845⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36805RawTerms
def rightRaw : List Term := Proof.Events142.exact36528RawTerms
def group : MergeGroup := .operator 36805 36528
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36805) (leftOrdinal := 0)
    (rightResult := 36528) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29845⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29845⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36811

namespace LeftMerge36812
def owner : Owner := ⟨.program ⟨214⟩, ⟨29847⟩⟩
def mergeEvent : Nat := 36812
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29845⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36805RawTerms
def rightRaw : List Term := Proof.Events142.exact36528RawTerms
def group : MergeGroup := .operator 36805 36528
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36805) (leftOrdinal := 1)
    (rightResult := 36528) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29845⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29845⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36812

namespace LeftMerge36814
def owner : Owner := ⟨.program ⟨214⟩, ⟨29847⟩⟩
def mergeEvent : Nat := 36814
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24735⟩⟩] } }
def rhsRaw : List Term := Proof.Events142.exact36525RawTerms
def group : MergeGroup := .relation 36813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36813) (rhsResult := 36525)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29845⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29845⟩⟩) ⟨24735⟩ 36525) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24735⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], [⟨.program ⟨214⟩, ⟨24735⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36814

namespace LeftMerge36828
def owner : Owner := ⟨.program ⟨214⟩, ⟨22707⟩⟩
def mergeEvent : Nat := 36828
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22704⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events143.exact36822RawTerms
def group : MergeGroup := .operator 36137 36822
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 36822) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22704⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22704⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36828

namespace LeftMerge36949
def owner : Owner := ⟨.program ⟨214⟩, ⟨16977⟩⟩
def mergeEvent : Nat := 36949
def frameStart : Nat := 36883
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events144.exact36945RawTerms
def rightRaw : List Term := Proof.Events144.exact36943RawTerms
def group : MergeGroup := .operator 36945 36943
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36945) (leftOrdinal := 0)
    (rightResult := 36943) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36949

namespace LeftMerge36961
def owner : Owner := ⟨.program ⟨214⟩, ⟨29846⟩⟩
def mergeEvent : Nat := 36961
def frameStart : Nat := 36883
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29845⟩⟩] } }
def leftRaw : List Term := Proof.Events144.exact36957RawTerms
def rightRaw : List Term := Proof.Events144.exact36934RawTerms
def group : MergeGroup := .operator 36957 36934
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36957) (leftOrdinal := 0)
    (rightResult := 36934) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29845⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29845⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36961

namespace LeftMerge36962
def owner : Owner := ⟨.program ⟨214⟩, ⟨29846⟩⟩
def mergeEvent : Nat := 36962
def frameStart : Nat := 36883
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29845⟩⟩] } }
def leftRaw : List Term := Proof.Events144.exact36957RawTerms
def rightRaw : List Term := Proof.Events144.exact36934RawTerms
def group : MergeGroup := .operator 36957 36934
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36957) (leftOrdinal := 1)
    (rightResult := 36934) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29845⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29845⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36962

namespace LeftMerge36964
def owner : Owner := ⟨.program ⟨214⟩, ⟨29846⟩⟩
def mergeEvent : Nat := 36964
def frameStart : Nat := 36883
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24735⟩⟩] } }
def rhsRaw : List Term := Proof.Events144.exact36931RawTerms
def group : MergeGroup := .relation 36963
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36963) (rhsResult := 36931)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29845⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29845⟩⟩) ⟨24735⟩ 36931) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24735⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16879⟩⟩], [⟨.program ⟨214⟩, ⟨24735⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36964

namespace LeftMerge36972
def owner : Owner := ⟨.program ⟨214⟩, ⟨17092⟩⟩
def mergeEvent : Nat := 36972
def frameStart : Nat := 36883
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17091⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events144.exact36945RawTerms
def rightRaw : List Term := Proof.Events144.exact36968RawTerms
def group : MergeGroup := .operator 36945 36968
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36945) (leftOrdinal := 0)
    (rightResult := 36968) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17091⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17091⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36972

namespace LeftMerge36989
def owner : Owner := ⟨.program ⟨214⟩, ⟨22707⟩⟩
def mergeEvent : Nat := 36989
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6741⟩⟩] } }
def rhsRaw : List Term := Proof.Events144.exact36986RawTerms
def group : MergeGroup := .relation 36988
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36988) (rhsResult := 36986)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22704⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 36987 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22704⟩⟩]⟩) (none) 36986) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6741⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36989

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
