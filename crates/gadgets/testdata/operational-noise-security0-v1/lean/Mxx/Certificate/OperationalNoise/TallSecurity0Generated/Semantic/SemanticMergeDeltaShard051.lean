import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge9573
def owner : Owner := ⟨.program ⟨214⟩, ⟨19835⟩⟩
def mergeEvent : Nat := 9573
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19832⟩⟩] } }
def leftRaw : List Term := Proof.Events025.exact6561RawTerms
def rightRaw : List Term := Proof.Events037.exact9567RawTerms
def group : MergeGroup := .operator 6561 9567
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6561) (leftOrdinal := 0)
    (rightResult := 9567) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19832⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19832⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9573

namespace LeftMerge9652
def owner : Owner := ⟨.program ⟨214⟩, ⟨11990⟩⟩
def mergeEvent : Nat := 9652
def frameStart : Nat := 9622
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9735⟩⟩, ⟨.program ⟨214⟩, ⟨11989⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events037.exact9648RawTerms
def rightRaw : List Term := Proof.Events037.exact9645RawTerms
def group : MergeGroup := .operator 9648 9645
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9648) (leftOrdinal := 0)
    (rightResult := 9645) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9735⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11989⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9735⟩⟩, ⟨.program ⟨214⟩, ⟨11989⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9652

namespace LeftMerge9682
def owner : Owner := ⟨.program ⟨214⟩, ⟨12071⟩⟩
def mergeEvent : Nat := 9682
def frameStart : Nat := 9622
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9735⟩⟩, ⟨.program ⟨214⟩, ⟨11989⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events037.exact9678RawTerms
def rightRaw : List Term := Proof.Events037.exact9676RawTerms
def group : MergeGroup := .operator 9678 9676
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9678) (leftOrdinal := 0)
    (rightResult := 9676) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9735⟩⟩, ⟨.program ⟨214⟩, ⟨11989⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9735⟩⟩, ⟨.program ⟨214⟩, ⟨11989⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9682

namespace LeftMerge9705
def owner : Owner := ⟨.program ⟨214⟩, ⟨7866⟩⟩
def mergeEvent : Nat := 9705
def frameStart : Nat := 9622
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩] } }
def leftRaw : List Term := Proof.Events037.exact9701RawTerms
def rightRaw : List Term := Proof.Events037.exact9698RawTerms
def group : MergeGroup := .operator 9701 9698
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9701) (leftOrdinal := 0)
    (rightResult := 9698) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7864⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9705

namespace LeftMerge9714
def owner : Owner := ⟨.program ⟨214⟩, ⟨25242⟩⟩
def mergeEvent : Nat := 9714
def frameStart : Nat := 9622
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9735⟩⟩, ⟨.program ⟨214⟩, ⟨11989⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25239⟩⟩] } }
def leftRaw : List Term := Proof.Events037.exact9710RawTerms
def rightRaw : List Term := Proof.Events037.exact9667RawTerms
def group : MergeGroup := .operator 9710 9667
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9710) (leftOrdinal := 1)
    (rightResult := 9667) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9735⟩⟩, ⟨.program ⟨214⟩, ⟨11989⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25239⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9735⟩⟩, ⟨.program ⟨214⟩, ⟨11989⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25239⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge9714

namespace LeftMerge9716
def owner : Owner := ⟨.program ⟨214⟩, ⟨25242⟩⟩
def mergeEvent : Nat := 9716
def frameStart : Nat := 9622
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9735⟩⟩, ⟨.program ⟨214⟩, ⟨11989⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23130⟩⟩] } }
def rhsRaw : List Term := Proof.Events037.exact9664RawTerms
def group : MergeGroup := .relation 9715
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 9715) (rhsResult := 9664)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9735⟩⟩, ⟨.program ⟨214⟩, ⟨11989⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25239⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25239⟩⟩) ⟨23130⟩ 9664) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23130⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9735⟩⟩, ⟨.program ⟨214⟩, ⟨11989⟩⟩], [⟨.program ⟨214⟩, ⟨23130⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge9716

namespace LeftMerge9717
def owner : Owner := ⟨.program ⟨214⟩, ⟨25242⟩⟩
def mergeEvent : Nat := 9717
def frameStart : Nat := 9622
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25239⟩⟩] } }
def leftRaw : List Term := Proof.Events037.exact9710RawTerms
def rightRaw : List Term := Proof.Events037.exact9667RawTerms
def group : MergeGroup := .operator 9710 9667
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9710) (leftOrdinal := 0)
    (rightResult := 9667) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25239⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25239⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9717

namespace LeftMerge9725
def owner : Owner := ⟨.program ⟨214⟩, ⟨16399⟩⟩
def mergeEvent : Nat := 9725
def frameStart : Nat := 9622
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events037.exact9678RawTerms
def rightRaw : List Term := Proof.Events037.exact9721RawTerms
def group : MergeGroup := .operator 9678 9721
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9678) (leftOrdinal := 0)
    (rightResult := 9721) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16397⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16397⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9725

namespace LeftMerge9742
def owner : Owner := ⟨.program ⟨214⟩, ⟨19835⟩⟩
def mergeEvent : Nat := 9742
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9735⟩⟩, ⟨.program ⟨214⟩, ⟨11989⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23130⟩⟩] } }
def rhsRaw : List Term := Proof.Events038.exact9739RawTerms
def group : MergeGroup := .relation 9741
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 9741) (rhsResult := 9739)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19832⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 9740 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19832⟩⟩]⟩) (none) 9739) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9735⟩⟩, ⟨.program ⟨214⟩, ⟨11989⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23130⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9735⟩⟩, ⟨.program ⟨214⟩, ⟨11989⟩⟩], [⟨.program ⟨214⟩, ⟨23130⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9742

namespace LeftMerge9743
def owner : Owner := ⟨.program ⟨214⟩, ⟨19835⟩⟩
def mergeEvent : Nat := 9743
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25239⟩⟩] } }
def rhsRaw : List Term := Proof.Events038.exact9739RawTerms
def group : MergeGroup := .relation 9741
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 9741) (rhsResult := 9739)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19832⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 9740 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19832⟩⟩]⟩) (none) 9739) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25239⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25239⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge9743

namespace LeftMerge9744
def owner : Owner := ⟨.program ⟨214⟩, ⟨19835⟩⟩
def mergeEvent : Nat := 9744
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events038.exact9739RawTerms
def group : MergeGroup := .relation 9741
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 9741) (rhsResult := 9739)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19832⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 9740 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19832⟩⟩]⟩) (none) 9739) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16397⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge9744

namespace LeftMerge9745
def owner : Owner := ⟨.program ⟨214⟩, ⟨19835⟩⟩
def mergeEvent : Nat := 9745
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩] } }
def rhsRaw : List Term := Proof.Events038.exact9739RawTerms
def group : MergeGroup := .relation 9741
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 9741) (rhsResult := 9739)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19832⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 9740 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19832⟩⟩]⟩) (none) 9739) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9745

namespace LeftMerge9750
def owner : Owner := ⟨.program ⟨214⟩, ⟨25241⟩⟩
def mergeEvent : Nat := 9750
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9735⟩⟩, ⟨.program ⟨214⟩, ⟨11989⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23130⟩⟩] } }
def leftRaw : List Term := Proof.Events038.exact9746RawTerms
def rightRaw : List Term := Proof.Events037.exact9560RawTerms
def group : MergeGroup := .operator 9746 9560
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9746) (leftOrdinal := 2)
    (rightResult := 9560) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9735⟩⟩, ⟨.program ⟨214⟩, ⟨11989⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23130⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9735⟩⟩, ⟨.program ⟨214⟩, ⟨11989⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23130⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9735⟩⟩, ⟨.program ⟨214⟩, ⟨11989⟩⟩], [⟨.program ⟨214⟩, ⟨23130⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge9750

namespace LeftMerge9751
def owner : Owner := ⟨.program ⟨214⟩, ⟨25241⟩⟩
def mergeEvent : Nat := 9751
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25239⟩⟩] } }
def leftRaw : List Term := Proof.Events038.exact9746RawTerms
def rightRaw : List Term := Proof.Events037.exact9560RawTerms
def group : MergeGroup := .operator 9746 9560
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9746) (leftOrdinal := 1)
    (rightResult := 9560) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25239⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25239⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25239⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9751

namespace LeftMerge9759
def owner : Owner := ⟨.program ⟨214⟩, ⟨28788⟩⟩
def mergeEvent : Nat := 9759
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28786⟩⟩] } }
def leftRaw : List Term := Proof.Events038.exact9753RawTerms
def rightRaw : List Term := Proof.Events036.exact9457RawTerms
def group : MergeGroup := .operator 9753 9457
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9753) (leftOrdinal := 1)
    (rightResult := 9457) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28786⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16397⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28786⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge9759

namespace LeftMerge9761
def owner : Owner := ⟨.program ⟨214⟩, ⟨28788⟩⟩
def mergeEvent : Nat := 9761
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24426⟩⟩] } }
def rhsRaw : List Term := Proof.Events036.exact9454RawTerms
def group : MergeGroup := .relation 9760
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 9760) (rhsResult := 9454)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16397⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28786⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28786⟩⟩) ⟨24426⟩ 9454) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24426⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16397⟩⟩], [⟨.program ⟨214⟩, ⟨24426⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge9761

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
