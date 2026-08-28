import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge105485
def owner : Owner := ⟨.program ⟨214⟩, ⟨28043⟩⟩
def mergeEvent : Nat := 105485
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98530RawTerms
def rightRaw : List Term := Proof.Events412.exact105478RawTerms
def group : MergeGroup := .operator 98530 105478
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98530) (leftOrdinal := 1)
    (rightResult := 105478) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28041⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105485

namespace LeftMerge105487
def owner : Owner := ⟨.program ⟨214⟩, ⟨28043⟩⟩
def mergeEvent : Nat := 105487
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24215⟩⟩] } }
def rhsRaw : List Term := Proof.Events412.exact105475RawTerms
def group : MergeGroup := .relation 105486
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 105486) (rhsResult := 105475)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28041⟩⟩) ⟨24215⟩ 105475) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24215⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨24215⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105487

namespace LeftMerge105501
def owner : Owner := ⟨.program ⟨214⟩, ⟨21464⟩⟩
def mergeEvent : Nat := 105501
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21461⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94462RawTerms
def rightRaw : List Term := Proof.Events412.exact105495RawTerms
def group : MergeGroup := .operator 94462 105495
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94462) (leftOrdinal := 0)
    (rightResult := 105495) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21461⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105501

namespace LeftMerge105598
def owner : Owner := ⟨.program ⟨214⟩, ⟨16128⟩⟩
def mergeEvent : Nat := 105598
def frameStart : Nat := 105544
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events412.exact105594RawTerms
def rightRaw : List Term := Proof.Events412.exact105592RawTerms
def group : MergeGroup := .operator 105594 105592
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105594) (leftOrdinal := 0)
    (rightResult := 105592) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105598

namespace LeftMerge105610
def owner : Owner := ⟨.program ⟨214⟩, ⟨28042⟩⟩
def mergeEvent : Nat := 105610
def frameStart : Nat := 105544
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩] } }
def leftRaw : List Term := Proof.Events412.exact105606RawTerms
def rightRaw : List Term := Proof.Events412.exact105583RawTerms
def group : MergeGroup := .operator 105606 105583
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105606) (leftOrdinal := 0)
    (rightResult := 105583) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28041⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105610

namespace LeftMerge105611
def owner : Owner := ⟨.program ⟨214⟩, ⟨28042⟩⟩
def mergeEvent : Nat := 105611
def frameStart : Nat := 105544
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩] } }
def leftRaw : List Term := Proof.Events412.exact105606RawTerms
def rightRaw : List Term := Proof.Events412.exact105583RawTerms
def group : MergeGroup := .operator 105606 105583
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105606) (leftOrdinal := 1)
    (rightResult := 105583) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28041⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105611

namespace LeftMerge105613
def owner : Owner := ⟨.program ⟨214⟩, ⟨28042⟩⟩
def mergeEvent : Nat := 105613
def frameStart : Nat := 105544
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24215⟩⟩] } }
def rhsRaw : List Term := Proof.Events412.exact105580RawTerms
def group : MergeGroup := .relation 105612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 105612) (rhsResult := 105580)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28041⟩⟩) ⟨24215⟩ 105580) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24215⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨24215⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105613

namespace LeftMerge105621
def owner : Owner := ⟨.program ⟨214⟩, ⟨18021⟩⟩
def mergeEvent : Nat := 105621
def frameStart : Nat := 105544
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18016⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events412.exact105594RawTerms
def rightRaw : List Term := Proof.Events412.exact105617RawTerms
def group : MergeGroup := .operator 105594 105617
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105594) (leftOrdinal := 0)
    (rightResult := 105617) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18016⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105621

namespace LeftMerge105638
def owner : Owner := ⟨.program ⟨214⟩, ⟨21464⟩⟩
def mergeEvent : Nat := 105638
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩] } }
def rhsRaw : List Term := Proof.Events412.exact105635RawTerms
def group : MergeGroup := .relation 105637
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 105637) (rhsResult := 105635)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 105636 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩) (none) 105635) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105638

namespace LeftMerge105639
def owner : Owner := ⟨.program ⟨214⟩, ⟨21464⟩⟩
def mergeEvent : Nat := 105639
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩] } }
def rhsRaw : List Term := Proof.Events412.exact105635RawTerms
def group : MergeGroup := .relation 105637
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 105637) (rhsResult := 105635)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 105636 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩) (none) 105635) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105639

namespace LeftMerge105640
def owner : Owner := ⟨.program ⟨214⟩, ⟨21464⟩⟩
def mergeEvent : Nat := 105640
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24215⟩⟩] } }
def rhsRaw : List Term := Proof.Events412.exact105635RawTerms
def group : MergeGroup := .relation 105637
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 105637) (rhsResult := 105635)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 105636 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩) (none) 105635) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24215⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨24215⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105640

namespace LeftMerge105641
def owner : Owner := ⟨.program ⟨214⟩, ⟨21464⟩⟩
def mergeEvent : Nat := 105641
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events412.exact105635RawTerms
def group : MergeGroup := .relation 105637
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 105637) (rhsResult := 105635)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 105636 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩) (none) 105635) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18016⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105641

namespace LeftMerge105646
def owner : Owner := ⟨.program ⟨214⟩, ⟨28044⟩⟩
def mergeEvent : Nat := 105646
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩] } }
def leftRaw : List Term := Proof.Events412.exact105642RawTerms
def rightRaw : List Term := Proof.Events412.exact105488RawTerms
def group : MergeGroup := .operator 105642 105488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105642) (leftOrdinal := 0)
    (rightResult := 105488) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105646

namespace LeftMerge105647
def owner : Owner := ⟨.program ⟨214⟩, ⟨28044⟩⟩
def mergeEvent : Nat := 105647
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24215⟩⟩] } }
def leftRaw : List Term := Proof.Events412.exact105642RawTerms
def rightRaw : List Term := Proof.Events412.exact105488RawTerms
def group : MergeGroup := .operator 105642 105488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105642) (leftOrdinal := 2)
    (rightResult := 105488) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24215⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24215⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨24215⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105647

namespace LeftMerge105655
def owner : Owner := ⟨.program ⟨214⟩, ⟨28045⟩⟩
def mergeEvent : Nat := 105655
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩] } }
def leftRaw : List Term := Proof.Events412.exact105649RawTerms
def rightRaw : List Term := Proof.Events022.exact5699RawTerms
def group : MergeGroup := .operator 105649 5699
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105649) (leftOrdinal := 0)
    (rightResult := 5699) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6637⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105655

namespace LeftMerge105656
def owner : Owner := ⟨.program ⟨214⟩, ⟨28045⟩⟩
def mergeEvent : Nat := 105656
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩] } }
def leftRaw : List Term := Proof.Events412.exact105649RawTerms
def rightRaw : List Term := Proof.Events022.exact5699RawTerms
def group : MergeGroup := .operator 105649 5699
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105649) (leftOrdinal := 1)
    (rightResult := 5699) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6637⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105656

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
