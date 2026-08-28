import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge27454
def owner : Owner := ⟨.program ⟨214⟩, ⟨15716⟩⟩
def mergeEvent : Nat := 27454
def frameStart : Nat := 27351
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15714⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27407RawTerms
def rightRaw : List Term := Proof.Events107.exact27450RawTerms
def group : MergeGroup := .operator 27407 27450
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27407) (leftOrdinal := 0)
    (rightResult := 27450) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15714⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27454

namespace LeftMerge27471
def owner : Owner := ⟨.program ⟨214⟩, ⟨19399⟩⟩
def mergeEvent : Nat := 27471
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27468RawTerms
def group : MergeGroup := .relation 27470
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27470) (rhsResult := 27468)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19396⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27469 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19396⟩⟩]⟩) (none) 27468) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27471

namespace LeftMerge27472
def owner : Owner := ⟨.program ⟨214⟩, ⟨19399⟩⟩
def mergeEvent : Nat := 27472
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25927⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27468RawTerms
def group : MergeGroup := .relation 27470
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27470) (rhsResult := 27468)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19396⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27469 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19396⟩⟩]⟩) (none) 27468) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25927⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25927⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27472

namespace LeftMerge27473
def owner : Owner := ⟨.program ⟨214⟩, ⟨19399⟩⟩
def mergeEvent : Nat := 27473
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23506⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27468RawTerms
def group : MergeGroup := .relation 27470
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27470) (rhsResult := 27468)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19396⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27469 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19396⟩⟩]⟩) (none) 27468) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11313⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23506⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨23506⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27473

namespace LeftMerge27474
def owner : Owner := ⟨.program ⟨214⟩, ⟨19399⟩⟩
def mergeEvent : Nat := 27474
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15714⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27468RawTerms
def group : MergeGroup := .relation 27470
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27470) (rhsResult := 27468)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19396⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27469 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19396⟩⟩]⟩) (none) 27468) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15714⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27474

namespace LeftMerge27479
def owner : Owner := ⟨.program ⟨214⟩, ⟨25929⟩⟩
def mergeEvent : Nat := 27479
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23506⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27475RawTerms
def rightRaw : List Term := Proof.Events106.exact27289RawTerms
def group : MergeGroup := .operator 27475 27289
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27475) (leftOrdinal := 2)
    (rightResult := 27289) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23506⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23506⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨23506⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27479

namespace LeftMerge27480
def owner : Owner := ⟨.program ⟨214⟩, ⟨25929⟩⟩
def mergeEvent : Nat := 27480
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25927⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27475RawTerms
def rightRaw : List Term := Proof.Events106.exact27289RawTerms
def group : MergeGroup := .operator 27475 27289
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27475) (leftOrdinal := 1)
    (rightResult := 27289) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25927⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25927⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25927⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27480

namespace LeftMerge27488
def owner : Owner := ⟨.program ⟨214⟩, ⟨27473⟩⟩
def mergeEvent : Nat := 27488
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27471⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27482RawTerms
def rightRaw : List Term := Proof.Events106.exact27205RawTerms
def group : MergeGroup := .operator 27482 27205
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27482) (leftOrdinal := 0)
    (rightResult := 27205) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27471⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27471⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27488

namespace LeftMerge27489
def owner : Owner := ⟨.program ⟨214⟩, ⟨27473⟩⟩
def mergeEvent : Nat := 27489
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15714⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27471⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27482RawTerms
def rightRaw : List Term := Proof.Events106.exact27205RawTerms
def group : MergeGroup := .operator 27482 27205
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27482) (leftOrdinal := 1)
    (rightResult := 27205) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15714⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27471⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27471⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27489

namespace LeftMerge27491
def owner : Owner := ⟨.program ⟨214⟩, ⟨27473⟩⟩
def mergeEvent : Nat := 27491
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15714⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24045⟩⟩] } }
def rhsRaw : List Term := Proof.Events106.exact27202RawTerms
def group : MergeGroup := .relation 27490
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27490) (rhsResult := 27202)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27471⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27471⟩⟩) ⟨24045⟩ 27202) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24045⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15714⟩⟩], [⟨.program ⟨214⟩, ⟨24045⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27491

namespace LeftMerge27505
def owner : Owner := ⟨.program ⟨214⟩, ⟨21127⟩⟩
def mergeEvent : Nat := 27505
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21124⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21512RawTerms
def rightRaw : List Term := Proof.Events107.exact27499RawTerms
def group : MergeGroup := .operator 21512 27499
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21512) (leftOrdinal := 0)
    (rightResult := 27499) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21124⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21124⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27505

namespace LeftMerge27626
def owner : Owner := ⟨.program ⟨214⟩, ⟨15791⟩⟩
def mergeEvent : Nat := 27626
def frameStart : Nat := 27560
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15714⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27622RawTerms
def rightRaw : List Term := Proof.Events107.exact27620RawTerms
def group : MergeGroup := .operator 27622 27620
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27622) (leftOrdinal := 0)
    (rightResult := 27620) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15714⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27626

namespace LeftMerge27638
def owner : Owner := ⟨.program ⟨214⟩, ⟨27472⟩⟩
def mergeEvent : Nat := 27638
def frameStart : Nat := 27560
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27471⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27634RawTerms
def rightRaw : List Term := Proof.Events107.exact27611RawTerms
def group : MergeGroup := .operator 27634 27611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27634) (leftOrdinal := 0)
    (rightResult := 27611) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27471⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27471⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27638

namespace LeftMerge27639
def owner : Owner := ⟨.program ⟨214⟩, ⟨27472⟩⟩
def mergeEvent : Nat := 27639
def frameStart : Nat := 27560
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15714⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27471⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27634RawTerms
def rightRaw : List Term := Proof.Events107.exact27611RawTerms
def group : MergeGroup := .operator 27634 27611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27634) (leftOrdinal := 1)
    (rightResult := 27611) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15714⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27471⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27471⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27639

namespace LeftMerge27641
def owner : Owner := ⟨.program ⟨214⟩, ⟨27472⟩⟩
def mergeEvent : Nat := 27641
def frameStart : Nat := 27560
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15714⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24045⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27608RawTerms
def group : MergeGroup := .relation 27640
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27640) (rhsResult := 27608)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27471⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27471⟩⟩) ⟨24045⟩ 27608) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24045⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15714⟩⟩], [⟨.program ⟨214⟩, ⟨24045⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27641

namespace LeftMerge27649
def owner : Owner := ⟨.program ⟨214⟩, ⟨15758⟩⟩
def mergeEvent : Nat := 27649
def frameStart : Nat := 27560
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15757⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27622RawTerms
def rightRaw : List Term := Proof.Events107.exact27645RawTerms
def group : MergeGroup := .operator 27622 27645
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27622) (leftOrdinal := 0)
    (rightResult := 27645) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15757⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27649

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
