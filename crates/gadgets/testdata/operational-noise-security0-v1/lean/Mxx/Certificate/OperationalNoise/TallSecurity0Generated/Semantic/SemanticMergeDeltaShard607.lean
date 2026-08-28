import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge98357
def owner : Owner := ⟨.program ⟨214⟩, ⟨26131⟩⟩
def mergeEvent : Nat := 98357
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26130⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98351RawTerms
def rightRaw : List Term := Proof.Events383.exact98287RawTerms
def group : MergeGroup := .operator 98351 98287
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98351) (leftOrdinal := 1)
    (rightResult := 98287) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26130⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26130⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98357

namespace LeftMerge98359
def owner : Owner := ⟨.program ⟨214⟩, ⟨26131⟩⟩
def mergeEvent : Nat := 98359
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23620⟩⟩] } }
def rhsRaw : List Term := Proof.Events383.exact98284RawTerms
def group : MergeGroup := .relation 98358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98358) (rhsResult := 98284)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26130⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26130⟩⟩) ⟨23620⟩ 98284) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], [⟨.program ⟨214⟩, ⟨23620⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98359

namespace LeftMerge98360
def owner : Owner := ⟨.program ⟨214⟩, ⟨26131⟩⟩
def mergeEvent : Nat := 98360
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26130⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98351RawTerms
def rightRaw : List Term := Proof.Events383.exact98287RawTerms
def group : MergeGroup := .operator 98351 98287
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98351) (leftOrdinal := 0)
    (rightResult := 98287) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26130⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26130⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98360

namespace LeftMerge98374
def owner : Owner := ⟨.program ⟨214⟩, ⟨19592⟩⟩
def mergeEvent : Nat := 98374
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19589⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94462RawTerms
def rightRaw : List Term := Proof.Events384.exact98368RawTerms
def group : MergeGroup := .operator 94462 98368
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94462) (leftOrdinal := 0)
    (rightResult := 98368) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19589⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19589⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98374

namespace LeftMerge98429
def owner : Owner := ⟨.program ⟨214⟩, ⟨14398⟩⟩
def mergeEvent : Nat := 98429
def frameStart : Nat := 98411
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events384.exact98425RawTerms
def rightRaw : List Term := Proof.Events384.exact98422RawTerms
def group : MergeGroup := .operator 98425 98422
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98425) (leftOrdinal := 0)
    (rightResult := 98422) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11541⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98429

namespace LeftMerge98459
def owner : Owner := ⟨.program ⟨214⟩, ⟨14525⟩⟩
def mergeEvent : Nat := 98459
def frameStart : Nat := 98411
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98455RawTerms
def rightRaw : List Term := Proof.Events384.exact98453RawTerms
def group : MergeGroup := .operator 98455 98453
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98455) (leftOrdinal := 0)
    (rightResult := 98453) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98459

namespace LeftMerge98482
def owner : Owner := ⟨.program ⟨214⟩, ⟨7857⟩⟩
def mergeEvent : Nat := 98482
def frameStart : Nat := 98411
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98478RawTerms
def rightRaw : List Term := Proof.Events384.exact98475RawTerms
def group : MergeGroup := .operator 98478 98475
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98478) (leftOrdinal := 0)
    (rightResult := 98475) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7855⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98482

namespace LeftMerge98491
def owner : Owner := ⟨.program ⟨214⟩, ⟨26133⟩⟩
def mergeEvent : Nat := 98491
def frameStart : Nat := 98411
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26130⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98487RawTerms
def rightRaw : List Term := Proof.Events384.exact98444RawTerms
def group : MergeGroup := .operator 98487 98444
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98487) (leftOrdinal := 0)
    (rightResult := 98444) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26130⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26130⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98491

namespace LeftMerge98492
def owner : Owner := ⟨.program ⟨214⟩, ⟨26133⟩⟩
def mergeEvent : Nat := 98492
def frameStart : Nat := 98411
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26130⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98487RawTerms
def rightRaw : List Term := Proof.Events384.exact98444RawTerms
def group : MergeGroup := .operator 98487 98444
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98487) (leftOrdinal := 1)
    (rightResult := 98444) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26130⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26130⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98492

namespace LeftMerge98494
def owner : Owner := ⟨.program ⟨214⟩, ⟨26133⟩⟩
def mergeEvent : Nat := 98494
def frameStart : Nat := 98411
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23620⟩⟩] } }
def rhsRaw : List Term := Proof.Events384.exact98441RawTerms
def group : MergeGroup := .relation 98493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98493) (rhsResult := 98441)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26130⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26130⟩⟩) ⟨23620⟩ 98441) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], [⟨.program ⟨214⟩, ⟨23620⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98494

namespace LeftMerge98502
def owner : Owner := ⟨.program ⟨214⟩, ⟨16051⟩⟩
def mergeEvent : Nat := 98502
def frameStart : Nat := 98411
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98455RawTerms
def rightRaw : List Term := Proof.Events384.exact98498RawTerms
def group : MergeGroup := .operator 98455 98498
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98455) (leftOrdinal := 0)
    (rightResult := 98498) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98502

namespace LeftMerge98519
def owner : Owner := ⟨.program ⟨214⟩, ⟨19592⟩⟩
def mergeEvent : Nat := 98519
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩] } }
def rhsRaw : List Term := Proof.Events384.exact98516RawTerms
def group : MergeGroup := .relation 98518
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98518) (rhsResult := 98516)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19589⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98517 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19589⟩⟩]⟩) (none) 98516) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98519

namespace LeftMerge98520
def owner : Owner := ⟨.program ⟨214⟩, ⟨19592⟩⟩
def mergeEvent : Nat := 98520
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26130⟩⟩] } }
def rhsRaw : List Term := Proof.Events384.exact98516RawTerms
def group : MergeGroup := .relation 98518
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98518) (rhsResult := 98516)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19589⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98517 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19589⟩⟩]⟩) (none) 98516) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26130⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26130⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98520

namespace LeftMerge98521
def owner : Owner := ⟨.program ⟨214⟩, ⟨19592⟩⟩
def mergeEvent : Nat := 98521
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23620⟩⟩] } }
def rhsRaw : List Term := Proof.Events384.exact98516RawTerms
def group : MergeGroup := .relation 98518
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98518) (rhsResult := 98516)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19589⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98517 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19589⟩⟩]⟩) (none) 98516) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], [⟨.program ⟨214⟩, ⟨23620⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98521

namespace LeftMerge98522
def owner : Owner := ⟨.program ⟨214⟩, ⟨19592⟩⟩
def mergeEvent : Nat := 98522
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events384.exact98516RawTerms
def group : MergeGroup := .relation 98518
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98518) (rhsResult := 98516)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19589⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98517 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19589⟩⟩]⟩) (none) 98516) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98522

namespace LeftMerge98527
def owner : Owner := ⟨.program ⟨214⟩, ⟨26132⟩⟩
def mergeEvent : Nat := 98527
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23620⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98523RawTerms
def rightRaw : List Term := Proof.Events384.exact98361RawTerms
def group : MergeGroup := .operator 98523 98361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98523) (leftOrdinal := 2)
    (rightResult := 98361) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23620⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23620⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], [⟨.program ⟨214⟩, ⟨23620⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98527

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
