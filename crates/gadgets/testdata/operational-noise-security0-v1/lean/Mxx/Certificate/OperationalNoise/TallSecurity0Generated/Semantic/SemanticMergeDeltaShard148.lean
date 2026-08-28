import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge25360
def owner : Owner := ⟨.program ⟨214⟩, ⟨26236⟩⟩
def mergeEvent : Nat := 25360
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26235⟩⟩] } }
def leftRaw : List Term := Proof.Events099.exact25351RawTerms
def rightRaw : List Term := Proof.Events098.exact25287RawTerms
def group : MergeGroup := .operator 25351 25287
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25351) (leftOrdinal := 0)
    (rightResult := 25287) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26235⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26235⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25360

namespace LeftMerge25374
def owner : Owner := ⟨.program ⟨214⟩, ⟨19687⟩⟩
def mergeEvent : Nat := 25374
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19684⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21512RawTerms
def rightRaw : List Term := Proof.Events099.exact25368RawTerms
def group : MergeGroup := .operator 21512 25368
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21512) (leftOrdinal := 0)
    (rightResult := 25368) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25374

namespace LeftMerge25453
def owner : Owner := ⟨.program ⟨214⟩, ⟨14669⟩⟩
def mergeEvent : Nat := 25453
def frameStart : Nat := 25423
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11649⟩⟩, ⟨.program ⟨214⟩, ⟨14668⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events099.exact25449RawTerms
def rightRaw : List Term := Proof.Events099.exact25446RawTerms
def group : MergeGroup := .operator 25449 25446
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25449) (leftOrdinal := 0)
    (rightResult := 25446) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14668⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11649⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11649⟩⟩, ⟨.program ⟨214⟩, ⟨14668⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25453

namespace LeftMerge25483
def owner : Owner := ⟨.program ⟨214⟩, ⟨14762⟩⟩
def mergeEvent : Nat := 25483
def frameStart : Nat := 25423
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11649⟩⟩, ⟨.program ⟨214⟩, ⟨14668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events099.exact25479RawTerms
def rightRaw : List Term := Proof.Events099.exact25477RawTerms
def group : MergeGroup := .operator 25479 25477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25479) (leftOrdinal := 0)
    (rightResult := 25477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11649⟩⟩, ⟨.program ⟨214⟩, ⟨14668⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11649⟩⟩, ⟨.program ⟨214⟩, ⟨14668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25483

namespace LeftMerge25506
def owner : Owner := ⟨.program ⟨214⟩, ⟨7860⟩⟩
def mergeEvent : Nat := 25506
def frameStart : Nat := 25423
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩] } }
def leftRaw : List Term := Proof.Events099.exact25502RawTerms
def rightRaw : List Term := Proof.Events099.exact25499RawTerms
def group : MergeGroup := .operator 25502 25499
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25502) (leftOrdinal := 0)
    (rightResult := 25499) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7858⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25506

namespace LeftMerge25515
def owner : Owner := ⟨.program ⟨214⟩, ⟨26238⟩⟩
def mergeEvent : Nat := 25515
def frameStart : Nat := 25423
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26235⟩⟩] } }
def leftRaw : List Term := Proof.Events099.exact25511RawTerms
def rightRaw : List Term := Proof.Events099.exact25468RawTerms
def group : MergeGroup := .operator 25511 25468
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25511) (leftOrdinal := 0)
    (rightResult := 25468) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26235⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26235⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25515

namespace LeftMerge25516
def owner : Owner := ⟨.program ⟨214⟩, ⟨26238⟩⟩
def mergeEvent : Nat := 25516
def frameStart : Nat := 25423
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11649⟩⟩, ⟨.program ⟨214⟩, ⟨14668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26235⟩⟩] } }
def leftRaw : List Term := Proof.Events099.exact25511RawTerms
def rightRaw : List Term := Proof.Events099.exact25468RawTerms
def group : MergeGroup := .operator 25511 25468
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25511) (leftOrdinal := 1)
    (rightResult := 25468) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11649⟩⟩, ⟨.program ⟨214⟩, ⟨14668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26235⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11649⟩⟩, ⟨.program ⟨214⟩, ⟨14668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26235⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25516

namespace LeftMerge25518
def owner : Owner := ⟨.program ⟨214⟩, ⟨26238⟩⟩
def mergeEvent : Nat := 25518
def frameStart : Nat := 25423
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11649⟩⟩, ⟨.program ⟨214⟩, ⟨14668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23674⟩⟩] } }
def rhsRaw : List Term := Proof.Events099.exact25465RawTerms
def group : MergeGroup := .relation 25517
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25517) (rhsResult := 25465)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11649⟩⟩, ⟨.program ⟨214⟩, ⟨14668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26235⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26235⟩⟩) ⟨23674⟩ 25465) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23674⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11649⟩⟩, ⟨.program ⟨214⟩, ⟨14668⟩⟩], [⟨.program ⟨214⟩, ⟨23674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25518

namespace LeftMerge25526
def owner : Owner := ⟨.program ⟨214⟩, ⟨16192⟩⟩
def mergeEvent : Nat := 25526
def frameStart : Nat := 25423
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events099.exact25479RawTerms
def rightRaw : List Term := Proof.Events099.exact25522RawTerms
def group : MergeGroup := .operator 25479 25522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25479) (leftOrdinal := 0)
    (rightResult := 25522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16190⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25526

namespace LeftMerge25543
def owner : Owner := ⟨.program ⟨214⟩, ⟨19687⟩⟩
def mergeEvent : Nat := 25543
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }
def rhsRaw : List Term := Proof.Events099.exact25540RawTerms
def group : MergeGroup := .relation 25542
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25542) (rhsResult := 25540)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19684⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 25541 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19684⟩⟩]⟩) (none) 25540) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25543

namespace LeftMerge25544
def owner : Owner := ⟨.program ⟨214⟩, ⟨19687⟩⟩
def mergeEvent : Nat := 25544
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26235⟩⟩] } }
def rhsRaw : List Term := Proof.Events099.exact25540RawTerms
def group : MergeGroup := .relation 25542
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25542) (rhsResult := 25540)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19684⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 25541 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19684⟩⟩]⟩) (none) 25540) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26235⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26235⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25544

namespace LeftMerge25545
def owner : Owner := ⟨.program ⟨214⟩, ⟨19687⟩⟩
def mergeEvent : Nat := 25545
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11649⟩⟩, ⟨.program ⟨214⟩, ⟨14668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23674⟩⟩] } }
def rhsRaw : List Term := Proof.Events099.exact25540RawTerms
def group : MergeGroup := .relation 25542
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25542) (rhsResult := 25540)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19684⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 25541 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19684⟩⟩]⟩) (none) 25540) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11649⟩⟩, ⟨.program ⟨214⟩, ⟨14668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23674⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11649⟩⟩, ⟨.program ⟨214⟩, ⟨14668⟩⟩], [⟨.program ⟨214⟩, ⟨23674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25545

namespace LeftMerge25546
def owner : Owner := ⟨.program ⟨214⟩, ⟨19687⟩⟩
def mergeEvent : Nat := 25546
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events099.exact25540RawTerms
def group : MergeGroup := .relation 25542
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25542) (rhsResult := 25540)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19684⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 25541 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19684⟩⟩]⟩) (none) 25540) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25546

namespace LeftMerge25551
def owner : Owner := ⟨.program ⟨214⟩, ⟨26237⟩⟩
def mergeEvent : Nat := 25551
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11649⟩⟩, ⟨.program ⟨214⟩, ⟨14668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23674⟩⟩] } }
def leftRaw : List Term := Proof.Events099.exact25547RawTerms
def rightRaw : List Term := Proof.Events099.exact25361RawTerms
def group : MergeGroup := .operator 25547 25361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25547) (leftOrdinal := 2)
    (rightResult := 25361) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11649⟩⟩, ⟨.program ⟨214⟩, ⟨14668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23674⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11649⟩⟩, ⟨.program ⟨214⟩, ⟨14668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23674⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11649⟩⟩, ⟨.program ⟨214⟩, ⟨14668⟩⟩], [⟨.program ⟨214⟩, ⟨23674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25551

namespace LeftMerge25552
def owner : Owner := ⟨.program ⟨214⟩, ⟨26237⟩⟩
def mergeEvent : Nat := 25552
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26235⟩⟩] } }
def leftRaw : List Term := Proof.Events099.exact25547RawTerms
def rightRaw : List Term := Proof.Events099.exact25361RawTerms
def group : MergeGroup := .operator 25547 25361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25547) (leftOrdinal := 1)
    (rightResult := 25361) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26235⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26235⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26235⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25552

namespace LeftMerge25560
def owner : Owner := ⟨.program ⟨214⟩, ⟨28341⟩⟩
def mergeEvent : Nat := 25560
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28339⟩⟩] } }
def leftRaw : List Term := Proof.Events099.exact25554RawTerms
def rightRaw : List Term := Proof.Events098.exact25277RawTerms
def group : MergeGroup := .operator 25554 25277
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25554) (leftOrdinal := 0)
    (rightResult := 25277) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28339⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28339⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25560

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
