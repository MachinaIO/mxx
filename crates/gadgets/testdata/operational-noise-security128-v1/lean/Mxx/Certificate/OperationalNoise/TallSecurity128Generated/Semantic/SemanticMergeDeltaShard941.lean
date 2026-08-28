import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge154560
def owner : Owner := ⟨.program ⟨257⟩, ⟨9534⟩⟩
def mergeEvent : Nat := 154560
def frameStart : Nat := 154477
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }
def leftRaw : List Term := Proof.Events603.exact154556RawTerms
def rightRaw : List Term := Proof.Events603.exact154553RawTerms
def group : MergeGroup := .operator 154556 154553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 154556) (leftOrdinal := 0)
    (rightResult := 154553) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9532⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge154560

namespace LeftMerge154569
def owner : Owner := ⟨.program ⟨257⟩, ⟨58449⟩⟩
def mergeEvent : Nat := 154569
def frameStart : Nat := 154477
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58446⟩⟩] } }
def leftRaw : List Term := Proof.Events603.exact154565RawTerms
def rightRaw : List Term := Proof.Events603.exact154522RawTerms
def group : MergeGroup := .operator 154565 154522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 154565) (leftOrdinal := 0)
    (rightResult := 154522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58446⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58446⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge154569

namespace LeftMerge154570
def owner : Owner := ⟨.program ⟨257⟩, ⟨58449⟩⟩
def mergeEvent : Nat := 154570
def frameStart : Nat := 154477
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24974⟩⟩, ⟨.program ⟨257⟩, ⟨56424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58446⟩⟩] } }
def leftRaw : List Term := Proof.Events603.exact154565RawTerms
def rightRaw : List Term := Proof.Events603.exact154522RawTerms
def group : MergeGroup := .operator 154565 154522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 154565) (leftOrdinal := 1)
    (rightResult := 154522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24974⟩⟩, ⟨.program ⟨257⟩, ⟨56424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58446⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24974⟩⟩, ⟨.program ⟨257⟩, ⟨56424⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58446⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge154570

namespace LeftMerge154572
def owner : Owner := ⟨.program ⟨257⟩, ⟨58449⟩⟩
def mergeEvent : Nat := 154572
def frameStart : Nat := 154477
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24974⟩⟩, ⟨.program ⟨257⟩, ⟨56424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57951⟩⟩] } }
def rhsRaw : List Term := Proof.Events603.exact154519RawTerms
def group : MergeGroup := .relation 154571
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 154571) (rhsResult := 154519)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24974⟩⟩, ⟨.program ⟨257⟩, ⟨56424⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58446⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58446⟩⟩) ⟨57951⟩ 154519) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57951⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24974⟩⟩, ⟨.program ⟨257⟩, ⟨56424⟩⟩], [⟨.program ⟨257⟩, ⟨57951⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge154572

namespace LeftMerge154580
def owner : Owner := ⟨.program ⟨257⟩, ⟨56826⟩⟩
def mergeEvent : Nat := 154580
def frameStart : Nat := 154477
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events603.exact154533RawTerms
def rightRaw : List Term := Proof.Events603.exact154576RawTerms
def group : MergeGroup := .operator 154533 154576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 154533) (leftOrdinal := 0)
    (rightResult := 154576) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56824⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge154580

namespace LeftMerge154597
def owner : Owner := ⟨.program ⟨257⟩, ⟨57382⟩⟩
def mergeEvent : Nat := 154597
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }
def rhsRaw : List Term := Proof.Events603.exact154594RawTerms
def group : MergeGroup := .relation 154596
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 154596) (rhsResult := 154594)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 154595 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57379⟩⟩]⟩) (none) 154594) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge154597

namespace LeftMerge154598
def owner : Owner := ⟨.program ⟨257⟩, ⟨57382⟩⟩
def mergeEvent : Nat := 154598
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58446⟩⟩] } }
def rhsRaw : List Term := Proof.Events603.exact154594RawTerms
def group : MergeGroup := .relation 154596
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 154596) (rhsResult := 154594)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 154595 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57379⟩⟩]⟩) (none) 154594) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58446⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58446⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge154598

namespace LeftMerge154599
def owner : Owner := ⟨.program ⟨257⟩, ⟨57382⟩⟩
def mergeEvent : Nat := 154599
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24974⟩⟩, ⟨.program ⟨257⟩, ⟨56424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57951⟩⟩] } }
def rhsRaw : List Term := Proof.Events603.exact154594RawTerms
def group : MergeGroup := .relation 154596
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 154596) (rhsResult := 154594)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 154595 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57379⟩⟩]⟩) (none) 154594) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24974⟩⟩, ⟨.program ⟨257⟩, ⟨56424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57951⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24974⟩⟩, ⟨.program ⟨257⟩, ⟨56424⟩⟩], [⟨.program ⟨257⟩, ⟨57951⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge154599

namespace LeftMerge154600
def owner : Owner := ⟨.program ⟨257⟩, ⟨57382⟩⟩
def mergeEvent : Nat := 154600
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨56824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events603.exact154594RawTerms
def group : MergeGroup := .relation 154596
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 154596) (rhsResult := 154594)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 154595 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57379⟩⟩]⟩) (none) 154594) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨56824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge154600

namespace LeftMerge154605
def owner : Owner := ⟨.program ⟨257⟩, ⟨58448⟩⟩
def mergeEvent : Nat := 154605
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24974⟩⟩, ⟨.program ⟨257⟩, ⟨56424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57951⟩⟩] } }
def leftRaw : List Term := Proof.Events603.exact154601RawTerms
def rightRaw : List Term := Proof.Events603.exact154415RawTerms
def group : MergeGroup := .operator 154601 154415
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 154601) (leftOrdinal := 2)
    (rightResult := 154415) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24974⟩⟩, ⟨.program ⟨257⟩, ⟨56424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57951⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24974⟩⟩, ⟨.program ⟨257⟩, ⟨56424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57951⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24974⟩⟩, ⟨.program ⟨257⟩, ⟨56424⟩⟩], [⟨.program ⟨257⟩, ⟨57951⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge154605

namespace LeftMerge154606
def owner : Owner := ⟨.program ⟨257⟩, ⟨58448⟩⟩
def mergeEvent : Nat := 154606
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58446⟩⟩] } }
def leftRaw : List Term := Proof.Events603.exact154601RawTerms
def rightRaw : List Term := Proof.Events603.exact154415RawTerms
def group : MergeGroup := .operator 154601 154415
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 154601) (leftOrdinal := 1)
    (rightResult := 154415) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58446⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58446⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58446⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge154606

namespace LeftMerge154614
def owner : Owner := ⟨.program ⟨257⟩, ⟨58821⟩⟩
def mergeEvent : Nat := 154614
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58819⟩⟩] } }
def leftRaw : List Term := Proof.Events603.exact154608RawTerms
def rightRaw : List Term := Proof.Events602.exact154331RawTerms
def group : MergeGroup := .operator 154608 154331
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 154608) (leftOrdinal := 0)
    (rightResult := 154331) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58819⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58819⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge154614

namespace LeftMerge154615
def owner : Owner := ⟨.program ⟨257⟩, ⟨58821⟩⟩
def mergeEvent : Nat := 154615
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨56824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58819⟩⟩] } }
def leftRaw : List Term := Proof.Events603.exact154608RawTerms
def rightRaw : List Term := Proof.Events602.exact154331RawTerms
def group : MergeGroup := .operator 154608 154331
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 154608) (leftOrdinal := 1)
    (rightResult := 154331) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨56824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58819⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨56824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58819⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge154615

namespace LeftMerge154617
def owner : Owner := ⟨.program ⟨257⟩, ⟨58821⟩⟩
def mergeEvent : Nat := 154617
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨56824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58094⟩⟩] } }
def rhsRaw : List Term := Proof.Events602.exact154328RawTerms
def group : MergeGroup := .relation 154616
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 154616) (rhsResult := 154328)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨56824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58819⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58819⟩⟩) ⟨58094⟩ 154328) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58094⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨56824⟩⟩], [⟨.program ⟨257⟩, ⟨58094⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge154617

namespace LeftMerge154631
def owner : Owner := ⟨.program ⟨257⟩, ⟨57659⟩⟩
def mergeEvent : Nat := 154631
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57656⟩⟩] } }
def leftRaw : List Term := Proof.Events582.exact149120RawTerms
def rightRaw : List Term := Proof.Events604.exact154625RawTerms
def group : MergeGroup := .operator 149120 154625
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149120) (leftOrdinal := 0)
    (rightResult := 154625) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57656⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57656⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge154631

namespace LeftMerge154752
def owner : Owner := ⟨.program ⟨257⟩, ⟨58316⟩⟩
def mergeEvent : Nat := 154752
def frameStart : Nat := 154686
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events604.exact154748RawTerms
def rightRaw : List Term := Proof.Events604.exact154746RawTerms
def group : MergeGroup := .operator 154748 154746
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 154748) (leftOrdinal := 0)
    (rightResult := 154746) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56824⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge154752

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
