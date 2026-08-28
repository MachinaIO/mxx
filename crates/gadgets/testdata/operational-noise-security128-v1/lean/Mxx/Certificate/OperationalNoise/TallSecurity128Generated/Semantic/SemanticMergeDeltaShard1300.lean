import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge211465
def owner : Owner := ⟨.program ⟨257⟩, ⟨69241⟩⟩
def mergeEvent : Nat := 211465
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69240⟩⟩] } }
def leftRaw : List Term := Proof.Events826.exact211459RawTerms
def rightRaw : List Term := Proof.Events825.exact211395RawTerms
def group : MergeGroup := .operator 211459 211395
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 211459) (leftOrdinal := 1)
    (rightResult := 211395) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69240⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69240⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge211465

namespace LeftMerge211467
def owner : Owner := ⟨.program ⟨257⟩, ⟨69241⟩⟩
def mergeEvent : Nat := 211467
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68530⟩⟩] } }
def rhsRaw : List Term := Proof.Events825.exact211392RawTerms
def group : MergeGroup := .relation 211466
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 211466) (rhsResult := 211392)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69240⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69240⟩⟩) ⟨68530⟩ 211392) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68530⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], [⟨.program ⟨257⟩, ⟨68530⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge211467

namespace LeftMerge211468
def owner : Owner := ⟨.program ⟨257⟩, ⟨69241⟩⟩
def mergeEvent : Nat := 211468
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69240⟩⟩] } }
def leftRaw : List Term := Proof.Events826.exact211459RawTerms
def rightRaw : List Term := Proof.Events825.exact211395RawTerms
def group : MergeGroup := .operator 211459 211395
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 211459) (leftOrdinal := 0)
    (rightResult := 211395) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69240⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69240⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge211468

namespace LeftMerge211482
def owner : Owner := ⟨.program ⟨257⟩, ⟨67773⟩⟩
def mergeEvent : Nat := 211482
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67770⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207620RawTerms
def rightRaw : List Term := Proof.Events826.exact211476RawTerms
def group : MergeGroup := .operator 207620 211476
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207620) (leftOrdinal := 0)
    (rightResult := 211476) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨67770⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67770⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge211482

namespace LeftMerge211561
def owner : Owner := ⟨.program ⟨257⟩, ⟨65446⟩⟩
def mergeEvent : Nat := 211561
def frameStart : Nat := 211531
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events826.exact211557RawTerms
def rightRaw : List Term := Proof.Events826.exact211554RawTerms
def group : MergeGroup := .operator 211557 211554
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 211557) (leftOrdinal := 0)
    (rightResult := 211554) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65445⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25730⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge211561

namespace LeftMerge211591
def owner : Owner := ⟨.program ⟨257⟩, ⟨68929⟩⟩
def mergeEvent : Nat := 211591
def frameStart : Nat := 211531
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events826.exact211587RawTerms
def rightRaw : List Term := Proof.Events826.exact211585RawTerms
def group : MergeGroup := .operator 211587 211585
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 211587) (leftOrdinal := 0)
    (rightResult := 211585) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge211591

namespace LeftMerge211614
def owner : Owner := ⟨.program ⟨257⟩, ⟨9543⟩⟩
def mergeEvent : Nat := 211614
def frameStart : Nat := 211531
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }
def leftRaw : List Term := Proof.Events826.exact211610RawTerms
def rightRaw : List Term := Proof.Events826.exact211607RawTerms
def group : MergeGroup := .operator 211610 211607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 211610) (leftOrdinal := 0)
    (rightResult := 211607) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9541⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge211614

namespace LeftMerge211623
def owner : Owner := ⟨.program ⟨257⟩, ⟨69243⟩⟩
def mergeEvent : Nat := 211623
def frameStart : Nat := 211531
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69240⟩⟩] } }
def leftRaw : List Term := Proof.Events826.exact211619RawTerms
def rightRaw : List Term := Proof.Events826.exact211576RawTerms
def group : MergeGroup := .operator 211619 211576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 211619) (leftOrdinal := 0)
    (rightResult := 211576) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69240⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69240⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge211623

namespace LeftMerge211624
def owner : Owner := ⟨.program ⟨257⟩, ⟨69243⟩⟩
def mergeEvent : Nat := 211624
def frameStart : Nat := 211531
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69240⟩⟩] } }
def leftRaw : List Term := Proof.Events826.exact211619RawTerms
def rightRaw : List Term := Proof.Events826.exact211576RawTerms
def group : MergeGroup := .operator 211619 211576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 211619) (leftOrdinal := 1)
    (rightResult := 211576) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69240⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69240⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge211624

namespace LeftMerge211626
def owner : Owner := ⟨.program ⟨257⟩, ⟨69243⟩⟩
def mergeEvent : Nat := 211626
def frameStart : Nat := 211531
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68530⟩⟩] } }
def rhsRaw : List Term := Proof.Events826.exact211573RawTerms
def group : MergeGroup := .relation 211625
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 211625) (rhsResult := 211573)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69240⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69240⟩⟩) ⟨68530⟩ 211573) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68530⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], [⟨.program ⟨257⟩, ⟨68530⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge211626

namespace LeftMerge211634
def owner : Owner := ⟨.program ⟨257⟩, ⟨65790⟩⟩
def mergeEvent : Nat := 211634
def frameStart : Nat := 211531
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events826.exact211587RawTerms
def rightRaw : List Term := Proof.Events826.exact211630RawTerms
def group : MergeGroup := .operator 211587 211630
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 211587) (leftOrdinal := 0)
    (rightResult := 211630) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65788⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge211634

namespace LeftMerge211651
def owner : Owner := ⟨.program ⟨257⟩, ⟨67773⟩⟩
def mergeEvent : Nat := 211651
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }
def rhsRaw : List Term := Proof.Events826.exact211648RawTerms
def group : MergeGroup := .relation 211650
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 211650) (rhsResult := 211648)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67770⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 211649 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67770⟩⟩]⟩) (none) 211648) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge211651

namespace LeftMerge211652
def owner : Owner := ⟨.program ⟨257⟩, ⟨67773⟩⟩
def mergeEvent : Nat := 211652
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69240⟩⟩] } }
def rhsRaw : List Term := Proof.Events826.exact211648RawTerms
def group : MergeGroup := .relation 211650
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 211650) (rhsResult := 211648)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67770⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 211649 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67770⟩⟩]⟩) (none) 211648) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69240⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69240⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge211652

namespace LeftMerge211653
def owner : Owner := ⟨.program ⟨257⟩, ⟨67773⟩⟩
def mergeEvent : Nat := 211653
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68530⟩⟩] } }
def rhsRaw : List Term := Proof.Events826.exact211648RawTerms
def group : MergeGroup := .relation 211650
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 211650) (rhsResult := 211648)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67770⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 211649 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67770⟩⟩]⟩) (none) 211648) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68530⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], [⟨.program ⟨257⟩, ⟨68530⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge211653

namespace LeftMerge211654
def owner : Owner := ⟨.program ⟨257⟩, ⟨67773⟩⟩
def mergeEvent : Nat := 211654
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨65788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events826.exact211648RawTerms
def group : MergeGroup := .relation 211650
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 211650) (rhsResult := 211648)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67770⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 211649 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67770⟩⟩]⟩) (none) 211648) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨65788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge211654

namespace LeftMerge211659
def owner : Owner := ⟨.program ⟨257⟩, ⟨69242⟩⟩
def mergeEvent : Nat := 211659
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68530⟩⟩] } }
def leftRaw : List Term := Proof.Events826.exact211655RawTerms
def rightRaw : List Term := Proof.Events826.exact211469RawTerms
def group : MergeGroup := .operator 211655 211469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 211655) (leftOrdinal := 2)
    (rightResult := 211469) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68530⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68530⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25730⟩⟩, ⟨.program ⟨257⟩, ⟨65445⟩⟩], [⟨.program ⟨257⟩, ⟨68530⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge211659

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
