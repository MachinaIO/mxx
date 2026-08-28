import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge195580
def owner : Owner := ⟨.program ⟨257⟩, ⟨35212⟩⟩
def mergeEvent : Nat := 195580
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }
def rhsRaw : List Term := Proof.Events763.exact195577RawTerms
def group : MergeGroup := .relation 195579
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 195579) (rhsResult := 195577)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35209⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 195578 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35209⟩⟩]⟩) (none) 195577) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195580

namespace LeftMerge195581
def owner : Owner := ⟨.program ⟨257⟩, ⟨35212⟩⟩
def mergeEvent : Nat := 195581
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36281⟩⟩] } }
def rhsRaw : List Term := Proof.Events763.exact195577RawTerms
def group : MergeGroup := .relation 195579
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 195579) (rhsResult := 195577)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35209⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 195578 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35209⟩⟩]⟩) (none) 195577) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36281⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36281⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195581

namespace LeftMerge195582
def owner : Owner := ⟨.program ⟨257⟩, ⟨35212⟩⟩
def mergeEvent : Nat := 195582
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13611⟩⟩, ⟨.program ⟨257⟩, ⟨34482⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35761⟩⟩] } }
def rhsRaw : List Term := Proof.Events763.exact195577RawTerms
def group : MergeGroup := .relation 195579
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 195579) (rhsResult := 195577)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35209⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 195578 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35209⟩⟩]⟩) (none) 195577) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13611⟩⟩, ⟨.program ⟨257⟩, ⟨34482⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35761⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13611⟩⟩, ⟨.program ⟨257⟩, ⟨34482⟩⟩], [⟨.program ⟨257⟩, ⟨35761⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195582

namespace LeftMerge195583
def owner : Owner := ⟨.program ⟨257⟩, ⟨35212⟩⟩
def mergeEvent : Nat := 195583
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events763.exact195577RawTerms
def group : MergeGroup := .relation 195579
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 195579) (rhsResult := 195577)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35209⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 195578 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35209⟩⟩]⟩) (none) 195577) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195583

namespace LeftMerge195588
def owner : Owner := ⟨.program ⟨257⟩, ⟨36283⟩⟩
def mergeEvent : Nat := 195588
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13611⟩⟩, ⟨.program ⟨257⟩, ⟨34482⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35761⟩⟩] } }
def leftRaw : List Term := Proof.Events764.exact195584RawTerms
def rightRaw : List Term := Proof.Events763.exact195398RawTerms
def group : MergeGroup := .operator 195584 195398
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195584) (leftOrdinal := 2)
    (rightResult := 195398) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13611⟩⟩, ⟨.program ⟨257⟩, ⟨34482⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35761⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13611⟩⟩, ⟨.program ⟨257⟩, ⟨34482⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35761⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13611⟩⟩, ⟨.program ⟨257⟩, ⟨34482⟩⟩], [⟨.program ⟨257⟩, ⟨35761⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195588

namespace LeftMerge195589
def owner : Owner := ⟨.program ⟨257⟩, ⟨36283⟩⟩
def mergeEvent : Nat := 195589
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36281⟩⟩] } }
def leftRaw : List Term := Proof.Events764.exact195584RawTerms
def rightRaw : List Term := Proof.Events763.exact195398RawTerms
def group : MergeGroup := .operator 195584 195398
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195584) (leftOrdinal := 1)
    (rightResult := 195398) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36281⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36281⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36281⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195589

namespace LeftMerge195597
def owner : Owner := ⟨.program ⟨257⟩, ⟨36681⟩⟩
def mergeEvent : Nat := 195597
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36679⟩⟩] } }
def leftRaw : List Term := Proof.Events764.exact195591RawTerms
def rightRaw : List Term := Proof.Events762.exact195314RawTerms
def group : MergeGroup := .operator 195591 195314
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195591) (leftOrdinal := 0)
    (rightResult := 195314) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36679⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36679⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195597

namespace LeftMerge195598
def owner : Owner := ⟨.program ⟨257⟩, ⟨36681⟩⟩
def mergeEvent : Nat := 195598
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36679⟩⟩] } }
def leftRaw : List Term := Proof.Events764.exact195591RawTerms
def rightRaw : List Term := Proof.Events762.exact195314RawTerms
def group : MergeGroup := .operator 195591 195314
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195591) (leftOrdinal := 1)
    (rightResult := 195314) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36679⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36679⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195598

namespace LeftMerge195600
def owner : Owner := ⟨.program ⟨257⟩, ⟨36681⟩⟩
def mergeEvent : Nat := 195600
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35919⟩⟩] } }
def rhsRaw : List Term := Proof.Events762.exact195311RawTerms
def group : MergeGroup := .relation 195599
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 195599) (rhsResult := 195311)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36679⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36679⟩⟩) ⟨35919⟩ 195311) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35919⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], [⟨.program ⟨257⟩, ⟨35919⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195600

namespace LeftMerge195614
def owner : Owner := ⟨.program ⟨257⟩, ⟨35539⟩⟩
def mergeEvent : Nat := 195614
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35536⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events764.exact195608RawTerms
def group : MergeGroup := .operator 192995 195608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 195608) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35536⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35536⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195614

namespace LeftMerge195735
def owner : Owner := ⟨.program ⟨257⟩, ⟨36116⟩⟩
def mergeEvent : Nat := 195735
def frameStart : Nat := 195669
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events764.exact195731RawTerms
def rightRaw : List Term := Proof.Events764.exact195729RawTerms
def group : MergeGroup := .operator 195731 195729
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195731) (leftOrdinal := 0)
    (rightResult := 195729) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195735

namespace LeftMerge195747
def owner : Owner := ⟨.program ⟨257⟩, ⟨36680⟩⟩
def mergeEvent : Nat := 195747
def frameStart : Nat := 195669
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36679⟩⟩] } }
def leftRaw : List Term := Proof.Events764.exact195743RawTerms
def rightRaw : List Term := Proof.Events764.exact195720RawTerms
def group : MergeGroup := .operator 195743 195720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195743) (leftOrdinal := 0)
    (rightResult := 195720) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36679⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36679⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195747

namespace LeftMerge195748
def owner : Owner := ⟨.program ⟨257⟩, ⟨36680⟩⟩
def mergeEvent : Nat := 195748
def frameStart : Nat := 195669
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36679⟩⟩] } }
def leftRaw : List Term := Proof.Events764.exact195743RawTerms
def rightRaw : List Term := Proof.Events764.exact195720RawTerms
def group : MergeGroup := .operator 195743 195720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195743) (leftOrdinal := 1)
    (rightResult := 195720) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36679⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36679⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195748

namespace LeftMerge195750
def owner : Owner := ⟨.program ⟨257⟩, ⟨36680⟩⟩
def mergeEvent : Nat := 195750
def frameStart : Nat := 195669
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35919⟩⟩] } }
def rhsRaw : List Term := Proof.Events764.exact195717RawTerms
def group : MergeGroup := .relation 195749
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 195749) (rhsResult := 195717)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨34764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36679⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36679⟩⟩) ⟨35919⟩ 195717) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35919⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨34764⟩⟩], [⟨.program ⟨257⟩, ⟨35919⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195750

namespace LeftMerge195758
def owner : Owner := ⟨.program ⟨257⟩, ⟨34990⟩⟩
def mergeEvent : Nat := 195758
def frameStart : Nat := 195669
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34989⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events764.exact195731RawTerms
def rightRaw : List Term := Proof.Events764.exact195754RawTerms
def group : MergeGroup := .operator 195731 195754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195731) (leftOrdinal := 0)
    (rightResult := 195754) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34989⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34989⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195758

namespace LeftMerge195775
def owner : Owner := ⟨.program ⟨257⟩, ⟨35539⟩⟩
def mergeEvent : Nat := 195775
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }
def rhsRaw : List Term := Proof.Events764.exact195772RawTerms
def group : MergeGroup := .relation 195774
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 195774) (rhsResult := 195772)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35536⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 195773 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35536⟩⟩]⟩) (none) 195772) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195775

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
