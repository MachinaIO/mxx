import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge185685
def owner : Owner := ⟨.program ⟨257⟩, ⟨21567⟩⟩
def mergeEvent : Nat := 185685
def frameStart : Nat := 185655
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21146⟩⟩, ⟨.program ⟨257⟩, ⟨21566⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events725.exact185681RawTerms
def rightRaw : List Term := Proof.Events725.exact185678RawTerms
def group : MergeGroup := .operator 185681 185678
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185681) (leftOrdinal := 0)
    (rightResult := 185678) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21146⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21566⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21146⟩⟩, ⟨.program ⟨257⟩, ⟨21566⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185685

namespace LeftMerge185715
def owner : Owner := ⟨.program ⟨257⟩, ⟨23220⟩⟩
def mergeEvent : Nat := 185715
def frameStart : Nat := 185655
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21146⟩⟩, ⟨.program ⟨257⟩, ⟨21566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events725.exact185711RawTerms
def rightRaw : List Term := Proof.Events725.exact185709RawTerms
def group : MergeGroup := .operator 185711 185709
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185711) (leftOrdinal := 0)
    (rightResult := 185709) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21146⟩⟩, ⟨.program ⟨257⟩, ⟨21566⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21146⟩⟩, ⟨.program ⟨257⟩, ⟨21566⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185715

namespace LeftMerge185738
def owner : Owner := ⟨.program ⟨257⟩, ⟨9576⟩⟩
def mergeEvent : Nat := 185738
def frameStart : Nat := 185655
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }
def leftRaw : List Term := Proof.Events725.exact185734RawTerms
def rightRaw : List Term := Proof.Events725.exact185731RawTerms
def group : MergeGroup := .operator 185734 185731
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185734) (leftOrdinal := 0)
    (rightResult := 185731) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9574⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185738

namespace LeftMerge185747
def owner : Owner := ⟨.program ⟨257⟩, ⟨23475⟩⟩
def mergeEvent : Nat := 185747
def frameStart : Nat := 185655
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23472⟩⟩] } }
def leftRaw : List Term := Proof.Events725.exact185743RawTerms
def rightRaw : List Term := Proof.Events725.exact185700RawTerms
def group : MergeGroup := .operator 185743 185700
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185743) (leftOrdinal := 0)
    (rightResult := 185700) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23472⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23472⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185747

namespace LeftMerge185748
def owner : Owner := ⟨.program ⟨257⟩, ⟨23475⟩⟩
def mergeEvent : Nat := 185748
def frameStart : Nat := 185655
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21146⟩⟩, ⟨.program ⟨257⟩, ⟨21566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23472⟩⟩] } }
def leftRaw : List Term := Proof.Events725.exact185743RawTerms
def rightRaw : List Term := Proof.Events725.exact185700RawTerms
def group : MergeGroup := .operator 185743 185700
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185743) (leftOrdinal := 1)
    (rightResult := 185700) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21146⟩⟩, ⟨.program ⟨257⟩, ⟨21566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23472⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21146⟩⟩, ⟨.program ⟨257⟩, ⟨21566⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23472⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185748

namespace LeftMerge185750
def owner : Owner := ⟨.program ⟨257⟩, ⟨23475⟩⟩
def mergeEvent : Nat := 185750
def frameStart : Nat := 185655
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21146⟩⟩, ⟨.program ⟨257⟩, ⟨21566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22947⟩⟩] } }
def rhsRaw : List Term := Proof.Events725.exact185697RawTerms
def group : MergeGroup := .relation 185749
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185749) (rhsResult := 185697)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21146⟩⟩, ⟨.program ⟨257⟩, ⟨21566⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23472⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23472⟩⟩) ⟨22947⟩ 185697) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22947⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21146⟩⟩, ⟨.program ⟨257⟩, ⟨21566⟩⟩], [⟨.program ⟨257⟩, ⟨22947⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185750

namespace LeftMerge185758
def owner : Owner := ⟨.program ⟨257⟩, ⟨21834⟩⟩
def mergeEvent : Nat := 185758
def frameStart : Nat := 185655
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events725.exact185711RawTerms
def rightRaw : List Term := Proof.Events725.exact185754RawTerms
def group : MergeGroup := .operator 185711 185754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185711) (leftOrdinal := 0)
    (rightResult := 185754) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21832⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185758

namespace LeftMerge185775
def owner : Owner := ⟨.program ⟨257⟩, ⟨22402⟩⟩
def mergeEvent : Nat := 185775
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }
def rhsRaw : List Term := Proof.Events725.exact185772RawTerms
def group : MergeGroup := .relation 185774
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185774) (rhsResult := 185772)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22399⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 185773 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22399⟩⟩]⟩) (none) 185772) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185775

namespace LeftMerge185776
def owner : Owner := ⟨.program ⟨257⟩, ⟨22402⟩⟩
def mergeEvent : Nat := 185776
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23472⟩⟩] } }
def rhsRaw : List Term := Proof.Events725.exact185772RawTerms
def group : MergeGroup := .relation 185774
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185774) (rhsResult := 185772)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22399⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 185773 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22399⟩⟩]⟩) (none) 185772) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23472⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23472⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185776

namespace LeftMerge185777
def owner : Owner := ⟨.program ⟨257⟩, ⟨22402⟩⟩
def mergeEvent : Nat := 185777
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨21146⟩⟩, ⟨.program ⟨257⟩, ⟨21566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22947⟩⟩] } }
def rhsRaw : List Term := Proof.Events725.exact185772RawTerms
def group : MergeGroup := .relation 185774
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185774) (rhsResult := 185772)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22399⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 185773 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22399⟩⟩]⟩) (none) 185772) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21146⟩⟩, ⟨.program ⟨257⟩, ⟨21566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22947⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨21146⟩⟩, ⟨.program ⟨257⟩, ⟨21566⟩⟩], [⟨.program ⟨257⟩, ⟨22947⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185777

namespace LeftMerge185778
def owner : Owner := ⟨.program ⟨257⟩, ⟨22402⟩⟩
def mergeEvent : Nat := 185778
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨21832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events725.exact185772RawTerms
def group : MergeGroup := .relation 185774
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185774) (rhsResult := 185772)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22399⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 185773 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22399⟩⟩]⟩) (none) 185772) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨21832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185778

namespace LeftMerge185783
def owner : Owner := ⟨.program ⟨257⟩, ⟨23474⟩⟩
def mergeEvent : Nat := 185783
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨21146⟩⟩, ⟨.program ⟨257⟩, ⟨21566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22947⟩⟩] } }
def leftRaw : List Term := Proof.Events725.exact185779RawTerms
def rightRaw : List Term := Proof.Events724.exact185593RawTerms
def group : MergeGroup := .operator 185779 185593
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185779) (leftOrdinal := 2)
    (rightResult := 185593) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨21146⟩⟩, ⟨.program ⟨257⟩, ⟨21566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22947⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨21146⟩⟩, ⟨.program ⟨257⟩, ⟨21566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22947⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨21146⟩⟩, ⟨.program ⟨257⟩, ⟨21566⟩⟩], [⟨.program ⟨257⟩, ⟨22947⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185783

namespace LeftMerge185784
def owner : Owner := ⟨.program ⟨257⟩, ⟨23474⟩⟩
def mergeEvent : Nat := 185784
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23472⟩⟩] } }
def leftRaw : List Term := Proof.Events725.exact185779RawTerms
def rightRaw : List Term := Proof.Events724.exact185593RawTerms
def group : MergeGroup := .operator 185779 185593
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185779) (leftOrdinal := 1)
    (rightResult := 185593) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23472⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23472⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23472⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185784

namespace LeftMerge185792
def owner : Owner := ⟨.program ⟨257⟩, ⟨23967⟩⟩
def mergeEvent : Nat := 185792
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23965⟩⟩] } }
def leftRaw : List Term := Proof.Events725.exact185786RawTerms
def rightRaw : List Term := Proof.Events724.exact185509RawTerms
def group : MergeGroup := .operator 185786 185509
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185786) (leftOrdinal := 0)
    (rightResult := 185509) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23965⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23965⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185792

namespace LeftMerge185793
def owner : Owner := ⟨.program ⟨257⟩, ⟨23967⟩⟩
def mergeEvent : Nat := 185793
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨21832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23965⟩⟩] } }
def leftRaw : List Term := Proof.Events725.exact185786RawTerms
def rightRaw : List Term := Proof.Events724.exact185509RawTerms
def group : MergeGroup := .operator 185786 185509
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185786) (leftOrdinal := 1)
    (rightResult := 185509) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨21832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23965⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨21832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23965⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185793

namespace LeftMerge185795
def owner : Owner := ⟨.program ⟨257⟩, ⟨23967⟩⟩
def mergeEvent : Nat := 185795
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨21832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23108⟩⟩] } }
def rhsRaw : List Term := Proof.Events724.exact185506RawTerms
def group : MergeGroup := .relation 185794
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185794) (rhsResult := 185506)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨21832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23965⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23965⟩⟩) ⟨23108⟩ 185506) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23108⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨21832⟩⟩], [⟨.program ⟨257⟩, ⟨23108⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185795

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
