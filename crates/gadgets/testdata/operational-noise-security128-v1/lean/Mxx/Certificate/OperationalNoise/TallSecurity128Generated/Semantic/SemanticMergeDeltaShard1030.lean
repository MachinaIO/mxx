import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge168571
def owner : Owner := ⟨.program ⟨257⟩, ⟨60432⟩⟩
def mergeEvent : Nat := 168571
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60429⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events658.exact168565RawTerms
def group : MergeGroup := .operator 163745 168565
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 168565) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60429⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60429⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge168571

namespace LeftMerge168650
def owner : Owner := ⟨.program ⟨257⟩, ⟨59594⟩⟩
def mergeEvent : Nat := 168650
def frameStart : Nat := 168620
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25298⟩⟩, ⟨.program ⟨257⟩, ⟨59593⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events658.exact168646RawTerms
def rightRaw : List Term := Proof.Events658.exact168643RawTerms
def group : MergeGroup := .operator 168646 168643
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 168646) (leftOrdinal := 0)
    (rightResult := 168643) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59593⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25298⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25298⟩⟩, ⟨.program ⟨257⟩, ⟨59593⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge168650

namespace LeftMerge168680
def owner : Owner := ⟨.program ⟨257⟩, ⟨61244⟩⟩
def mergeEvent : Nat := 168680
def frameStart : Nat := 168620
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25298⟩⟩, ⟨.program ⟨257⟩, ⟨59593⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events658.exact168676RawTerms
def rightRaw : List Term := Proof.Events658.exact168674RawTerms
def group : MergeGroup := .operator 168676 168674
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 168676) (leftOrdinal := 0)
    (rightResult := 168674) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25298⟩⟩, ⟨.program ⟨257⟩, ⟨59593⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25298⟩⟩, ⟨.program ⟨257⟩, ⟨59593⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge168680

namespace LeftMerge168703
def owner : Owner := ⟨.program ⟨257⟩, ⟨9537⟩⟩
def mergeEvent : Nat := 168703
def frameStart : Nat := 168620
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }
def leftRaw : List Term := Proof.Events658.exact168699RawTerms
def rightRaw : List Term := Proof.Events658.exact168696RawTerms
def group : MergeGroup := .operator 168699 168696
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 168699) (leftOrdinal := 0)
    (rightResult := 168696) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9535⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge168703

namespace LeftMerge168712
def owner : Owner := ⟨.program ⟨257⟩, ⟨61506⟩⟩
def mergeEvent : Nat := 168712
def frameStart : Nat := 168620
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61503⟩⟩] } }
def leftRaw : List Term := Proof.Events659.exact168708RawTerms
def rightRaw : List Term := Proof.Events658.exact168665RawTerms
def group : MergeGroup := .operator 168708 168665
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 168708) (leftOrdinal := 0)
    (rightResult := 168665) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61503⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61503⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge168712

namespace LeftMerge168713
def owner : Owner := ⟨.program ⟨257⟩, ⟨61506⟩⟩
def mergeEvent : Nat := 168713
def frameStart : Nat := 168620
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25298⟩⟩, ⟨.program ⟨257⟩, ⟨59593⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61503⟩⟩] } }
def leftRaw : List Term := Proof.Events659.exact168708RawTerms
def rightRaw : List Term := Proof.Events658.exact168665RawTerms
def group : MergeGroup := .operator 168708 168665
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 168708) (leftOrdinal := 1)
    (rightResult := 168665) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25298⟩⟩, ⟨.program ⟨257⟩, ⟨59593⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61503⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25298⟩⟩, ⟨.program ⟨257⟩, ⟨59593⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61503⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge168713

namespace LeftMerge168715
def owner : Owner := ⟨.program ⟨257⟩, ⟨61506⟩⟩
def mergeEvent : Nat := 168715
def frameStart : Nat := 168620
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25298⟩⟩, ⟨.program ⟨257⟩, ⟨59593⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60973⟩⟩] } }
def rhsRaw : List Term := Proof.Events658.exact168662RawTerms
def group : MergeGroup := .relation 168714
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 168714) (rhsResult := 168662)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25298⟩⟩, ⟨.program ⟨257⟩, ⟨59593⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61503⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61503⟩⟩) ⟨60973⟩ 168662) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60973⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25298⟩⟩, ⟨.program ⟨257⟩, ⟨59593⟩⟩], [⟨.program ⟨257⟩, ⟨60973⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge168715

namespace LeftMerge168723
def owner : Owner := ⟨.program ⟨257⟩, ⟨59862⟩⟩
def mergeEvent : Nat := 168723
def frameStart : Nat := 168620
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events658.exact168676RawTerms
def rightRaw : List Term := Proof.Events659.exact168719RawTerms
def group : MergeGroup := .operator 168676 168719
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 168676) (leftOrdinal := 0)
    (rightResult := 168719) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59860⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge168723

namespace LeftMerge168740
def owner : Owner := ⟨.program ⟨257⟩, ⟨60432⟩⟩
def mergeEvent : Nat := 168740
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }
def rhsRaw : List Term := Proof.Events659.exact168737RawTerms
def group : MergeGroup := .relation 168739
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 168739) (rhsResult := 168737)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 168738 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60429⟩⟩]⟩) (none) 168737) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge168740

namespace LeftMerge168741
def owner : Owner := ⟨.program ⟨257⟩, ⟨60432⟩⟩
def mergeEvent : Nat := 168741
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61503⟩⟩] } }
def rhsRaw : List Term := Proof.Events659.exact168737RawTerms
def group : MergeGroup := .relation 168739
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 168739) (rhsResult := 168737)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 168738 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60429⟩⟩]⟩) (none) 168737) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61503⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61503⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge168741

namespace LeftMerge168742
def owner : Owner := ⟨.program ⟨257⟩, ⟨60432⟩⟩
def mergeEvent : Nat := 168742
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25298⟩⟩, ⟨.program ⟨257⟩, ⟨59593⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60973⟩⟩] } }
def rhsRaw : List Term := Proof.Events659.exact168737RawTerms
def group : MergeGroup := .relation 168739
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 168739) (rhsResult := 168737)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 168738 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60429⟩⟩]⟩) (none) 168737) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25298⟩⟩, ⟨.program ⟨257⟩, ⟨59593⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60973⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25298⟩⟩, ⟨.program ⟨257⟩, ⟨59593⟩⟩], [⟨.program ⟨257⟩, ⟨60973⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge168742

namespace LeftMerge168743
def owner : Owner := ⟨.program ⟨257⟩, ⟨60432⟩⟩
def mergeEvent : Nat := 168743
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨59860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events659.exact168737RawTerms
def group : MergeGroup := .relation 168739
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 168739) (rhsResult := 168737)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 168738 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60429⟩⟩]⟩) (none) 168737) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨59860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge168743

namespace LeftMerge168748
def owner : Owner := ⟨.program ⟨257⟩, ⟨61505⟩⟩
def mergeEvent : Nat := 168748
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25298⟩⟩, ⟨.program ⟨257⟩, ⟨59593⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60973⟩⟩] } }
def leftRaw : List Term := Proof.Events659.exact168744RawTerms
def rightRaw : List Term := Proof.Events658.exact168558RawTerms
def group : MergeGroup := .operator 168744 168558
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 168744) (leftOrdinal := 2)
    (rightResult := 168558) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25298⟩⟩, ⟨.program ⟨257⟩, ⟨59593⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60973⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25298⟩⟩, ⟨.program ⟨257⟩, ⟨59593⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60973⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25298⟩⟩, ⟨.program ⟨257⟩, ⟨59593⟩⟩], [⟨.program ⟨257⟩, ⟨60973⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge168748

namespace LeftMerge168749
def owner : Owner := ⟨.program ⟨257⟩, ⟨61505⟩⟩
def mergeEvent : Nat := 168749
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61503⟩⟩] } }
def leftRaw : List Term := Proof.Events659.exact168744RawTerms
def rightRaw : List Term := Proof.Events658.exact168558RawTerms
def group : MergeGroup := .operator 168744 168558
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 168744) (leftOrdinal := 1)
    (rightResult := 168558) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61503⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61503⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61503⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge168749

namespace LeftMerge168757
def owner : Owner := ⟨.program ⟨257⟩, ⟨62018⟩⟩
def mergeEvent : Nat := 168757
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62016⟩⟩] } }
def leftRaw : List Term := Proof.Events659.exact168751RawTerms
def rightRaw : List Term := Proof.Events658.exact168474RawTerms
def group : MergeGroup := .operator 168751 168474
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 168751) (leftOrdinal := 0)
    (rightResult := 168474) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨62016⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62016⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge168757

namespace LeftMerge168758
def owner : Owner := ⟨.program ⟨257⟩, ⟨62018⟩⟩
def mergeEvent : Nat := 168758
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨59860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62016⟩⟩] } }
def leftRaw : List Term := Proof.Events659.exact168751RawTerms
def rightRaw : List Term := Proof.Events658.exact168474RawTerms
def group : MergeGroup := .operator 168751 168474
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 168751) (leftOrdinal := 1)
    (rightResult := 168474) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨59860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨62016⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨59860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62016⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge168758

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
