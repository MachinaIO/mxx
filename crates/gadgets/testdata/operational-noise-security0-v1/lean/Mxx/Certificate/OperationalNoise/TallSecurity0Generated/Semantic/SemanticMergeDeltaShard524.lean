import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge85474
def owner : Owner := ⟨.program ⟨214⟩, ⟨25991⟩⟩
def mergeEvent : Nat := 85474
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25989⟩⟩] } }
def leftRaw : List Term := Proof.Events333.exact85469RawTerms
def rightRaw : List Term := Proof.Events333.exact85285RawTerms
def group : MergeGroup := .operator 85469 85285
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85469) (leftOrdinal := 1)
    (rightResult := 85285) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25989⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25989⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25989⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85474

namespace LeftMerge85482
def owner : Owner := ⟨.program ⟨214⟩, ⟨27651⟩⟩
def mergeEvent : Nat := 85482
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩] } }
def leftRaw : List Term := Proof.Events333.exact85476RawTerms
def rightRaw : List Term := Proof.Events332.exact85201RawTerms
def group : MergeGroup := .operator 85476 85201
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85476) (leftOrdinal := 0)
    (rightResult := 85201) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27649⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85482

namespace LeftMerge85483
def owner : Owner := ⟨.program ⟨214⟩, ⟨27651⟩⟩
def mergeEvent : Nat := 85483
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15821⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩] } }
def leftRaw : List Term := Proof.Events333.exact85476RawTerms
def rightRaw : List Term := Proof.Events332.exact85201RawTerms
def group : MergeGroup := .operator 85476 85201
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85476) (leftOrdinal := 1)
    (rightResult := 85201) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15821⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27649⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15821⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85483

namespace LeftMerge85485
def owner : Owner := ⟨.program ⟨214⟩, ⟨27651⟩⟩
def mergeEvent : Nat := 85485
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15821⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24099⟩⟩] } }
def rhsRaw : List Term := Proof.Events332.exact85198RawTerms
def group : MergeGroup := .relation 85484
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 85484) (rhsResult := 85198)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15821⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27649⟩⟩) ⟨24099⟩ 85198) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24099⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15821⟩⟩], [⟨.program ⟨214⟩, ⟨24099⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85485

namespace LeftMerge85499
def owner : Owner := ⟨.program ⟨214⟩, ⟨21259⟩⟩
def mergeEvent : Nat := 85499
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21256⟩⟩] } }
def leftRaw : List Term := Proof.Events312.exact80012RawTerms
def rightRaw : List Term := Proof.Events333.exact85493RawTerms
def group : MergeGroup := .operator 80012 85493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80012) (leftOrdinal := 0)
    (rightResult := 85493) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21256⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21256⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85499

namespace LeftMerge85620
def owner : Owner := ⟨.program ⟨214⟩, ⟨15898⟩⟩
def mergeEvent : Nat := 85620
def frameStart : Nat := 85554
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15821⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events334.exact85616RawTerms
def rightRaw : List Term := Proof.Events334.exact85614RawTerms
def group : MergeGroup := .operator 85616 85614
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85616) (leftOrdinal := 0)
    (rightResult := 85614) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15821⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15821⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85620

namespace LeftMerge85632
def owner : Owner := ⟨.program ⟨214⟩, ⟨27650⟩⟩
def mergeEvent : Nat := 85632
def frameStart : Nat := 85554
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩] } }
def leftRaw : List Term := Proof.Events334.exact85628RawTerms
def rightRaw : List Term := Proof.Events334.exact85605RawTerms
def group : MergeGroup := .operator 85628 85605
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85628) (leftOrdinal := 0)
    (rightResult := 85605) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27649⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85632

namespace LeftMerge85633
def owner : Owner := ⟨.program ⟨214⟩, ⟨27650⟩⟩
def mergeEvent : Nat := 85633
def frameStart : Nat := 85554
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15821⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩] } }
def leftRaw : List Term := Proof.Events334.exact85628RawTerms
def rightRaw : List Term := Proof.Events334.exact85605RawTerms
def group : MergeGroup := .operator 85628 85605
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85628) (leftOrdinal := 1)
    (rightResult := 85605) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15821⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27649⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15821⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85633

namespace LeftMerge85635
def owner : Owner := ⟨.program ⟨214⟩, ⟨27650⟩⟩
def mergeEvent : Nat := 85635
def frameStart : Nat := 85554
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15821⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24099⟩⟩] } }
def rhsRaw : List Term := Proof.Events334.exact85602RawTerms
def group : MergeGroup := .relation 85634
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 85634) (rhsResult := 85602)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15821⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27649⟩⟩) ⟨24099⟩ 85602) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24099⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15821⟩⟩], [⟨.program ⟨214⟩, ⟨24099⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85635

namespace LeftMerge85643
def owner : Owner := ⟨.program ⟨214⟩, ⟨15868⟩⟩
def mergeEvent : Nat := 85643
def frameStart : Nat := 85554
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events334.exact85616RawTerms
def rightRaw : List Term := Proof.Events334.exact85639RawTerms
def group : MergeGroup := .operator 85616 85639
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85616) (leftOrdinal := 0)
    (rightResult := 85639) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85643

namespace LeftMerge85660
def owner : Owner := ⟨.program ⟨214⟩, ⟨21259⟩⟩
def mergeEvent : Nat := 85660
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }
def rhsRaw : List Term := Proof.Events334.exact85657RawTerms
def group : MergeGroup := .relation 85659
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 85659) (rhsResult := 85657)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21256⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 85658 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21256⟩⟩]⟩) (none) 85657) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85660

namespace LeftMerge85661
def owner : Owner := ⟨.program ⟨214⟩, ⟨21259⟩⟩
def mergeEvent : Nat := 85661
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩] } }
def rhsRaw : List Term := Proof.Events334.exact85657RawTerms
def group : MergeGroup := .relation 85659
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 85659) (rhsResult := 85657)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21256⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 85658 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21256⟩⟩]⟩) (none) 85657) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85661

namespace LeftMerge85662
def owner : Owner := ⟨.program ⟨214⟩, ⟨21259⟩⟩
def mergeEvent : Nat := 85662
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15821⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24099⟩⟩] } }
def rhsRaw : List Term := Proof.Events334.exact85657RawTerms
def group : MergeGroup := .relation 85659
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 85659) (rhsResult := 85657)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21256⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 85658 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21256⟩⟩]⟩) (none) 85657) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15821⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24099⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15821⟩⟩], [⟨.program ⟨214⟩, ⟨24099⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85662

namespace LeftMerge85663
def owner : Owner := ⟨.program ⟨214⟩, ⟨21259⟩⟩
def mergeEvent : Nat := 85663
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events334.exact85657RawTerms
def group : MergeGroup := .relation 85659
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 85659) (rhsResult := 85657)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21256⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 85658 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21256⟩⟩]⟩) (none) 85657) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85663

namespace LeftMerge85668
def owner : Owner := ⟨.program ⟨214⟩, ⟨27652⟩⟩
def mergeEvent : Nat := 85668
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩] } }
def leftRaw : List Term := Proof.Events334.exact85664RawTerms
def rightRaw : List Term := Proof.Events333.exact85486RawTerms
def group : MergeGroup := .operator 85664 85486
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85664) (leftOrdinal := 0)
    (rightResult := 85486) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85668

namespace LeftMerge85669
def owner : Owner := ⟨.program ⟨214⟩, ⟨27652⟩⟩
def mergeEvent : Nat := 85669
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15821⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24099⟩⟩] } }
def leftRaw : List Term := Proof.Events334.exact85664RawTerms
def rightRaw : List Term := Proof.Events333.exact85486RawTerms
def group : MergeGroup := .operator 85664 85486
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85664) (leftOrdinal := 2)
    (rightResult := 85486) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15821⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24099⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15821⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24099⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15821⟩⟩], [⟨.program ⟨214⟩, ⟨24099⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85669

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
