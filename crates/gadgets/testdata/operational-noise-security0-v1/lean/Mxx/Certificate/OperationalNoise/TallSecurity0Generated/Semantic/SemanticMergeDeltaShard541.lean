import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge88647
def owner : Owner := ⟨.program ⟨214⟩, ⟨30121⟩⟩
def mergeEvent : Nat := 88647
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17085⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events346.exact88636RawTerms
def rightRaw : List Term := Proof.Events312.exact79895RawTerms
def group : MergeGroup := .operator 88636 79895
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 88636) (leftOrdinal := 29)
    (rightResult := 79895) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17085⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge88647

namespace LeftMerge88649
def owner : Owner := ⟨.program ⟨214⟩, ⟨30121⟩⟩
def mergeEvent : Nat := 88649
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17085⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events312.exact79892RawTerms
def group : MergeGroup := .relation 88648
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 88648) (rhsResult := 79892)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge88649

namespace LeftMerge88650
def owner : Owner := ⟨.program ⟨214⟩, ⟨30121⟩⟩
def mergeEvent : Nat := 88650
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events346.exact88636RawTerms
def rightRaw : List Term := Proof.Events312.exact79895RawTerms
def group : MergeGroup := .operator 88636 79895
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 88636) (leftOrdinal := 15)
    (rightResult := 79895) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge88650

namespace LeftMerge88651
def owner : Owner := ⟨.program ⟨214⟩, ⟨30121⟩⟩
def mergeEvent : Nat := 88651
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16798⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events346.exact88636RawTerms
def rightRaw : List Term := Proof.Events312.exact79895RawTerms
def group : MergeGroup := .operator 88636 79895
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 88636) (leftOrdinal := 28)
    (rightResult := 79895) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16798⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16798⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge88651

namespace LeftMerge88653
def owner : Owner := ⟨.program ⟨214⟩, ⟨30121⟩⟩
def mergeEvent : Nat := 88653
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16798⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events312.exact79892RawTerms
def group : MergeGroup := .relation 88652
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 88652) (rhsResult := 79892)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16798⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16798⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge88653

namespace LeftMerge88654
def owner : Owner := ⟨.program ⟨214⟩, ⟨30121⟩⟩
def mergeEvent : Nat := 88654
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events346.exact88636RawTerms
def rightRaw : List Term := Proof.Events312.exact79895RawTerms
def group : MergeGroup := .operator 88636 79895
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 88636) (leftOrdinal := 14)
    (rightResult := 79895) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge88654

namespace LeftMerge88655
def owner : Owner := ⟨.program ⟨214⟩, ⟨30121⟩⟩
def mergeEvent : Nat := 88655
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16679⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events346.exact88636RawTerms
def rightRaw : List Term := Proof.Events312.exact79895RawTerms
def group : MergeGroup := .operator 88636 79895
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 88636) (leftOrdinal := 27)
    (rightResult := 79895) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16679⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16679⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge88655

namespace LeftMerge88657
def owner : Owner := ⟨.program ⟨214⟩, ⟨30121⟩⟩
def mergeEvent : Nat := 88657
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16679⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events312.exact79892RawTerms
def group : MergeGroup := .relation 88656
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 88656) (rhsResult := 79892)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16679⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16679⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge88657

namespace LeftMerge88658
def owner : Owner := ⟨.program ⟨214⟩, ⟨30121⟩⟩
def mergeEvent : Nat := 88658
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events346.exact88636RawTerms
def rightRaw : List Term := Proof.Events312.exact79895RawTerms
def group : MergeGroup := .operator 88636 79895
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 88636) (leftOrdinal := 13)
    (rightResult := 79895) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge88658

namespace LeftMerge88659
def owner : Owner := ⟨.program ⟨214⟩, ⟨30121⟩⟩
def mergeEvent : Nat := 88659
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18205⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events346.exact88636RawTerms
def rightRaw : List Term := Proof.Events312.exact79895RawTerms
def group : MergeGroup := .operator 88636 79895
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 88636) (leftOrdinal := 34)
    (rightResult := 79895) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18205⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge88659

namespace LeftMerge88661
def owner : Owner := ⟨.program ⟨214⟩, ⟨30121⟩⟩
def mergeEvent : Nat := 88661
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18205⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events312.exact79892RawTerms
def group : MergeGroup := .relation 88660
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 88660) (rhsResult := 79892)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge88661

namespace LeftMerge88662
def owner : Owner := ⟨.program ⟨214⟩, ⟨30121⟩⟩
def mergeEvent : Nat := 88662
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events346.exact88636RawTerms
def rightRaw : List Term := Proof.Events312.exact79895RawTerms
def group : MergeGroup := .operator 88636 79895
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 88636) (leftOrdinal := 12)
    (rightResult := 79895) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge88662

namespace LeftMerge88663
def owner : Owner := ⟨.program ⟨214⟩, ⟨30121⟩⟩
def mergeEvent : Nat := 88663
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events346.exact88636RawTerms
def rightRaw : List Term := Proof.Events312.exact79895RawTerms
def group : MergeGroup := .operator 88636 79895
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 88636) (leftOrdinal := 32)
    (rightResult := 79895) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge88663

namespace LeftMerge88665
def owner : Owner := ⟨.program ⟨214⟩, ⟨30121⟩⟩
def mergeEvent : Nat := 88665
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events312.exact79892RawTerms
def group : MergeGroup := .relation 88664
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 88664) (rhsResult := 79892)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge88665

namespace LeftMerge88666
def owner : Owner := ⟨.program ⟨214⟩, ⟨30121⟩⟩
def mergeEvent : Nat := 88666
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events346.exact88636RawTerms
def rightRaw : List Term := Proof.Events312.exact79895RawTerms
def group : MergeGroup := .operator 88636 79895
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 88636) (leftOrdinal := 11)
    (rightResult := 79895) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge88666

namespace LeftMerge88667
def owner : Owner := ⟨.program ⟨214⟩, ⟨30121⟩⟩
def mergeEvent : Nat := 88667
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events346.exact88636RawTerms
def rightRaw : List Term := Proof.Events312.exact79895RawTerms
def group : MergeGroup := .operator 88636 79895
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 88636) (leftOrdinal := 30)
    (rightResult := 79895) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge88667

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
