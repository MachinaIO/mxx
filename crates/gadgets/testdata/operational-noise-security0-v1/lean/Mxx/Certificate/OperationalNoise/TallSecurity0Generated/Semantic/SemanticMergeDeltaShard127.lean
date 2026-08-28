import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge21505
def owner : Owner := ⟨.program ⟨214⟩, ⟨5558⟩⟩
def mergeEvent : Nat := 21505
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21290RawTerms
def rightRaw : List Term := Proof.Events025.exact6550RawTerms
def group : MergeGroup := .operator 21290 6550
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21290) (leftOrdinal := 0)
    (rightResult := 6550) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21505

namespace LeftMerge21518
def owner : Owner := ⟨.program ⟨214⟩, ⟨20263⟩⟩
def mergeEvent : Nat := 21518
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20260⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21512RawTerms
def rightRaw : List Term := Proof.Events083.exact21501RawTerms
def group : MergeGroup := .operator 21512 21501
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21512) (leftOrdinal := 0)
    (rightResult := 21501) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20260⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20260⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21518

namespace LeftMerge21597
def owner : Owner := ⟨.program ⟨214⟩, ⟨13375⟩⟩
def mergeEvent : Nat := 21597
def frameStart : Nat := 21567
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events084.exact21593RawTerms
def rightRaw : List Term := Proof.Events084.exact21590RawTerms
def group : MergeGroup := .operator 21593 21590
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21593) (leftOrdinal := 0)
    (rightResult := 21590) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10360⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13374⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21597

namespace LeftMerge21627
def owner : Owner := ⟨.program ⟨214⟩, ⟨13460⟩⟩
def mergeEvent : Nat := 21627
def frameStart : Nat := 21567
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21623RawTerms
def rightRaw : List Term := Proof.Events084.exact21621RawTerms
def group : MergeGroup := .operator 21623 21621
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21623) (leftOrdinal := 0)
    (rightResult := 21621) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21627

namespace LeftMerge21650
def owner : Owner := ⟨.program ⟨214⟩, ⟨7884⟩⟩
def mergeEvent : Nat := 21650
def frameStart : Nat := 21567
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21646RawTerms
def rightRaw : List Term := Proof.Events084.exact21643RawTerms
def group : MergeGroup := .operator 21646 21643
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21646) (leftOrdinal := 0)
    (rightResult := 21643) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7882⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21650

namespace LeftMerge21659
def owner : Owner := ⟨.program ⟨214⟩, ⟨25776⟩⟩
def mergeEvent : Nat := 21659
def frameStart : Nat := 21567
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21655RawTerms
def rightRaw : List Term := Proof.Events084.exact21612RawTerms
def group : MergeGroup := .operator 21655 21612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21655) (leftOrdinal := 0)
    (rightResult := 21612) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25773⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21659

namespace LeftMerge21660
def owner : Owner := ⟨.program ⟨214⟩, ⟨25776⟩⟩
def mergeEvent : Nat := 21660
def frameStart : Nat := 21567
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21655RawTerms
def rightRaw : List Term := Proof.Events084.exact21612RawTerms
def group : MergeGroup := .operator 21655 21612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21655) (leftOrdinal := 1)
    (rightResult := 21612) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25773⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21660

namespace LeftMerge21662
def owner : Owner := ⟨.program ⟨214⟩, ⟨25776⟩⟩
def mergeEvent : Nat := 21662
def frameStart : Nat := 21567
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23422⟩⟩] } }
def rhsRaw : List Term := Proof.Events084.exact21609RawTerms
def group : MergeGroup := .relation 21661
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 21661) (rhsResult := 21609)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25773⟩⟩) ⟨23422⟩ 21609) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23422⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], [⟨.program ⟨214⟩, ⟨23422⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21662

namespace LeftMerge21670
def owner : Owner := ⟨.program ⟨214⟩, ⟨17025⟩⟩
def mergeEvent : Nat := 21670
def frameStart : Nat := 21567
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17023⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21623RawTerms
def rightRaw : List Term := Proof.Events084.exact21666RawTerms
def group : MergeGroup := .operator 21623 21666
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21623) (leftOrdinal := 0)
    (rightResult := 21666) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17023⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17023⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21670

namespace LeftMerge21687
def owner : Owner := ⟨.program ⟨214⟩, ⟨20263⟩⟩
def mergeEvent : Nat := 21687
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }
def rhsRaw : List Term := Proof.Events084.exact21684RawTerms
def group : MergeGroup := .relation 21686
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 21686) (rhsResult := 21684)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20260⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 21685 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20260⟩⟩]⟩) (none) 21684) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21687

namespace LeftMerge21688
def owner : Owner := ⟨.program ⟨214⟩, ⟨20263⟩⟩
def mergeEvent : Nat := 21688
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩] } }
def rhsRaw : List Term := Proof.Events084.exact21684RawTerms
def group : MergeGroup := .relation 21686
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 21686) (rhsResult := 21684)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20260⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 21685 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20260⟩⟩]⟩) (none) 21684) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21688

namespace LeftMerge21689
def owner : Owner := ⟨.program ⟨214⟩, ⟨20263⟩⟩
def mergeEvent : Nat := 21689
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23422⟩⟩] } }
def rhsRaw : List Term := Proof.Events084.exact21684RawTerms
def group : MergeGroup := .relation 21686
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 21686) (rhsResult := 21684)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20260⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 21685 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20260⟩⟩]⟩) (none) 21684) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23422⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], [⟨.program ⟨214⟩, ⟨23422⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21689

namespace LeftMerge21690
def owner : Owner := ⟨.program ⟨214⟩, ⟨20263⟩⟩
def mergeEvent : Nat := 21690
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17023⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events084.exact21684RawTerms
def group : MergeGroup := .relation 21686
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 21686) (rhsResult := 21684)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20260⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 21685 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20260⟩⟩]⟩) (none) 21684) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17023⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17023⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21690

namespace LeftMerge21695
def owner : Owner := ⟨.program ⟨214⟩, ⟨25775⟩⟩
def mergeEvent : Nat := 21695
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23422⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21691RawTerms
def rightRaw : List Term := Proof.Events083.exact21494RawTerms
def group : MergeGroup := .operator 21691 21494
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21691) (leftOrdinal := 2)
    (rightResult := 21494) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23422⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23422⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], [⟨.program ⟨214⟩, ⟨23422⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21695

namespace LeftMerge21696
def owner : Owner := ⟨.program ⟨214⟩, ⟨25775⟩⟩
def mergeEvent : Nat := 21696
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21691RawTerms
def rightRaw : List Term := Proof.Events083.exact21494RawTerms
def group : MergeGroup := .operator 21691 21494
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21691) (leftOrdinal := 1)
    (rightResult := 21494) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21696

namespace LeftMerge21704
def owner : Owner := ⟨.program ⟨214⟩, ⟨30185⟩⟩
def mergeEvent : Nat := 21704
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30183⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21698RawTerms
def rightRaw : List Term := Proof.Events083.exact21405RawTerms
def group : MergeGroup := .operator 21698 21405
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21698) (leftOrdinal := 0)
    (rightResult := 21405) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30183⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30183⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21704

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
