import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge124641
def owner : Owner := ⟨.program ⟨257⟩, ⟨59381⟩⟩
def mergeEvent : Nat := 124641
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events021.exact5563RawTerms
def rightRaw : List Term := Proof.Events467.exact119778RawTerms
def group : MergeGroup := .operator 5563 119778
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5563) (leftOrdinal := 0)
    (rightResult := 119778) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124641

namespace LeftMerge124646
def owner : Owner := ⟨.program ⟨257⟩, ⟨8141⟩⟩
def mergeEvent : Nat := 124646
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }
def leftRaw : List Term := Proof.Events467.exact119648RawTerms
def rightRaw : List Term := Proof.Events086.exact22131RawTerms
def group : MergeGroup := .operator 119648 22131
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119648) (leftOrdinal := 0)
    (rightResult := 22131) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124646

namespace LeftMerge124663
def owner : Owner := ⟨.program ⟨257⟩, ⟨59384⟩⟩
def mergeEvent : Nat := 124663
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }
def leftRaw : List Term := Proof.Events486.exact124657RawTerms
def rightRaw : List Term := Proof.Events086.exact22120RawTerms
def group : MergeGroup := .operator 124657 22120
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124657) (leftOrdinal := 1)
    (rightResult := 22120) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9535⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124663

namespace LeftMerge124665
def owner : Owner := ⟨.program ⟨257⟩, ⟨59384⟩⟩
def mergeEvent : Nat := 124665
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }
def rhsRaw : List Term := Proof.Events086.exact22090RawTerms
def group : MergeGroup := .relation 124664
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 124664) (rhsResult := 22090)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9535⟩⟩) ⟨7274⟩ 22090) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124665

namespace LeftMerge124666
def owner : Owner := ⟨.program ⟨257⟩, ⟨59384⟩⟩
def mergeEvent : Nat := 124666
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }
def leftRaw : List Term := Proof.Events486.exact124657RawTerms
def rightRaw : List Term := Proof.Events086.exact22120RawTerms
def group : MergeGroup := .operator 124657 22120
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124657) (leftOrdinal := 0)
    (rightResult := 22120) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9535⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124666

namespace LeftMerge124671
def owner : Owner := ⟨.program ⟨257⟩, ⟨59385⟩⟩
def mergeEvent : Nat := 124671
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }
def leftRaw : List Term := Proof.Events486.exact124667RawTerms
def rightRaw : List Term := Proof.Events486.exact124637RawTerms
def group : MergeGroup := .operator 124667 124637
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124667) (leftOrdinal := 1)
    (rightResult := 124637) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124671

namespace LeftMerge124679
def owner : Owner := ⟨.program ⟨257⟩, ⟨61416⟩⟩
def mergeEvent : Nat := 124679
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61415⟩⟩] } }
def leftRaw : List Term := Proof.Events487.exact124673RawTerms
def rightRaw : List Term := Proof.Events486.exact124609RawTerms
def group : MergeGroup := .operator 124673 124609
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124673) (leftOrdinal := 1)
    (rightResult := 124609) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61415⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61415⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124679

namespace LeftMerge124681
def owner : Owner := ⟨.program ⟨257⟩, ⟨61416⟩⟩
def mergeEvent : Nat := 124681
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60925⟩⟩] } }
def rhsRaw : List Term := Proof.Events486.exact124606RawTerms
def group : MergeGroup := .relation 124680
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 124680) (rhsResult := 124606)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61415⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61415⟩⟩) ⟨60925⟩ 124606) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60925⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], [⟨.program ⟨257⟩, ⟨60925⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124681

namespace LeftMerge124682
def owner : Owner := ⟨.program ⟨257⟩, ⟨61416⟩⟩
def mergeEvent : Nat := 124682
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61415⟩⟩] } }
def leftRaw : List Term := Proof.Events487.exact124673RawTerms
def rightRaw : List Term := Proof.Events486.exact124609RawTerms
def group : MergeGroup := .operator 124673 124609
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124673) (leftOrdinal := 0)
    (rightResult := 124609) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61415⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61415⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124682

namespace LeftMerge124696
def owner : Owner := ⟨.program ⟨257⟩, ⟨60352⟩⟩
def mergeEvent : Nat := 124696
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60349⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119870RawTerms
def rightRaw : List Term := Proof.Events487.exact124690RawTerms
def group : MergeGroup := .operator 119870 124690
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119870) (leftOrdinal := 0)
    (rightResult := 124690) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60349⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60349⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124696

namespace LeftMerge124775
def owner : Owner := ⟨.program ⟨257⟩, ⟨59378⟩⟩
def mergeEvent : Nat := 124775
def frameStart : Nat := 124745
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events487.exact124771RawTerms
def rightRaw : List Term := Proof.Events487.exact124768RawTerms
def group : MergeGroup := .operator 124771 124768
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124771) (leftOrdinal := 0)
    (rightResult := 124768) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25202⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124775

namespace LeftMerge124805
def owner : Owner := ⟨.program ⟨257⟩, ⟨61212⟩⟩
def mergeEvent : Nat := 124805
def frameStart : Nat := 124745
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events487.exact124801RawTerms
def rightRaw : List Term := Proof.Events487.exact124799RawTerms
def group : MergeGroup := .operator 124801 124799
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124801) (leftOrdinal := 0)
    (rightResult := 124799) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124805

namespace LeftMerge124828
def owner : Owner := ⟨.program ⟨257⟩, ⟨9537⟩⟩
def mergeEvent : Nat := 124828
def frameStart : Nat := 124745
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }
def leftRaw : List Term := Proof.Events487.exact124824RawTerms
def rightRaw : List Term := Proof.Events487.exact124821RawTerms
def group : MergeGroup := .operator 124824 124821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124824) (leftOrdinal := 0)
    (rightResult := 124821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9535⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124828

namespace LeftMerge124837
def owner : Owner := ⟨.program ⟨257⟩, ⟨61418⟩⟩
def mergeEvent : Nat := 124837
def frameStart : Nat := 124745
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61415⟩⟩] } }
def leftRaw : List Term := Proof.Events487.exact124833RawTerms
def rightRaw : List Term := Proof.Events487.exact124790RawTerms
def group : MergeGroup := .operator 124833 124790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124833) (leftOrdinal := 0)
    (rightResult := 124790) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61415⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61415⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124837

namespace LeftMerge124838
def owner : Owner := ⟨.program ⟨257⟩, ⟨61418⟩⟩
def mergeEvent : Nat := 124838
def frameStart : Nat := 124745
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61415⟩⟩] } }
def leftRaw : List Term := Proof.Events487.exact124833RawTerms
def rightRaw : List Term := Proof.Events487.exact124790RawTerms
def group : MergeGroup := .operator 124833 124790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124833) (leftOrdinal := 1)
    (rightResult := 124790) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61415⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61415⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124838

namespace LeftMerge124840
def owner : Owner := ⟨.program ⟨257⟩, ⟨61418⟩⟩
def mergeEvent : Nat := 124840
def frameStart : Nat := 124745
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60925⟩⟩] } }
def rhsRaw : List Term := Proof.Events487.exact124787RawTerms
def group : MergeGroup := .relation 124839
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 124839) (rhsResult := 124787)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61415⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61415⟩⟩) ⟨60925⟩ 124787) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60925⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], [⟨.program ⟨257⟩, ⟨60925⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124840

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
