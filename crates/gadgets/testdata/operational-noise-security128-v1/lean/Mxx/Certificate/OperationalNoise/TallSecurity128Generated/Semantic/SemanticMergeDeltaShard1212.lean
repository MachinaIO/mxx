import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge197528
def owner : Owner := ⟨.program ⟨257⟩, ⟨64936⟩⟩
def mergeEvent : Nat := 197528
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64099⟩⟩] } }
def rhsRaw : List Term := Proof.Events770.exact197239RawTerms
def group : MergeGroup := .relation 197527
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 197527) (rhsResult := 197239)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64934⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64934⟩⟩) ⟨64099⟩ 197239) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64099⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], [⟨.program ⟨257⟩, ⟨64099⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge197528

namespace LeftMerge197542
def owner : Owner := ⟨.program ⟨257⟩, ⟨63719⟩⟩
def mergeEvent : Nat := 197542
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63716⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events771.exact197536RawTerms
def group : MergeGroup := .operator 192995 197536
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 197536) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63716⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63716⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197542

namespace LeftMerge197663
def owner : Owner := ⟨.program ⟨257⟩, ⟨64296⟩⟩
def mergeEvent : Nat := 197663
def frameStart : Nat := 197597
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events772.exact197659RawTerms
def rightRaw : List Term := Proof.Events772.exact197657RawTerms
def group : MergeGroup := .operator 197659 197657
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197659) (leftOrdinal := 0)
    (rightResult := 197657) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197663

namespace LeftMerge197675
def owner : Owner := ⟨.program ⟨257⟩, ⟨64935⟩⟩
def mergeEvent : Nat := 197675
def frameStart : Nat := 197597
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64934⟩⟩] } }
def leftRaw : List Term := Proof.Events772.exact197671RawTerms
def rightRaw : List Term := Proof.Events772.exact197648RawTerms
def group : MergeGroup := .operator 197671 197648
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197671) (leftOrdinal := 0)
    (rightResult := 197648) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64934⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197675

namespace LeftMerge197676
def owner : Owner := ⟨.program ⟨257⟩, ⟨64935⟩⟩
def mergeEvent : Nat := 197676
def frameStart : Nat := 197597
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64934⟩⟩] } }
def leftRaw : List Term := Proof.Events772.exact197671RawTerms
def rightRaw : List Term := Proof.Events772.exact197648RawTerms
def group : MergeGroup := .operator 197671 197648
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197671) (leftOrdinal := 1)
    (rightResult := 197648) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge197676

namespace LeftMerge197678
def owner : Owner := ⟨.program ⟨257⟩, ⟨64935⟩⟩
def mergeEvent : Nat := 197678
def frameStart : Nat := 197597
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64099⟩⟩] } }
def rhsRaw : List Term := Proof.Events772.exact197645RawTerms
def group : MergeGroup := .relation 197677
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 197677) (rhsResult := 197645)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨62824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64934⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64934⟩⟩) ⟨64099⟩ 197645) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64099⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨62824⟩⟩], [⟨.program ⟨257⟩, ⟨64099⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge197678

namespace LeftMerge197686
def owner : Owner := ⟨.program ⟨257⟩, ⟨63121⟩⟩
def mergeEvent : Nat := 197686
def frameStart : Nat := 197597
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63119⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events772.exact197659RawTerms
def rightRaw : List Term := Proof.Events772.exact197682RawTerms
def group : MergeGroup := .operator 197659 197682
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197659) (leftOrdinal := 0)
    (rightResult := 197682) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63119⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨63119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197686

namespace LeftMerge197703
def owner : Owner := ⟨.program ⟨257⟩, ⟨63719⟩⟩
def mergeEvent : Nat := 197703
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }
def rhsRaw : List Term := Proof.Events772.exact197700RawTerms
def group : MergeGroup := .relation 197702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 197702) (rhsResult := 197700)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63716⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 197701 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63716⟩⟩]⟩) (none) 197700) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197703

namespace LeftMerge197704
def owner : Owner := ⟨.program ⟨257⟩, ⟨63719⟩⟩
def mergeEvent : Nat := 197704
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64934⟩⟩] } }
def rhsRaw : List Term := Proof.Events772.exact197700RawTerms
def group : MergeGroup := .relation 197702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 197702) (rhsResult := 197700)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63716⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 197701 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63716⟩⟩]⟩) (none) 197700) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64934⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge197704

namespace LeftMerge197705
def owner : Owner := ⟨.program ⟨257⟩, ⟨63719⟩⟩
def mergeEvent : Nat := 197705
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64099⟩⟩] } }
def rhsRaw : List Term := Proof.Events772.exact197700RawTerms
def group : MergeGroup := .relation 197702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 197702) (rhsResult := 197700)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63716⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 197701 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63716⟩⟩]⟩) (none) 197700) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64099⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], [⟨.program ⟨257⟩, ⟨64099⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197705

namespace LeftMerge197706
def owner : Owner := ⟨.program ⟨257⟩, ⟨63719⟩⟩
def mergeEvent : Nat := 197706
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63119⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events772.exact197700RawTerms
def group : MergeGroup := .relation 197702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 197702) (rhsResult := 197700)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63716⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 197701 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63716⟩⟩]⟩) (none) 197700) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63119⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge197706

namespace LeftMerge197711
def owner : Owner := ⟨.program ⟨257⟩, ⟨64937⟩⟩
def mergeEvent : Nat := 197711
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64934⟩⟩] } }
def leftRaw : List Term := Proof.Events772.exact197707RawTerms
def rightRaw : List Term := Proof.Events771.exact197529RawTerms
def group : MergeGroup := .operator 197707 197529
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197707) (leftOrdinal := 0)
    (rightResult := 197529) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64934⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197711

namespace LeftMerge197712
def owner : Owner := ⟨.program ⟨257⟩, ⟨64937⟩⟩
def mergeEvent : Nat := 197712
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64099⟩⟩] } }
def leftRaw : List Term := Proof.Events772.exact197707RawTerms
def rightRaw : List Term := Proof.Events771.exact197529RawTerms
def group : MergeGroup := .operator 197707 197529
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197707) (leftOrdinal := 2)
    (rightResult := 197529) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64099⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64099⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], [⟨.program ⟨257⟩, ⟨64099⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge197712

namespace LeftMerge197738
def owner : Owner := ⟨.program ⟨257⟩, ⟨25275⟩⟩
def mergeEvent : Nat := 197738
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events036.exact9300RawTerms
def rightRaw : List Term := Proof.Events753.exact192903RawTerms
def group : MergeGroup := .operator 9300 192903
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9300) (leftOrdinal := 0)
    (rightResult := 192903) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25274⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197738

namespace LeftMerge197743
def owner : Owner := ⟨.program ⟨257⟩, ⟨8808⟩⟩
def mergeEvent : Nat := 197743
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192773RawTerms
def rightRaw : List Term := Proof.Events086.exact22090RawTerms
def group : MergeGroup := .operator 192773 22090
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192773) (leftOrdinal := 0)
    (rightResult := 22090) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197743

namespace LeftMerge197760
def owner : Owner := ⟨.program ⟨257⟩, ⟨59542⟩⟩
def mergeEvent : Nat := 197760
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25274⟩⟩, ⟨.program ⟨257⟩, ⟨59539⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events772.exact197754RawTerms
def rightRaw : List Term := Proof.Events036.exact9303RawTerms
def group : MergeGroup := .operator 197754 9303
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197754) (leftOrdinal := 1)
    (rightResult := 9303) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59539⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25274⟩⟩, ⟨.program ⟨257⟩, ⟨59539⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge197760

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
