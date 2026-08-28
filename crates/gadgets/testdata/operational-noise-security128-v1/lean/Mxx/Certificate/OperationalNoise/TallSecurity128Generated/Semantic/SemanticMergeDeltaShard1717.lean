import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge278459
def owner : Owner := ⟨.program ⟨257⟩, ⟨61630⟩⟩
def mergeEvent : Nat := 278459
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61628⟩⟩] } }
def leftRaw : List Term := Proof.Events1059.exact271126RawTerms
def rightRaw : List Term := Proof.Events1087.exact278452RawTerms
def group : MergeGroup := .operator 271126 278452
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271126) (leftOrdinal := 1)
    (rightResult := 278452) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61628⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61628⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge278459

namespace LeftMerge278461
def owner : Owner := ⟨.program ⟨257⟩, ⟨61630⟩⟩
def mergeEvent : Nat := 278461
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61025⟩⟩] } }
def rhsRaw : List Term := Proof.Events1087.exact278449RawTerms
def group : MergeGroup := .relation 278460
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 278460) (rhsResult := 278449)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61628⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61628⟩⟩) ⟨61025⟩ 278449) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61025⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], [⟨.program ⟨257⟩, ⟨61025⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge278461

namespace LeftMerge278475
def owner : Owner := ⟨.program ⟨257⟩, ⟨60529⟩⟩
def mergeEvent : Nat := 278475
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60526⟩⟩] } }
def leftRaw : List Term := Proof.Events1039.exact266120RawTerms
def rightRaw : List Term := Proof.Events1087.exact278469RawTerms
def group : MergeGroup := .operator 266120 278469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266120) (leftOrdinal := 0)
    (rightResult := 278469) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60526⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60526⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge278475

namespace LeftMerge278596
def owner : Owner := ⟨.program ⟨257⟩, ⟨61276⟩⟩
def mergeEvent : Nat := 278596
def frameStart : Nat := 278530
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1088.exact278592RawTerms
def rightRaw : List Term := Proof.Events1088.exact278590RawTerms
def group : MergeGroup := .operator 278592 278590
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 278592) (leftOrdinal := 0)
    (rightResult := 278590) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59762⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge278596

namespace LeftMerge278608
def owner : Owner := ⟨.program ⟨257⟩, ⟨61629⟩⟩
def mergeEvent : Nat := 278608
def frameStart : Nat := 278530
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61628⟩⟩] } }
def leftRaw : List Term := Proof.Events1088.exact278604RawTerms
def rightRaw : List Term := Proof.Events1088.exact278581RawTerms
def group : MergeGroup := .operator 278604 278581
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 278604) (leftOrdinal := 0)
    (rightResult := 278581) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61628⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61628⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge278608

namespace LeftMerge278609
def owner : Owner := ⟨.program ⟨257⟩, ⟨61629⟩⟩
def mergeEvent : Nat := 278609
def frameStart : Nat := 278530
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61628⟩⟩] } }
def leftRaw : List Term := Proof.Events1088.exact278604RawTerms
def rightRaw : List Term := Proof.Events1088.exact278581RawTerms
def group : MergeGroup := .operator 278604 278581
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 278604) (leftOrdinal := 1)
    (rightResult := 278581) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61628⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59762⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61628⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge278609

namespace LeftMerge278611
def owner : Owner := ⟨.program ⟨257⟩, ⟨61629⟩⟩
def mergeEvent : Nat := 278611
def frameStart : Nat := 278530
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61025⟩⟩] } }
def rhsRaw : List Term := Proof.Events1088.exact278578RawTerms
def group : MergeGroup := .relation 278610
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 278610) (rhsResult := 278578)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨59762⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61628⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61628⟩⟩) ⟨61025⟩ 278578) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61025⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨59762⟩⟩], [⟨.program ⟨257⟩, ⟨61025⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge278611

namespace LeftMerge278619
def owner : Owner := ⟨.program ⟨257⟩, ⟨59951⟩⟩
def mergeEvent : Nat := 278619
def frameStart : Nat := 278530
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59948⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1088.exact278592RawTerms
def rightRaw : List Term := Proof.Events1088.exact278615RawTerms
def group : MergeGroup := .operator 278592 278615
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 278592) (leftOrdinal := 0)
    (rightResult := 278615) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59948⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge278619

namespace LeftMerge278636
def owner : Owner := ⟨.program ⟨257⟩, ⟨60529⟩⟩
def mergeEvent : Nat := 278636
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7211⟩⟩] } }
def rhsRaw : List Term := Proof.Events1088.exact278633RawTerms
def group : MergeGroup := .relation 278635
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 278635) (rhsResult := 278633)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60526⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 278634 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60526⟩⟩]⟩) (none) 278633) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7211⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge278636

namespace LeftMerge278637
def owner : Owner := ⟨.program ⟨257⟩, ⟨60529⟩⟩
def mergeEvent : Nat := 278637
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61628⟩⟩] } }
def rhsRaw : List Term := Proof.Events1088.exact278633RawTerms
def group : MergeGroup := .relation 278635
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 278635) (rhsResult := 278633)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60526⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 278634 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60526⟩⟩]⟩) (none) 278633) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61628⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61628⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge278637

namespace LeftMerge278638
def owner : Owner := ⟨.program ⟨257⟩, ⟨60529⟩⟩
def mergeEvent : Nat := 278638
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61025⟩⟩] } }
def rhsRaw : List Term := Proof.Events1088.exact278633RawTerms
def group : MergeGroup := .relation 278635
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 278635) (rhsResult := 278633)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60526⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 278634 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60526⟩⟩]⟩) (none) 278633) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61025⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], [⟨.program ⟨257⟩, ⟨61025⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge278638

namespace LeftMerge278639
def owner : Owner := ⟨.program ⟨257⟩, ⟨60529⟩⟩
def mergeEvent : Nat := 278639
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1088.exact278633RawTerms
def group : MergeGroup := .relation 278635
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 278635) (rhsResult := 278633)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60526⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 278634 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60526⟩⟩]⟩) (none) 278633) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59948⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge278639

namespace LeftMerge278644
def owner : Owner := ⟨.program ⟨257⟩, ⟨61631⟩⟩
def mergeEvent : Nat := 278644
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61628⟩⟩] } }
def leftRaw : List Term := Proof.Events1088.exact278640RawTerms
def rightRaw : List Term := Proof.Events1087.exact278462RawTerms
def group : MergeGroup := .operator 278640 278462
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 278640) (leftOrdinal := 0)
    (rightResult := 278462) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61628⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61628⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61628⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge278644

namespace LeftMerge278645
def owner : Owner := ⟨.program ⟨257⟩, ⟨61631⟩⟩
def mergeEvent : Nat := 278645
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61025⟩⟩] } }
def leftRaw : List Term := Proof.Events1088.exact278640RawTerms
def rightRaw : List Term := Proof.Events1087.exact278462RawTerms
def group : MergeGroup := .operator 278640 278462
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 278640) (leftOrdinal := 2)
    (rightResult := 278462) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61025⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61025⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], [⟨.program ⟨257⟩, ⟨61025⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge278645

namespace LeftMerge278653
def owner : Owner := ⟨.program ⟨257⟩, ⟨61632⟩⟩
def mergeEvent : Nat := 278653
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩] } }
def leftRaw : List Term := Proof.Events1088.exact278647RawTerms
def rightRaw : List Term := Proof.Events061.exact15742RawTerms
def group : MergeGroup := .operator 278647 15742
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 278647) (leftOrdinal := 0)
    (rightResult := 15742) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7211⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7103⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge278653

namespace LeftMerge278654
def owner : Owner := ⟨.program ⟨257⟩, ⟨61632⟩⟩
def mergeEvent : Nat := 278654
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩] } }
def leftRaw : List Term := Proof.Events1088.exact278647RawTerms
def rightRaw : List Term := Proof.Events061.exact15742RawTerms
def group : MergeGroup := .operator 278647 15742
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 278647) (leftOrdinal := 1)
    (rightResult := 15742) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7103⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge278654

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
