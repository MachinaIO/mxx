import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge270450
def owner : Owner := ⟨.program ⟨257⟩, ⟨64349⟩⟩
def mergeEvent : Nat := 270450
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64348⟩⟩] } }
def leftRaw : List Term := Proof.Events1056.exact270441RawTerms
def rightRaw : List Term := Proof.Events1056.exact270377RawTerms
def group : MergeGroup := .operator 270441 270377
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 270441) (leftOrdinal := 0)
    (rightResult := 270377) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64348⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64348⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge270450

namespace LeftMerge270464
def owner : Owner := ⟨.program ⟨257⟩, ⟨63289⟩⟩
def mergeEvent : Nat := 270464
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63286⟩⟩] } }
def leftRaw : List Term := Proof.Events1039.exact266120RawTerms
def rightRaw : List Term := Proof.Events1056.exact270458RawTerms
def group : MergeGroup := .operator 266120 270458
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266120) (leftOrdinal := 0)
    (rightResult := 270458) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63286⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63286⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge270464

namespace LeftMerge270543
def owner : Owner := ⟨.program ⟨257⟩, ⟨62241⟩⟩
def mergeEvent : Nat := 270543
def frameStart : Nat := 270513
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1056.exact270539RawTerms
def rightRaw : List Term := Proof.Events1056.exact270536RawTerms
def group : MergeGroup := .operator 270539 270536
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 270539) (leftOrdinal := 0)
    (rightResult := 270536) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62240⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25390⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge270543

namespace LeftMerge270573
def owner : Owner := ⟨.program ⟨257⟩, ⟨64176⟩⟩
def mergeEvent : Nat := 270573
def frameStart : Nat := 270513
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1056.exact270569RawTerms
def rightRaw : List Term := Proof.Events1056.exact270567RawTerms
def group : MergeGroup := .operator 270569 270567
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 270569) (leftOrdinal := 0)
    (rightResult := 270567) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge270573

namespace LeftMerge270596
def owner : Owner := ⟨.program ⟨257⟩, ⟨9540⟩⟩
def mergeEvent : Nat := 270596
def frameStart : Nat := 270513
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }
def leftRaw : List Term := Proof.Events1057.exact270592RawTerms
def rightRaw : List Term := Proof.Events1056.exact270589RawTerms
def group : MergeGroup := .operator 270592 270589
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 270592) (leftOrdinal := 0)
    (rightResult := 270589) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9538⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge270596

namespace LeftMerge270605
def owner : Owner := ⟨.program ⟨257⟩, ⟨64351⟩⟩
def mergeEvent : Nat := 270605
def frameStart : Nat := 270513
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64348⟩⟩] } }
def leftRaw : List Term := Proof.Events1057.exact270601RawTerms
def rightRaw : List Term := Proof.Events1056.exact270558RawTerms
def group : MergeGroup := .operator 270601 270558
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 270601) (leftOrdinal := 0)
    (rightResult := 270558) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64348⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64348⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge270605

namespace LeftMerge270606
def owner : Owner := ⟨.program ⟨257⟩, ⟨64351⟩⟩
def mergeEvent : Nat := 270606
def frameStart : Nat := 270513
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64348⟩⟩] } }
def leftRaw : List Term := Proof.Events1057.exact270601RawTerms
def rightRaw : List Term := Proof.Events1056.exact270558RawTerms
def group : MergeGroup := .operator 270601 270558
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 270601) (leftOrdinal := 1)
    (rightResult := 270558) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64348⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64348⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge270606

namespace LeftMerge270608
def owner : Owner := ⟨.program ⟨257⟩, ⟨64351⟩⟩
def mergeEvent : Nat := 270608
def frameStart : Nat := 270513
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63879⟩⟩] } }
def rhsRaw : List Term := Proof.Events1056.exact270555RawTerms
def group : MergeGroup := .relation 270607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 270607) (rhsResult := 270555)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64348⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64348⟩⟩) ⟨63879⟩ 270555) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63879⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], [⟨.program ⟨257⟩, ⟨63879⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge270608

namespace LeftMerge270616
def owner : Owner := ⟨.program ⟨257⟩, ⟨62744⟩⟩
def mergeEvent : Nat := 270616
def frameStart : Nat := 270513
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62742⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1056.exact270569RawTerms
def rightRaw : List Term := Proof.Events1057.exact270612RawTerms
def group : MergeGroup := .operator 270569 270612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 270569) (leftOrdinal := 0)
    (rightResult := 270612) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62742⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62742⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge270616

namespace LeftMerge270633
def owner : Owner := ⟨.program ⟨257⟩, ⟨63289⟩⟩
def mergeEvent : Nat := 270633
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }
def rhsRaw : List Term := Proof.Events1057.exact270630RawTerms
def group : MergeGroup := .relation 270632
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 270632) (rhsResult := 270630)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63286⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 270631 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63286⟩⟩]⟩) (none) 270630) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge270633

namespace LeftMerge270634
def owner : Owner := ⟨.program ⟨257⟩, ⟨63289⟩⟩
def mergeEvent : Nat := 270634
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64348⟩⟩] } }
def rhsRaw : List Term := Proof.Events1057.exact270630RawTerms
def group : MergeGroup := .relation 270632
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 270632) (rhsResult := 270630)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63286⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 270631 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63286⟩⟩]⟩) (none) 270630) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64348⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64348⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge270634

namespace LeftMerge270635
def owner : Owner := ⟨.program ⟨257⟩, ⟨63289⟩⟩
def mergeEvent : Nat := 270635
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63879⟩⟩] } }
def rhsRaw : List Term := Proof.Events1057.exact270630RawTerms
def group : MergeGroup := .relation 270632
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 270632) (rhsResult := 270630)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63286⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 270631 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63286⟩⟩]⟩) (none) 270630) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63879⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], [⟨.program ⟨257⟩, ⟨63879⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge270635

namespace LeftMerge270636
def owner : Owner := ⟨.program ⟨257⟩, ⟨63289⟩⟩
def mergeEvent : Nat := 270636
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62742⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1057.exact270630RawTerms
def group : MergeGroup := .relation 270632
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 270632) (rhsResult := 270630)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63286⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 270631 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63286⟩⟩]⟩) (none) 270630) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62742⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62742⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge270636

namespace LeftMerge270641
def owner : Owner := ⟨.program ⟨257⟩, ⟨64350⟩⟩
def mergeEvent : Nat := 270641
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63879⟩⟩] } }
def leftRaw : List Term := Proof.Events1057.exact270637RawTerms
def rightRaw : List Term := Proof.Events1056.exact270451RawTerms
def group : MergeGroup := .operator 270637 270451
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 270637) (leftOrdinal := 2)
    (rightResult := 270451) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63879⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63879⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], [⟨.program ⟨257⟩, ⟨63879⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge270641

namespace LeftMerge270642
def owner : Owner := ⟨.program ⟨257⟩, ⟨64350⟩⟩
def mergeEvent : Nat := 270642
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64348⟩⟩] } }
def leftRaw : List Term := Proof.Events1057.exact270637RawTerms
def rightRaw : List Term := Proof.Events1056.exact270451RawTerms
def group : MergeGroup := .operator 270637 270451
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 270637) (leftOrdinal := 1)
    (rightResult := 270451) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64348⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64348⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64348⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge270642

namespace LeftMerge270650
def owner : Owner := ⟨.program ⟨257⟩, ⟨64617⟩⟩
def mergeEvent : Nat := 270650
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64615⟩⟩] } }
def leftRaw : List Term := Proof.Events1057.exact270644RawTerms
def rightRaw : List Term := Proof.Events1056.exact270367RawTerms
def group : MergeGroup := .operator 270644 270367
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 270644) (leftOrdinal := 0)
    (rightResult := 270367) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64615⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64615⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge270650

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
