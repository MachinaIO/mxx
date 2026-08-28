import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge245573
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245573
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨63043⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events959.exact245530RawTerms
def rightRaw : List Term := Proof.Events924.exact236753RawTerms
def group : MergeGroup := .operator 245530 236753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245530) (leftOrdinal := 34)
    (rightResult := 236753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨63043⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨63043⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245573

namespace LeftMerge245575
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245575
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨63043⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def rhsRaw : List Term := Proof.Events924.exact236750RawTerms
def group : MergeGroup := .relation 245574
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 245574) (rhsResult := 236750)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨63043⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71172⟩⟩) ⟨68818⟩ 236750) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨63043⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245575

namespace LeftMerge245576
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245576
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events959.exact245530RawTerms
def rightRaw : List Term := Proof.Events924.exact236753RawTerms
def group : MergeGroup := .operator 245530 236753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245530) (leftOrdinal := 7)
    (rightResult := 236753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge245576

namespace LeftMerge245577
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245577
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨60063⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events959.exact245530RawTerms
def rightRaw : List Term := Proof.Events924.exact236753RawTerms
def group : MergeGroup := .operator 245530 236753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245530) (leftOrdinal := 33)
    (rightResult := 236753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨60063⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨60063⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245577

namespace LeftMerge245579
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245579
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨60063⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def rhsRaw : List Term := Proof.Events924.exact236750RawTerms
def group : MergeGroup := .relation 245578
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 245578) (rhsResult := 236750)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨60063⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71172⟩⟩) ⟨68818⟩ 236750) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨60063⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245579

namespace LeftMerge245580
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245580
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events959.exact245530RawTerms
def rightRaw : List Term := Proof.Events924.exact236753RawTerms
def group : MergeGroup := .operator 245530 236753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245530) (leftOrdinal := 6)
    (rightResult := 236753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge245580

namespace LeftMerge245581
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245581
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨57083⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events959.exact245530RawTerms
def rightRaw : List Term := Proof.Events924.exact236753RawTerms
def group : MergeGroup := .operator 245530 236753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245530) (leftOrdinal := 32)
    (rightResult := 236753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨57083⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨57083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245581

namespace LeftMerge245583
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245583
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨57083⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def rhsRaw : List Term := Proof.Events924.exact236750RawTerms
def group : MergeGroup := .relation 245582
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 245582) (rhsResult := 236750)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨57083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71172⟩⟩) ⟨68818⟩ 236750) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨57083⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245583

namespace LeftMerge245584
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245584
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events959.exact245530RawTerms
def rightRaw : List Term := Proof.Events924.exact236753RawTerms
def group : MergeGroup := .operator 245530 236753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245530) (leftOrdinal := 5)
    (rightResult := 236753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge245584

namespace LeftMerge245585
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245585
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨54103⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events959.exact245530RawTerms
def rightRaw : List Term := Proof.Events924.exact236753RawTerms
def group : MergeGroup := .operator 245530 236753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245530) (leftOrdinal := 31)
    (rightResult := 236753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨54103⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨54103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245585

namespace LeftMerge245587
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245587
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨54103⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def rhsRaw : List Term := Proof.Events924.exact236750RawTerms
def group : MergeGroup := .relation 245586
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 245586) (rhsResult := 236750)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨54103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71172⟩⟩) ⟨68818⟩ 236750) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨54103⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245587

namespace LeftMerge245588
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245588
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events959.exact245530RawTerms
def rightRaw : List Term := Proof.Events924.exact236753RawTerms
def group : MergeGroup := .operator 245530 236753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245530) (leftOrdinal := 4)
    (rightResult := 236753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge245588

namespace LeftMerge245589
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245589
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨51123⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events959.exact245530RawTerms
def rightRaw : List Term := Proof.Events924.exact236753RawTerms
def group : MergeGroup := .operator 245530 236753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245530) (leftOrdinal := 30)
    (rightResult := 236753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨51123⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨51123⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245589

namespace LeftMerge245591
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245591
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨51123⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def rhsRaw : List Term := Proof.Events924.exact236750RawTerms
def group : MergeGroup := .relation 245590
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 245590) (rhsResult := 236750)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨51123⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71172⟩⟩) ⟨68818⟩ 236750) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨51123⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245591

namespace LeftMerge245592
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245592
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events959.exact245530RawTerms
def rightRaw : List Term := Proof.Events924.exact236753RawTerms
def group : MergeGroup := .operator 245530 236753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245530) (leftOrdinal := 3)
    (rightResult := 236753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge245592

namespace LeftMerge245593
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245593
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨32068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events959.exact245530RawTerms
def rightRaw : List Term := Proof.Events924.exact236753RawTerms
def group : MergeGroup := .operator 245530 236753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245530) (leftOrdinal := 23)
    (rightResult := 236753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨32068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨32068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245593

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
