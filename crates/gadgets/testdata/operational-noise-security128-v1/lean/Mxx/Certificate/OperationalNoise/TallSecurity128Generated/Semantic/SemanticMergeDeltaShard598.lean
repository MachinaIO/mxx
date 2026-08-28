import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge100687
def owner : Owner := ⟨.program ⟨257⟩, ⟨71406⟩⟩
def mergeEvent : Nat := 100687
def frameStart : Nat := 99961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63176⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }
def rhsRaw : List Term := Proof.Events392.exact100474RawTerms
def group : MergeGroup := .relation 100686
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 100686) (rhsResult := 100474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨63176⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71405⟩⟩) ⟨68860⟩ 100474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨63176⟩⟩], [⟨.program ⟨257⟩, ⟨68860⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100687

namespace LeftMerge100688
def owner : Owner := ⟨.program ⟨257⟩, ⟨71406⟩⟩
def mergeEvent : Nat := 100688
def frameStart : Nat := 99961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60196⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩] } }
def leftRaw : List Term := Proof.Events393.exact100636RawTerms
def rightRaw : List Term := Proof.Events392.exact100477RawTerms
def group : MergeGroup := .operator 100636 100477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100636) (leftOrdinal := 33)
    (rightResult := 100477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60196⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71405⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨60196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100688

namespace LeftMerge100690
def owner : Owner := ⟨.program ⟨257⟩, ⟨71406⟩⟩
def mergeEvent : Nat := 100690
def frameStart : Nat := 99961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60196⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }
def rhsRaw : List Term := Proof.Events392.exact100474RawTerms
def group : MergeGroup := .relation 100689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 100689) (rhsResult := 100474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨60196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71405⟩⟩) ⟨68860⟩ 100474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨60196⟩⟩], [⟨.program ⟨257⟩, ⟨68860⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100690

namespace LeftMerge100691
def owner : Owner := ⟨.program ⟨257⟩, ⟨71406⟩⟩
def mergeEvent : Nat := 100691
def frameStart : Nat := 99961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57216⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩] } }
def leftRaw : List Term := Proof.Events393.exact100636RawTerms
def rightRaw : List Term := Proof.Events392.exact100477RawTerms
def group : MergeGroup := .operator 100636 100477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100636) (leftOrdinal := 32)
    (rightResult := 100477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57216⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71405⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨57216⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100691

namespace LeftMerge100693
def owner : Owner := ⟨.program ⟨257⟩, ⟨71406⟩⟩
def mergeEvent : Nat := 100693
def frameStart : Nat := 99961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57216⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }
def rhsRaw : List Term := Proof.Events392.exact100474RawTerms
def group : MergeGroup := .relation 100692
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 100692) (rhsResult := 100474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨57216⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71405⟩⟩) ⟨68860⟩ 100474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨57216⟩⟩], [⟨.program ⟨257⟩, ⟨68860⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100693

namespace LeftMerge100694
def owner : Owner := ⟨.program ⟨257⟩, ⟨71406⟩⟩
def mergeEvent : Nat := 100694
def frameStart : Nat := 99961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54236⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩] } }
def leftRaw : List Term := Proof.Events393.exact100636RawTerms
def rightRaw : List Term := Proof.Events392.exact100477RawTerms
def group : MergeGroup := .operator 100636 100477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100636) (leftOrdinal := 31)
    (rightResult := 100477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54236⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71405⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨54236⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100694

namespace LeftMerge100696
def owner : Owner := ⟨.program ⟨257⟩, ⟨71406⟩⟩
def mergeEvent : Nat := 100696
def frameStart : Nat := 99961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54236⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }
def rhsRaw : List Term := Proof.Events392.exact100474RawTerms
def group : MergeGroup := .relation 100695
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 100695) (rhsResult := 100474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨54236⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71405⟩⟩) ⟨68860⟩ 100474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨54236⟩⟩], [⟨.program ⟨257⟩, ⟨68860⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100696

namespace LeftMerge100697
def owner : Owner := ⟨.program ⟨257⟩, ⟨71406⟩⟩
def mergeEvent : Nat := 100697
def frameStart : Nat := 99961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩] } }
def leftRaw : List Term := Proof.Events393.exact100636RawTerms
def rightRaw : List Term := Proof.Events392.exact100477RawTerms
def group : MergeGroup := .operator 100636 100477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100636) (leftOrdinal := 30)
    (rightResult := 100477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71405⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨51256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100697

namespace LeftMerge100699
def owner : Owner := ⟨.program ⟨257⟩, ⟨71406⟩⟩
def mergeEvent : Nat := 100699
def frameStart : Nat := 99961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }
def rhsRaw : List Term := Proof.Events392.exact100474RawTerms
def group : MergeGroup := .relation 100698
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 100698) (rhsResult := 100474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨51256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71405⟩⟩) ⟨68860⟩ 100474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨51256⟩⟩], [⟨.program ⟨257⟩, ⟨68860⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100699

namespace LeftMerge100700
def owner : Owner := ⟨.program ⟨257⟩, ⟨71406⟩⟩
def mergeEvent : Nat := 100700
def frameStart : Nat := 99961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32201⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩] } }
def leftRaw : List Term := Proof.Events393.exact100636RawTerms
def rightRaw : List Term := Proof.Events392.exact100477RawTerms
def group : MergeGroup := .operator 100636 100477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100636) (leftOrdinal := 23)
    (rightResult := 100477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32201⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71405⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100700

namespace LeftMerge100702
def owner : Owner := ⟨.program ⟨257⟩, ⟨71406⟩⟩
def mergeEvent : Nat := 100702
def frameStart : Nat := 99961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32201⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }
def rhsRaw : List Term := Proof.Events392.exact100474RawTerms
def group : MergeGroup := .relation 100701
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 100701) (rhsResult := 100474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨32201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71405⟩⟩) ⟨68860⟩ 100474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨32201⟩⟩], [⟨.program ⟨257⟩, ⟨68860⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100702

namespace LeftMerge100703
def owner : Owner := ⟨.program ⟨257⟩, ⟨71406⟩⟩
def mergeEvent : Nat := 100703
def frameStart : Nat := 99961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22181⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩] } }
def leftRaw : List Term := Proof.Events393.exact100636RawTerms
def rightRaw : List Term := Proof.Events392.exact100477RawTerms
def group : MergeGroup := .operator 100636 100477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100636) (leftOrdinal := 20)
    (rightResult := 100477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22181⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71405⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨22181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100703

namespace LeftMerge100705
def owner : Owner := ⟨.program ⟨257⟩, ⟨71406⟩⟩
def mergeEvent : Nat := 100705
def frameStart : Nat := 99961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22181⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }
def rhsRaw : List Term := Proof.Events392.exact100474RawTerms
def group : MergeGroup := .relation 100704
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 100704) (rhsResult := 100474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨22181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71405⟩⟩) ⟨68860⟩ 100474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨22181⟩⟩], [⟨.program ⟨257⟩, ⟨68860⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100705

namespace LeftMerge100706
def owner : Owner := ⟨.program ⟨257⟩, ⟨71406⟩⟩
def mergeEvent : Nat := 100706
def frameStart : Nat := 99961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18961⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩] } }
def leftRaw : List Term := Proof.Events393.exact100636RawTerms
def rightRaw : List Term := Proof.Events392.exact100477RawTerms
def group : MergeGroup := .operator 100636 100477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100636) (leftOrdinal := 19)
    (rightResult := 100477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18961⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71405⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18961⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100706

namespace LeftMerge100708
def owner : Owner := ⟨.program ⟨257⟩, ⟨71406⟩⟩
def mergeEvent : Nat := 100708
def frameStart : Nat := 99961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18961⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }
def rhsRaw : List Term := Proof.Events392.exact100474RawTerms
def group : MergeGroup := .relation 100707
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 100707) (rhsResult := 100474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨18961⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71405⟩⟩) ⟨68860⟩ 100474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨18961⟩⟩], [⟨.program ⟨257⟩, ⟨68860⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100708

namespace LeftMerge100709
def owner : Owner := ⟨.program ⟨257⟩, ⟨71406⟩⟩
def mergeEvent : Nat := 100709
def frameStart : Nat := 99961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16115⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩] } }
def leftRaw : List Term := Proof.Events393.exact100636RawTerms
def rightRaw : List Term := Proof.Events392.exact100477RawTerms
def group : MergeGroup := .operator 100636 100477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100636) (leftOrdinal := 18)
    (rightResult := 100477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16115⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71405⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨16115⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100709

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
