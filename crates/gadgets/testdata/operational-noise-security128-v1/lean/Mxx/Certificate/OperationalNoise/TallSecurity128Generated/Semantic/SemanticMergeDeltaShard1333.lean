import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge217699
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217699
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }
def rhsRaw : List Term := Proof.Events849.exact217474RawTerms
def group : MergeGroup := .relation 217698
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 217698) (rhsResult := 217474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217699

namespace LeftMerge217700
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217700
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 23)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217700

namespace LeftMerge217702
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217702
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }
def rhsRaw : List Term := Proof.Events849.exact217474RawTerms
def group : MergeGroup := .relation 217701
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 217701) (rhsResult := 217474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217702

namespace LeftMerge217703
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217703
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 20)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217703

namespace LeftMerge217705
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217705
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }
def rhsRaw : List Term := Proof.Events849.exact217474RawTerms
def group : MergeGroup := .relation 217704
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 217704) (rhsResult := 217474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217705

namespace LeftMerge217706
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217706
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 19)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217706

namespace LeftMerge217708
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217708
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }
def rhsRaw : List Term := Proof.Events849.exact217474RawTerms
def group : MergeGroup := .relation 217707
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 217707) (rhsResult := 217474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217708

namespace LeftMerge217709
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217709
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 18)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217709

namespace LeftMerge217711
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217711
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }
def rhsRaw : List Term := Proof.Events849.exact217474RawTerms
def group : MergeGroup := .relation 217710
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 217710) (rhsResult := 217474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217711

namespace LeftMerge217719
def owner : Owner := ⟨.program ⟨257⟩, ⟨67459⟩⟩
def mergeEvent : Nat := 217719
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨67457⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events849.exact217488RawTerms
def rightRaw : List Term := Proof.Events850.exact217715RawTerms
def group : MergeGroup := .operator 217488 217715
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217488) (leftOrdinal := 0)
    (rightResult := 217715) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67457⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217719

namespace LeftMerge217736
def owner : Owner := ⟨.program ⟨257⟩, ⟨68373⟩⟩
def mergeEvent : Nat := 217736
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩] } }
def rhsRaw : List Term := Proof.Events850.exact217733RawTerms
def group : MergeGroup := .relation 217735
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 217735) (rhsResult := 217733)
    (sourceTermOrdinal := 18) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68370⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 217734 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68370⟩⟩]⟩) (none) 217733) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217736

namespace LeftMerge217737
def owner : Owner := ⟨.program ⟨257⟩, ⟨68373⟩⟩
def mergeEvent : Nat := 217737
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def rhsRaw : List Term := Proof.Events850.exact217733RawTerms
def group : MergeGroup := .relation 217735
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 217735) (rhsResult := 217733)
    (sourceTermOrdinal := 17) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68370⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 217734 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68370⟩⟩]⟩) (none) 217733) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217737

namespace LeftMerge217738
def owner : Owner := ⟨.program ⟨257⟩, ⟨68373⟩⟩
def mergeEvent : Nat := 217738
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def rhsRaw : List Term := Proof.Events850.exact217733RawTerms
def group : MergeGroup := .relation 217735
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 217735) (rhsResult := 217733)
    (sourceTermOrdinal := 16) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68370⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 217734 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68370⟩⟩]⟩) (none) 217733) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217738

namespace LeftMerge217739
def owner : Owner := ⟨.program ⟨257⟩, ⟨68373⟩⟩
def mergeEvent : Nat := 217739
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def rhsRaw : List Term := Proof.Events850.exact217733RawTerms
def group : MergeGroup := .relation 217735
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 217735) (rhsResult := 217733)
    (sourceTermOrdinal := 15) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68370⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 217734 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68370⟩⟩]⟩) (none) 217733) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217739

namespace LeftMerge217740
def owner : Owner := ⟨.program ⟨257⟩, ⟨68373⟩⟩
def mergeEvent : Nat := 217740
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def rhsRaw : List Term := Proof.Events850.exact217733RawTerms
def group : MergeGroup := .relation 217735
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 217735) (rhsResult := 217733)
    (sourceTermOrdinal := 14) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68370⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 217734 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68370⟩⟩]⟩) (none) 217733) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217740

namespace LeftMerge217741
def owner : Owner := ⟨.program ⟨257⟩, ⟨68373⟩⟩
def mergeEvent : Nat := 217741
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def rhsRaw : List Term := Proof.Events850.exact217733RawTerms
def group : MergeGroup := .relation 217735
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 217735) (rhsResult := 217733)
    (sourceTermOrdinal := 13) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68370⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 217734 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68370⟩⟩]⟩) (none) 217733) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217741

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
