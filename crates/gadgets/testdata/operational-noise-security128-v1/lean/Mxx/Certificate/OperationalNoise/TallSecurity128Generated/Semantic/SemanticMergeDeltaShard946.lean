import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge155378
def owner : Owner := ⟨.program ⟨257⟩, ⟨52487⟩⟩
def mergeEvent : Nat := 155378
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52486⟩⟩] } }
def leftRaw : List Term := Proof.Events606.exact155369RawTerms
def rightRaw : List Term := Proof.Events606.exact155305RawTerms
def group : MergeGroup := .operator 155369 155305
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 155369) (leftOrdinal := 0)
    (rightResult := 155305) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52486⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52486⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge155378

namespace LeftMerge155392
def owner : Owner := ⟨.program ⟨257⟩, ⟨51422⟩⟩
def mergeEvent : Nat := 155392
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51419⟩⟩] } }
def leftRaw : List Term := Proof.Events582.exact149120RawTerms
def rightRaw : List Term := Proof.Events606.exact155386RawTerms
def group : MergeGroup := .operator 149120 155386
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149120) (leftOrdinal := 0)
    (rightResult := 155386) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51419⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51419⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge155392

namespace LeftMerge155471
def owner : Owner := ⟨.program ⟨257⟩, ⟨50465⟩⟩
def mergeEvent : Nat := 155471
def frameStart : Nat := 155441
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24494⟩⟩, ⟨.program ⟨257⟩, ⟨50464⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events607.exact155467RawTerms
def rightRaw : List Term := Proof.Events607.exact155464RawTerms
def group : MergeGroup := .operator 155467 155464
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 155467) (leftOrdinal := 0)
    (rightResult := 155464) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50464⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24494⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24494⟩⟩, ⟨.program ⟨257⟩, ⟨50464⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge155471

namespace LeftMerge155501
def owner : Owner := ⟨.program ⟨257⟩, ⟨52276⟩⟩
def mergeEvent : Nat := 155501
def frameStart : Nat := 155441
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24494⟩⟩, ⟨.program ⟨257⟩, ⟨50464⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events607.exact155497RawTerms
def rightRaw : List Term := Proof.Events607.exact155495RawTerms
def group : MergeGroup := .operator 155497 155495
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 155497) (leftOrdinal := 0)
    (rightResult := 155495) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24494⟩⟩, ⟨.program ⟨257⟩, ⟨50464⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24494⟩⟩, ⟨.program ⟨257⟩, ⟨50464⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge155501

namespace LeftMerge155524
def owner : Owner := ⟨.program ⟨257⟩, ⟨9582⟩⟩
def mergeEvent : Nat := 155524
def frameStart : Nat := 155441
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }
def leftRaw : List Term := Proof.Events607.exact155520RawTerms
def rightRaw : List Term := Proof.Events607.exact155517RawTerms
def group : MergeGroup := .operator 155520 155517
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 155520) (leftOrdinal := 0)
    (rightResult := 155517) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9580⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge155524

namespace LeftMerge155533
def owner : Owner := ⟨.program ⟨257⟩, ⟨52489⟩⟩
def mergeEvent : Nat := 155533
def frameStart : Nat := 155441
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52486⟩⟩] } }
def leftRaw : List Term := Proof.Events607.exact155529RawTerms
def rightRaw : List Term := Proof.Events607.exact155486RawTerms
def group : MergeGroup := .operator 155529 155486
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 155529) (leftOrdinal := 0)
    (rightResult := 155486) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52486⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52486⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge155533

namespace LeftMerge155534
def owner : Owner := ⟨.program ⟨257⟩, ⟨52489⟩⟩
def mergeEvent : Nat := 155534
def frameStart : Nat := 155441
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24494⟩⟩, ⟨.program ⟨257⟩, ⟨50464⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52486⟩⟩] } }
def leftRaw : List Term := Proof.Events607.exact155529RawTerms
def rightRaw : List Term := Proof.Events607.exact155486RawTerms
def group : MergeGroup := .operator 155529 155486
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 155529) (leftOrdinal := 1)
    (rightResult := 155486) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24494⟩⟩, ⟨.program ⟨257⟩, ⟨50464⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52486⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24494⟩⟩, ⟨.program ⟨257⟩, ⟨50464⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52486⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge155534

namespace LeftMerge155536
def owner : Owner := ⟨.program ⟨257⟩, ⟨52489⟩⟩
def mergeEvent : Nat := 155536
def frameStart : Nat := 155441
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24494⟩⟩, ⟨.program ⟨257⟩, ⟨50464⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51991⟩⟩] } }
def rhsRaw : List Term := Proof.Events607.exact155483RawTerms
def group : MergeGroup := .relation 155535
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 155535) (rhsResult := 155483)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24494⟩⟩, ⟨.program ⟨257⟩, ⟨50464⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52486⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52486⟩⟩) ⟨51991⟩ 155483) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51991⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24494⟩⟩, ⟨.program ⟨257⟩, ⟨50464⟩⟩], [⟨.program ⟨257⟩, ⟨51991⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge155536

namespace LeftMerge155544
def owner : Owner := ⟨.program ⟨257⟩, ⟨50866⟩⟩
def mergeEvent : Nat := 155544
def frameStart : Nat := 155441
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events607.exact155497RawTerms
def rightRaw : List Term := Proof.Events607.exact155540RawTerms
def group : MergeGroup := .operator 155497 155540
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 155497) (leftOrdinal := 0)
    (rightResult := 155540) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50864⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge155544

namespace LeftMerge155561
def owner : Owner := ⟨.program ⟨257⟩, ⟨51422⟩⟩
def mergeEvent : Nat := 155561
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }
def rhsRaw : List Term := Proof.Events607.exact155558RawTerms
def group : MergeGroup := .relation 155560
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 155560) (rhsResult := 155558)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 155559 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51419⟩⟩]⟩) (none) 155558) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge155561

namespace LeftMerge155562
def owner : Owner := ⟨.program ⟨257⟩, ⟨51422⟩⟩
def mergeEvent : Nat := 155562
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52486⟩⟩] } }
def rhsRaw : List Term := Proof.Events607.exact155558RawTerms
def group : MergeGroup := .relation 155560
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 155560) (rhsResult := 155558)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 155559 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51419⟩⟩]⟩) (none) 155558) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52486⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52486⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge155562

namespace LeftMerge155563
def owner : Owner := ⟨.program ⟨257⟩, ⟨51422⟩⟩
def mergeEvent : Nat := 155563
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24494⟩⟩, ⟨.program ⟨257⟩, ⟨50464⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51991⟩⟩] } }
def rhsRaw : List Term := Proof.Events607.exact155558RawTerms
def group : MergeGroup := .relation 155560
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 155560) (rhsResult := 155558)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 155559 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51419⟩⟩]⟩) (none) 155558) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24494⟩⟩, ⟨.program ⟨257⟩, ⟨50464⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51991⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24494⟩⟩, ⟨.program ⟨257⟩, ⟨50464⟩⟩], [⟨.program ⟨257⟩, ⟨51991⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge155563

namespace LeftMerge155564
def owner : Owner := ⟨.program ⟨257⟩, ⟨51422⟩⟩
def mergeEvent : Nat := 155564
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨50864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events607.exact155558RawTerms
def group : MergeGroup := .relation 155560
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 155560) (rhsResult := 155558)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 155559 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51419⟩⟩]⟩) (none) 155558) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨50864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge155564

namespace LeftMerge155569
def owner : Owner := ⟨.program ⟨257⟩, ⟨52488⟩⟩
def mergeEvent : Nat := 155569
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24494⟩⟩, ⟨.program ⟨257⟩, ⟨50464⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51991⟩⟩] } }
def leftRaw : List Term := Proof.Events607.exact155565RawTerms
def rightRaw : List Term := Proof.Events606.exact155379RawTerms
def group : MergeGroup := .operator 155565 155379
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 155565) (leftOrdinal := 2)
    (rightResult := 155379) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24494⟩⟩, ⟨.program ⟨257⟩, ⟨50464⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51991⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24494⟩⟩, ⟨.program ⟨257⟩, ⟨50464⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51991⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24494⟩⟩, ⟨.program ⟨257⟩, ⟨50464⟩⟩], [⟨.program ⟨257⟩, ⟨51991⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge155569

namespace LeftMerge155570
def owner : Owner := ⟨.program ⟨257⟩, ⟨52488⟩⟩
def mergeEvent : Nat := 155570
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52486⟩⟩] } }
def leftRaw : List Term := Proof.Events607.exact155565RawTerms
def rightRaw : List Term := Proof.Events606.exact155379RawTerms
def group : MergeGroup := .operator 155565 155379
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 155565) (leftOrdinal := 1)
    (rightResult := 155379) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52486⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52486⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52486⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge155570

namespace LeftMerge155578
def owner : Owner := ⟨.program ⟨257⟩, ⟨52861⟩⟩
def mergeEvent : Nat := 155578
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52859⟩⟩] } }
def leftRaw : List Term := Proof.Events607.exact155572RawTerms
def rightRaw : List Term := Proof.Events606.exact155295RawTerms
def group : MergeGroup := .operator 155572 155295
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 155572) (leftOrdinal := 0)
    (rightResult := 155295) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52859⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52859⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge155578

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
