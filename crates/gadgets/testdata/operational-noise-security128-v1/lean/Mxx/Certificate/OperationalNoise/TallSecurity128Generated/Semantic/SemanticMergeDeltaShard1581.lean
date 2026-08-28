import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge256430
def owner : Owner := ⟨.program ⟨257⟩, ⟨61208⟩⟩
def mergeEvent : Nat := 256430
def frameStart : Nat := 256370
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25190⟩⟩, ⟨.program ⟨257⟩, ⟨59350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1001.exact256426RawTerms
def rightRaw : List Term := Proof.Events1001.exact256424RawTerms
def group : MergeGroup := .operator 256426 256424
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256426) (leftOrdinal := 0)
    (rightResult := 256424) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25190⟩⟩, ⟨.program ⟨257⟩, ⟨59350⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25190⟩⟩, ⟨.program ⟨257⟩, ⟨59350⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256430

namespace LeftMerge256453
def owner : Owner := ⟨.program ⟨257⟩, ⟨9537⟩⟩
def mergeEvent : Nat := 256453
def frameStart : Nat := 256370
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }
def leftRaw : List Term := Proof.Events1001.exact256449RawTerms
def rightRaw : List Term := Proof.Events1001.exact256446RawTerms
def group : MergeGroup := .operator 256449 256446
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256449) (leftOrdinal := 0)
    (rightResult := 256446) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9535⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256453

namespace LeftMerge256462
def owner : Owner := ⟨.program ⟨257⟩, ⟨61407⟩⟩
def mergeEvent : Nat := 256462
def frameStart : Nat := 256370
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61404⟩⟩] } }
def leftRaw : List Term := Proof.Events1001.exact256458RawTerms
def rightRaw : List Term := Proof.Events1001.exact256415RawTerms
def group : MergeGroup := .operator 256458 256415
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256458) (leftOrdinal := 0)
    (rightResult := 256415) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61404⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61404⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256462

namespace LeftMerge256463
def owner : Owner := ⟨.program ⟨257⟩, ⟨61407⟩⟩
def mergeEvent : Nat := 256463
def frameStart : Nat := 256370
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25190⟩⟩, ⟨.program ⟨257⟩, ⟨59350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61404⟩⟩] } }
def leftRaw : List Term := Proof.Events1001.exact256458RawTerms
def rightRaw : List Term := Proof.Events1001.exact256415RawTerms
def group : MergeGroup := .operator 256458 256415
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256458) (leftOrdinal := 1)
    (rightResult := 256415) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25190⟩⟩, ⟨.program ⟨257⟩, ⟨59350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61404⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25190⟩⟩, ⟨.program ⟨257⟩, ⟨59350⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61404⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256463

namespace LeftMerge256465
def owner : Owner := ⟨.program ⟨257⟩, ⟨61407⟩⟩
def mergeEvent : Nat := 256465
def frameStart : Nat := 256370
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25190⟩⟩, ⟨.program ⟨257⟩, ⟨59350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60919⟩⟩] } }
def rhsRaw : List Term := Proof.Events1001.exact256412RawTerms
def group : MergeGroup := .relation 256464
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 256464) (rhsResult := 256412)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25190⟩⟩, ⟨.program ⟨257⟩, ⟨59350⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61404⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61404⟩⟩) ⟨60919⟩ 256412) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60919⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25190⟩⟩, ⟨.program ⟨257⟩, ⟨59350⟩⟩], [⟨.program ⟨257⟩, ⟨60919⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256465

namespace LeftMerge256473
def owner : Owner := ⟨.program ⟨257⟩, ⟨59790⟩⟩
def mergeEvent : Nat := 256473
def frameStart : Nat := 256370
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1001.exact256426RawTerms
def rightRaw : List Term := Proof.Events1001.exact256469RawTerms
def group : MergeGroup := .operator 256426 256469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256426) (leftOrdinal := 0)
    (rightResult := 256469) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59788⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256473

namespace LeftMerge256490
def owner : Owner := ⟨.program ⟨257⟩, ⟨60342⟩⟩
def mergeEvent : Nat := 256490
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }
def rhsRaw : List Term := Proof.Events1001.exact256487RawTerms
def group : MergeGroup := .relation 256489
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 256489) (rhsResult := 256487)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60339⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 256488 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60339⟩⟩]⟩) (none) 256487) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256490

namespace LeftMerge256491
def owner : Owner := ⟨.program ⟨257⟩, ⟨60342⟩⟩
def mergeEvent : Nat := 256491
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61404⟩⟩] } }
def rhsRaw : List Term := Proof.Events1001.exact256487RawTerms
def group : MergeGroup := .relation 256489
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 256489) (rhsResult := 256487)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60339⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 256488 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60339⟩⟩]⟩) (none) 256487) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61404⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61404⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256491

namespace LeftMerge256492
def owner : Owner := ⟨.program ⟨257⟩, ⟨60342⟩⟩
def mergeEvent : Nat := 256492
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25190⟩⟩, ⟨.program ⟨257⟩, ⟨59350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60919⟩⟩] } }
def rhsRaw : List Term := Proof.Events1001.exact256487RawTerms
def group : MergeGroup := .relation 256489
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 256489) (rhsResult := 256487)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60339⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 256488 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60339⟩⟩]⟩) (none) 256487) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25190⟩⟩, ⟨.program ⟨257⟩, ⟨59350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60919⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25190⟩⟩, ⟨.program ⟨257⟩, ⟨59350⟩⟩], [⟨.program ⟨257⟩, ⟨60919⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256492

namespace LeftMerge256493
def owner : Owner := ⟨.program ⟨257⟩, ⟨60342⟩⟩
def mergeEvent : Nat := 256493
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨59788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1001.exact256487RawTerms
def group : MergeGroup := .relation 256489
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 256489) (rhsResult := 256487)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60339⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 256488 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60339⟩⟩]⟩) (none) 256487) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨59788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256493

namespace LeftMerge256498
def owner : Owner := ⟨.program ⟨257⟩, ⟨61406⟩⟩
def mergeEvent : Nat := 256498
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25190⟩⟩, ⟨.program ⟨257⟩, ⟨59350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60919⟩⟩] } }
def leftRaw : List Term := Proof.Events1001.exact256494RawTerms
def rightRaw : List Term := Proof.Events1001.exact256308RawTerms
def group : MergeGroup := .operator 256494 256308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256494) (leftOrdinal := 2)
    (rightResult := 256308) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25190⟩⟩, ⟨.program ⟨257⟩, ⟨59350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60919⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25190⟩⟩, ⟨.program ⟨257⟩, ⟨59350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60919⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25190⟩⟩, ⟨.program ⟨257⟩, ⟨59350⟩⟩], [⟨.program ⟨257⟩, ⟨60919⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256498

namespace LeftMerge256499
def owner : Owner := ⟨.program ⟨257⟩, ⟨61406⟩⟩
def mergeEvent : Nat := 256499
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61404⟩⟩] } }
def leftRaw : List Term := Proof.Events1001.exact256494RawTerms
def rightRaw : List Term := Proof.Events1001.exact256308RawTerms
def group : MergeGroup := .operator 256494 256308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256494) (leftOrdinal := 1)
    (rightResult := 256308) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61404⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61404⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61404⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256499

namespace LeftMerge256507
def owner : Owner := ⟨.program ⟨257⟩, ⟨61739⟩⟩
def mergeEvent : Nat := 256507
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61737⟩⟩] } }
def leftRaw : List Term := Proof.Events1001.exact256501RawTerms
def rightRaw : List Term := Proof.Events1000.exact256224RawTerms
def group : MergeGroup := .operator 256501 256224
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256501) (leftOrdinal := 0)
    (rightResult := 256224) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61737⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61737⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256507

namespace LeftMerge256508
def owner : Owner := ⟨.program ⟨257⟩, ⟨61739⟩⟩
def mergeEvent : Nat := 256508
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨59788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61737⟩⟩] } }
def leftRaw : List Term := Proof.Events1001.exact256501RawTerms
def rightRaw : List Term := Proof.Events1000.exact256224RawTerms
def group : MergeGroup := .operator 256501 256224
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256501) (leftOrdinal := 1)
    (rightResult := 256224) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨59788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61737⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨59788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61737⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256508

namespace LeftMerge256510
def owner : Owner := ⟨.program ⟨257⟩, ⟨61739⟩⟩
def mergeEvent : Nat := 256510
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨59788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61056⟩⟩] } }
def rhsRaw : List Term := Proof.Events1000.exact256221RawTerms
def group : MergeGroup := .relation 256509
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 256509) (rhsResult := 256221)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨59788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61737⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61737⟩⟩) ⟨61056⟩ 256221) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61056⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨59788⟩⟩], [⟨.program ⟨257⟩, ⟨61056⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256510

namespace LeftMerge256524
def owner : Owner := ⟨.program ⟨257⟩, ⟨60599⟩⟩
def mergeEvent : Nat := 256524
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60596⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events1002.exact256518RawTerms
def group : MergeGroup := .operator 251495 256518
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 256518) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60596⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60596⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256524

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
