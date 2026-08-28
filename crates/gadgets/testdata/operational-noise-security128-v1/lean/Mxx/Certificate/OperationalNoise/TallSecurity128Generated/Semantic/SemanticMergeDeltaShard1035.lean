import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge169510
def owner : Owner := ⟨.program ⟨257⟩, ⟨53641⟩⟩
def mergeEvent : Nat := 169510
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }
def leftRaw : List Term := Proof.Events662.exact169506RawTerms
def rightRaw : List Term := Proof.Events662.exact169476RawTerms
def group : MergeGroup := .operator 169506 169476
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169506) (leftOrdinal := 1)
    (rightResult := 169476) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge169510

namespace LeftMerge169518
def owner : Owner := ⟨.program ⟨257⟩, ⟨55544⟩⟩
def mergeEvent : Nat := 169518
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨24818⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55543⟩⟩] } }
def leftRaw : List Term := Proof.Events662.exact169512RawTerms
def rightRaw : List Term := Proof.Events661.exact169448RawTerms
def group : MergeGroup := .operator 169512 169448
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169512) (leftOrdinal := 1)
    (rightResult := 169448) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨24818⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55543⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨24818⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55543⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge169518

namespace LeftMerge169520
def owner : Owner := ⟨.program ⟨257⟩, ⟨55544⟩⟩
def mergeEvent : Nat := 169520
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨24818⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55013⟩⟩] } }
def rhsRaw : List Term := Proof.Events661.exact169445RawTerms
def group : MergeGroup := .relation 169519
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 169519) (rhsResult := 169445)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨24818⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55543⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55543⟩⟩) ⟨55013⟩ 169445) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55013⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨24818⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], [⟨.program ⟨257⟩, ⟨55013⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge169520

namespace LeftMerge169521
def owner : Owner := ⟨.program ⟨257⟩, ⟨55544⟩⟩
def mergeEvent : Nat := 169521
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55543⟩⟩] } }
def leftRaw : List Term := Proof.Events662.exact169512RawTerms
def rightRaw : List Term := Proof.Events661.exact169448RawTerms
def group : MergeGroup := .operator 169512 169448
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169512) (leftOrdinal := 0)
    (rightResult := 169448) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55543⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55543⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge169521

namespace LeftMerge169535
def owner : Owner := ⟨.program ⟨257⟩, ⟨54472⟩⟩
def mergeEvent : Nat := 169535
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54469⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events662.exact169529RawTerms
def group : MergeGroup := .operator 163745 169529
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 169529) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge169535

namespace LeftMerge169614
def owner : Owner := ⟨.program ⟨257⟩, ⟨53634⟩⟩
def mergeEvent : Nat := 169614
def frameStart : Nat := 169584
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24818⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events662.exact169610RawTerms
def rightRaw : List Term := Proof.Events662.exact169607RawTerms
def group : MergeGroup := .operator 169610 169607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169610) (leftOrdinal := 0)
    (rightResult := 169607) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53633⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24818⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24818⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge169614

namespace LeftMerge169644
def owner : Owner := ⟨.program ⟨257⟩, ⟨55284⟩⟩
def mergeEvent : Nat := 169644
def frameStart : Nat := 169584
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24818⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events662.exact169640RawTerms
def rightRaw : List Term := Proof.Events662.exact169638RawTerms
def group : MergeGroup := .operator 169640 169638
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169640) (leftOrdinal := 0)
    (rightResult := 169638) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24818⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24818⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge169644

namespace LeftMerge169667
def owner : Owner := ⟨.program ⟨257⟩, ⟨9531⟩⟩
def mergeEvent : Nat := 169667
def frameStart : Nat := 169584
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }
def leftRaw : List Term := Proof.Events662.exact169663RawTerms
def rightRaw : List Term := Proof.Events662.exact169660RawTerms
def group : MergeGroup := .operator 169663 169660
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169663) (leftOrdinal := 0)
    (rightResult := 169660) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9529⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge169667

namespace LeftMerge169676
def owner : Owner := ⟨.program ⟨257⟩, ⟨55546⟩⟩
def mergeEvent : Nat := 169676
def frameStart : Nat := 169584
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55543⟩⟩] } }
def leftRaw : List Term := Proof.Events662.exact169672RawTerms
def rightRaw : List Term := Proof.Events662.exact169629RawTerms
def group : MergeGroup := .operator 169672 169629
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169672) (leftOrdinal := 0)
    (rightResult := 169629) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55543⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55543⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge169676

namespace LeftMerge169677
def owner : Owner := ⟨.program ⟨257⟩, ⟨55546⟩⟩
def mergeEvent : Nat := 169677
def frameStart : Nat := 169584
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24818⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55543⟩⟩] } }
def leftRaw : List Term := Proof.Events662.exact169672RawTerms
def rightRaw : List Term := Proof.Events662.exact169629RawTerms
def group : MergeGroup := .operator 169672 169629
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169672) (leftOrdinal := 1)
    (rightResult := 169629) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24818⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55543⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24818⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55543⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge169677

namespace LeftMerge169679
def owner : Owner := ⟨.program ⟨257⟩, ⟨55546⟩⟩
def mergeEvent : Nat := 169679
def frameStart : Nat := 169584
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24818⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55013⟩⟩] } }
def rhsRaw : List Term := Proof.Events662.exact169626RawTerms
def group : MergeGroup := .relation 169678
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 169678) (rhsResult := 169626)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24818⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55543⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55543⟩⟩) ⟨55013⟩ 169626) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55013⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24818⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], [⟨.program ⟨257⟩, ⟨55013⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge169679

namespace LeftMerge169687
def owner : Owner := ⟨.program ⟨257⟩, ⟨53902⟩⟩
def mergeEvent : Nat := 169687
def frameStart : Nat := 169584
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events662.exact169640RawTerms
def rightRaw : List Term := Proof.Events662.exact169683RawTerms
def group : MergeGroup := .operator 169640 169683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169640) (leftOrdinal := 0)
    (rightResult := 169683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53900⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53900⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge169687

namespace LeftMerge169704
def owner : Owner := ⟨.program ⟨257⟩, ⟨54472⟩⟩
def mergeEvent : Nat := 169704
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }
def rhsRaw : List Term := Proof.Events662.exact169701RawTerms
def group : MergeGroup := .relation 169703
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 169703) (rhsResult := 169701)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54469⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 169702 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54469⟩⟩]⟩) (none) 169701) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge169704

namespace LeftMerge169705
def owner : Owner := ⟨.program ⟨257⟩, ⟨54472⟩⟩
def mergeEvent : Nat := 169705
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55543⟩⟩] } }
def rhsRaw : List Term := Proof.Events662.exact169701RawTerms
def group : MergeGroup := .relation 169703
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 169703) (rhsResult := 169701)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54469⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 169702 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54469⟩⟩]⟩) (none) 169701) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55543⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55543⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge169705

namespace LeftMerge169706
def owner : Owner := ⟨.program ⟨257⟩, ⟨54472⟩⟩
def mergeEvent : Nat := 169706
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨24818⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55013⟩⟩] } }
def rhsRaw : List Term := Proof.Events662.exact169701RawTerms
def group : MergeGroup := .relation 169703
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 169703) (rhsResult := 169701)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54469⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 169702 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54469⟩⟩]⟩) (none) 169701) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24818⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55013⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨24818⟩⟩, ⟨.program ⟨257⟩, ⟨53633⟩⟩], [⟨.program ⟨257⟩, ⟨55013⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge169706

namespace LeftMerge169707
def owner : Owner := ⟨.program ⟨257⟩, ⟨54472⟩⟩
def mergeEvent : Nat := 169707
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨53900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events662.exact169701RawTerms
def group : MergeGroup := .relation 169703
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 169703) (rhsResult := 169701)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54469⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 169702 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54469⟩⟩]⟩) (none) 169701) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨53900⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge169707

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
