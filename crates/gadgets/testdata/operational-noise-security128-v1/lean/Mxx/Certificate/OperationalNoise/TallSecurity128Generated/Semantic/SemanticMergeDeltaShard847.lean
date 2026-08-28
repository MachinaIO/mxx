import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge139492
def owner : Owner := ⟨.program ⟨257⟩, ⟨60322⟩⟩
def mergeEvent : Nat := 139492
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25166⟩⟩, ⟨.program ⟨257⟩, ⟨59296⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60907⟩⟩] } }
def rhsRaw : List Term := Proof.Events544.exact139487RawTerms
def group : MergeGroup := .relation 139489
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 139489) (rhsResult := 139487)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60319⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 139488 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60319⟩⟩]⟩) (none) 139487) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25166⟩⟩, ⟨.program ⟨257⟩, ⟨59296⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60907⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25166⟩⟩, ⟨.program ⟨257⟩, ⟨59296⟩⟩], [⟨.program ⟨257⟩, ⟨60907⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139492

namespace LeftMerge139493
def owner : Owner := ⟨.program ⟨257⟩, ⟨60322⟩⟩
def mergeEvent : Nat := 139493
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events544.exact139487RawTerms
def group : MergeGroup := .relation 139489
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 139489) (rhsResult := 139487)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60319⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 139488 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60319⟩⟩]⟩) (none) 139487) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge139493

namespace LeftMerge139498
def owner : Owner := ⟨.program ⟨257⟩, ⟨61384⟩⟩
def mergeEvent : Nat := 139498
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25166⟩⟩, ⟨.program ⟨257⟩, ⟨59296⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60907⟩⟩] } }
def leftRaw : List Term := Proof.Events544.exact139494RawTerms
def rightRaw : List Term := Proof.Events544.exact139308RawTerms
def group : MergeGroup := .operator 139494 139308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139494) (leftOrdinal := 2)
    (rightResult := 139308) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25166⟩⟩, ⟨.program ⟨257⟩, ⟨59296⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60907⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25166⟩⟩, ⟨.program ⟨257⟩, ⟨59296⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60907⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25166⟩⟩, ⟨.program ⟨257⟩, ⟨59296⟩⟩], [⟨.program ⟨257⟩, ⟨60907⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge139498

namespace LeftMerge139499
def owner : Owner := ⟨.program ⟨257⟩, ⟨61384⟩⟩
def mergeEvent : Nat := 139499
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61382⟩⟩] } }
def leftRaw : List Term := Proof.Events544.exact139494RawTerms
def rightRaw : List Term := Proof.Events544.exact139308RawTerms
def group : MergeGroup := .operator 139494 139308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139494) (leftOrdinal := 1)
    (rightResult := 139308) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61382⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61382⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61382⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139499

namespace LeftMerge139507
def owner : Owner := ⟨.program ⟨257⟩, ⟨61677⟩⟩
def mergeEvent : Nat := 139507
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61675⟩⟩] } }
def leftRaw : List Term := Proof.Events544.exact139501RawTerms
def rightRaw : List Term := Proof.Events543.exact139224RawTerms
def group : MergeGroup := .operator 139501 139224
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139501) (leftOrdinal := 0)
    (rightResult := 139224) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61675⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61675⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139507

namespace LeftMerge139508
def owner : Owner := ⟨.program ⟨257⟩, ⟨61677⟩⟩
def mergeEvent : Nat := 139508
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61675⟩⟩] } }
def leftRaw : List Term := Proof.Events544.exact139501RawTerms
def rightRaw : List Term := Proof.Events543.exact139224RawTerms
def group : MergeGroup := .operator 139501 139224
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139501) (leftOrdinal := 1)
    (rightResult := 139224) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61675⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61675⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge139508

namespace LeftMerge139510
def owner : Owner := ⟨.program ⟨257⟩, ⟨61677⟩⟩
def mergeEvent : Nat := 139510
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61038⟩⟩] } }
def rhsRaw : List Term := Proof.Events543.exact139221RawTerms
def group : MergeGroup := .relation 139509
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 139509) (rhsResult := 139221)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61675⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61675⟩⟩) ⟨61038⟩ 139221) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61038⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59772⟩⟩], [⟨.program ⟨257⟩, ⟨61038⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge139510

namespace LeftMerge139524
def owner : Owner := ⟨.program ⟨257⟩, ⟨60559⟩⟩
def mergeEvent : Nat := 139524
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60556⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134495RawTerms
def rightRaw : List Term := Proof.Events544.exact139518RawTerms
def group : MergeGroup := .operator 134495 139518
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134495) (leftOrdinal := 0)
    (rightResult := 139518) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60556⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139524

namespace LeftMerge139645
def owner : Owner := ⟨.program ⟨257⟩, ⟨61280⟩⟩
def mergeEvent : Nat := 139645
def frameStart : Nat := 139579
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events545.exact139641RawTerms
def rightRaw : List Term := Proof.Events545.exact139639RawTerms
def group : MergeGroup := .operator 139641 139639
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139641) (leftOrdinal := 0)
    (rightResult := 139639) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59772⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139645

namespace LeftMerge139657
def owner : Owner := ⟨.program ⟨257⟩, ⟨61676⟩⟩
def mergeEvent : Nat := 139657
def frameStart : Nat := 139579
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61675⟩⟩] } }
def leftRaw : List Term := Proof.Events545.exact139653RawTerms
def rightRaw : List Term := Proof.Events545.exact139630RawTerms
def group : MergeGroup := .operator 139653 139630
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139653) (leftOrdinal := 0)
    (rightResult := 139630) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61675⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61675⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139657

namespace LeftMerge139658
def owner : Owner := ⟨.program ⟨257⟩, ⟨61676⟩⟩
def mergeEvent : Nat := 139658
def frameStart : Nat := 139579
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61675⟩⟩] } }
def leftRaw : List Term := Proof.Events545.exact139653RawTerms
def rightRaw : List Term := Proof.Events545.exact139630RawTerms
def group : MergeGroup := .operator 139653 139630
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139653) (leftOrdinal := 1)
    (rightResult := 139630) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61675⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61675⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge139658

namespace LeftMerge139660
def owner : Owner := ⟨.program ⟨257⟩, ⟨61676⟩⟩
def mergeEvent : Nat := 139660
def frameStart : Nat := 139579
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61038⟩⟩] } }
def rhsRaw : List Term := Proof.Events545.exact139627RawTerms
def group : MergeGroup := .relation 139659
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 139659) (rhsResult := 139627)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨59772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61675⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61675⟩⟩) ⟨61038⟩ 139627) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61038⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨59772⟩⟩], [⟨.program ⟨257⟩, ⟨61038⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge139660

namespace LeftMerge139668
def owner : Owner := ⟨.program ⟨257⟩, ⟨59970⟩⟩
def mergeEvent : Nat := 139668
def frameStart : Nat := 139579
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59968⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events545.exact139641RawTerms
def rightRaw : List Term := Proof.Events545.exact139664RawTerms
def group : MergeGroup := .operator 139641 139664
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139641) (leftOrdinal := 0)
    (rightResult := 139664) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59968⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139668

namespace LeftMerge139685
def owner : Owner := ⟨.program ⟨257⟩, ⟨60559⟩⟩
def mergeEvent : Nat := 139685
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }
def rhsRaw : List Term := Proof.Events545.exact139682RawTerms
def group : MergeGroup := .relation 139684
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 139684) (rhsResult := 139682)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60556⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 139683 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60556⟩⟩]⟩) (none) 139682) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139685

namespace LeftMerge139686
def owner : Owner := ⟨.program ⟨257⟩, ⟨60559⟩⟩
def mergeEvent : Nat := 139686
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61675⟩⟩] } }
def rhsRaw : List Term := Proof.Events545.exact139682RawTerms
def group : MergeGroup := .relation 139684
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 139684) (rhsResult := 139682)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60556⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 139683 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60556⟩⟩]⟩) (none) 139682) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61675⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61675⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge139686

namespace LeftMerge139687
def owner : Owner := ⟨.program ⟨257⟩, ⟨60559⟩⟩
def mergeEvent : Nat := 139687
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61038⟩⟩] } }
def rhsRaw : List Term := Proof.Events545.exact139682RawTerms
def group : MergeGroup := .relation 139684
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 139684) (rhsResult := 139682)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60556⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 139683 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60556⟩⟩]⟩) (none) 139682) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61038⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59772⟩⟩], [⟨.program ⟨257⟩, ⟨61038⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139687

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
