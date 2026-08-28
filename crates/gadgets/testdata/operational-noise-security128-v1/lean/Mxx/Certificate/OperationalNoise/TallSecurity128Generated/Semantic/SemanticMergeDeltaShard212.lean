import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge38544
def owner : Owner := ⟨.program ⟨257⟩, ⟨50962⟩⟩
def mergeEvent : Nat := 38544
def frameStart : Nat := 38441
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50960⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events150.exact38497RawTerms
def rightRaw : List Term := Proof.Events150.exact38540RawTerms
def group : MergeGroup := .operator 38497 38540
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38497) (leftOrdinal := 0)
    (rightResult := 38540) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50960⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38544

namespace LeftMerge38561
def owner : Owner := ⟨.program ⟨257⟩, ⟨51542⟩⟩
def mergeEvent : Nat := 38561
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }
def rhsRaw : List Term := Proof.Events150.exact38558RawTerms
def group : MergeGroup := .relation 38560
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38560) (rhsResult := 38558)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51539⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 38559 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51539⟩⟩]⟩) (none) 38558) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38561

namespace LeftMerge38562
def owner : Owner := ⟨.program ⟨257⟩, ⟨51542⟩⟩
def mergeEvent : Nat := 38562
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52618⟩⟩] } }
def rhsRaw : List Term := Proof.Events150.exact38558RawTerms
def group : MergeGroup := .relation 38560
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38560) (rhsResult := 38558)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51539⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 38559 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51539⟩⟩]⟩) (none) 38558) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52618⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52618⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38562

namespace LeftMerge38563
def owner : Owner := ⟨.program ⟨257⟩, ⟨51542⟩⟩
def mergeEvent : Nat := 38563
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24638⟩⟩, ⟨.program ⟨257⟩, ⟨50788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52063⟩⟩] } }
def rhsRaw : List Term := Proof.Events150.exact38558RawTerms
def group : MergeGroup := .relation 38560
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38560) (rhsResult := 38558)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51539⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 38559 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51539⟩⟩]⟩) (none) 38558) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24638⟩⟩, ⟨.program ⟨257⟩, ⟨50788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52063⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24638⟩⟩, ⟨.program ⟨257⟩, ⟨50788⟩⟩], [⟨.program ⟨257⟩, ⟨52063⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38563

namespace LeftMerge38564
def owner : Owner := ⟨.program ⟨257⟩, ⟨51542⟩⟩
def mergeEvent : Nat := 38564
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨50960⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events150.exact38558RawTerms
def group : MergeGroup := .relation 38560
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38560) (rhsResult := 38558)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51539⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 38559 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51539⟩⟩]⟩) (none) 38558) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50960⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38564

namespace LeftMerge38569
def owner : Owner := ⟨.program ⟨257⟩, ⟨52620⟩⟩
def mergeEvent : Nat := 38569
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24638⟩⟩, ⟨.program ⟨257⟩, ⟨50788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52063⟩⟩] } }
def leftRaw : List Term := Proof.Events150.exact38565RawTerms
def rightRaw : List Term := Proof.Events149.exact38379RawTerms
def group : MergeGroup := .operator 38565 38379
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38565) (leftOrdinal := 2)
    (rightResult := 38379) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24638⟩⟩, ⟨.program ⟨257⟩, ⟨50788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52063⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24638⟩⟩, ⟨.program ⟨257⟩, ⟨50788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52063⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24638⟩⟩, ⟨.program ⟨257⟩, ⟨50788⟩⟩], [⟨.program ⟨257⟩, ⟨52063⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38569

namespace LeftMerge38570
def owner : Owner := ⟨.program ⟨257⟩, ⟨52620⟩⟩
def mergeEvent : Nat := 38570
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52618⟩⟩] } }
def leftRaw : List Term := Proof.Events150.exact38565RawTerms
def rightRaw : List Term := Proof.Events149.exact38379RawTerms
def group : MergeGroup := .operator 38565 38379
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38565) (leftOrdinal := 1)
    (rightResult := 38379) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52618⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52618⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52618⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38570

namespace LeftMerge38578
def owner : Owner := ⟨.program ⟨257⟩, ⟨53233⟩⟩
def mergeEvent : Nat := 38578
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53231⟩⟩] } }
def leftRaw : List Term := Proof.Events150.exact38572RawTerms
def rightRaw : List Term := Proof.Events149.exact38295RawTerms
def group : MergeGroup := .operator 38572 38295
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38572) (leftOrdinal := 0)
    (rightResult := 38295) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53231⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53231⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38578

namespace LeftMerge38579
def owner : Owner := ⟨.program ⟨257⟩, ⟨53233⟩⟩
def mergeEvent : Nat := 38579
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨50960⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53231⟩⟩] } }
def leftRaw : List Term := Proof.Events150.exact38572RawTerms
def rightRaw : List Term := Proof.Events149.exact38295RawTerms
def group : MergeGroup := .operator 38572 38295
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38572) (leftOrdinal := 1)
    (rightResult := 38295) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨50960⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53231⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53231⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38579

namespace LeftMerge38581
def owner : Owner := ⟨.program ⟨257⟩, ⟨53233⟩⟩
def mergeEvent : Nat := 38581
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨50960⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52242⟩⟩] } }
def rhsRaw : List Term := Proof.Events149.exact38292RawTerms
def group : MergeGroup := .relation 38580
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38580) (rhsResult := 38292)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53231⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨53231⟩⟩) ⟨52242⟩ 38292) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52242⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨52242⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38581

namespace LeftMerge38595
def owner : Owner := ⟨.program ⟨257⟩, ⟨51939⟩⟩
def mergeEvent : Nat := 38595
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51936⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32120RawTerms
def rightRaw : List Term := Proof.Events150.exact38589RawTerms
def group : MergeGroup := .operator 32120 38589
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32120) (leftOrdinal := 0)
    (rightResult := 38589) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51936⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51936⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38595

namespace LeftMerge38716
def owner : Owner := ⟨.program ⟨257⟩, ⟨52404⟩⟩
def mergeEvent : Nat := 38716
def frameStart : Nat := 38650
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50960⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events151.exact38712RawTerms
def rightRaw : List Term := Proof.Events151.exact38710RawTerms
def group : MergeGroup := .operator 38712 38710
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38712) (leftOrdinal := 0)
    (rightResult := 38710) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50960⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38716

namespace LeftMerge38728
def owner : Owner := ⟨.program ⟨257⟩, ⟨53232⟩⟩
def mergeEvent : Nat := 38728
def frameStart : Nat := 38650
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53231⟩⟩] } }
def leftRaw : List Term := Proof.Events151.exact38724RawTerms
def rightRaw : List Term := Proof.Events151.exact38701RawTerms
def group : MergeGroup := .operator 38724 38701
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38724) (leftOrdinal := 0)
    (rightResult := 38701) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53231⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53231⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38728

namespace LeftMerge38729
def owner : Owner := ⟨.program ⟨257⟩, ⟨53232⟩⟩
def mergeEvent : Nat := 38729
def frameStart : Nat := 38650
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50960⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53231⟩⟩] } }
def leftRaw : List Term := Proof.Events151.exact38724RawTerms
def rightRaw : List Term := Proof.Events151.exact38701RawTerms
def group : MergeGroup := .operator 38724 38701
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38724) (leftOrdinal := 1)
    (rightResult := 38701) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50960⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53231⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53231⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38729

namespace LeftMerge38731
def owner : Owner := ⟨.program ⟨257⟩, ⟨53232⟩⟩
def mergeEvent : Nat := 38731
def frameStart : Nat := 38650
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50960⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52242⟩⟩] } }
def rhsRaw : List Term := Proof.Events151.exact38698RawTerms
def group : MergeGroup := .relation 38730
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38730) (rhsResult := 38698)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53231⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨53231⟩⟩) ⟨52242⟩ 38698) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52242⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨52242⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38731

namespace LeftMerge38739
def owner : Owner := ⟨.program ⟨257⟩, ⟨51334⟩⟩
def mergeEvent : Nat := 38739
def frameStart : Nat := 38650
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51332⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events151.exact38712RawTerms
def rightRaw : List Term := Proof.Events151.exact38735RawTerms
def group : MergeGroup := .operator 38712 38735
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38712) (leftOrdinal := 0)
    (rightResult := 38735) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51332⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨51332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38739

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
