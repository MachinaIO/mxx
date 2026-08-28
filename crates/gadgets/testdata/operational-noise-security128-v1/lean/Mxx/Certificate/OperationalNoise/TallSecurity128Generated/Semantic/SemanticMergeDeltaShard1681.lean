import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge272544
def owner : Owner := ⟨.program ⟨257⟩, ⟨50824⟩⟩
def mergeEvent : Nat := 272544
def frameStart : Nat := 272441
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50822⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1064.exact272497RawTerms
def rightRaw : List Term := Proof.Events1064.exact272540RawTerms
def group : MergeGroup := .operator 272497 272540
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272497) (leftOrdinal := 0)
    (rightResult := 272540) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50822⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50822⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272544

namespace LeftMerge272561
def owner : Owner := ⟨.program ⟨257⟩, ⟨51369⟩⟩
def mergeEvent : Nat := 272561
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }
def rhsRaw : List Term := Proof.Events1064.exact272558RawTerms
def group : MergeGroup := .relation 272560
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272560) (rhsResult := 272558)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51366⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 272559 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51366⟩⟩]⟩) (none) 272558) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272561

namespace LeftMerge272562
def owner : Owner := ⟨.program ⟨257⟩, ⟨51369⟩⟩
def mergeEvent : Nat := 272562
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52428⟩⟩] } }
def rhsRaw : List Term := Proof.Events1064.exact272558RawTerms
def group : MergeGroup := .relation 272560
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272560) (rhsResult := 272558)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51366⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 272559 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51366⟩⟩]⟩) (none) 272558) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52428⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52428⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272562

namespace LeftMerge272563
def owner : Owner := ⟨.program ⟨257⟩, ⟨51369⟩⟩
def mergeEvent : Nat := 272563
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51959⟩⟩] } }
def rhsRaw : List Term := Proof.Events1064.exact272558RawTerms
def group : MergeGroup := .relation 272560
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272560) (rhsResult := 272558)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51366⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 272559 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51366⟩⟩]⟩) (none) 272558) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51959⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], [⟨.program ⟨257⟩, ⟨51959⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272563

namespace LeftMerge272564
def owner : Owner := ⟨.program ⟨257⟩, ⟨51369⟩⟩
def mergeEvent : Nat := 272564
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50822⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1064.exact272558RawTerms
def group : MergeGroup := .relation 272560
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272560) (rhsResult := 272558)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51366⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 272559 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51366⟩⟩]⟩) (none) 272558) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50822⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50822⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272564

namespace LeftMerge272569
def owner : Owner := ⟨.program ⟨257⟩, ⟨52430⟩⟩
def mergeEvent : Nat := 272569
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51959⟩⟩] } }
def leftRaw : List Term := Proof.Events1064.exact272565RawTerms
def rightRaw : List Term := Proof.Events1063.exact272379RawTerms
def group : MergeGroup := .operator 272565 272379
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272565) (leftOrdinal := 2)
    (rightResult := 272379) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51959⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51959⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], [⟨.program ⟨257⟩, ⟨51959⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272569

namespace LeftMerge272570
def owner : Owner := ⟨.program ⟨257⟩, ⟨52430⟩⟩
def mergeEvent : Nat := 272570
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52428⟩⟩] } }
def leftRaw : List Term := Proof.Events1064.exact272565RawTerms
def rightRaw : List Term := Proof.Events1063.exact272379RawTerms
def group : MergeGroup := .operator 272565 272379
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272565) (leftOrdinal := 1)
    (rightResult := 272379) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52428⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52428⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52428⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272570

namespace LeftMerge272578
def owner : Owner := ⟨.program ⟨257⟩, ⟨52697⟩⟩
def mergeEvent : Nat := 272578
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52695⟩⟩] } }
def leftRaw : List Term := Proof.Events1064.exact272572RawTerms
def rightRaw : List Term := Proof.Events1063.exact272295RawTerms
def group : MergeGroup := .operator 272572 272295
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272572) (leftOrdinal := 0)
    (rightResult := 272295) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52695⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52695⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272578

namespace LeftMerge272579
def owner : Owner := ⟨.program ⟨257⟩, ⟨52697⟩⟩
def mergeEvent : Nat := 272579
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50822⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52695⟩⟩] } }
def leftRaw : List Term := Proof.Events1064.exact272572RawTerms
def rightRaw : List Term := Proof.Events1063.exact272295RawTerms
def group : MergeGroup := .operator 272572 272295
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272572) (leftOrdinal := 1)
    (rightResult := 272295) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50822⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52695⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50822⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52695⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272579

namespace LeftMerge272581
def owner : Owner := ⟨.program ⟨257⟩, ⟨52697⟩⟩
def mergeEvent : Nat := 272581
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50822⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52086⟩⟩] } }
def rhsRaw : List Term := Proof.Events1063.exact272292RawTerms
def group : MergeGroup := .relation 272580
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272580) (rhsResult := 272292)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50822⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52695⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52695⟩⟩) ⟨52086⟩ 272292) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52086⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50822⟩⟩], [⟨.program ⟨257⟩, ⟨52086⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272581

namespace LeftMerge272595
def owner : Owner := ⟨.program ⟨257⟩, ⟨51593⟩⟩
def mergeEvent : Nat := 272595
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51590⟩⟩] } }
def leftRaw : List Term := Proof.Events1039.exact266120RawTerms
def rightRaw : List Term := Proof.Events1064.exact272589RawTerms
def group : MergeGroup := .operator 266120 272589
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266120) (leftOrdinal := 0)
    (rightResult := 272589) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51590⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51590⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272595

namespace LeftMerge272716
def owner : Owner := ⟨.program ⟨257⟩, ⟨52336⟩⟩
def mergeEvent : Nat := 272716
def frameStart : Nat := 272650
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50822⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1065.exact272712RawTerms
def rightRaw : List Term := Proof.Events1065.exact272710RawTerms
def group : MergeGroup := .operator 272712 272710
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272712) (leftOrdinal := 0)
    (rightResult := 272710) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50822⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50822⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272716

namespace LeftMerge272728
def owner : Owner := ⟨.program ⟨257⟩, ⟨52696⟩⟩
def mergeEvent : Nat := 272728
def frameStart : Nat := 272650
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52695⟩⟩] } }
def leftRaw : List Term := Proof.Events1065.exact272724RawTerms
def rightRaw : List Term := Proof.Events1065.exact272701RawTerms
def group : MergeGroup := .operator 272724 272701
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272724) (leftOrdinal := 0)
    (rightResult := 272701) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52695⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52695⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272728

namespace LeftMerge272729
def owner : Owner := ⟨.program ⟨257⟩, ⟨52696⟩⟩
def mergeEvent : Nat := 272729
def frameStart : Nat := 272650
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50822⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52695⟩⟩] } }
def leftRaw : List Term := Proof.Events1065.exact272724RawTerms
def rightRaw : List Term := Proof.Events1065.exact272701RawTerms
def group : MergeGroup := .operator 272724 272701
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272724) (leftOrdinal := 1)
    (rightResult := 272701) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50822⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52695⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50822⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52695⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272729

namespace LeftMerge272731
def owner : Owner := ⟨.program ⟨257⟩, ⟨52696⟩⟩
def mergeEvent : Nat := 272731
def frameStart : Nat := 272650
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50822⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52086⟩⟩] } }
def rhsRaw : List Term := Proof.Events1065.exact272698RawTerms
def group : MergeGroup := .relation 272730
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272730) (rhsResult := 272698)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨50822⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52695⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52695⟩⟩) ⟨52086⟩ 272698) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52086⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨50822⟩⟩], [⟨.program ⟨257⟩, ⟨52086⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272731

namespace LeftMerge272739
def owner : Owner := ⟨.program ⟨257⟩, ⟨51006⟩⟩
def mergeEvent : Nat := 272739
def frameStart : Nat := 272650
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51004⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1065.exact272712RawTerms
def rightRaw : List Term := Proof.Events1065.exact272735RawTerms
def group : MergeGroup := .operator 272712 272735
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272712) (leftOrdinal := 0)
    (rightResult := 272735) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51004⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272739

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
