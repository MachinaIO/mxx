import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge243569
def owner : Owner := ⟨.program ⟨257⟩, ⟨31435⟩⟩
def mergeEvent : Nat := 243569
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events045.exact11639RawTerms
def rightRaw : List Term := Proof.Events924.exact236778RawTerms
def group : MergeGroup := .operator 11639 236778
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11639) (leftOrdinal := 0)
    (rightResult := 236778) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243569

namespace LeftMerge243574
def owner : Owner := ⟨.program ⟨257⟩, ⟨8365⟩⟩
def mergeEvent : Nat := 243574
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }
def leftRaw : List Term := Proof.Events924.exact236648RawTerms
def rightRaw : List Term := Proof.Events094.exact24135RawTerms
def group : MergeGroup := .operator 236648 24135
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236648) (leftOrdinal := 0)
    (rightResult := 24135) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243574

namespace LeftMerge243591
def owner : Owner := ⟨.program ⟨257⟩, ⟨31438⟩⟩
def mergeEvent : Nat := 243591
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }
def leftRaw : List Term := Proof.Events951.exact243585RawTerms
def rightRaw : List Term := Proof.Events094.exact24124RawTerms
def group : MergeGroup := .operator 243585 24124
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243585) (leftOrdinal := 1)
    (rightResult := 24124) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9577⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243591

namespace LeftMerge243593
def owner : Owner := ⟨.program ⟨257⟩, ⟨31438⟩⟩
def mergeEvent : Nat := 243593
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }
def rhsRaw : List Term := Proof.Events094.exact24094RawTerms
def group : MergeGroup := .relation 243592
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243592) (rhsResult := 24094)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9577⟩⟩) ⟨7307⟩ 24094) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243593

namespace LeftMerge243594
def owner : Owner := ⟨.program ⟨257⟩, ⟨31438⟩⟩
def mergeEvent : Nat := 243594
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }
def leftRaw : List Term := Proof.Events951.exact243585RawTerms
def rightRaw : List Term := Proof.Events094.exact24124RawTerms
def group : MergeGroup := .operator 243585 24124
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243585) (leftOrdinal := 0)
    (rightResult := 24124) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9577⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243594

namespace LeftMerge243599
def owner : Owner := ⟨.program ⟨257⟩, ⟨31439⟩⟩
def mergeEvent : Nat := 243599
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }
def leftRaw : List Term := Proof.Events951.exact243595RawTerms
def rightRaw : List Term := Proof.Events951.exact243565RawTerms
def group : MergeGroup := .operator 243595 243565
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243595) (leftOrdinal := 1)
    (rightResult := 243565) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243599

namespace LeftMerge243607
def owner : Owner := ⟨.program ⟨257⟩, ⟨33438⟩⟩
def mergeEvent : Nat := 243607
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33437⟩⟩] } }
def leftRaw : List Term := Proof.Events951.exact243601RawTerms
def rightRaw : List Term := Proof.Events951.exact243537RawTerms
def group : MergeGroup := .operator 243601 243537
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243601) (leftOrdinal := 1)
    (rightResult := 243537) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243607

namespace LeftMerge243609
def owner : Owner := ⟨.program ⟨257⟩, ⟨33438⟩⟩
def mergeEvent : Nat := 243609
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32937⟩⟩] } }
def rhsRaw : List Term := Proof.Events951.exact243534RawTerms
def group : MergeGroup := .relation 243608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243608) (rhsResult := 243534)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33437⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33437⟩⟩) ⟨32937⟩ 243534) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32937⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], [⟨.program ⟨257⟩, ⟨32937⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243609

namespace LeftMerge243610
def owner : Owner := ⟨.program ⟨257⟩, ⟨33438⟩⟩
def mergeEvent : Nat := 243610
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33437⟩⟩] } }
def leftRaw : List Term := Proof.Events951.exact243601RawTerms
def rightRaw : List Term := Proof.Events951.exact243537RawTerms
def group : MergeGroup := .operator 243601 243537
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243601) (leftOrdinal := 0)
    (rightResult := 243537) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243610

namespace LeftMerge243624
def owner : Owner := ⟨.program ⟨257⟩, ⟨32372⟩⟩
def mergeEvent : Nat := 243624
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32369⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236870RawTerms
def rightRaw : List Term := Proof.Events951.exact243618RawTerms
def group : MergeGroup := .operator 236870 243618
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236870) (leftOrdinal := 0)
    (rightResult := 243618) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32369⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32369⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243624

namespace LeftMerge243703
def owner : Owner := ⟨.program ⟨257⟩, ⟨31432⟩⟩
def mergeEvent : Nat := 243703
def frameStart : Nat := 243673
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events951.exact243699RawTerms
def rightRaw : List Term := Proof.Events951.exact243696RawTerms
def group : MergeGroup := .operator 243699 243696
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243699) (leftOrdinal := 0)
    (rightResult := 243696) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24266⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243703

namespace LeftMerge243733
def owner : Owner := ⟨.program ⟨257⟩, ⟨33220⟩⟩
def mergeEvent : Nat := 243733
def frameStart : Nat := 243673
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events952.exact243729RawTerms
def rightRaw : List Term := Proof.Events952.exact243727RawTerms
def group : MergeGroup := .operator 243729 243727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243729) (leftOrdinal := 0)
    (rightResult := 243727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243733

namespace LeftMerge243756
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def mergeEvent : Nat := 243756
def frameStart : Nat := 243673
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }
def leftRaw : List Term := Proof.Events952.exact243752RawTerms
def rightRaw : List Term := Proof.Events952.exact243749RawTerms
def group : MergeGroup := .operator 243752 243749
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243752) (leftOrdinal := 0)
    (rightResult := 243749) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9577⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243756

namespace LeftMerge243765
def owner : Owner := ⟨.program ⟨257⟩, ⟨33440⟩⟩
def mergeEvent : Nat := 243765
def frameStart : Nat := 243673
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33437⟩⟩] } }
def leftRaw : List Term := Proof.Events952.exact243761RawTerms
def rightRaw : List Term := Proof.Events952.exact243718RawTerms
def group : MergeGroup := .operator 243761 243718
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243761) (leftOrdinal := 0)
    (rightResult := 243718) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33437⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243765

namespace LeftMerge243766
def owner : Owner := ⟨.program ⟨257⟩, ⟨33440⟩⟩
def mergeEvent : Nat := 243766
def frameStart : Nat := 243673
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33437⟩⟩] } }
def leftRaw : List Term := Proof.Events952.exact243761RawTerms
def rightRaw : List Term := Proof.Events952.exact243718RawTerms
def group : MergeGroup := .operator 243761 243718
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243761) (leftOrdinal := 1)
    (rightResult := 243718) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243766

namespace LeftMerge243768
def owner : Owner := ⟨.program ⟨257⟩, ⟨33440⟩⟩
def mergeEvent : Nat := 243768
def frameStart : Nat := 243673
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32937⟩⟩] } }
def rhsRaw : List Term := Proof.Events952.exact243715RawTerms
def group : MergeGroup := .relation 243767
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243767) (rhsResult := 243715)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33437⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33437⟩⟩) ⟨32937⟩ 243715) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32937⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], [⟨.program ⟨257⟩, ⟨32937⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243768

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
