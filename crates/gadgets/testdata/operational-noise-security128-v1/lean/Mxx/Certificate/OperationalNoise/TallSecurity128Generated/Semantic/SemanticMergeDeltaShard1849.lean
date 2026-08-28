import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge298818
def owner : Owner := ⟨.program ⟨257⟩, ⟨67673⟩⟩
def mergeEvent : Nat := 298818
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }
def rhsRaw : List Term := Proof.Events1167.exact298815RawTerms
def group : MergeGroup := .relation 298817
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 298817) (rhsResult := 298815)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67670⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 298816 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67670⟩⟩]⟩) (none) 298815) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298818

namespace LeftMerge298819
def owner : Owner := ⟨.program ⟨257⟩, ⟨67673⟩⟩
def mergeEvent : Nat := 298819
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69130⟩⟩] } }
def rhsRaw : List Term := Proof.Events1167.exact298815RawTerms
def group : MergeGroup := .relation 298817
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 298817) (rhsResult := 298815)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67670⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 298816 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67670⟩⟩]⟩) (none) 298815) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69130⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69130⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge298819

namespace LeftMerge298820
def owner : Owner := ⟨.program ⟨257⟩, ⟨67673⟩⟩
def mergeEvent : Nat := 298820
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68470⟩⟩] } }
def rhsRaw : List Term := Proof.Events1167.exact298815RawTerms
def group : MergeGroup := .relation 298817
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 298817) (rhsResult := 298815)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67670⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 298816 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67670⟩⟩]⟩) (none) 298815) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68470⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], [⟨.program ⟨257⟩, ⟨68470⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298820

namespace LeftMerge298821
def owner : Owner := ⟨.program ⟨257⟩, ⟨67673⟩⟩
def mergeEvent : Nat := 298821
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1167.exact298815RawTerms
def group : MergeGroup := .relation 298817
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 298817) (rhsResult := 298815)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67670⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 298816 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67670⟩⟩]⟩) (none) 298815) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge298821

namespace LeftMerge298826
def owner : Owner := ⟨.program ⟨257⟩, ⟨69132⟩⟩
def mergeEvent : Nat := 298826
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68470⟩⟩] } }
def leftRaw : List Term := Proof.Events1167.exact298822RawTerms
def rightRaw : List Term := Proof.Events1166.exact298660RawTerms
def group : MergeGroup := .operator 298822 298660
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298822) (leftOrdinal := 2)
    (rightResult := 298660) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68470⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68470⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], [⟨.program ⟨257⟩, ⟨68470⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge298826

namespace LeftMerge298827
def owner : Owner := ⟨.program ⟨257⟩, ⟨69132⟩⟩
def mergeEvent : Nat := 298827
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69130⟩⟩] } }
def leftRaw : List Term := Proof.Events1167.exact298822RawTerms
def rightRaw : List Term := Proof.Events1166.exact298660RawTerms
def group : MergeGroup := .operator 298822 298660
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298822) (leftOrdinal := 1)
    (rightResult := 298660) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69130⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69130⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69130⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298827

namespace LeftMerge298835
def owner : Owner := ⟨.program ⟨257⟩, ⟨69389⟩⟩
def mergeEvent : Nat := 298835
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩] } }
def leftRaw : List Term := Proof.Events1167.exact298829RawTerms
def rightRaw : List Term := Proof.Events1166.exact298576RawTerms
def group : MergeGroup := .operator 298829 298576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298829) (leftOrdinal := 0)
    (rightResult := 298576) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69387⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298835

namespace LeftMerge298836
def owner : Owner := ⟨.program ⟨257⟩, ⟨69389⟩⟩
def mergeEvent : Nat := 298836
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩] } }
def leftRaw : List Term := Proof.Events1167.exact298829RawTerms
def rightRaw : List Term := Proof.Events1166.exact298576RawTerms
def group : MergeGroup := .operator 298829 298576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298829) (leftOrdinal := 1)
    (rightResult := 298576) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69387⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge298836

namespace LeftMerge298838
def owner : Owner := ⟨.program ⟨257⟩, ⟨69389⟩⟩
def mergeEvent : Nat := 298838
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68592⟩⟩] } }
def rhsRaw : List Term := Proof.Events1166.exact298573RawTerms
def group : MergeGroup := .relation 298837
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 298837) (rhsResult := 298573)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69387⟩⟩) ⟨68592⟩ 298573) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68592⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨68592⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge298838

namespace LeftMerge298852
def owner : Owner := ⟨.program ⟨257⟩, ⟨67880⟩⟩
def mergeEvent : Nat := 298852
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67877⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1167.exact298846RawTerms
def group : MergeGroup := .operator 295195 298846
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 298846) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨67877⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67877⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298852

namespace LeftMerge298949
def owner : Owner := ⟨.program ⟨257⟩, ⟨68969⟩⟩
def mergeEvent : Nat := 298949
def frameStart : Nat := 298895
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1167.exact298945RawTerms
def rightRaw : List Term := Proof.Events1167.exact298943RawTerms
def group : MergeGroup := .operator 298945 298943
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298945) (leftOrdinal := 0)
    (rightResult := 298943) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65708⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298949

namespace LeftMerge298961
def owner : Owner := ⟨.program ⟨257⟩, ⟨69388⟩⟩
def mergeEvent : Nat := 298961
def frameStart : Nat := 298895
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩] } }
def leftRaw : List Term := Proof.Events1167.exact298957RawTerms
def rightRaw : List Term := Proof.Events1167.exact298934RawTerms
def group : MergeGroup := .operator 298957 298934
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298957) (leftOrdinal := 0)
    (rightResult := 298934) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69387⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298961

namespace LeftMerge298962
def owner : Owner := ⟨.program ⟨257⟩, ⟨69388⟩⟩
def mergeEvent : Nat := 298962
def frameStart : Nat := 298895
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩] } }
def leftRaw : List Term := Proof.Events1167.exact298957RawTerms
def rightRaw : List Term := Proof.Events1167.exact298934RawTerms
def group : MergeGroup := .operator 298957 298934
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298957) (leftOrdinal := 1)
    (rightResult := 298934) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69387⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge298962

namespace LeftMerge298964
def owner : Owner := ⟨.program ⟨257⟩, ⟨69388⟩⟩
def mergeEvent : Nat := 298964
def frameStart : Nat := 298895
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68592⟩⟩] } }
def rhsRaw : List Term := Proof.Events1167.exact298931RawTerms
def group : MergeGroup := .relation 298963
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 298963) (rhsResult := 298931)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69387⟩⟩) ⟨68592⟩ 298931) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68592⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨68592⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge298964

namespace LeftMerge298972
def owner : Owner := ⟨.program ⟨257⟩, ⟨65912⟩⟩
def mergeEvent : Nat := 298972
def frameStart : Nat := 298895
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65901⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1167.exact298945RawTerms
def rightRaw : List Term := Proof.Events1167.exact298968RawTerms
def group : MergeGroup := .operator 298945 298968
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298945) (leftOrdinal := 0)
    (rightResult := 298968) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65901⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298972

namespace LeftMerge298989
def owner : Owner := ⟨.program ⟨257⟩, ⟨67880⟩⟩
def mergeEvent : Nat := 298989
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }
def rhsRaw : List Term := Proof.Events1167.exact298986RawTerms
def group : MergeGroup := .relation 298988
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 298988) (rhsResult := 298986)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67877⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 298987 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67877⟩⟩]⟩) (none) 298986) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298989

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
