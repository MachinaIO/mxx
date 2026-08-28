import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge63769
def owner : Owner := ⟨.program ⟨257⟩, ⟨36337⟩⟩
def mergeEvent : Nat := 63769
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36336⟩⟩] } }
def leftRaw : List Term := Proof.Events249.exact63763RawTerms
def rightRaw : List Term := Proof.Events248.exact63699RawTerms
def group : MergeGroup := .operator 63763 63699
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63763) (leftOrdinal := 1)
    (rightResult := 63699) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36336⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36336⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63769

namespace LeftMerge63771
def owner : Owner := ⟨.program ⟨257⟩, ⟨36337⟩⟩
def mergeEvent : Nat := 63771
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35791⟩⟩] } }
def rhsRaw : List Term := Proof.Events248.exact63696RawTerms
def group : MergeGroup := .relation 63770
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63770) (rhsResult := 63696)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36336⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36336⟩⟩) ⟨35791⟩ 63696) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35791⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], [⟨.program ⟨257⟩, ⟨35791⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63771

namespace LeftMerge63772
def owner : Owner := ⟨.program ⟨257⟩, ⟨36337⟩⟩
def mergeEvent : Nat := 63772
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36336⟩⟩] } }
def leftRaw : List Term := Proof.Events249.exact63763RawTerms
def rightRaw : List Term := Proof.Events248.exact63699RawTerms
def group : MergeGroup := .operator 63763 63699
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63763) (leftOrdinal := 0)
    (rightResult := 63699) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36336⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36336⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63772

namespace LeftMerge63786
def owner : Owner := ⟨.program ⟨257⟩, ⟨35262⟩⟩
def mergeEvent : Nat := 63786
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35259⟩⟩] } }
def leftRaw : List Term := Proof.Events239.exact61370RawTerms
def rightRaw : List Term := Proof.Events249.exact63780RawTerms
def group : MergeGroup := .operator 61370 63780
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61370) (leftOrdinal := 0)
    (rightResult := 63780) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35259⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63786

namespace LeftMerge63865
def owner : Owner := ⟨.program ⟨257⟩, ⟨34603⟩⟩
def mergeEvent : Nat := 63865
def frameStart : Nat := 63835
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events249.exact63861RawTerms
def rightRaw : List Term := Proof.Events249.exact63858RawTerms
def group : MergeGroup := .operator 63861 63858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63861) (leftOrdinal := 0)
    (rightResult := 63858) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13686⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34602⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63865

namespace LeftMerge63895
def owner : Owner := ⟨.program ⟨257⟩, ⟨36056⟩⟩
def mergeEvent : Nat := 63895
def frameStart : Nat := 63835
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events249.exact63891RawTerms
def rightRaw : List Term := Proof.Events249.exact63889RawTerms
def group : MergeGroup := .operator 63891 63889
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63891) (leftOrdinal := 0)
    (rightResult := 63889) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63895

namespace LeftMerge63918
def owner : Owner := ⟨.program ⟨257⟩, ⟨9552⟩⟩
def mergeEvent : Nat := 63918
def frameStart : Nat := 63835
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }
def leftRaw : List Term := Proof.Events249.exact63914RawTerms
def rightRaw : List Term := Proof.Events249.exact63911RawTerms
def group : MergeGroup := .operator 63914 63911
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63914) (leftOrdinal := 0)
    (rightResult := 63911) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9550⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63918

namespace LeftMerge63927
def owner : Owner := ⟨.program ⟨257⟩, ⟨36339⟩⟩
def mergeEvent : Nat := 63927
def frameStart : Nat := 63835
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36336⟩⟩] } }
def leftRaw : List Term := Proof.Events249.exact63923RawTerms
def rightRaw : List Term := Proof.Events249.exact63880RawTerms
def group : MergeGroup := .operator 63923 63880
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63923) (leftOrdinal := 0)
    (rightResult := 63880) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36336⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36336⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63927

namespace LeftMerge63928
def owner : Owner := ⟨.program ⟨257⟩, ⟨36339⟩⟩
def mergeEvent : Nat := 63928
def frameStart : Nat := 63835
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36336⟩⟩] } }
def leftRaw : List Term := Proof.Events249.exact63923RawTerms
def rightRaw : List Term := Proof.Events249.exact63880RawTerms
def group : MergeGroup := .operator 63923 63880
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63923) (leftOrdinal := 1)
    (rightResult := 63880) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36336⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36336⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63928

namespace LeftMerge63930
def owner : Owner := ⟨.program ⟨257⟩, ⟨36339⟩⟩
def mergeEvent : Nat := 63930
def frameStart : Nat := 63835
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35791⟩⟩] } }
def rhsRaw : List Term := Proof.Events249.exact63877RawTerms
def group : MergeGroup := .relation 63929
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63929) (rhsResult := 63877)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36336⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36336⟩⟩) ⟨35791⟩ 63877) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35791⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], [⟨.program ⟨257⟩, ⟨35791⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63930

namespace LeftMerge63938
def owner : Owner := ⟨.program ⟨257⟩, ⟨34806⟩⟩
def mergeEvent : Nat := 63938
def frameStart : Nat := 63835
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events249.exact63891RawTerms
def rightRaw : List Term := Proof.Events249.exact63934RawTerms
def group : MergeGroup := .operator 63891 63934
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63891) (leftOrdinal := 0)
    (rightResult := 63934) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34804⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63938

namespace LeftMerge63955
def owner : Owner := ⟨.program ⟨257⟩, ⟨35262⟩⟩
def mergeEvent : Nat := 63955
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }
def rhsRaw : List Term := Proof.Events249.exact63952RawTerms
def group : MergeGroup := .relation 63954
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63954) (rhsResult := 63952)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35259⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 63953 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35259⟩⟩]⟩) (none) 63952) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63955

namespace LeftMerge63956
def owner : Owner := ⟨.program ⟨257⟩, ⟨35262⟩⟩
def mergeEvent : Nat := 63956
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36336⟩⟩] } }
def rhsRaw : List Term := Proof.Events249.exact63952RawTerms
def group : MergeGroup := .relation 63954
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63954) (rhsResult := 63952)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35259⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 63953 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35259⟩⟩]⟩) (none) 63952) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36336⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36336⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63956

namespace LeftMerge63957
def owner : Owner := ⟨.program ⟨257⟩, ⟨35262⟩⟩
def mergeEvent : Nat := 63957
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35791⟩⟩] } }
def rhsRaw : List Term := Proof.Events249.exact63952RawTerms
def group : MergeGroup := .relation 63954
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63954) (rhsResult := 63952)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35259⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 63953 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35259⟩⟩]⟩) (none) 63952) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35791⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], [⟨.program ⟨257⟩, ⟨35791⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63957

namespace LeftMerge63958
def owner : Owner := ⟨.program ⟨257⟩, ⟨35262⟩⟩
def mergeEvent : Nat := 63958
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨34804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events249.exact63952RawTerms
def group : MergeGroup := .relation 63954
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63954) (rhsResult := 63952)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35259⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 63953 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35259⟩⟩]⟩) (none) 63952) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨34804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63958

namespace LeftMerge63963
def owner : Owner := ⟨.program ⟨257⟩, ⟨36338⟩⟩
def mergeEvent : Nat := 63963
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35791⟩⟩] } }
def leftRaw : List Term := Proof.Events249.exact63959RawTerms
def rightRaw : List Term := Proof.Events249.exact63773RawTerms
def group : MergeGroup := .operator 63959 63773
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63959) (leftOrdinal := 2)
    (rightResult := 63773) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35791⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35791⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], [⟨.program ⟨257⟩, ⟨35791⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63963

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
