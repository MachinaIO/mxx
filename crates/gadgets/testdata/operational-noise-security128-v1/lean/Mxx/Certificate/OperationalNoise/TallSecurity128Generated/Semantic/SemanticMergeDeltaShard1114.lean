import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge181829
def owner : Owner := ⟨.program ⟨257⟩, ⟨26167⟩⟩
def mergeEvent : Nat := 181829
def frameStart : Nat := 181799
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13026⟩⟩, ⟨.program ⟨257⟩, ⟨26166⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events710.exact181825RawTerms
def rightRaw : List Term := Proof.Events710.exact181822RawTerms
def group : MergeGroup := .operator 181825 181822
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181825) (leftOrdinal := 0)
    (rightResult := 181822) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13026⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26166⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13026⟩⟩, ⟨.program ⟨257⟩, ⟨26166⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181829

namespace LeftMerge181859
def owner : Owner := ⟨.program ⟨257⟩, ⟨27700⟩⟩
def mergeEvent : Nat := 181859
def frameStart : Nat := 181799
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13026⟩⟩, ⟨.program ⟨257⟩, ⟨26166⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events710.exact181855RawTerms
def rightRaw : List Term := Proof.Events710.exact181853RawTerms
def group : MergeGroup := .operator 181855 181853
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181855) (leftOrdinal := 0)
    (rightResult := 181853) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13026⟩⟩, ⟨.program ⟨257⟩, ⟨26166⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13026⟩⟩, ⟨.program ⟨257⟩, ⟨26166⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181859

namespace LeftMerge181882
def owner : Owner := ⟨.program ⟨257⟩, ⟨9546⟩⟩
def mergeEvent : Nat := 181882
def frameStart : Nat := 181799
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }
def leftRaw : List Term := Proof.Events710.exact181878RawTerms
def rightRaw : List Term := Proof.Events710.exact181875RawTerms
def group : MergeGroup := .operator 181878 181875
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181878) (leftOrdinal := 0)
    (rightResult := 181875) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9544⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181882

namespace LeftMerge181891
def owner : Owner := ⟨.program ⟨257⟩, ⟨27955⟩⟩
def mergeEvent : Nat := 181891
def frameStart : Nat := 181799
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27952⟩⟩] } }
def leftRaw : List Term := Proof.Events710.exact181887RawTerms
def rightRaw : List Term := Proof.Events710.exact181844RawTerms
def group : MergeGroup := .operator 181887 181844
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181887) (leftOrdinal := 0)
    (rightResult := 181844) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27952⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27952⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181891

namespace LeftMerge181892
def owner : Owner := ⟨.program ⟨257⟩, ⟨27955⟩⟩
def mergeEvent : Nat := 181892
def frameStart : Nat := 181799
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13026⟩⟩, ⟨.program ⟨257⟩, ⟨26166⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27952⟩⟩] } }
def leftRaw : List Term := Proof.Events710.exact181887RawTerms
def rightRaw : List Term := Proof.Events710.exact181844RawTerms
def group : MergeGroup := .operator 181887 181844
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181887) (leftOrdinal := 1)
    (rightResult := 181844) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13026⟩⟩, ⟨.program ⟨257⟩, ⟨26166⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27952⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13026⟩⟩, ⟨.program ⟨257⟩, ⟨26166⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27952⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge181892

namespace LeftMerge181894
def owner : Owner := ⟨.program ⟨257⟩, ⟨27955⟩⟩
def mergeEvent : Nat := 181894
def frameStart : Nat := 181799
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13026⟩⟩, ⟨.program ⟨257⟩, ⟨26166⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27427⟩⟩] } }
def rhsRaw : List Term := Proof.Events710.exact181841RawTerms
def group : MergeGroup := .relation 181893
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 181893) (rhsResult := 181841)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13026⟩⟩, ⟨.program ⟨257⟩, ⟨26166⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27952⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27952⟩⟩) ⟨27427⟩ 181841) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27427⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13026⟩⟩, ⟨.program ⟨257⟩, ⟨26166⟩⟩], [⟨.program ⟨257⟩, ⟨27427⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge181894

namespace LeftMerge181902
def owner : Owner := ⟨.program ⟨257⟩, ⟨26434⟩⟩
def mergeEvent : Nat := 181902
def frameStart : Nat := 181799
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26432⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events710.exact181855RawTerms
def rightRaw : List Term := Proof.Events710.exact181898RawTerms
def group : MergeGroup := .operator 181855 181898
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181855) (leftOrdinal := 0)
    (rightResult := 181898) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26432⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26432⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181902

namespace LeftMerge181919
def owner : Owner := ⟨.program ⟨257⟩, ⟨26882⟩⟩
def mergeEvent : Nat := 181919
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }
def rhsRaw : List Term := Proof.Events710.exact181916RawTerms
def group : MergeGroup := .relation 181918
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 181918) (rhsResult := 181916)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26879⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 181917 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26879⟩⟩]⟩) (none) 181916) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181919

namespace LeftMerge181920
def owner : Owner := ⟨.program ⟨257⟩, ⟨26882⟩⟩
def mergeEvent : Nat := 181920
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27952⟩⟩] } }
def rhsRaw : List Term := Proof.Events710.exact181916RawTerms
def group : MergeGroup := .relation 181918
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 181918) (rhsResult := 181916)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26879⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 181917 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26879⟩⟩]⟩) (none) 181916) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27952⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27952⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge181920

namespace LeftMerge181921
def owner : Owner := ⟨.program ⟨257⟩, ⟨26882⟩⟩
def mergeEvent : Nat := 181921
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13026⟩⟩, ⟨.program ⟨257⟩, ⟨26166⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27427⟩⟩] } }
def rhsRaw : List Term := Proof.Events710.exact181916RawTerms
def group : MergeGroup := .relation 181918
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 181918) (rhsResult := 181916)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26879⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 181917 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26879⟩⟩]⟩) (none) 181916) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13026⟩⟩, ⟨.program ⟨257⟩, ⟨26166⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27427⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13026⟩⟩, ⟨.program ⟨257⟩, ⟨26166⟩⟩], [⟨.program ⟨257⟩, ⟨27427⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181921

namespace LeftMerge181922
def owner : Owner := ⟨.program ⟨257⟩, ⟨26882⟩⟩
def mergeEvent : Nat := 181922
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26432⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events710.exact181916RawTerms
def group : MergeGroup := .relation 181918
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 181918) (rhsResult := 181916)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26879⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 181917 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26879⟩⟩]⟩) (none) 181916) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26432⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26432⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge181922

namespace LeftMerge181927
def owner : Owner := ⟨.program ⟨257⟩, ⟨27954⟩⟩
def mergeEvent : Nat := 181927
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13026⟩⟩, ⟨.program ⟨257⟩, ⟨26166⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27427⟩⟩] } }
def leftRaw : List Term := Proof.Events710.exact181923RawTerms
def rightRaw : List Term := Proof.Events709.exact181737RawTerms
def group : MergeGroup := .operator 181923 181737
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181923) (leftOrdinal := 2)
    (rightResult := 181737) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13026⟩⟩, ⟨.program ⟨257⟩, ⟨26166⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27427⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13026⟩⟩, ⟨.program ⟨257⟩, ⟨26166⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27427⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13026⟩⟩, ⟨.program ⟨257⟩, ⟨26166⟩⟩], [⟨.program ⟨257⟩, ⟨27427⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge181927

namespace LeftMerge181928
def owner : Owner := ⟨.program ⟨257⟩, ⟨27954⟩⟩
def mergeEvent : Nat := 181928
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27952⟩⟩] } }
def leftRaw : List Term := Proof.Events710.exact181923RawTerms
def rightRaw : List Term := Proof.Events709.exact181737RawTerms
def group : MergeGroup := .operator 181923 181737
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181923) (leftOrdinal := 1)
    (rightResult := 181737) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27952⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27952⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27952⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181928

namespace LeftMerge181936
def owner : Owner := ⟨.program ⟨257⟩, ⟨28366⟩⟩
def mergeEvent : Nat := 181936
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28364⟩⟩] } }
def leftRaw : List Term := Proof.Events710.exact181930RawTerms
def rightRaw : List Term := Proof.Events709.exact181653RawTerms
def group : MergeGroup := .operator 181930 181653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181930) (leftOrdinal := 0)
    (rightResult := 181653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28364⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28364⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181936

namespace LeftMerge181937
def owner : Owner := ⟨.program ⟨257⟩, ⟨28366⟩⟩
def mergeEvent : Nat := 181937
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26432⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28364⟩⟩] } }
def leftRaw : List Term := Proof.Events710.exact181930RawTerms
def rightRaw : List Term := Proof.Events709.exact181653RawTerms
def group : MergeGroup := .operator 181930 181653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181930) (leftOrdinal := 1)
    (rightResult := 181653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26432⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28364⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26432⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28364⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge181937

namespace LeftMerge181939
def owner : Owner := ⟨.program ⟨257⟩, ⟨28366⟩⟩
def mergeEvent : Nat := 181939
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26432⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27588⟩⟩] } }
def rhsRaw : List Term := Proof.Events709.exact181650RawTerms
def group : MergeGroup := .relation 181938
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 181938) (rhsResult := 181650)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26432⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28364⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28364⟩⟩) ⟨27588⟩ 181650) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27588⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26432⟩⟩], [⟨.program ⟨257⟩, ⟨27588⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge181939

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
