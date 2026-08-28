import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge282810
def owner : Owner := ⟨.program ⟨257⟩, ⟨38876⟩⟩
def mergeEvent : Nat := 282810
def frameStart : Nat := 282720
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38873⟩⟩] } }
def leftRaw : List Term := Proof.Events1104.exact282806RawTerms
def rightRaw : List Term := Proof.Events1104.exact282765RawTerms
def group : MergeGroup := .operator 282806 282765
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 282806) (leftOrdinal := 0)
    (rightResult := 282765) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38873⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38873⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge282810

namespace LeftMerge282811
def owner : Owner := ⟨.program ⟨257⟩, ⟨38876⟩⟩
def mergeEvent : Nat := 282811
def frameStart : Nat := 282720
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13791⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38873⟩⟩] } }
def leftRaw : List Term := Proof.Events1104.exact282806RawTerms
def rightRaw : List Term := Proof.Events1104.exact282765RawTerms
def group : MergeGroup := .operator 282806 282765
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 282806) (leftOrdinal := 1)
    (rightResult := 282765) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13791⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38873⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13791⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38873⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge282811

namespace LeftMerge282813
def owner : Owner := ⟨.program ⟨257⟩, ⟨38876⟩⟩
def mergeEvent : Nat := 282813
def frameStart : Nat := 282720
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13791⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38393⟩⟩] } }
def rhsRaw : List Term := Proof.Events1104.exact282762RawTerms
def group : MergeGroup := .relation 282812
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 282812) (rhsResult := 282762)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13791⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38873⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38873⟩⟩) ⟨38393⟩ 282762) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38393⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13791⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], [⟨.program ⟨257⟩, ⟨38393⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge282813

namespace LeftMerge282821
def owner : Owner := ⟨.program ⟨257⟩, ⟨37382⟩⟩
def mergeEvent : Nat := 282821
def frameStart : Nat := 282720
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37380⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1104.exact282776RawTerms
def rightRaw : List Term := Proof.Events1104.exact282817RawTerms
def group : MergeGroup := .operator 282776 282817
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 282776) (leftOrdinal := 0)
    (rightResult := 282817) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37380⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge282821

namespace LeftMerge282838
def owner : Owner := ⟨.program ⟨257⟩, ⟨37812⟩⟩
def mergeEvent : Nat := 282838
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }
def rhsRaw : List Term := Proof.Events1104.exact282835RawTerms
def group : MergeGroup := .relation 282837
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 282837) (rhsResult := 282835)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37809⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 282836 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37809⟩⟩]⟩) (none) 282835) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge282838

namespace LeftMerge282839
def owner : Owner := ⟨.program ⟨257⟩, ⟨37812⟩⟩
def mergeEvent : Nat := 282839
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38873⟩⟩] } }
def rhsRaw : List Term := Proof.Events1104.exact282835RawTerms
def group : MergeGroup := .relation 282837
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 282837) (rhsResult := 282835)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37809⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 282836 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37809⟩⟩]⟩) (none) 282835) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38873⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38873⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge282839

namespace LeftMerge282840
def owner : Owner := ⟨.program ⟨257⟩, ⟨37812⟩⟩
def mergeEvent : Nat := 282840
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13791⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38393⟩⟩] } }
def rhsRaw : List Term := Proof.Events1104.exact282835RawTerms
def group : MergeGroup := .relation 282837
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 282837) (rhsResult := 282835)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37809⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 282836 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37809⟩⟩]⟩) (none) 282835) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13791⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38393⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13791⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], [⟨.program ⟨257⟩, ⟨38393⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge282840

namespace LeftMerge282841
def owner : Owner := ⟨.program ⟨257⟩, ⟨37812⟩⟩
def mergeEvent : Nat := 282841
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37380⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1104.exact282835RawTerms
def group : MergeGroup := .relation 282837
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 282837) (rhsResult := 282835)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37809⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 282836 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37809⟩⟩]⟩) (none) 282835) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37380⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge282841

namespace LeftMerge282846
def owner : Owner := ⟨.program ⟨257⟩, ⟨38875⟩⟩
def mergeEvent : Nat := 282846
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13791⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38393⟩⟩] } }
def leftRaw : List Term := Proof.Events1104.exact282842RawTerms
def rightRaw : List Term := Proof.Events1104.exact282658RawTerms
def group : MergeGroup := .operator 282842 282658
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 282842) (leftOrdinal := 2)
    (rightResult := 282658) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13791⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38393⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13791⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38393⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13791⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], [⟨.program ⟨257⟩, ⟨38393⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge282846

namespace LeftMerge282847
def owner : Owner := ⟨.program ⟨257⟩, ⟨38875⟩⟩
def mergeEvent : Nat := 282847
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38873⟩⟩] } }
def leftRaw : List Term := Proof.Events1104.exact282842RawTerms
def rightRaw : List Term := Proof.Events1104.exact282658RawTerms
def group : MergeGroup := .operator 282842 282658
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 282842) (leftOrdinal := 1)
    (rightResult := 282658) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38873⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38873⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38873⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge282847

namespace LeftMerge282855
def owner : Owner := ⟨.program ⟨257⟩, ⟨39161⟩⟩
def mergeEvent : Nat := 282855
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩] } }
def leftRaw : List Term := Proof.Events1104.exact282849RawTerms
def rightRaw : List Term := Proof.Events1103.exact282574RawTerms
def group : MergeGroup := .operator 282849 282574
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 282849) (leftOrdinal := 0)
    (rightResult := 282574) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39159⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge282855

namespace LeftMerge282856
def owner : Owner := ⟨.program ⟨257⟩, ⟨39161⟩⟩
def mergeEvent : Nat := 282856
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37380⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩] } }
def leftRaw : List Term := Proof.Events1104.exact282849RawTerms
def rightRaw : List Term := Proof.Events1103.exact282574RawTerms
def group : MergeGroup := .operator 282849 282574
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 282849) (leftOrdinal := 1)
    (rightResult := 282574) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37380⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39159⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge282856

namespace LeftMerge282858
def owner : Owner := ⟨.program ⟨257⟩, ⟨39161⟩⟩
def mergeEvent : Nat := 282858
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37380⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38527⟩⟩] } }
def rhsRaw : List Term := Proof.Events1103.exact282571RawTerms
def group : MergeGroup := .relation 282857
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 282857) (rhsResult := 282571)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39159⟩⟩) ⟨38527⟩ 282571) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38527⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨38527⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge282858

namespace LeftMerge282872
def owner : Owner := ⟨.program ⟨257⟩, ⟨38059⟩⟩
def mergeEvent : Nat := 282872
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38056⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280745RawTerms
def rightRaw : List Term := Proof.Events1104.exact282866RawTerms
def group : MergeGroup := .operator 280745 282866
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280745) (leftOrdinal := 0)
    (rightResult := 282866) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38056⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38056⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge282872

namespace LeftMerge282993
def owner : Owner := ⟨.program ⟨257⟩, ⟨38764⟩⟩
def mergeEvent : Nat := 282993
def frameStart : Nat := 282927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37380⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1105.exact282989RawTerms
def rightRaw : List Term := Proof.Events1105.exact282987RawTerms
def group : MergeGroup := .operator 282989 282987
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 282989) (leftOrdinal := 0)
    (rightResult := 282987) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37380⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge282993

namespace LeftMerge283005
def owner : Owner := ⟨.program ⟨257⟩, ⟨39160⟩⟩
def mergeEvent : Nat := 283005
def frameStart : Nat := 282927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩] } }
def leftRaw : List Term := Proof.Events1105.exact283001RawTerms
def rightRaw : List Term := Proof.Events1105.exact282978RawTerms
def group : MergeGroup := .operator 283001 282978
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 283001) (leftOrdinal := 0)
    (rightResult := 282978) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39159⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge283005

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
