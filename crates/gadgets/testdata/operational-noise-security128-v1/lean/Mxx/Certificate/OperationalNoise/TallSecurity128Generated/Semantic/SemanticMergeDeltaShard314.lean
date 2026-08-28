import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge55054
def owner : Owner := ⟨.program ⟨257⟩, ⟨17160⟩⟩
def mergeEvent : Nat := 55054
def frameStart : Nat := 54994
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12501⟩⟩, ⟨.program ⟨257⟩, ⟨15666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55050RawTerms
def rightRaw : List Term := Proof.Events215.exact55048RawTerms
def group : MergeGroup := .operator 55050 55048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55050) (leftOrdinal := 0)
    (rightResult := 55048) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12501⟩⟩, ⟨.program ⟨257⟩, ⟨15666⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12501⟩⟩, ⟨.program ⟨257⟩, ⟨15666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55054

namespace LeftMerge55077
def owner : Owner := ⟨.program ⟨257⟩, ⟨9570⟩⟩
def mergeEvent : Nat := 55077
def frameStart : Nat := 54994
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55073RawTerms
def rightRaw : List Term := Proof.Events215.exact55070RawTerms
def group : MergeGroup := .operator 55073 55070
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55073) (leftOrdinal := 0)
    (rightResult := 55070) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9568⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55077

namespace LeftMerge55086
def owner : Owner := ⟨.program ⟨257⟩, ⟨17450⟩⟩
def mergeEvent : Nat := 55086
def frameStart : Nat := 54994
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17447⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55082RawTerms
def rightRaw : List Term := Proof.Events214.exact55039RawTerms
def group : MergeGroup := .operator 55082 55039
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55082) (leftOrdinal := 0)
    (rightResult := 55039) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17447⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17447⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55086

namespace LeftMerge55087
def owner : Owner := ⟨.program ⟨257⟩, ⟨17450⟩⟩
def mergeEvent : Nat := 55087
def frameStart : Nat := 54994
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12501⟩⟩, ⟨.program ⟨257⟩, ⟨15666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17447⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55082RawTerms
def rightRaw : List Term := Proof.Events214.exact55039RawTerms
def group : MergeGroup := .operator 55082 55039
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55082) (leftOrdinal := 1)
    (rightResult := 55039) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12501⟩⟩, ⟨.program ⟨257⟩, ⟨15666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17447⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12501⟩⟩, ⟨.program ⟨257⟩, ⟨15666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17447⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55087

namespace LeftMerge55089
def owner : Owner := ⟨.program ⟨257⟩, ⟨17450⟩⟩
def mergeEvent : Nat := 55089
def frameStart : Nat := 54994
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12501⟩⟩, ⟨.program ⟨257⟩, ⟨15666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16897⟩⟩] } }
def rhsRaw : List Term := Proof.Events214.exact55036RawTerms
def group : MergeGroup := .relation 55088
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55088) (rhsResult := 55036)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12501⟩⟩, ⟨.program ⟨257⟩, ⟨15666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17447⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17447⟩⟩) ⟨16897⟩ 55036) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16897⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12501⟩⟩, ⟨.program ⟨257⟩, ⟨15666⟩⟩], [⟨.program ⟨257⟩, ⟨16897⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55089

namespace LeftMerge55097
def owner : Owner := ⟨.program ⟨257⟩, ⟨15854⟩⟩
def mergeEvent : Nat := 55097
def frameStart : Nat := 54994
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55050RawTerms
def rightRaw : List Term := Proof.Events215.exact55093RawTerms
def group : MergeGroup := .operator 55050 55093
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55050) (leftOrdinal := 0)
    (rightResult := 55093) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15852⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55097

namespace LeftMerge55114
def owner : Owner := ⟨.program ⟨257⟩, ⟨16372⟩⟩
def mergeEvent : Nat := 55114
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }
def rhsRaw : List Term := Proof.Events215.exact55111RawTerms
def group : MergeGroup := .relation 55113
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55113) (rhsResult := 55111)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 55112 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16369⟩⟩]⟩) (none) 55111) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55114

namespace LeftMerge55115
def owner : Owner := ⟨.program ⟨257⟩, ⟨16372⟩⟩
def mergeEvent : Nat := 55115
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17447⟩⟩] } }
def rhsRaw : List Term := Proof.Events215.exact55111RawTerms
def group : MergeGroup := .relation 55113
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55113) (rhsResult := 55111)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 55112 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16369⟩⟩]⟩) (none) 55111) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17447⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17447⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55115

namespace LeftMerge55116
def owner : Owner := ⟨.program ⟨257⟩, ⟨16372⟩⟩
def mergeEvent : Nat := 55116
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨12501⟩⟩, ⟨.program ⟨257⟩, ⟨15666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16897⟩⟩] } }
def rhsRaw : List Term := Proof.Events215.exact55111RawTerms
def group : MergeGroup := .relation 55113
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55113) (rhsResult := 55111)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 55112 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16369⟩⟩]⟩) (none) 55111) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12501⟩⟩, ⟨.program ⟨257⟩, ⟨15666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16897⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨12501⟩⟩, ⟨.program ⟨257⟩, ⟨15666⟩⟩], [⟨.program ⟨257⟩, ⟨16897⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55116

namespace LeftMerge55117
def owner : Owner := ⟨.program ⟨257⟩, ⟨16372⟩⟩
def mergeEvent : Nat := 55117
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events215.exact55111RawTerms
def group : MergeGroup := .relation 55113
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55113) (rhsResult := 55111)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 55112 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16369⟩⟩]⟩) (none) 55111) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55117

namespace LeftMerge55122
def owner : Owner := ⟨.program ⟨257⟩, ⟨17449⟩⟩
def mergeEvent : Nat := 55122
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨12501⟩⟩, ⟨.program ⟨257⟩, ⟨15666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16897⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55118RawTerms
def rightRaw : List Term := Proof.Events214.exact54932RawTerms
def group : MergeGroup := .operator 55118 54932
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55118) (leftOrdinal := 2)
    (rightResult := 54932) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨12501⟩⟩, ⟨.program ⟨257⟩, ⟨15666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16897⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨12501⟩⟩, ⟨.program ⟨257⟩, ⟨15666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16897⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨12501⟩⟩, ⟨.program ⟨257⟩, ⟨15666⟩⟩], [⟨.program ⟨257⟩, ⟨16897⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55122

namespace LeftMerge55123
def owner : Owner := ⟨.program ⟨257⟩, ⟨17449⟩⟩
def mergeEvent : Nat := 55123
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17447⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55118RawTerms
def rightRaw : List Term := Proof.Events214.exact54932RawTerms
def group : MergeGroup := .operator 55118 54932
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55118) (leftOrdinal := 1)
    (rightResult := 54932) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17447⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17447⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17447⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55123

namespace LeftMerge55131
def owner : Owner := ⟨.program ⟨257⟩, ⟨17987⟩⟩
def mergeEvent : Nat := 55131
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17985⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55125RawTerms
def rightRaw : List Term := Proof.Events214.exact54848RawTerms
def group : MergeGroup := .operator 55125 54848
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55125) (leftOrdinal := 0)
    (rightResult := 54848) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17985⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17985⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55131

namespace LeftMerge55132
def owner : Owner := ⟨.program ⟨257⟩, ⟨17987⟩⟩
def mergeEvent : Nat := 55132
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17985⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55125RawTerms
def rightRaw : List Term := Proof.Events214.exact54848RawTerms
def group : MergeGroup := .operator 55125 54848
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55125) (leftOrdinal := 1)
    (rightResult := 54848) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17985⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17985⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55132

namespace LeftMerge55134
def owner : Owner := ⟨.program ⟨257⟩, ⟨17987⟩⟩
def mergeEvent : Nat := 55134
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17073⟩⟩] } }
def rhsRaw : List Term := Proof.Events214.exact54845RawTerms
def group : MergeGroup := .relation 55133
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55133) (rhsResult := 54845)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17985⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17985⟩⟩) ⟨17073⟩ 54845) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17073⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨17073⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55134

namespace LeftMerge55148
def owner : Owner := ⟨.program ⟨257⟩, ⟨16759⟩⟩
def mergeEvent : Nat := 55148
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16756⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46745RawTerms
def rightRaw : List Term := Proof.Events215.exact55142RawTerms
def group : MergeGroup := .operator 46745 55142
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46745) (leftOrdinal := 0)
    (rightResult := 55142) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16756⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55148

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
