import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge50065
def owner : Owner := ⟨.program ⟨257⟩, ⟨26292⟩⟩
def mergeEvent : Nat := 50065
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }
def leftRaw : List Term := Proof.Events195.exact50058RawTerms
def rightRaw : List Term := Proof.Events006.exact1754RawTerms
def group : MergeGroup := .operator 50058 1754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50058) (leftOrdinal := 0)
    (rightResult := 1754) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13101⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50065

namespace LeftMerge50070
def owner : Owner := ⟨.program ⟨257⟩, ⟨13102⟩⟩
def mergeEvent : Nat := 50070
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events006.exact1754RawTerms
def rightRaw : List Term := Proof.Events182.exact46653RawTerms
def group : MergeGroup := .operator 1754 46653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1754) (leftOrdinal := 0)
    (rightResult := 46653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13101⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50070

namespace LeftMerge50075
def owner : Owner := ⟨.program ⟨257⟩, ⟨11201⟩⟩
def mergeEvent : Nat := 50075
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46523RawTerms
def rightRaw : List Term := Proof.Events080.exact20628RawTerms
def group : MergeGroup := .operator 46523 20628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46523) (leftOrdinal := 0)
    (rightResult := 20628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50075

namespace LeftMerge50092
def owner : Owner := ⟨.program ⟨257⟩, ⟨13105⟩⟩
def mergeEvent : Nat := 50092
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }
def leftRaw : List Term := Proof.Events195.exact50086RawTerms
def rightRaw : List Term := Proof.Events080.exact20617RawTerms
def group : MergeGroup := .operator 50086 20617
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50086) (leftOrdinal := 1)
    (rightResult := 20617) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9544⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50092

namespace LeftMerge50094
def owner : Owner := ⟨.program ⟨257⟩, ⟨13105⟩⟩
def mergeEvent : Nat := 50094
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }
def rhsRaw : List Term := Proof.Events080.exact20587RawTerms
def group : MergeGroup := .relation 50093
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50093) (rhsResult := 20587)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9544⟩⟩) ⟨7278⟩ 20587) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50094

namespace LeftMerge50095
def owner : Owner := ⟨.program ⟨257⟩, ⟨13105⟩⟩
def mergeEvent : Nat := 50095
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }
def leftRaw : List Term := Proof.Events195.exact50086RawTerms
def rightRaw : List Term := Proof.Events080.exact20617RawTerms
def group : MergeGroup := .operator 50086 20617
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50086) (leftOrdinal := 0)
    (rightResult := 20617) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9544⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50095

namespace LeftMerge50100
def owner : Owner := ⟨.program ⟨257⟩, ⟨26293⟩⟩
def mergeEvent : Nat := 50100
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }
def leftRaw : List Term := Proof.Events195.exact50096RawTerms
def rightRaw : List Term := Proof.Events195.exact50066RawTerms
def group : MergeGroup := .operator 50096 50066
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50096) (leftOrdinal := 1)
    (rightResult := 50066) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50100

namespace LeftMerge50108
def owner : Owner := ⟨.program ⟨257⟩, ⟨28008⟩⟩
def mergeEvent : Nat := 50108
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28007⟩⟩] } }
def leftRaw : List Term := Proof.Events195.exact50102RawTerms
def rightRaw : List Term := Proof.Events195.exact50038RawTerms
def group : MergeGroup := .operator 50102 50038
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50102) (leftOrdinal := 1)
    (rightResult := 50038) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28007⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28007⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50108

namespace LeftMerge50110
def owner : Owner := ⟨.program ⟨257⟩, ⟨28008⟩⟩
def mergeEvent : Nat := 50110
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27457⟩⟩] } }
def rhsRaw : List Term := Proof.Events195.exact50035RawTerms
def group : MergeGroup := .relation 50109
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50109) (rhsResult := 50035)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28007⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28007⟩⟩) ⟨27457⟩ 50035) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27457⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], [⟨.program ⟨257⟩, ⟨27457⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50110

namespace LeftMerge50111
def owner : Owner := ⟨.program ⟨257⟩, ⟨28008⟩⟩
def mergeEvent : Nat := 50111
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨28007⟩⟩] } }
def leftRaw : List Term := Proof.Events195.exact50102RawTerms
def rightRaw : List Term := Proof.Events195.exact50038RawTerms
def group : MergeGroup := .operator 50102 50038
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50102) (leftOrdinal := 0)
    (rightResult := 50038) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28007⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨28007⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50111

namespace LeftMerge50125
def owner : Owner := ⟨.program ⟨257⟩, ⟨26932⟩⟩
def mergeEvent : Nat := 50125
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26929⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46745RawTerms
def rightRaw : List Term := Proof.Events195.exact50119RawTerms
def group : MergeGroup := .operator 46745 50119
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46745) (leftOrdinal := 0)
    (rightResult := 50119) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨26929⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26929⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50125

namespace LeftMerge50204
def owner : Owner := ⟨.program ⟨257⟩, ⟨26287⟩⟩
def mergeEvent : Nat := 50204
def frameStart : Nat := 50174
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events196.exact50200RawTerms
def rightRaw : List Term := Proof.Events196.exact50197RawTerms
def group : MergeGroup := .operator 50200 50197
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50200) (leftOrdinal := 0)
    (rightResult := 50197) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13101⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26286⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50204

namespace LeftMerge50234
def owner : Owner := ⟨.program ⟨257⟩, ⟨27720⟩⟩
def mergeEvent : Nat := 50234
def frameStart : Nat := 50174
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events196.exact50230RawTerms
def rightRaw : List Term := Proof.Events196.exact50228RawTerms
def group : MergeGroup := .operator 50230 50228
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50230) (leftOrdinal := 0)
    (rightResult := 50228) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50234

namespace LeftMerge50257
def owner : Owner := ⟨.program ⟨257⟩, ⟨9546⟩⟩
def mergeEvent : Nat := 50257
def frameStart : Nat := 50174
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }
def leftRaw : List Term := Proof.Events196.exact50253RawTerms
def rightRaw : List Term := Proof.Events196.exact50250RawTerms
def group : MergeGroup := .operator 50253 50250
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50253) (leftOrdinal := 0)
    (rightResult := 50250) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9544⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50257

namespace LeftMerge50266
def owner : Owner := ⟨.program ⟨257⟩, ⟨28010⟩⟩
def mergeEvent : Nat := 50266
def frameStart : Nat := 50174
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨28007⟩⟩] } }
def leftRaw : List Term := Proof.Events196.exact50262RawTerms
def rightRaw : List Term := Proof.Events196.exact50219RawTerms
def group : MergeGroup := .operator 50262 50219
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50262) (leftOrdinal := 0)
    (rightResult := 50219) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28007⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨28007⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50266

namespace LeftMerge50267
def owner : Owner := ⟨.program ⟨257⟩, ⟨28010⟩⟩
def mergeEvent : Nat := 50267
def frameStart : Nat := 50174
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28007⟩⟩] } }
def leftRaw : List Term := Proof.Events196.exact50262RawTerms
def rightRaw : List Term := Proof.Events196.exact50219RawTerms
def group : MergeGroup := .operator 50262 50219
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50262) (leftOrdinal := 1)
    (rightResult := 50219) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28007⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28007⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50267

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
