import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge210945
def owner : Owner := ⟨.program ⟨257⟩, ⟨12982⟩⟩
def mergeEvent : Nat := 210945
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12981⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events038.exact9982RawTerms
def rightRaw : List Term := Proof.Events810.exact207528RawTerms
def group : MergeGroup := .operator 9982 207528
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9982) (leftOrdinal := 0)
    (rightResult := 207528) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12981⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12981⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210945

namespace LeftMerge210950
def owner : Owner := ⟨.program ⟨257⟩, ⟨8601⟩⟩
def mergeEvent : Nat := 210950
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }
def leftRaw : List Term := Proof.Events810.exact207398RawTerms
def rightRaw : List Term := Proof.Events080.exact20628RawTerms
def group : MergeGroup := .operator 207398 20628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207398) (leftOrdinal := 0)
    (rightResult := 20628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210950

namespace LeftMerge210967
def owner : Owner := ⟨.program ⟨257⟩, ⟨12985⟩⟩
def mergeEvent : Nat := 210967
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12981⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }
def leftRaw : List Term := Proof.Events824.exact210961RawTerms
def rightRaw : List Term := Proof.Events080.exact20617RawTerms
def group : MergeGroup := .operator 210961 20617
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210961) (leftOrdinal := 1)
    (rightResult := 20617) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12981⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9544⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12981⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge210967

namespace LeftMerge210969
def owner : Owner := ⟨.program ⟨257⟩, ⟨12985⟩⟩
def mergeEvent : Nat := 210969
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12981⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }
def rhsRaw : List Term := Proof.Events080.exact20587RawTerms
def group : MergeGroup := .relation 210968
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 210968) (rhsResult := 20587)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12981⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9544⟩⟩) ⟨7278⟩ 20587) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12981⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge210969

namespace LeftMerge210970
def owner : Owner := ⟨.program ⟨257⟩, ⟨12985⟩⟩
def mergeEvent : Nat := 210970
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }
def leftRaw : List Term := Proof.Events824.exact210961RawTerms
def rightRaw : List Term := Proof.Events080.exact20617RawTerms
def group : MergeGroup := .operator 210961 20617
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210961) (leftOrdinal := 0)
    (rightResult := 20617) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9544⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210970

namespace LeftMerge210975
def owner : Owner := ⟨.program ⟨257⟩, ⟨26101⟩⟩
def mergeEvent : Nat := 210975
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12981⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }
def leftRaw : List Term := Proof.Events824.exact210971RawTerms
def rightRaw : List Term := Proof.Events823.exact210941RawTerms
def group : MergeGroup := .operator 210971 210941
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210971) (leftOrdinal := 1)
    (rightResult := 210941) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12981⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12981⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12981⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210975

namespace LeftMerge210983
def owner : Owner := ⟨.program ⟨257⟩, ⟨27920⟩⟩
def mergeEvent : Nat := 210983
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12981⟩⟩, ⟨.program ⟨257⟩, ⟨26094⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27919⟩⟩] } }
def leftRaw : List Term := Proof.Events824.exact210977RawTerms
def rightRaw : List Term := Proof.Events823.exact210913RawTerms
def group : MergeGroup := .operator 210977 210913
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210977) (leftOrdinal := 1)
    (rightResult := 210913) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12981⟩⟩, ⟨.program ⟨257⟩, ⟨26094⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27919⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12981⟩⟩, ⟨.program ⟨257⟩, ⟨26094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27919⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge210983

namespace LeftMerge210985
def owner : Owner := ⟨.program ⟨257⟩, ⟨27920⟩⟩
def mergeEvent : Nat := 210985
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12981⟩⟩, ⟨.program ⟨257⟩, ⟨26094⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27409⟩⟩] } }
def rhsRaw : List Term := Proof.Events823.exact210910RawTerms
def group : MergeGroup := .relation 210984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 210984) (rhsResult := 210910)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12981⟩⟩, ⟨.program ⟨257⟩, ⟨26094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27919⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27919⟩⟩) ⟨27409⟩ 210910) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27409⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12981⟩⟩, ⟨.program ⟨257⟩, ⟨26094⟩⟩], [⟨.program ⟨257⟩, ⟨27409⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge210985

namespace LeftMerge210986
def owner : Owner := ⟨.program ⟨257⟩, ⟨27920⟩⟩
def mergeEvent : Nat := 210986
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27919⟩⟩] } }
def leftRaw : List Term := Proof.Events824.exact210977RawTerms
def rightRaw : List Term := Proof.Events823.exact210913RawTerms
def group : MergeGroup := .operator 210977 210913
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210977) (leftOrdinal := 0)
    (rightResult := 210913) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27919⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27919⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210986

namespace LeftMerge211000
def owner : Owner := ⟨.program ⟨257⟩, ⟨26852⟩⟩
def mergeEvent : Nat := 211000
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26849⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207620RawTerms
def rightRaw : List Term := Proof.Events824.exact210994RawTerms
def group : MergeGroup := .operator 207620 210994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207620) (leftOrdinal := 0)
    (rightResult := 210994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨26849⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26849⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge211000

namespace LeftMerge211079
def owner : Owner := ⟨.program ⟨257⟩, ⟨26095⟩⟩
def mergeEvent : Nat := 211079
def frameStart : Nat := 211049
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12981⟩⟩, ⟨.program ⟨257⟩, ⟨26094⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events824.exact211075RawTerms
def rightRaw : List Term := Proof.Events824.exact211072RawTerms
def group : MergeGroup := .operator 211075 211072
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 211075) (leftOrdinal := 0)
    (rightResult := 211072) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12981⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26094⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12981⟩⟩, ⟨.program ⟨257⟩, ⟨26094⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge211079

namespace LeftMerge211109
def owner : Owner := ⟨.program ⟨257⟩, ⟨27688⟩⟩
def mergeEvent : Nat := 211109
def frameStart : Nat := 211049
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12981⟩⟩, ⟨.program ⟨257⟩, ⟨26094⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events824.exact211105RawTerms
def rightRaw : List Term := Proof.Events824.exact211103RawTerms
def group : MergeGroup := .operator 211105 211103
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 211105) (leftOrdinal := 0)
    (rightResult := 211103) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12981⟩⟩, ⟨.program ⟨257⟩, ⟨26094⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12981⟩⟩, ⟨.program ⟨257⟩, ⟨26094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge211109

namespace LeftMerge211132
def owner : Owner := ⟨.program ⟨257⟩, ⟨9546⟩⟩
def mergeEvent : Nat := 211132
def frameStart : Nat := 211049
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }
def leftRaw : List Term := Proof.Events824.exact211128RawTerms
def rightRaw : List Term := Proof.Events824.exact211125RawTerms
def group : MergeGroup := .operator 211128 211125
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 211128) (leftOrdinal := 0)
    (rightResult := 211125) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9544⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge211132

namespace LeftMerge211141
def owner : Owner := ⟨.program ⟨257⟩, ⟨27922⟩⟩
def mergeEvent : Nat := 211141
def frameStart : Nat := 211049
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27919⟩⟩] } }
def leftRaw : List Term := Proof.Events824.exact211137RawTerms
def rightRaw : List Term := Proof.Events824.exact211094RawTerms
def group : MergeGroup := .operator 211137 211094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 211137) (leftOrdinal := 0)
    (rightResult := 211094) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27919⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27919⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge211141

namespace LeftMerge211142
def owner : Owner := ⟨.program ⟨257⟩, ⟨27922⟩⟩
def mergeEvent : Nat := 211142
def frameStart : Nat := 211049
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12981⟩⟩, ⟨.program ⟨257⟩, ⟨26094⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27919⟩⟩] } }
def leftRaw : List Term := Proof.Events824.exact211137RawTerms
def rightRaw : List Term := Proof.Events824.exact211094RawTerms
def group : MergeGroup := .operator 211137 211094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 211137) (leftOrdinal := 1)
    (rightResult := 211094) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12981⟩⟩, ⟨.program ⟨257⟩, ⟨26094⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27919⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12981⟩⟩, ⟨.program ⟨257⟩, ⟨26094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27919⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge211142

namespace LeftMerge211144
def owner : Owner := ⟨.program ⟨257⟩, ⟨27922⟩⟩
def mergeEvent : Nat := 211144
def frameStart : Nat := 211049
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12981⟩⟩, ⟨.program ⟨257⟩, ⟨26094⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27409⟩⟩] } }
def rhsRaw : List Term := Proof.Events824.exact211091RawTerms
def group : MergeGroup := .relation 211143
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 211143) (rhsResult := 211091)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12981⟩⟩, ⟨.program ⟨257⟩, ⟨26094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27919⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27919⟩⟩) ⟨27409⟩ 211091) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27409⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12981⟩⟩, ⟨.program ⟨257⟩, ⟨26094⟩⟩], [⟨.program ⟨257⟩, ⟨27409⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge211144

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
