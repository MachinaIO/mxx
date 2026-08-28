import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge298130
def owner : Owner := ⟨.program ⟨257⟩, ⟨30722⟩⟩
def mergeEvent : Nat := 298130
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨29008⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30151⟩⟩] } }
def leftRaw : List Term := Proof.Events1164.exact298125RawTerms
def rightRaw : List Term := Proof.Events1163.exact297971RawTerms
def group : MergeGroup := .operator 298125 297971
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298125) (leftOrdinal := 2)
    (rightResult := 297971) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨29008⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30151⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨29008⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30151⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨29008⟩⟩], [⟨.program ⟨257⟩, ⟨30151⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge298130

namespace LeftMerge298156
def owner : Owner := ⟨.program ⟨257⟩, ⟨25857⟩⟩
def mergeEvent : Nat := 298156
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events056.exact14451RawTerms
def rightRaw : List Term := Proof.Events000.exact32RawTerms
def group : MergeGroup := .operator 14451 32
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14451) (leftOrdinal := 0)
    (rightResult := 32) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25854⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298156

namespace LeftMerge298161
def owner : Owner := ⟨.program ⟨257⟩, ⟨7426⟩⟩
def mergeEvent : Nat := 298161
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact27RawTerms
def rightRaw : List Term := Proof.Events080.exact20587RawTerms
def group : MergeGroup := .operator 27 20587
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27) (leftOrdinal := 0)
    (rightResult := 20587) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298161

namespace LeftMerge298178
def owner : Owner := ⟨.program ⟨257⟩, ⟨25860⟩⟩
def mergeEvent : Nat := 298178
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1164.exact298172RawTerms
def rightRaw : List Term := Proof.Events056.exact14454RawTerms
def group : MergeGroup := .operator 298172 14454
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298172) (leftOrdinal := 1)
    (rightResult := 14454) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12831⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge298178

namespace LeftMerge298179
def owner : Owner := ⟨.program ⟨257⟩, ⟨25860⟩⟩
def mergeEvent : Nat := 298179
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }
def leftRaw : List Term := Proof.Events1164.exact298172RawTerms
def rightRaw : List Term := Proof.Events056.exact14454RawTerms
def group : MergeGroup := .operator 298172 14454
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298172) (leftOrdinal := 0)
    (rightResult := 14454) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12831⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298179

namespace LeftMerge298184
def owner : Owner := ⟨.program ⟨257⟩, ⟨12832⟩⟩
def mergeEvent : Nat := 298184
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events056.exact14454RawTerms
def rightRaw : List Term := Proof.Events000.exact32RawTerms
def group : MergeGroup := .operator 14454 32
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14454) (leftOrdinal := 0)
    (rightResult := 32) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12831⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298184

namespace LeftMerge298189
def owner : Owner := ⟨.program ⟨257⟩, ⟨7443⟩⟩
def mergeEvent : Nat := 298189
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact27RawTerms
def rightRaw : List Term := Proof.Events080.exact20628RawTerms
def group : MergeGroup := .operator 27 20628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27) (leftOrdinal := 0)
    (rightResult := 20628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298189

namespace LeftMerge298206
def owner : Owner := ⟨.program ⟨257⟩, ⟨12835⟩⟩
def mergeEvent : Nat := 298206
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }
def leftRaw : List Term := Proof.Events1164.exact298200RawTerms
def rightRaw : List Term := Proof.Events080.exact20617RawTerms
def group : MergeGroup := .operator 298200 20617
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298200) (leftOrdinal := 1)
    (rightResult := 20617) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9544⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge298206

namespace LeftMerge298208
def owner : Owner := ⟨.program ⟨257⟩, ⟨12835⟩⟩
def mergeEvent : Nat := 298208
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }
def rhsRaw : List Term := Proof.Events080.exact20587RawTerms
def group : MergeGroup := .relation 298207
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 298207) (rhsResult := 20587)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9544⟩⟩) ⟨7278⟩ 20587) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge298208

namespace LeftMerge298209
def owner : Owner := ⟨.program ⟨257⟩, ⟨12835⟩⟩
def mergeEvent : Nat := 298209
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }
def leftRaw : List Term := Proof.Events1164.exact298200RawTerms
def rightRaw : List Term := Proof.Events080.exact20617RawTerms
def group : MergeGroup := .operator 298200 20617
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298200) (leftOrdinal := 0)
    (rightResult := 20617) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9544⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298209

namespace LeftMerge298214
def owner : Owner := ⟨.program ⟨257⟩, ⟨25861⟩⟩
def mergeEvent : Nat := 298214
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }
def leftRaw : List Term := Proof.Events1164.exact298210RawTerms
def rightRaw : List Term := Proof.Events1164.exact298180RawTerms
def group : MergeGroup := .operator 298210 298180
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298210) (leftOrdinal := 1)
    (rightResult := 298180) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298214

namespace LeftMerge298222
def owner : Owner := ⟨.program ⟨257⟩, ⟨27810⟩⟩
def mergeEvent : Nat := 298222
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27809⟩⟩] } }
def leftRaw : List Term := Proof.Events1164.exact298216RawTerms
def rightRaw : List Term := Proof.Events1164.exact298152RawTerms
def group : MergeGroup := .operator 298216 298152
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298216) (leftOrdinal := 1)
    (rightResult := 298152) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27809⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27809⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge298222

namespace LeftMerge298224
def owner : Owner := ⟨.program ⟨257⟩, ⟨27810⟩⟩
def mergeEvent : Nat := 298224
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27349⟩⟩] } }
def rhsRaw : List Term := Proof.Events1164.exact298149RawTerms
def group : MergeGroup := .relation 298223
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 298223) (rhsResult := 298149)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27809⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27809⟩⟩) ⟨27349⟩ 298149) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27349⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], [⟨.program ⟨257⟩, ⟨27349⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge298224

namespace LeftMerge298225
def owner : Owner := ⟨.program ⟨257⟩, ⟨27810⟩⟩
def mergeEvent : Nat := 298225
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27809⟩⟩] } }
def leftRaw : List Term := Proof.Events1164.exact298216RawTerms
def rightRaw : List Term := Proof.Events1164.exact298152RawTerms
def group : MergeGroup := .operator 298216 298152
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298216) (leftOrdinal := 0)
    (rightResult := 298152) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27809⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27809⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298225

namespace LeftMerge298239
def owner : Owner := ⟨.program ⟨257⟩, ⟨26752⟩⟩
def mergeEvent : Nat := 298239
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26749⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1164.exact298233RawTerms
def group : MergeGroup := .operator 295195 298233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 298233) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨26749⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26749⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298239

namespace LeftMerge298294
def owner : Owner := ⟨.program ⟨257⟩, ⟨25855⟩⟩
def mergeEvent : Nat := 298294
def frameStart : Nat := 298276
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1165.exact298290RawTerms
def rightRaw : List Term := Proof.Events1165.exact298287RawTerms
def group : MergeGroup := .operator 298290 298287
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 298290) (leftOrdinal := 0)
    (rightResult := 298287) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12831⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25854⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12831⟩⟩, ⟨.program ⟨257⟩, ⟨25854⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge298294

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
