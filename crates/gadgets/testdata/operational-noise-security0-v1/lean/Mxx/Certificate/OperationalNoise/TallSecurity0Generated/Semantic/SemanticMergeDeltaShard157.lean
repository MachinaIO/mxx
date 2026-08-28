import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge27023
def owner : Owner := ⟨.program ⟨214⟩, ⟨21271⟩⟩
def mergeEvent : Nat := 27023
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21268⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21512RawTerms
def rightRaw : List Term := Proof.Events105.exact27017RawTerms
def group : MergeGroup := .operator 21512 27017
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21512) (leftOrdinal := 0)
    (rightResult := 27017) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21268⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27023

namespace LeftMerge27144
def owner : Owner := ⟨.program ⟨214⟩, ⟨15910⟩⟩
def mergeEvent : Nat := 27144
def frameStart : Nat := 27078
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15833⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events106.exact27140RawTerms
def rightRaw : List Term := Proof.Events106.exact27138RawTerms
def group : MergeGroup := .operator 27140 27138
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27140) (leftOrdinal := 0)
    (rightResult := 27138) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15833⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27144

namespace LeftMerge27156
def owner : Owner := ⟨.program ⟨214⟩, ⟨27689⟩⟩
def mergeEvent : Nat := 27156
def frameStart : Nat := 27078
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩] } }
def leftRaw : List Term := Proof.Events106.exact27152RawTerms
def rightRaw : List Term := Proof.Events105.exact27129RawTerms
def group : MergeGroup := .operator 27152 27129
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27152) (leftOrdinal := 0)
    (rightResult := 27129) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27688⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27156

namespace LeftMerge27157
def owner : Owner := ⟨.program ⟨214⟩, ⟨27689⟩⟩
def mergeEvent : Nat := 27157
def frameStart : Nat := 27078
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15833⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩] } }
def leftRaw : List Term := Proof.Events106.exact27152RawTerms
def rightRaw : List Term := Proof.Events105.exact27129RawTerms
def group : MergeGroup := .operator 27152 27129
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27152) (leftOrdinal := 1)
    (rightResult := 27129) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15833⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27688⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27157

namespace LeftMerge27159
def owner : Owner := ⟨.program ⟨214⟩, ⟨27689⟩⟩
def mergeEvent : Nat := 27159
def frameStart : Nat := 27078
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15833⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24108⟩⟩] } }
def rhsRaw : List Term := Proof.Events105.exact27126RawTerms
def group : MergeGroup := .relation 27158
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27158) (rhsResult := 27126)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27688⟩⟩) ⟨24108⟩ 27126) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24108⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨24108⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27159

namespace LeftMerge27167
def owner : Owner := ⟨.program ⟨214⟩, ⟨15877⟩⟩
def mergeEvent : Nat := 27167
def frameStart : Nat := 27078
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15876⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events106.exact27140RawTerms
def rightRaw : List Term := Proof.Events106.exact27163RawTerms
def group : MergeGroup := .operator 27140 27163
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27140) (leftOrdinal := 0)
    (rightResult := 27163) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15876⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27167

namespace LeftMerge27184
def owner : Owner := ⟨.program ⟨214⟩, ⟨21271⟩⟩
def mergeEvent : Nat := 27184
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }
def rhsRaw : List Term := Proof.Events106.exact27181RawTerms
def group : MergeGroup := .relation 27183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27183) (rhsResult := 27181)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27182 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩) (none) 27181) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27184

namespace LeftMerge27185
def owner : Owner := ⟨.program ⟨214⟩, ⟨21271⟩⟩
def mergeEvent : Nat := 27185
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩] } }
def rhsRaw : List Term := Proof.Events106.exact27181RawTerms
def group : MergeGroup := .relation 27183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27183) (rhsResult := 27181)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27182 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩) (none) 27181) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27185

namespace LeftMerge27186
def owner : Owner := ⟨.program ⟨214⟩, ⟨21271⟩⟩
def mergeEvent : Nat := 27186
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15833⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24108⟩⟩] } }
def rhsRaw : List Term := Proof.Events106.exact27181RawTerms
def group : MergeGroup := .relation 27183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27183) (rhsResult := 27181)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27182 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩) (none) 27181) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15833⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24108⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨24108⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27186

namespace LeftMerge27187
def owner : Owner := ⟨.program ⟨214⟩, ⟨21271⟩⟩
def mergeEvent : Nat := 27187
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15876⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events106.exact27181RawTerms
def group : MergeGroup := .relation 27183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27183) (rhsResult := 27181)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27182 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩) (none) 27181) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15876⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27187

namespace LeftMerge27192
def owner : Owner := ⟨.program ⟨214⟩, ⟨27691⟩⟩
def mergeEvent : Nat := 27192
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩] } }
def leftRaw : List Term := Proof.Events106.exact27188RawTerms
def rightRaw : List Term := Proof.Events105.exact27010RawTerms
def group : MergeGroup := .operator 27188 27010
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27188) (leftOrdinal := 0)
    (rightResult := 27010) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27192

namespace LeftMerge27193
def owner : Owner := ⟨.program ⟨214⟩, ⟨27691⟩⟩
def mergeEvent : Nat := 27193
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15833⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24108⟩⟩] } }
def leftRaw : List Term := Proof.Events106.exact27188RawTerms
def rightRaw : List Term := Proof.Events105.exact27010RawTerms
def group : MergeGroup := .operator 27188 27010
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27188) (leftOrdinal := 2)
    (rightResult := 27010) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15833⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24108⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15833⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24108⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨24108⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27193

namespace LeftMerge27219
def owner : Owner := ⟨.program ⟨214⟩, ⟨11314⟩⟩
def mergeEvent : Nat := 27219
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events004.exact1118RawTerms
def rightRaw : List Term := Proof.Events083.exact21420RawTerms
def group : MergeGroup := .operator 1118 21420
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1118) (leftOrdinal := 0)
    (rightResult := 21420) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11313⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27219

namespace LeftMerge27224
def owner : Owner := ⟨.program ⟨214⟩, ⟨7347⟩⟩
def mergeEvent : Nat := 27224
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21290RawTerms
def rightRaw : List Term := Proof.Events048.exact12484RawTerms
def group : MergeGroup := .operator 21290 12484
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21290) (leftOrdinal := 0)
    (rightResult := 12484) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27224

namespace LeftMerge27241
def owner : Owner := ⟨.program ⟨214⟩, ⟨13803⟩⟩
def mergeEvent : Nat := 27241
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events106.exact27235RawTerms
def rightRaw : List Term := Proof.Events004.exact1121RawTerms
def group : MergeGroup := .operator 27235 1121
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27235) (leftOrdinal := 1)
    (rightResult := 1121) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13800⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27241

namespace LeftMerge27242
def owner : Owner := ⟨.program ⟨214⟩, ⟨13803⟩⟩
def mergeEvent : Nat := 27242
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } }
def leftRaw : List Term := Proof.Events106.exact27235RawTerms
def rightRaw : List Term := Proof.Events004.exact1121RawTerms
def group : MergeGroup := .operator 27235 1121
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27235) (leftOrdinal := 0)
    (rightResult := 1121) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13800⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27242

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
