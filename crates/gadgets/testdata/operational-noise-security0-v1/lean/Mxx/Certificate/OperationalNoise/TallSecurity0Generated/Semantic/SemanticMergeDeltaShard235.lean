import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge39179
def owner : Owner := ⟨.program ⟨214⟩, ⟨25232⟩⟩
def mergeEvent : Nat := 39179
def frameStart : Nat := 39084
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9725⟩⟩, ⟨.program ⟨214⟩, ⟨11973⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23126⟩⟩] } }
def rhsRaw : List Term := Proof.Events152.exact39126RawTerms
def group : MergeGroup := .relation 39178
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39178) (rhsResult := 39126)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9725⟩⟩, ⟨.program ⟨214⟩, ⟨11973⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25229⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25229⟩⟩) ⟨23126⟩ 39126) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23126⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9725⟩⟩, ⟨.program ⟨214⟩, ⟨11973⟩⟩], [⟨.program ⟨214⟩, ⟨23126⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39179

namespace LeftMerge39187
def owner : Owner := ⟨.program ⟨214⟩, ⟨16391⟩⟩
def mergeEvent : Nat := 39187
def frameStart : Nat := 39084
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16389⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events152.exact39140RawTerms
def rightRaw : List Term := Proof.Events153.exact39183RawTerms
def group : MergeGroup := .operator 39140 39183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39140) (leftOrdinal := 0)
    (rightResult := 39183) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16389⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16389⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39187

namespace LeftMerge39204
def owner : Owner := ⟨.program ⟨214⟩, ⟨19827⟩⟩
def mergeEvent : Nat := 39204
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩] } }
def rhsRaw : List Term := Proof.Events153.exact39201RawTerms
def group : MergeGroup := .relation 39203
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39203) (rhsResult := 39201)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19824⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 39202 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19824⟩⟩]⟩) (none) 39201) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39204

namespace LeftMerge39205
def owner : Owner := ⟨.program ⟨214⟩, ⟨19827⟩⟩
def mergeEvent : Nat := 39205
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25229⟩⟩] } }
def rhsRaw : List Term := Proof.Events153.exact39201RawTerms
def group : MergeGroup := .relation 39203
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39203) (rhsResult := 39201)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19824⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 39202 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19824⟩⟩]⟩) (none) 39201) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25229⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25229⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39205

namespace LeftMerge39206
def owner : Owner := ⟨.program ⟨214⟩, ⟨19827⟩⟩
def mergeEvent : Nat := 39206
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9725⟩⟩, ⟨.program ⟨214⟩, ⟨11973⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23126⟩⟩] } }
def rhsRaw : List Term := Proof.Events153.exact39201RawTerms
def group : MergeGroup := .relation 39203
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39203) (rhsResult := 39201)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19824⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 39202 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19824⟩⟩]⟩) (none) 39201) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9725⟩⟩, ⟨.program ⟨214⟩, ⟨11973⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23126⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9725⟩⟩, ⟨.program ⟨214⟩, ⟨11973⟩⟩], [⟨.program ⟨214⟩, ⟨23126⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39206

namespace LeftMerge39207
def owner : Owner := ⟨.program ⟨214⟩, ⟨19827⟩⟩
def mergeEvent : Nat := 39207
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16389⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events153.exact39201RawTerms
def group : MergeGroup := .relation 39203
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39203) (rhsResult := 39201)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19824⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 39202 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19824⟩⟩]⟩) (none) 39201) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16389⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16389⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39207

namespace LeftMerge39212
def owner : Owner := ⟨.program ⟨214⟩, ⟨25231⟩⟩
def mergeEvent : Nat := 39212
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9725⟩⟩, ⟨.program ⟨214⟩, ⟨11973⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23126⟩⟩] } }
def leftRaw : List Term := Proof.Events153.exact39208RawTerms
def rightRaw : List Term := Proof.Events152.exact39022RawTerms
def group : MergeGroup := .operator 39208 39022
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39208) (leftOrdinal := 2)
    (rightResult := 39022) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9725⟩⟩, ⟨.program ⟨214⟩, ⟨11973⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23126⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9725⟩⟩, ⟨.program ⟨214⟩, ⟨11973⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23126⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9725⟩⟩, ⟨.program ⟨214⟩, ⟨11973⟩⟩], [⟨.program ⟨214⟩, ⟨23126⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39212

namespace LeftMerge39213
def owner : Owner := ⟨.program ⟨214⟩, ⟨25231⟩⟩
def mergeEvent : Nat := 39213
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25229⟩⟩] } }
def leftRaw : List Term := Proof.Events153.exact39208RawTerms
def rightRaw : List Term := Proof.Events152.exact39022RawTerms
def group : MergeGroup := .operator 39208 39022
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39208) (leftOrdinal := 1)
    (rightResult := 39022) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25229⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25229⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25229⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39213

namespace LeftMerge39221
def owner : Owner := ⟨.program ⟨214⟩, ⟨28762⟩⟩
def mergeEvent : Nat := 39221
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28760⟩⟩] } }
def leftRaw : List Term := Proof.Events153.exact39215RawTerms
def rightRaw : List Term := Proof.Events152.exact38938RawTerms
def group : MergeGroup := .operator 39215 38938
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39215) (leftOrdinal := 0)
    (rightResult := 38938) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28760⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28760⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39221

namespace LeftMerge39222
def owner : Owner := ⟨.program ⟨214⟩, ⟨28762⟩⟩
def mergeEvent : Nat := 39222
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16389⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28760⟩⟩] } }
def leftRaw : List Term := Proof.Events153.exact39215RawTerms
def rightRaw : List Term := Proof.Events152.exact38938RawTerms
def group : MergeGroup := .operator 39215 38938
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39215) (leftOrdinal := 1)
    (rightResult := 38938) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16389⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28760⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16389⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28760⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39222

namespace LeftMerge39224
def owner : Owner := ⟨.program ⟨214⟩, ⟨28762⟩⟩
def mergeEvent : Nat := 39224
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16389⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24420⟩⟩] } }
def rhsRaw : List Term := Proof.Events152.exact38935RawTerms
def group : MergeGroup := .relation 39223
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39223) (rhsResult := 38935)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16389⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28760⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28760⟩⟩) ⟨24420⟩ 38935) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24420⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16389⟩⟩], [⟨.program ⟨214⟩, ⟨24420⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39224

namespace LeftMerge39238
def owner : Owner := ⟨.program ⟨214⟩, ⟨21987⟩⟩
def mergeEvent : Nat := 39238
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21984⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events153.exact39232RawTerms
def group : MergeGroup := .operator 36137 39232
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 39232) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21984⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21984⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39238

namespace LeftMerge39359
def owner : Owner := ⟨.program ⟨214⟩, ⟨16431⟩⟩
def mergeEvent : Nat := 39359
def frameStart : Nat := 39293
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16389⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events153.exact39355RawTerms
def rightRaw : List Term := Proof.Events153.exact39353RawTerms
def group : MergeGroup := .operator 39355 39353
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39355) (leftOrdinal := 0)
    (rightResult := 39353) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16389⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16389⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39359

namespace LeftMerge39371
def owner : Owner := ⟨.program ⟨214⟩, ⟨28761⟩⟩
def mergeEvent : Nat := 39371
def frameStart : Nat := 39293
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28760⟩⟩] } }
def leftRaw : List Term := Proof.Events153.exact39367RawTerms
def rightRaw : List Term := Proof.Events153.exact39344RawTerms
def group : MergeGroup := .operator 39367 39344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39367) (leftOrdinal := 0)
    (rightResult := 39344) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28760⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28760⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39371

namespace LeftMerge39372
def owner : Owner := ⟨.program ⟨214⟩, ⟨28761⟩⟩
def mergeEvent : Nat := 39372
def frameStart : Nat := 39293
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16389⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28760⟩⟩] } }
def leftRaw : List Term := Proof.Events153.exact39367RawTerms
def rightRaw : List Term := Proof.Events153.exact39344RawTerms
def group : MergeGroup := .operator 39367 39344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39367) (leftOrdinal := 1)
    (rightResult := 39344) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16389⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28760⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16389⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28760⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39372

namespace LeftMerge39374
def owner : Owner := ⟨.program ⟨214⟩, ⟨28761⟩⟩
def mergeEvent : Nat := 39374
def frameStart : Nat := 39293
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16389⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24420⟩⟩] } }
def rhsRaw : List Term := Proof.Events153.exact39341RawTerms
def group : MergeGroup := .relation 39373
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39373) (rhsResult := 39341)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16389⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28760⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28760⟩⟩) ⟨24420⟩ 39341) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24420⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16389⟩⟩], [⟨.program ⟨214⟩, ⟨24420⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39374

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
