import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge57166
def owner : Owner := ⟨.program ⟨214⟩, ⟨7845⟩⟩
def mergeEvent : Nat := 57166
def frameStart : Nat := 57083
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩] } }
def leftRaw : List Term := Proof.Events223.exact57162RawTerms
def rightRaw : List Term := Proof.Events223.exact57159RawTerms
def group : MergeGroup := .operator 57162 57159
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57162) (leftOrdinal := 0)
    (rightResult := 57159) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7843⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57166

namespace LeftMerge57175
def owner : Owner := ⟨.program ⟨214⟩, ⟨25843⟩⟩
def mergeEvent : Nat := 57175
def frameStart : Nat := 57083
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25840⟩⟩] } }
def leftRaw : List Term := Proof.Events223.exact57171RawTerms
def rightRaw : List Term := Proof.Events223.exact57128RawTerms
def group : MergeGroup := .operator 57171 57128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57171) (leftOrdinal := 0)
    (rightResult := 57128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25840⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25840⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57175

namespace LeftMerge57176
def owner : Owner := ⟨.program ⟨214⟩, ⟨25843⟩⟩
def mergeEvent : Nat := 57176
def frameStart : Nat := 57083
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11221⟩⟩, ⟨.program ⟨214⟩, ⟨13565⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25840⟩⟩] } }
def leftRaw : List Term := Proof.Events223.exact57171RawTerms
def rightRaw : List Term := Proof.Events223.exact57128RawTerms
def group : MergeGroup := .operator 57171 57128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57171) (leftOrdinal := 1)
    (rightResult := 57128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11221⟩⟩, ⟨.program ⟨214⟩, ⟨13565⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25840⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11221⟩⟩, ⟨.program ⟨214⟩, ⟨13565⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25840⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57176

namespace LeftMerge57178
def owner : Owner := ⟨.program ⟨214⟩, ⟨25843⟩⟩
def mergeEvent : Nat := 57178
def frameStart : Nat := 57083
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11221⟩⟩, ⟨.program ⟨214⟩, ⟨13565⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23460⟩⟩] } }
def rhsRaw : List Term := Proof.Events223.exact57125RawTerms
def group : MergeGroup := .relation 57177
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 57177) (rhsResult := 57125)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11221⟩⟩, ⟨.program ⟨214⟩, ⟨13565⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25840⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25840⟩⟩) ⟨23460⟩ 57125) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23460⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11221⟩⟩, ⟨.program ⟨214⟩, ⟨13565⟩⟩], [⟨.program ⟨214⟩, ⟨23460⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57178

namespace LeftMerge57186
def owner : Owner := ⟨.program ⟨214⟩, ⟨15589⟩⟩
def mergeEvent : Nat := 57186
def frameStart : Nat := 57083
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15587⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events223.exact57139RawTerms
def rightRaw : List Term := Proof.Events223.exact57182RawTerms
def group : MergeGroup := .operator 57139 57182
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57139) (leftOrdinal := 0)
    (rightResult := 57182) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15587⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15587⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57186

namespace LeftMerge57203
def owner : Owner := ⟨.program ⟨214⟩, ⟨19319⟩⟩
def mergeEvent : Nat := 57203
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6694⟩⟩] } }
def rhsRaw : List Term := Proof.Events223.exact57200RawTerms
def group : MergeGroup := .relation 57202
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 57202) (rhsResult := 57200)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19316⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 57201 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19316⟩⟩]⟩) (none) 57200) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6694⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57203

namespace LeftMerge57204
def owner : Owner := ⟨.program ⟨214⟩, ⟨19319⟩⟩
def mergeEvent : Nat := 57204
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25840⟩⟩] } }
def rhsRaw : List Term := Proof.Events223.exact57200RawTerms
def group : MergeGroup := .relation 57202
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 57202) (rhsResult := 57200)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19316⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 57201 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19316⟩⟩]⟩) (none) 57200) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25840⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25840⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57204

namespace LeftMerge57205
def owner : Owner := ⟨.program ⟨214⟩, ⟨19319⟩⟩
def mergeEvent : Nat := 57205
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11221⟩⟩, ⟨.program ⟨214⟩, ⟨13565⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23460⟩⟩] } }
def rhsRaw : List Term := Proof.Events223.exact57200RawTerms
def group : MergeGroup := .relation 57202
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 57202) (rhsResult := 57200)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19316⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 57201 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19316⟩⟩]⟩) (none) 57200) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11221⟩⟩, ⟨.program ⟨214⟩, ⟨13565⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23460⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11221⟩⟩, ⟨.program ⟨214⟩, ⟨13565⟩⟩], [⟨.program ⟨214⟩, ⟨23460⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57205

namespace LeftMerge57206
def owner : Owner := ⟨.program ⟨214⟩, ⟨19319⟩⟩
def mergeEvent : Nat := 57206
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15587⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events223.exact57200RawTerms
def group : MergeGroup := .relation 57202
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 57202) (rhsResult := 57200)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19316⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 57201 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19316⟩⟩]⟩) (none) 57200) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15587⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15587⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57206

namespace LeftMerge57211
def owner : Owner := ⟨.program ⟨214⟩, ⟨25842⟩⟩
def mergeEvent : Nat := 57211
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11221⟩⟩, ⟨.program ⟨214⟩, ⟨13565⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23460⟩⟩] } }
def leftRaw : List Term := Proof.Events223.exact57207RawTerms
def rightRaw : List Term := Proof.Events222.exact57021RawTerms
def group : MergeGroup := .operator 57207 57021
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57207) (leftOrdinal := 2)
    (rightResult := 57021) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11221⟩⟩, ⟨.program ⟨214⟩, ⟨13565⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23460⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11221⟩⟩, ⟨.program ⟨214⟩, ⟨13565⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23460⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11221⟩⟩, ⟨.program ⟨214⟩, ⟨13565⟩⟩], [⟨.program ⟨214⟩, ⟨23460⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57211

namespace LeftMerge57212
def owner : Owner := ⟨.program ⟨214⟩, ⟨25842⟩⟩
def mergeEvent : Nat := 57212
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25840⟩⟩] } }
def leftRaw : List Term := Proof.Events223.exact57207RawTerms
def rightRaw : List Term := Proof.Events222.exact57021RawTerms
def group : MergeGroup := .operator 57207 57021
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57207) (leftOrdinal := 1)
    (rightResult := 57021) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25840⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25840⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25840⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57212

namespace LeftMerge57220
def owner : Owner := ⟨.program ⟨214⟩, ⟨27230⟩⟩
def mergeEvent : Nat := 57220
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27228⟩⟩] } }
def leftRaw : List Term := Proof.Events223.exact57214RawTerms
def rightRaw : List Term := Proof.Events222.exact56937RawTerms
def group : MergeGroup := .operator 57214 56937
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57214) (leftOrdinal := 0)
    (rightResult := 56937) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6694⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27228⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27228⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57220

namespace LeftMerge57221
def owner : Owner := ⟨.program ⟨214⟩, ⟨27230⟩⟩
def mergeEvent : Nat := 57221
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15587⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27228⟩⟩] } }
def leftRaw : List Term := Proof.Events223.exact57214RawTerms
def rightRaw : List Term := Proof.Events222.exact56937RawTerms
def group : MergeGroup := .operator 57214 56937
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57214) (leftOrdinal := 1)
    (rightResult := 56937) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15587⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27228⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15587⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27228⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57221

namespace LeftMerge57223
def owner : Owner := ⟨.program ⟨214⟩, ⟨27230⟩⟩
def mergeEvent : Nat := 57223
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15587⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23976⟩⟩] } }
def rhsRaw : List Term := Proof.Events222.exact56934RawTerms
def group : MergeGroup := .relation 57222
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 57222) (rhsResult := 56934)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15587⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27228⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27228⟩⟩) ⟨23976⟩ 56934) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23976⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15587⟩⟩], [⟨.program ⟨214⟩, ⟨23976⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57223

namespace LeftMerge57237
def owner : Owner := ⟨.program ⟨214⟩, ⟨20975⟩⟩
def mergeEvent : Nat := 57237
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20972⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events223.exact57231RawTerms
def group : MergeGroup := .operator 50762 57231
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 57231) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20972⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20972⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57237

namespace LeftMerge57358
def owner : Owner := ⟨.program ⟨214⟩, ⟨15664⟩⟩
def mergeEvent : Nat := 57358
def frameStart : Nat := 57292
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15587⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events224.exact57354RawTerms
def rightRaw : List Term := Proof.Events224.exact57352RawTerms
def group : MergeGroup := .operator 57354 57352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57354) (leftOrdinal := 0)
    (rightResult := 57352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15587⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15587⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57358

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
