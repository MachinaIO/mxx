import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge14226
def owner : Owner := ⟨.program ⟨214⟩, ⟨25088⟩⟩
def mergeEvent : Nat := 14226
def frameStart : Nat := 14131
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25085⟩⟩] } }
def leftRaw : List Term := Proof.Events055.exact14219RawTerms
def rightRaw : List Term := Proof.Events055.exact14176RawTerms
def group : MergeGroup := .operator 14219 14176
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14219) (leftOrdinal := 0)
    (rightResult := 14176) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25085⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25085⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14226

namespace LeftMerge14234
def owner : Owner := ⟨.program ⟨214⟩, ⟨15132⟩⟩
def mergeEvent : Nat := 14234
def frameStart : Nat := 14131
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15130⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events055.exact14187RawTerms
def rightRaw : List Term := Proof.Events055.exact14230RawTerms
def group : MergeGroup := .operator 14187 14230
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14187) (leftOrdinal := 0)
    (rightResult := 14230) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15130⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15130⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14234

namespace LeftMerge14251
def owner : Owner := ⟨.program ⟨214⟩, ⟨19187⟩⟩
def mergeEvent : Nat := 14251
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10862⟩⟩, ⟨.program ⟨214⟩, ⟨11009⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23046⟩⟩] } }
def rhsRaw : List Term := Proof.Events055.exact14248RawTerms
def group : MergeGroup := .relation 14250
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 14250) (rhsResult := 14248)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19184⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 14249 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19184⟩⟩]⟩) (none) 14248) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10862⟩⟩, ⟨.program ⟨214⟩, ⟨11009⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23046⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10862⟩⟩, ⟨.program ⟨214⟩, ⟨11009⟩⟩], [⟨.program ⟨214⟩, ⟨23046⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14251

namespace LeftMerge14252
def owner : Owner := ⟨.program ⟨214⟩, ⟨19187⟩⟩
def mergeEvent : Nat := 14252
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25085⟩⟩] } }
def rhsRaw : List Term := Proof.Events055.exact14248RawTerms
def group : MergeGroup := .relation 14250
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 14250) (rhsResult := 14248)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19184⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 14249 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19184⟩⟩]⟩) (none) 14248) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25085⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25085⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge14252

namespace LeftMerge14253
def owner : Owner := ⟨.program ⟨214⟩, ⟨19187⟩⟩
def mergeEvent : Nat := 14253
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15130⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events055.exact14248RawTerms
def group : MergeGroup := .relation 14250
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 14250) (rhsResult := 14248)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19184⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 14249 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19184⟩⟩]⟩) (none) 14248) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15130⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15130⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge14253

namespace LeftMerge14254
def owner : Owner := ⟨.program ⟨214⟩, ⟨19187⟩⟩
def mergeEvent : Nat := 14254
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩] } }
def rhsRaw : List Term := Proof.Events055.exact14248RawTerms
def group : MergeGroup := .relation 14250
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 14250) (rhsResult := 14248)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19184⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 14249 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19184⟩⟩]⟩) (none) 14248) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14254

namespace LeftMerge14259
def owner : Owner := ⟨.program ⟨214⟩, ⟨25087⟩⟩
def mergeEvent : Nat := 14259
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10862⟩⟩, ⟨.program ⟨214⟩, ⟨11009⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23046⟩⟩] } }
def leftRaw : List Term := Proof.Events055.exact14255RawTerms
def rightRaw : List Term := Proof.Events054.exact14069RawTerms
def group : MergeGroup := .operator 14255 14069
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14255) (leftOrdinal := 2)
    (rightResult := 14069) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10862⟩⟩, ⟨.program ⟨214⟩, ⟨11009⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23046⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10862⟩⟩, ⟨.program ⟨214⟩, ⟨11009⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23046⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10862⟩⟩, ⟨.program ⟨214⟩, ⟨11009⟩⟩], [⟨.program ⟨214⟩, ⟨23046⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge14259

namespace LeftMerge14260
def owner : Owner := ⟨.program ⟨214⟩, ⟨25087⟩⟩
def mergeEvent : Nat := 14260
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25085⟩⟩] } }
def leftRaw : List Term := Proof.Events055.exact14255RawTerms
def rightRaw : List Term := Proof.Events054.exact14069RawTerms
def group : MergeGroup := .operator 14255 14069
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14255) (leftOrdinal := 1)
    (rightResult := 14069) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25085⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25085⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25085⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14260

namespace LeftMerge14268
def owner : Owner := ⟨.program ⟨214⟩, ⟨26835⟩⟩
def mergeEvent : Nat := 14268
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15130⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26833⟩⟩] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events054.exact13966RawTerms
def group : MergeGroup := .operator 14262 13966
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 1)
    (rightResult := 13966) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15130⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26833⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15130⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26833⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge14268

namespace LeftMerge14270
def owner : Owner := ⟨.program ⟨214⟩, ⟨26835⟩⟩
def mergeEvent : Nat := 14270
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15130⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23859⟩⟩] } }
def rhsRaw : List Term := Proof.Events054.exact13963RawTerms
def group : MergeGroup := .relation 14269
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 14269) (rhsResult := 13963)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15130⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26833⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26833⟩⟩) ⟨23859⟩ 13963) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23859⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15130⟩⟩], [⟨.program ⟨214⟩, ⟨23859⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge14270

namespace LeftMerge14271
def owner : Owner := ⟨.program ⟨214⟩, ⟨26835⟩⟩
def mergeEvent : Nat := 14271
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26833⟩⟩] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events054.exact13966RawTerms
def group : MergeGroup := .operator 14262 13966
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 0)
    (rightResult := 13966) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26833⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26833⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14271

namespace LeftMerge14285
def owner : Owner := ⟨.program ⟨214⟩, ⟨20699⟩⟩
def mergeEvent : Nat := 14285
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20696⟩⟩] } }
def leftRaw : List Term := Proof.Events025.exact6561RawTerms
def rightRaw : List Term := Proof.Events055.exact14279RawTerms
def group : MergeGroup := .operator 6561 14279
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6561) (leftOrdinal := 0)
    (rightResult := 14279) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20696⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20696⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14285

namespace LeftMerge14406
def owner : Owner := ⟨.program ⟨214⟩, ⟨15172⟩⟩
def mergeEvent : Nat := 14406
def frameStart : Nat := 14340
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15130⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events056.exact14402RawTerms
def rightRaw : List Term := Proof.Events056.exact14400RawTerms
def group : MergeGroup := .operator 14402 14400
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14402) (leftOrdinal := 0)
    (rightResult := 14400) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15130⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15130⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14406

namespace LeftMerge14418
def owner : Owner := ⟨.program ⟨214⟩, ⟨26834⟩⟩
def mergeEvent : Nat := 14418
def frameStart : Nat := 14340
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15130⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26833⟩⟩] } }
def leftRaw : List Term := Proof.Events056.exact14414RawTerms
def rightRaw : List Term := Proof.Events056.exact14391RawTerms
def group : MergeGroup := .operator 14414 14391
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14414) (leftOrdinal := 1)
    (rightResult := 14391) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15130⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26833⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15130⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26833⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge14418

namespace LeftMerge14420
def owner : Owner := ⟨.program ⟨214⟩, ⟨26834⟩⟩
def mergeEvent : Nat := 14420
def frameStart : Nat := 14340
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15130⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23859⟩⟩] } }
def rhsRaw : List Term := Proof.Events056.exact14388RawTerms
def group : MergeGroup := .relation 14419
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 14419) (rhsResult := 14388)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15130⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26833⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26833⟩⟩) ⟨23859⟩ 14388) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23859⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15130⟩⟩], [⟨.program ⟨214⟩, ⟨23859⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge14420

namespace LeftMerge14421
def owner : Owner := ⟨.program ⟨214⟩, ⟨26834⟩⟩
def mergeEvent : Nat := 14421
def frameStart : Nat := 14340
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26833⟩⟩] } }
def leftRaw : List Term := Proof.Events056.exact14414RawTerms
def rightRaw : List Term := Proof.Events056.exact14391RawTerms
def group : MergeGroup := .operator 14414 14391
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14414) (leftOrdinal := 0)
    (rightResult := 14391) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26833⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26833⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14421

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
