import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge227046
def owner : Owner := ⟨.program ⟨257⟩, ⟨59466⟩⟩
def mergeEvent : Nat := 227046
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }
def leftRaw : List Term := Proof.Events886.exact227042RawTerms
def rightRaw : List Term := Proof.Events886.exact227012RawTerms
def group : MergeGroup := .operator 227042 227012
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227042) (leftOrdinal := 1)
    (rightResult := 227012) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227046

namespace LeftMerge227054
def owner : Owner := ⟨.program ⟨257⟩, ⟨61449⟩⟩
def mergeEvent : Nat := 227054
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61448⟩⟩] } }
def leftRaw : List Term := Proof.Events886.exact227048RawTerms
def rightRaw : List Term := Proof.Events886.exact226984RawTerms
def group : MergeGroup := .operator 227048 226984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227048) (leftOrdinal := 1)
    (rightResult := 226984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61448⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61448⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge227054

namespace LeftMerge227056
def owner : Owner := ⟨.program ⟨257⟩, ⟨61449⟩⟩
def mergeEvent : Nat := 227056
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60943⟩⟩] } }
def rhsRaw : List Term := Proof.Events886.exact226981RawTerms
def group : MergeGroup := .relation 227055
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 227055) (rhsResult := 226981)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61448⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61448⟩⟩) ⟨60943⟩ 226981) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60943⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], [⟨.program ⟨257⟩, ⟨60943⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge227056

namespace LeftMerge227057
def owner : Owner := ⟨.program ⟨257⟩, ⟨61449⟩⟩
def mergeEvent : Nat := 227057
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61448⟩⟩] } }
def leftRaw : List Term := Proof.Events886.exact227048RawTerms
def rightRaw : List Term := Proof.Events886.exact226984RawTerms
def group : MergeGroup := .operator 227048 226984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227048) (leftOrdinal := 0)
    (rightResult := 226984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61448⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61448⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227057

namespace LeftMerge227071
def owner : Owner := ⟨.program ⟨257⟩, ⟨60382⟩⟩
def mergeEvent : Nat := 227071
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60379⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events886.exact227065RawTerms
def group : MergeGroup := .operator 222245 227065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 227065) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60379⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60379⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227071

namespace LeftMerge227150
def owner : Owner := ⟨.program ⟨257⟩, ⟨59459⟩⟩
def mergeEvent : Nat := 227150
def frameStart : Nat := 227120
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events887.exact227146RawTerms
def rightRaw : List Term := Proof.Events887.exact227143RawTerms
def group : MergeGroup := .operator 227146 227143
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227146) (leftOrdinal := 0)
    (rightResult := 227143) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59458⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25238⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227150

namespace LeftMerge227180
def owner : Owner := ⟨.program ⟨257⟩, ⟨61224⟩⟩
def mergeEvent : Nat := 227180
def frameStart : Nat := 227120
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events887.exact227176RawTerms
def rightRaw : List Term := Proof.Events887.exact227174RawTerms
def group : MergeGroup := .operator 227176 227174
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227176) (leftOrdinal := 0)
    (rightResult := 227174) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227180

namespace LeftMerge227203
def owner : Owner := ⟨.program ⟨257⟩, ⟨9537⟩⟩
def mergeEvent : Nat := 227203
def frameStart : Nat := 227120
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }
def leftRaw : List Term := Proof.Events887.exact227199RawTerms
def rightRaw : List Term := Proof.Events887.exact227196RawTerms
def group : MergeGroup := .operator 227199 227196
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227199) (leftOrdinal := 0)
    (rightResult := 227196) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9535⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227203

namespace LeftMerge227212
def owner : Owner := ⟨.program ⟨257⟩, ⟨61451⟩⟩
def mergeEvent : Nat := 227212
def frameStart : Nat := 227120
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61448⟩⟩] } }
def leftRaw : List Term := Proof.Events887.exact227208RawTerms
def rightRaw : List Term := Proof.Events887.exact227165RawTerms
def group : MergeGroup := .operator 227208 227165
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227208) (leftOrdinal := 0)
    (rightResult := 227165) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61448⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61448⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227212

namespace LeftMerge227213
def owner : Owner := ⟨.program ⟨257⟩, ⟨61451⟩⟩
def mergeEvent : Nat := 227213
def frameStart : Nat := 227120
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61448⟩⟩] } }
def leftRaw : List Term := Proof.Events887.exact227208RawTerms
def rightRaw : List Term := Proof.Events887.exact227165RawTerms
def group : MergeGroup := .operator 227208 227165
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227208) (leftOrdinal := 1)
    (rightResult := 227165) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61448⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61448⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge227213

namespace LeftMerge227215
def owner : Owner := ⟨.program ⟨257⟩, ⟨61451⟩⟩
def mergeEvent : Nat := 227215
def frameStart : Nat := 227120
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60943⟩⟩] } }
def rhsRaw : List Term := Proof.Events887.exact227162RawTerms
def group : MergeGroup := .relation 227214
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 227214) (rhsResult := 227162)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61448⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61448⟩⟩) ⟨60943⟩ 227162) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60943⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], [⟨.program ⟨257⟩, ⟨60943⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge227215

namespace LeftMerge227223
def owner : Owner := ⟨.program ⟨257⟩, ⟨59822⟩⟩
def mergeEvent : Nat := 227223
def frameStart : Nat := 227120
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events887.exact227176RawTerms
def rightRaw : List Term := Proof.Events887.exact227219RawTerms
def group : MergeGroup := .operator 227176 227219
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227176) (leftOrdinal := 0)
    (rightResult := 227219) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59820⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227223

namespace LeftMerge227240
def owner : Owner := ⟨.program ⟨257⟩, ⟨60382⟩⟩
def mergeEvent : Nat := 227240
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }
def rhsRaw : List Term := Proof.Events887.exact227237RawTerms
def group : MergeGroup := .relation 227239
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 227239) (rhsResult := 227237)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 227238 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60379⟩⟩]⟩) (none) 227237) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227240

namespace LeftMerge227241
def owner : Owner := ⟨.program ⟨257⟩, ⟨60382⟩⟩
def mergeEvent : Nat := 227241
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61448⟩⟩] } }
def rhsRaw : List Term := Proof.Events887.exact227237RawTerms
def group : MergeGroup := .relation 227239
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 227239) (rhsResult := 227237)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 227238 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60379⟩⟩]⟩) (none) 227237) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61448⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61448⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge227241

namespace LeftMerge227242
def owner : Owner := ⟨.program ⟨257⟩, ⟨60382⟩⟩
def mergeEvent : Nat := 227242
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60943⟩⟩] } }
def rhsRaw : List Term := Proof.Events887.exact227237RawTerms
def group : MergeGroup := .relation 227239
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 227239) (rhsResult := 227237)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 227238 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60379⟩⟩]⟩) (none) 227237) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60943⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], [⟨.program ⟨257⟩, ⟨60943⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227242

namespace LeftMerge227243
def owner : Owner := ⟨.program ⟨257⟩, ⟨60382⟩⟩
def mergeEvent : Nat := 227243
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events887.exact227237RawTerms
def group : MergeGroup := .relation 227239
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 227239) (rhsResult := 227237)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 227238 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60379⟩⟩]⟩) (none) 227237) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge227243

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
