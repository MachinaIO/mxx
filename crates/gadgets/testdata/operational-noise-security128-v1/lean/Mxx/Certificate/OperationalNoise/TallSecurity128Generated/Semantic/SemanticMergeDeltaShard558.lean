import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge93168
def owner : Owner := ⟨.program ⟨257⟩, ⟨9552⟩⟩
def mergeEvent : Nat := 93168
def frameStart : Nat := 93085
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }
def leftRaw : List Term := Proof.Events363.exact93164RawTerms
def rightRaw : List Term := Proof.Events363.exact93161RawTerms
def group : MergeGroup := .operator 93164 93161
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93164) (leftOrdinal := 0)
    (rightResult := 93161) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9550⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93168

namespace LeftMerge93177
def owner : Owner := ⟨.program ⟨257⟩, ⟨36317⟩⟩
def mergeEvent : Nat := 93177
def frameStart : Nat := 93085
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36314⟩⟩] } }
def leftRaw : List Term := Proof.Events363.exact93173RawTerms
def rightRaw : List Term := Proof.Events363.exact93130RawTerms
def group : MergeGroup := .operator 93173 93130
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93173) (leftOrdinal := 0)
    (rightResult := 93130) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36314⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36314⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93177

namespace LeftMerge93178
def owner : Owner := ⟨.program ⟨257⟩, ⟨36317⟩⟩
def mergeEvent : Nat := 93178
def frameStart : Nat := 93085
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13656⟩⟩, ⟨.program ⟨257⟩, ⟨34554⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36314⟩⟩] } }
def leftRaw : List Term := Proof.Events363.exact93173RawTerms
def rightRaw : List Term := Proof.Events363.exact93130RawTerms
def group : MergeGroup := .operator 93173 93130
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93173) (leftOrdinal := 1)
    (rightResult := 93130) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13656⟩⟩, ⟨.program ⟨257⟩, ⟨34554⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36314⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13656⟩⟩, ⟨.program ⟨257⟩, ⟨34554⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36314⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93178

namespace LeftMerge93180
def owner : Owner := ⟨.program ⟨257⟩, ⟨36317⟩⟩
def mergeEvent : Nat := 93180
def frameStart : Nat := 93085
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13656⟩⟩, ⟨.program ⟨257⟩, ⟨34554⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35779⟩⟩] } }
def rhsRaw : List Term := Proof.Events363.exact93127RawTerms
def group : MergeGroup := .relation 93179
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 93179) (rhsResult := 93127)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13656⟩⟩, ⟨.program ⟨257⟩, ⟨34554⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36314⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36314⟩⟩) ⟨35779⟩ 93127) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35779⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13656⟩⟩, ⟨.program ⟨257⟩, ⟨34554⟩⟩], [⟨.program ⟨257⟩, ⟨35779⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93180

namespace LeftMerge93188
def owner : Owner := ⟨.program ⟨257⟩, ⟨34790⟩⟩
def mergeEvent : Nat := 93188
def frameStart : Nat := 93085
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events363.exact93141RawTerms
def rightRaw : List Term := Proof.Events364.exact93184RawTerms
def group : MergeGroup := .operator 93141 93184
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93141) (leftOrdinal := 0)
    (rightResult := 93184) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34788⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93188

namespace LeftMerge93205
def owner : Owner := ⟨.program ⟨257⟩, ⟨35242⟩⟩
def mergeEvent : Nat := 93205
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }
def rhsRaw : List Term := Proof.Events364.exact93202RawTerms
def group : MergeGroup := .relation 93204
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 93204) (rhsResult := 93202)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35239⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 93203 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35239⟩⟩]⟩) (none) 93202) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93205

namespace LeftMerge93206
def owner : Owner := ⟨.program ⟨257⟩, ⟨35242⟩⟩
def mergeEvent : Nat := 93206
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36314⟩⟩] } }
def rhsRaw : List Term := Proof.Events364.exact93202RawTerms
def group : MergeGroup := .relation 93204
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 93204) (rhsResult := 93202)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35239⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 93203 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35239⟩⟩]⟩) (none) 93202) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36314⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36314⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93206

namespace LeftMerge93207
def owner : Owner := ⟨.program ⟨257⟩, ⟨35242⟩⟩
def mergeEvent : Nat := 93207
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13656⟩⟩, ⟨.program ⟨257⟩, ⟨34554⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35779⟩⟩] } }
def rhsRaw : List Term := Proof.Events364.exact93202RawTerms
def group : MergeGroup := .relation 93204
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 93204) (rhsResult := 93202)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35239⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 93203 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35239⟩⟩]⟩) (none) 93202) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13656⟩⟩, ⟨.program ⟨257⟩, ⟨34554⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35779⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13656⟩⟩, ⟨.program ⟨257⟩, ⟨34554⟩⟩], [⟨.program ⟨257⟩, ⟨35779⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93207

namespace LeftMerge93208
def owner : Owner := ⟨.program ⟨257⟩, ⟨35242⟩⟩
def mergeEvent : Nat := 93208
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events364.exact93202RawTerms
def group : MergeGroup := .relation 93204
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 93204) (rhsResult := 93202)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35239⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 93203 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35239⟩⟩]⟩) (none) 93202) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93208

namespace LeftMerge93213
def owner : Owner := ⟨.program ⟨257⟩, ⟨36316⟩⟩
def mergeEvent : Nat := 93213
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13656⟩⟩, ⟨.program ⟨257⟩, ⟨34554⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35779⟩⟩] } }
def leftRaw : List Term := Proof.Events364.exact93209RawTerms
def rightRaw : List Term := Proof.Events363.exact93023RawTerms
def group : MergeGroup := .operator 93209 93023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93209) (leftOrdinal := 2)
    (rightResult := 93023) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13656⟩⟩, ⟨.program ⟨257⟩, ⟨34554⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35779⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13656⟩⟩, ⟨.program ⟨257⟩, ⟨34554⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35779⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13656⟩⟩, ⟨.program ⟨257⟩, ⟨34554⟩⟩], [⟨.program ⟨257⟩, ⟨35779⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93213

namespace LeftMerge93214
def owner : Owner := ⟨.program ⟨257⟩, ⟨36316⟩⟩
def mergeEvent : Nat := 93214
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36314⟩⟩] } }
def leftRaw : List Term := Proof.Events364.exact93209RawTerms
def rightRaw : List Term := Proof.Events363.exact93023RawTerms
def group : MergeGroup := .operator 93209 93023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93209) (leftOrdinal := 1)
    (rightResult := 93023) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36314⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36314⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36314⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93214

namespace LeftMerge93222
def owner : Owner := ⟨.program ⟨257⟩, ⟨36756⟩⟩
def mergeEvent : Nat := 93222
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36754⟩⟩] } }
def leftRaw : List Term := Proof.Events364.exact93216RawTerms
def rightRaw : List Term := Proof.Events363.exact92939RawTerms
def group : MergeGroup := .operator 93216 92939
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93216) (leftOrdinal := 0)
    (rightResult := 92939) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36754⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36754⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93222

namespace LeftMerge93223
def owner : Owner := ⟨.program ⟨257⟩, ⟨36756⟩⟩
def mergeEvent : Nat := 93223
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36754⟩⟩] } }
def leftRaw : List Term := Proof.Events364.exact93216RawTerms
def rightRaw : List Term := Proof.Events363.exact92939RawTerms
def group : MergeGroup := .operator 93216 92939
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93216) (leftOrdinal := 1)
    (rightResult := 92939) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36754⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36754⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93223

namespace LeftMerge93225
def owner : Owner := ⟨.program ⟨257⟩, ⟨36756⟩⟩
def mergeEvent : Nat := 93225
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35946⟩⟩] } }
def rhsRaw : List Term := Proof.Events363.exact92936RawTerms
def group : MergeGroup := .relation 93224
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 93224) (rhsResult := 92936)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36754⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36754⟩⟩) ⟨35946⟩ 92936) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35946⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨35946⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93225

namespace LeftMerge93239
def owner : Owner := ⟨.program ⟨257⟩, ⟨35599⟩⟩
def mergeEvent : Nat := 93239
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35596⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90620RawTerms
def rightRaw : List Term := Proof.Events364.exact93233RawTerms
def group : MergeGroup := .operator 90620 93233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90620) (leftOrdinal := 0)
    (rightResult := 93233) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35596⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35596⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93239

namespace LeftMerge93360
def owner : Owner := ⟨.program ⟨257⟩, ⟨36128⟩⟩
def mergeEvent : Nat := 93360
def frameStart : Nat := 93294
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events364.exact93356RawTerms
def rightRaw : List Term := Proof.Events364.exact93354RawTerms
def group : MergeGroup := .operator 93356 93354
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93356) (leftOrdinal := 0)
    (rightResult := 93354) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34788⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93360

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
