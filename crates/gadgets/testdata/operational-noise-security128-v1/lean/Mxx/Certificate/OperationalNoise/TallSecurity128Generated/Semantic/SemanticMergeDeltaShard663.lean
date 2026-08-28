import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge110212
def owner : Owner := ⟨.program ⟨257⟩, ⟨61473⟩⟩
def mergeEvent : Nat := 110212
def frameStart : Nat := 110120
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61470⟩⟩] } }
def leftRaw : List Term := Proof.Events430.exact110208RawTerms
def rightRaw : List Term := Proof.Events430.exact110165RawTerms
def group : MergeGroup := .operator 110208 110165
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 110208) (leftOrdinal := 0)
    (rightResult := 110165) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61470⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61470⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge110212

namespace LeftMerge110213
def owner : Owner := ⟨.program ⟨257⟩, ⟨61473⟩⟩
def mergeEvent : Nat := 110213
def frameStart : Nat := 110120
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25262⟩⟩, ⟨.program ⟨257⟩, ⟨59512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61470⟩⟩] } }
def leftRaw : List Term := Proof.Events430.exact110208RawTerms
def rightRaw : List Term := Proof.Events430.exact110165RawTerms
def group : MergeGroup := .operator 110208 110165
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 110208) (leftOrdinal := 1)
    (rightResult := 110165) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25262⟩⟩, ⟨.program ⟨257⟩, ⟨59512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61470⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25262⟩⟩, ⟨.program ⟨257⟩, ⟨59512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61470⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge110213

namespace LeftMerge110215
def owner : Owner := ⟨.program ⟨257⟩, ⟨61473⟩⟩
def mergeEvent : Nat := 110215
def frameStart : Nat := 110120
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25262⟩⟩, ⟨.program ⟨257⟩, ⟨59512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60955⟩⟩] } }
def rhsRaw : List Term := Proof.Events430.exact110162RawTerms
def group : MergeGroup := .relation 110214
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 110214) (rhsResult := 110162)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25262⟩⟩, ⟨.program ⟨257⟩, ⟨59512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61470⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61470⟩⟩) ⟨60955⟩ 110162) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60955⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25262⟩⟩, ⟨.program ⟨257⟩, ⟨59512⟩⟩], [⟨.program ⟨257⟩, ⟨60955⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge110215

namespace LeftMerge110223
def owner : Owner := ⟨.program ⟨257⟩, ⟨59838⟩⟩
def mergeEvent : Nat := 110223
def frameStart : Nat := 110120
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events430.exact110176RawTerms
def rightRaw : List Term := Proof.Events430.exact110219RawTerms
def group : MergeGroup := .operator 110176 110219
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 110176) (leftOrdinal := 0)
    (rightResult := 110219) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59836⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge110223

namespace LeftMerge110240
def owner : Owner := ⟨.program ⟨257⟩, ⟨60402⟩⟩
def mergeEvent : Nat := 110240
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }
def rhsRaw : List Term := Proof.Events430.exact110237RawTerms
def group : MergeGroup := .relation 110239
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 110239) (rhsResult := 110237)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60399⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 110238 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60399⟩⟩]⟩) (none) 110237) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge110240

namespace LeftMerge110241
def owner : Owner := ⟨.program ⟨257⟩, ⟨60402⟩⟩
def mergeEvent : Nat := 110241
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61470⟩⟩] } }
def rhsRaw : List Term := Proof.Events430.exact110237RawTerms
def group : MergeGroup := .relation 110239
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 110239) (rhsResult := 110237)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60399⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 110238 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60399⟩⟩]⟩) (none) 110237) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61470⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61470⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge110241

namespace LeftMerge110242
def owner : Owner := ⟨.program ⟨257⟩, ⟨60402⟩⟩
def mergeEvent : Nat := 110242
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25262⟩⟩, ⟨.program ⟨257⟩, ⟨59512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60955⟩⟩] } }
def rhsRaw : List Term := Proof.Events430.exact110237RawTerms
def group : MergeGroup := .relation 110239
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 110239) (rhsResult := 110237)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60399⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 110238 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60399⟩⟩]⟩) (none) 110237) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25262⟩⟩, ⟨.program ⟨257⟩, ⟨59512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60955⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25262⟩⟩, ⟨.program ⟨257⟩, ⟨59512⟩⟩], [⟨.program ⟨257⟩, ⟨60955⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge110242

namespace LeftMerge110243
def owner : Owner := ⟨.program ⟨257⟩, ⟨60402⟩⟩
def mergeEvent : Nat := 110243
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨59836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events430.exact110237RawTerms
def group : MergeGroup := .relation 110239
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 110239) (rhsResult := 110237)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60399⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 110238 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60399⟩⟩]⟩) (none) 110237) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨59836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge110243

namespace LeftMerge110248
def owner : Owner := ⟨.program ⟨257⟩, ⟨61472⟩⟩
def mergeEvent : Nat := 110248
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25262⟩⟩, ⟨.program ⟨257⟩, ⟨59512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60955⟩⟩] } }
def leftRaw : List Term := Proof.Events430.exact110244RawTerms
def rightRaw : List Term := Proof.Events429.exact110058RawTerms
def group : MergeGroup := .operator 110244 110058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 110244) (leftOrdinal := 2)
    (rightResult := 110058) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25262⟩⟩, ⟨.program ⟨257⟩, ⟨59512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60955⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25262⟩⟩, ⟨.program ⟨257⟩, ⟨59512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60955⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25262⟩⟩, ⟨.program ⟨257⟩, ⟨59512⟩⟩], [⟨.program ⟨257⟩, ⟨60955⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge110248

namespace LeftMerge110249
def owner : Owner := ⟨.program ⟨257⟩, ⟨61472⟩⟩
def mergeEvent : Nat := 110249
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61470⟩⟩] } }
def leftRaw : List Term := Proof.Events430.exact110244RawTerms
def rightRaw : List Term := Proof.Events429.exact110058RawTerms
def group : MergeGroup := .operator 110244 110058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 110244) (leftOrdinal := 1)
    (rightResult := 110058) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61470⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61470⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61470⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge110249

namespace LeftMerge110257
def owner : Owner := ⟨.program ⟨257⟩, ⟨61925⟩⟩
def mergeEvent : Nat := 110257
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61923⟩⟩] } }
def leftRaw : List Term := Proof.Events430.exact110251RawTerms
def rightRaw : List Term := Proof.Events429.exact109974RawTerms
def group : MergeGroup := .operator 110251 109974
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 110251) (leftOrdinal := 0)
    (rightResult := 109974) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61923⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61923⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge110257

namespace LeftMerge110258
def owner : Owner := ⟨.program ⟨257⟩, ⟨61925⟩⟩
def mergeEvent : Nat := 110258
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨59836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61923⟩⟩] } }
def leftRaw : List Term := Proof.Events430.exact110251RawTerms
def rightRaw : List Term := Proof.Events429.exact109974RawTerms
def group : MergeGroup := .operator 110251 109974
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 110251) (leftOrdinal := 1)
    (rightResult := 109974) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨59836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61923⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨59836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61923⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge110258

namespace LeftMerge110260
def owner : Owner := ⟨.program ⟨257⟩, ⟨61925⟩⟩
def mergeEvent : Nat := 110260
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨59836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61110⟩⟩] } }
def rhsRaw : List Term := Proof.Events429.exact109971RawTerms
def group : MergeGroup := .relation 110259
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 110259) (rhsResult := 109971)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨59836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61923⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61923⟩⟩) ⟨61110⟩ 109971) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61110⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨59836⟩⟩], [⟨.program ⟨257⟩, ⟨61110⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge110260

namespace LeftMerge110274
def owner : Owner := ⟨.program ⟨257⟩, ⟨60719⟩⟩
def mergeEvent : Nat := 110274
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60716⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105245RawTerms
def rightRaw : List Term := Proof.Events430.exact110268RawTerms
def group : MergeGroup := .operator 105245 110268
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105245) (leftOrdinal := 0)
    (rightResult := 110268) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60716⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60716⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge110274

namespace LeftMerge110395
def owner : Owner := ⟨.program ⟨257⟩, ⟨61312⟩⟩
def mergeEvent : Nat := 110395
def frameStart : Nat := 110329
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events431.exact110391RawTerms
def rightRaw : List Term := Proof.Events431.exact110389RawTerms
def group : MergeGroup := .operator 110391 110389
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 110391) (leftOrdinal := 0)
    (rightResult := 110389) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59836⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge110395

namespace LeftMerge110407
def owner : Owner := ⟨.program ⟨257⟩, ⟨61924⟩⟩
def mergeEvent : Nat := 110407
def frameStart : Nat := 110329
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61923⟩⟩] } }
def leftRaw : List Term := Proof.Events431.exact110403RawTerms
def rightRaw : List Term := Proof.Events431.exact110380RawTerms
def group : MergeGroup := .operator 110403 110380
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 110403) (leftOrdinal := 0)
    (rightResult := 110380) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61923⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61923⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge110407

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
