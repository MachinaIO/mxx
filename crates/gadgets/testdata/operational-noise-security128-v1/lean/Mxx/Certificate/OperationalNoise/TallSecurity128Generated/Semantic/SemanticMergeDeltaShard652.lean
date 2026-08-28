import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge108118
def owner : Owner := ⟨.program ⟨257⟩, ⟨28805⟩⟩
def mergeEvent : Nat := 108118
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13296⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }
def leftRaw : List Term := Proof.Events422.exact108114RawTerms
def rightRaw : List Term := Proof.Events422.exact108084RawTerms
def group : MergeGroup := .operator 108114 108084
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108114) (leftOrdinal := 1)
    (rightResult := 108084) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13296⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13296⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13296⟩⟩], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108118

namespace LeftMerge108126
def owner : Owner := ⟨.program ⟨257⟩, ⟨30611⟩⟩
def mergeEvent : Nat := 108126
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30610⟩⟩] } }
def leftRaw : List Term := Proof.Events422.exact108120RawTerms
def rightRaw : List Term := Proof.Events422.exact108056RawTerms
def group : MergeGroup := .operator 108120 108056
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108120) (leftOrdinal := 1)
    (rightResult := 108056) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30610⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30610⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108126

namespace LeftMerge108128
def owner : Owner := ⟨.program ⟨257⟩, ⟨30611⟩⟩
def mergeEvent : Nat := 108128
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30095⟩⟩] } }
def rhsRaw : List Term := Proof.Events422.exact108053RawTerms
def group : MergeGroup := .relation 108127
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108127) (rhsResult := 108053)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30610⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30610⟩⟩) ⟨30095⟩ 108053) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30095⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], [⟨.program ⟨257⟩, ⟨30095⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108128

namespace LeftMerge108129
def owner : Owner := ⟨.program ⟨257⟩, ⟨30611⟩⟩
def mergeEvent : Nat := 108129
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30610⟩⟩] } }
def leftRaw : List Term := Proof.Events422.exact108120RawTerms
def rightRaw : List Term := Proof.Events422.exact108056RawTerms
def group : MergeGroup := .operator 108120 108056
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108120) (leftOrdinal := 0)
    (rightResult := 108056) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30610⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30610⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108129

namespace LeftMerge108143
def owner : Owner := ⟨.program ⟨257⟩, ⟨29542⟩⟩
def mergeEvent : Nat := 108143
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29539⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105245RawTerms
def rightRaw : List Term := Proof.Events422.exact108137RawTerms
def group : MergeGroup := .operator 105245 108137
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105245) (leftOrdinal := 0)
    (rightResult := 108137) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨29539⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29539⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108143

namespace LeftMerge108222
def owner : Owner := ⟨.program ⟨257⟩, ⟨28799⟩⟩
def mergeEvent : Nat := 108222
def frameStart : Nat := 108192
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events422.exact108218RawTerms
def rightRaw : List Term := Proof.Events422.exact108215RawTerms
def group : MergeGroup := .operator 108218 108215
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108218) (leftOrdinal := 0)
    (rightResult := 108215) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13296⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨28798⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108222

namespace LeftMerge108252
def owner : Owner := ⟨.program ⟨257⟩, ⟨30372⟩⟩
def mergeEvent : Nat := 108252
def frameStart : Nat := 108192
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events422.exact108248RawTerms
def rightRaw : List Term := Proof.Events422.exact108246RawTerms
def group : MergeGroup := .operator 108248 108246
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108248) (leftOrdinal := 0)
    (rightResult := 108246) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108252

namespace LeftMerge108275
def owner : Owner := ⟨.program ⟨257⟩, ⟨9549⟩⟩
def mergeEvent : Nat := 108275
def frameStart : Nat := 108192
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }
def leftRaw : List Term := Proof.Events422.exact108271RawTerms
def rightRaw : List Term := Proof.Events422.exact108268RawTerms
def group : MergeGroup := .operator 108271 108268
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108271) (leftOrdinal := 0)
    (rightResult := 108268) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9547⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108275

namespace LeftMerge108284
def owner : Owner := ⟨.program ⟨257⟩, ⟨30613⟩⟩
def mergeEvent : Nat := 108284
def frameStart : Nat := 108192
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30610⟩⟩] } }
def leftRaw : List Term := Proof.Events422.exact108280RawTerms
def rightRaw : List Term := Proof.Events422.exact108237RawTerms
def group : MergeGroup := .operator 108280 108237
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108280) (leftOrdinal := 0)
    (rightResult := 108237) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30610⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30610⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108284

namespace LeftMerge108285
def owner : Owner := ⟨.program ⟨257⟩, ⟨30613⟩⟩
def mergeEvent : Nat := 108285
def frameStart : Nat := 108192
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30610⟩⟩] } }
def leftRaw : List Term := Proof.Events422.exact108280RawTerms
def rightRaw : List Term := Proof.Events422.exact108237RawTerms
def group : MergeGroup := .operator 108280 108237
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108280) (leftOrdinal := 1)
    (rightResult := 108237) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30610⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30610⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108285

namespace LeftMerge108287
def owner : Owner := ⟨.program ⟨257⟩, ⟨30613⟩⟩
def mergeEvent : Nat := 108287
def frameStart : Nat := 108192
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30095⟩⟩] } }
def rhsRaw : List Term := Proof.Events422.exact108234RawTerms
def group : MergeGroup := .relation 108286
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108286) (rhsResult := 108234)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30610⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30610⟩⟩) ⟨30095⟩ 108234) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30095⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], [⟨.program ⟨257⟩, ⟨30095⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108287

namespace LeftMerge108295
def owner : Owner := ⟨.program ⟨257⟩, ⟨29098⟩⟩
def mergeEvent : Nat := 108295
def frameStart : Nat := 108192
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29096⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events422.exact108248RawTerms
def rightRaw : List Term := Proof.Events423.exact108291RawTerms
def group : MergeGroup := .operator 108248 108291
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108248) (leftOrdinal := 0)
    (rightResult := 108291) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29096⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29096⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108295

namespace LeftMerge108312
def owner : Owner := ⟨.program ⟨257⟩, ⟨29542⟩⟩
def mergeEvent : Nat := 108312
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }
def rhsRaw : List Term := Proof.Events423.exact108309RawTerms
def group : MergeGroup := .relation 108311
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108311) (rhsResult := 108309)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29539⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 108310 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29539⟩⟩]⟩) (none) 108309) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108312

namespace LeftMerge108313
def owner : Owner := ⟨.program ⟨257⟩, ⟨29542⟩⟩
def mergeEvent : Nat := 108313
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30610⟩⟩] } }
def rhsRaw : List Term := Proof.Events423.exact108309RawTerms
def group : MergeGroup := .relation 108311
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108311) (rhsResult := 108309)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29539⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 108310 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29539⟩⟩]⟩) (none) 108309) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30610⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30610⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108313

namespace LeftMerge108314
def owner : Owner := ⟨.program ⟨257⟩, ⟨29542⟩⟩
def mergeEvent : Nat := 108314
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30095⟩⟩] } }
def rhsRaw : List Term := Proof.Events423.exact108309RawTerms
def group : MergeGroup := .relation 108311
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108311) (rhsResult := 108309)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29539⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 108310 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29539⟩⟩]⟩) (none) 108309) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30095⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], [⟨.program ⟨257⟩, ⟨30095⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108314

namespace LeftMerge108315
def owner : Owner := ⟨.program ⟨257⟩, ⟨29542⟩⟩
def mergeEvent : Nat := 108315
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29096⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events423.exact108309RawTerms
def group : MergeGroup := .relation 108311
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108311) (rhsResult := 108309)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29539⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 108310 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29539⟩⟩]⟩) (none) 108309) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29096⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29096⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108315

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
