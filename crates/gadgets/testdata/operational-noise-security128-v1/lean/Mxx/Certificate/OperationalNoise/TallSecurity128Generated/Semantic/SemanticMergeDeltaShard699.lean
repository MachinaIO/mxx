import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge116083
def owner : Owner := ⟨.program ⟨257⟩, ⟨44692⟩⟩
def mergeEvent : Nat := 116083
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨43015⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩] } }
def leftRaw : List Term := Proof.Events453.exact116076RawTerms
def rightRaw : List Term := Proof.Events060.exact15582RawTerms
def group : MergeGroup := .operator 116076 15582
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 116076) (leftOrdinal := 1)
    (rightResult := 15582) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨43015⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7153⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨43015⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge116083

namespace LeftMerge116085
def owner : Owner := ⟨.program ⟨257⟩, ⟨44692⟩⟩
def mergeEvent : Nat := 116085
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨43015⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events060.exact15575RawTerms
def group : MergeGroup := .relation 116084
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 116084) (rhsResult := 15575)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨43015⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7153⟩⟩) ⟨7042⟩ 15575) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨43015⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge116085

namespace LeftMerge116099
def owner : Owner := ⟨.program ⟨257⟩, ⟨42010⟩⟩
def mergeEvent : Nat := 116099
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42008⟩⟩] } }
def leftRaw : List Term := Proof.Events417.exact106877RawTerms
def rightRaw : List Term := Proof.Events453.exact116093RawTerms
def group : MergeGroup := .operator 106877 116093
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 106877) (leftOrdinal := 0)
    (rightResult := 116093) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42008⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42008⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge116099

namespace LeftMerge116100
def owner : Owner := ⟨.program ⟨257⟩, ⟨42010⟩⟩
def mergeEvent : Nat := 116100
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40116⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42008⟩⟩] } }
def leftRaw : List Term := Proof.Events417.exact106877RawTerms
def rightRaw : List Term := Proof.Events453.exact116093RawTerms
def group : MergeGroup := .operator 106877 116093
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 106877) (leftOrdinal := 1)
    (rightResult := 116093) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40116⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42008⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42008⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge116100

namespace LeftMerge116102
def owner : Owner := ⟨.program ⟨257⟩, ⟨42010⟩⟩
def mergeEvent : Nat := 116102
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40116⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41269⟩⟩] } }
def rhsRaw : List Term := Proof.Events453.exact116090RawTerms
def group : MergeGroup := .relation 116101
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 116101) (rhsResult := 116090)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42008⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42008⟩⟩) ⟨41269⟩ 116090) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41269⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨41269⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge116102

namespace LeftMerge116116
def owner : Owner := ⟨.program ⟨257⟩, ⟨40875⟩⟩
def mergeEvent : Nat := 116116
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40872⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105245RawTerms
def rightRaw : List Term := Proof.Events453.exact116110RawTerms
def group : MergeGroup := .operator 105245 116110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105245) (leftOrdinal := 0)
    (rightResult := 116110) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40872⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40872⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge116116

namespace LeftMerge116237
def owner : Owner := ⟨.program ⟨257⟩, ⟨41472⟩⟩
def mergeEvent : Nat := 116237
def frameStart : Nat := 116171
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40116⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events454.exact116233RawTerms
def rightRaw : List Term := Proof.Events454.exact116231RawTerms
def group : MergeGroup := .operator 116233 116231
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 116233) (leftOrdinal := 0)
    (rightResult := 116231) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40116⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge116237

namespace LeftMerge116249
def owner : Owner := ⟨.program ⟨257⟩, ⟨42009⟩⟩
def mergeEvent : Nat := 116249
def frameStart : Nat := 116171
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42008⟩⟩] } }
def leftRaw : List Term := Proof.Events454.exact116245RawTerms
def rightRaw : List Term := Proof.Events453.exact116222RawTerms
def group : MergeGroup := .operator 116245 116222
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 116245) (leftOrdinal := 0)
    (rightResult := 116222) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42008⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42008⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge116249

namespace LeftMerge116250
def owner : Owner := ⟨.program ⟨257⟩, ⟨42009⟩⟩
def mergeEvent : Nat := 116250
def frameStart : Nat := 116171
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40116⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42008⟩⟩] } }
def leftRaw : List Term := Proof.Events454.exact116245RawTerms
def rightRaw : List Term := Proof.Events453.exact116222RawTerms
def group : MergeGroup := .operator 116245 116222
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 116245) (leftOrdinal := 1)
    (rightResult := 116222) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40116⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42008⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42008⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge116250

namespace LeftMerge116252
def owner : Owner := ⟨.program ⟨257⟩, ⟨42009⟩⟩
def mergeEvent : Nat := 116252
def frameStart : Nat := 116171
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40116⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41269⟩⟩] } }
def rhsRaw : List Term := Proof.Events453.exact116219RawTerms
def group : MergeGroup := .relation 116251
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 116251) (rhsResult := 116219)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42008⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42008⟩⟩) ⟨41269⟩ 116219) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41269⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨41269⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge116252

namespace LeftMerge116260
def owner : Owner := ⟨.program ⟨257⟩, ⟨40337⟩⟩
def mergeEvent : Nat := 116260
def frameStart : Nat := 116171
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40335⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events454.exact116233RawTerms
def rightRaw : List Term := Proof.Events454.exact116256RawTerms
def group : MergeGroup := .operator 116233 116256
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 116233) (leftOrdinal := 0)
    (rightResult := 116256) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40335⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40335⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge116260

namespace LeftMerge116277
def owner : Owner := ⟨.program ⟨257⟩, ⟨40875⟩⟩
def mergeEvent : Nat := 116277
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩] } }
def rhsRaw : List Term := Proof.Events454.exact116274RawTerms
def group : MergeGroup := .relation 116276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 116276) (rhsResult := 116274)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40872⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 116275 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40872⟩⟩]⟩) (none) 116274) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge116277

namespace LeftMerge116278
def owner : Owner := ⟨.program ⟨257⟩, ⟨40875⟩⟩
def mergeEvent : Nat := 116278
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42008⟩⟩] } }
def rhsRaw : List Term := Proof.Events454.exact116274RawTerms
def group : MergeGroup := .relation 116276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 116276) (rhsResult := 116274)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40872⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 116275 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40872⟩⟩]⟩) (none) 116274) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42008⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42008⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge116278

namespace LeftMerge116279
def owner : Owner := ⟨.program ⟨257⟩, ⟨40875⟩⟩
def mergeEvent : Nat := 116279
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40116⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41269⟩⟩] } }
def rhsRaw : List Term := Proof.Events454.exact116274RawTerms
def group : MergeGroup := .relation 116276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 116276) (rhsResult := 116274)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40872⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 116275 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40872⟩⟩]⟩) (none) 116274) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40116⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41269⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨41269⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge116279

namespace LeftMerge116280
def owner : Owner := ⟨.program ⟨257⟩, ⟨40875⟩⟩
def mergeEvent : Nat := 116280
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40335⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events454.exact116274RawTerms
def group : MergeGroup := .relation 116276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 116276) (rhsResult := 116274)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40872⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 116275 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40872⟩⟩]⟩) (none) 116274) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40335⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40335⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge116280

namespace LeftMerge116285
def owner : Owner := ⟨.program ⟨257⟩, ⟨42011⟩⟩
def mergeEvent : Nat := 116285
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42008⟩⟩] } }
def leftRaw : List Term := Proof.Events454.exact116281RawTerms
def rightRaw : List Term := Proof.Events453.exact116103RawTerms
def group : MergeGroup := .operator 116281 116103
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 116281) (leftOrdinal := 0)
    (rightResult := 116103) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42008⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42008⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42008⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge116285

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
