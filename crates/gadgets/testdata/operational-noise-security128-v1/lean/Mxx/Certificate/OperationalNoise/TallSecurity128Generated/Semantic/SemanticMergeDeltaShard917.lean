import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge150073
def owner : Owner := ⟨.program ⟨257⟩, ⟨44267⟩⟩
def mergeEvent : Nat := 150073
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44266⟩⟩] } }
def leftRaw : List Term := Proof.Events586.exact150067RawTerms
def rightRaw : List Term := Proof.Events585.exact150003RawTerms
def group : MergeGroup := .operator 150067 150003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 150067) (leftOrdinal := 1)
    (rightResult := 150003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44266⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44266⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge150073

namespace LeftMerge150075
def owner : Owner := ⟨.program ⟨257⟩, ⟨44267⟩⟩
def mergeEvent : Nat := 150075
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43771⟩⟩] } }
def rhsRaw : List Term := Proof.Events585.exact150000RawTerms
def group : MergeGroup := .relation 150074
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 150074) (rhsResult := 150000)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44266⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44266⟩⟩) ⟨43771⟩ 150000) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43771⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], [⟨.program ⟨257⟩, ⟨43771⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge150075

namespace LeftMerge150076
def owner : Owner := ⟨.program ⟨257⟩, ⟨44267⟩⟩
def mergeEvent : Nat := 150076
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44266⟩⟩] } }
def leftRaw : List Term := Proof.Events586.exact150067RawTerms
def rightRaw : List Term := Proof.Events585.exact150003RawTerms
def group : MergeGroup := .operator 150067 150003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 150067) (leftOrdinal := 0)
    (rightResult := 150003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44266⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44266⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge150076

namespace LeftMerge150090
def owner : Owner := ⟨.program ⟨257⟩, ⟨43202⟩⟩
def mergeEvent : Nat := 150090
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43199⟩⟩] } }
def leftRaw : List Term := Proof.Events582.exact149120RawTerms
def rightRaw : List Term := Proof.Events586.exact150084RawTerms
def group : MergeGroup := .operator 149120 150084
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149120) (leftOrdinal := 0)
    (rightResult := 150084) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43199⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43199⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge150090

namespace LeftMerge150169
def owner : Owner := ⟨.program ⟨257⟩, ⟨42403⟩⟩
def mergeEvent : Nat := 150169
def frameStart : Nat := 150139
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events586.exact150165RawTerms
def rightRaw : List Term := Proof.Events586.exact150162RawTerms
def group : MergeGroup := .operator 150165 150162
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 150165) (leftOrdinal := 0)
    (rightResult := 150162) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14436⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42402⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge150169

namespace LeftMerge150199
def owner : Owner := ⟨.program ⟨257⟩, ⟨44056⟩⟩
def mergeEvent : Nat := 150199
def frameStart : Nat := 150139
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events586.exact150195RawTerms
def rightRaw : List Term := Proof.Events586.exact150193RawTerms
def group : MergeGroup := .operator 150195 150193
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 150195) (leftOrdinal := 0)
    (rightResult := 150193) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge150199

namespace LeftMerge150222
def owner : Owner := ⟨.program ⟨257⟩, ⟨9561⟩⟩
def mergeEvent : Nat := 150222
def frameStart : Nat := 150139
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }
def leftRaw : List Term := Proof.Events586.exact150218RawTerms
def rightRaw : List Term := Proof.Events586.exact150215RawTerms
def group : MergeGroup := .operator 150218 150215
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 150218) (leftOrdinal := 0)
    (rightResult := 150215) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9559⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge150222

namespace LeftMerge150231
def owner : Owner := ⟨.program ⟨257⟩, ⟨44269⟩⟩
def mergeEvent : Nat := 150231
def frameStart : Nat := 150139
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44266⟩⟩] } }
def leftRaw : List Term := Proof.Events586.exact150227RawTerms
def rightRaw : List Term := Proof.Events586.exact150184RawTerms
def group : MergeGroup := .operator 150227 150184
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 150227) (leftOrdinal := 0)
    (rightResult := 150184) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44266⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44266⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge150231

namespace LeftMerge150232
def owner : Owner := ⟨.program ⟨257⟩, ⟨44269⟩⟩
def mergeEvent : Nat := 150232
def frameStart : Nat := 150139
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44266⟩⟩] } }
def leftRaw : List Term := Proof.Events586.exact150227RawTerms
def rightRaw : List Term := Proof.Events586.exact150184RawTerms
def group : MergeGroup := .operator 150227 150184
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 150227) (leftOrdinal := 1)
    (rightResult := 150184) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44266⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44266⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge150232

namespace LeftMerge150234
def owner : Owner := ⟨.program ⟨257⟩, ⟨44269⟩⟩
def mergeEvent : Nat := 150234
def frameStart : Nat := 150139
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43771⟩⟩] } }
def rhsRaw : List Term := Proof.Events586.exact150181RawTerms
def group : MergeGroup := .relation 150233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 150233) (rhsResult := 150181)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44266⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44266⟩⟩) ⟨43771⟩ 150181) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43771⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], [⟨.program ⟨257⟩, ⟨43771⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge150234

namespace LeftMerge150242
def owner : Owner := ⟨.program ⟨257⟩, ⟨42766⟩⟩
def mergeEvent : Nat := 150242
def frameStart : Nat := 150139
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events586.exact150195RawTerms
def rightRaw : List Term := Proof.Events586.exact150238RawTerms
def group : MergeGroup := .operator 150195 150238
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 150195) (leftOrdinal := 0)
    (rightResult := 150238) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42764⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge150242

namespace LeftMerge150259
def owner : Owner := ⟨.program ⟨257⟩, ⟨43202⟩⟩
def mergeEvent : Nat := 150259
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }
def rhsRaw : List Term := Proof.Events586.exact150256RawTerms
def group : MergeGroup := .relation 150258
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 150258) (rhsResult := 150256)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43199⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 150257 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43199⟩⟩]⟩) (none) 150256) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge150259

namespace LeftMerge150260
def owner : Owner := ⟨.program ⟨257⟩, ⟨43202⟩⟩
def mergeEvent : Nat := 150260
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44266⟩⟩] } }
def rhsRaw : List Term := Proof.Events586.exact150256RawTerms
def group : MergeGroup := .relation 150258
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 150258) (rhsResult := 150256)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43199⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 150257 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43199⟩⟩]⟩) (none) 150256) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44266⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44266⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge150260

namespace LeftMerge150261
def owner : Owner := ⟨.program ⟨257⟩, ⟨43202⟩⟩
def mergeEvent : Nat := 150261
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43771⟩⟩] } }
def rhsRaw : List Term := Proof.Events586.exact150256RawTerms
def group : MergeGroup := .relation 150258
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 150258) (rhsResult := 150256)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43199⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 150257 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43199⟩⟩]⟩) (none) 150256) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43771⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], [⟨.program ⟨257⟩, ⟨43771⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge150261

namespace LeftMerge150262
def owner : Owner := ⟨.program ⟨257⟩, ⟨43202⟩⟩
def mergeEvent : Nat := 150262
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨42764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events586.exact150256RawTerms
def group : MergeGroup := .relation 150258
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 150258) (rhsResult := 150256)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43199⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 150257 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43199⟩⟩]⟩) (none) 150256) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨42764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge150262

namespace LeftMerge150267
def owner : Owner := ⟨.program ⟨257⟩, ⟨44268⟩⟩
def mergeEvent : Nat := 150267
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43771⟩⟩] } }
def leftRaw : List Term := Proof.Events586.exact150263RawTerms
def rightRaw : List Term := Proof.Events586.exact150077RawTerms
def group : MergeGroup := .operator 150263 150077
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 150263) (leftOrdinal := 2)
    (rightResult := 150077) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43771⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43771⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], [⟨.program ⟨257⟩, ⟨43771⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge150267

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
