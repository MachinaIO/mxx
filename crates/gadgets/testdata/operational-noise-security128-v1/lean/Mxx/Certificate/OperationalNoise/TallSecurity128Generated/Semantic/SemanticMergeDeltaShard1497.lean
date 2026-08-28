import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge243221
def owner : Owner := ⟨.program ⟨257⟩, ⟨50492⟩⟩
def mergeEvent : Nat := 243221
def frameStart : Nat := 243191
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24506⟩⟩, ⟨.program ⟨257⟩, ⟨50491⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events950.exact243217RawTerms
def rightRaw : List Term := Proof.Events950.exact243214RawTerms
def group : MergeGroup := .operator 243217 243214
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243217) (leftOrdinal := 0)
    (rightResult := 243214) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50491⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24506⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24506⟩⟩, ⟨.program ⟨257⟩, ⟨50491⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243221

namespace LeftMerge243251
def owner : Owner := ⟨.program ⟨257⟩, ⟨52280⟩⟩
def mergeEvent : Nat := 243251
def frameStart : Nat := 243191
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24506⟩⟩, ⟨.program ⟨257⟩, ⟨50491⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events950.exact243247RawTerms
def rightRaw : List Term := Proof.Events950.exact243245RawTerms
def group : MergeGroup := .operator 243247 243245
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243247) (leftOrdinal := 0)
    (rightResult := 243245) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24506⟩⟩, ⟨.program ⟨257⟩, ⟨50491⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24506⟩⟩, ⟨.program ⟨257⟩, ⟨50491⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243251

namespace LeftMerge243274
def owner : Owner := ⟨.program ⟨257⟩, ⟨9582⟩⟩
def mergeEvent : Nat := 243274
def frameStart : Nat := 243191
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }
def leftRaw : List Term := Proof.Events950.exact243270RawTerms
def rightRaw : List Term := Proof.Events950.exact243267RawTerms
def group : MergeGroup := .operator 243270 243267
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243270) (leftOrdinal := 0)
    (rightResult := 243267) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9580⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243274

namespace LeftMerge243283
def owner : Owner := ⟨.program ⟨257⟩, ⟨52500⟩⟩
def mergeEvent : Nat := 243283
def frameStart : Nat := 243191
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52497⟩⟩] } }
def leftRaw : List Term := Proof.Events950.exact243279RawTerms
def rightRaw : List Term := Proof.Events950.exact243236RawTerms
def group : MergeGroup := .operator 243279 243236
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243279) (leftOrdinal := 0)
    (rightResult := 243236) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52497⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52497⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243283

namespace LeftMerge243284
def owner : Owner := ⟨.program ⟨257⟩, ⟨52500⟩⟩
def mergeEvent : Nat := 243284
def frameStart : Nat := 243191
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24506⟩⟩, ⟨.program ⟨257⟩, ⟨50491⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52497⟩⟩] } }
def leftRaw : List Term := Proof.Events950.exact243279RawTerms
def rightRaw : List Term := Proof.Events950.exact243236RawTerms
def group : MergeGroup := .operator 243279 243236
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243279) (leftOrdinal := 1)
    (rightResult := 243236) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24506⟩⟩, ⟨.program ⟨257⟩, ⟨50491⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52497⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24506⟩⟩, ⟨.program ⟨257⟩, ⟨50491⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52497⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243284

namespace LeftMerge243286
def owner : Owner := ⟨.program ⟨257⟩, ⟨52500⟩⟩
def mergeEvent : Nat := 243286
def frameStart : Nat := 243191
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24506⟩⟩, ⟨.program ⟨257⟩, ⟨50491⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51997⟩⟩] } }
def rhsRaw : List Term := Proof.Events950.exact243233RawTerms
def group : MergeGroup := .relation 243285
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243285) (rhsResult := 243233)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24506⟩⟩, ⟨.program ⟨257⟩, ⟨50491⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52497⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52497⟩⟩) ⟨51997⟩ 243233) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51997⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24506⟩⟩, ⟨.program ⟨257⟩, ⟨50491⟩⟩], [⟨.program ⟨257⟩, ⟨51997⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243286

namespace LeftMerge243294
def owner : Owner := ⟨.program ⟨257⟩, ⟨50874⟩⟩
def mergeEvent : Nat := 243294
def frameStart : Nat := 243191
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events950.exact243247RawTerms
def rightRaw : List Term := Proof.Events950.exact243290RawTerms
def group : MergeGroup := .operator 243247 243290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243247) (leftOrdinal := 0)
    (rightResult := 243290) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243294

namespace LeftMerge243311
def owner : Owner := ⟨.program ⟨257⟩, ⟨51432⟩⟩
def mergeEvent : Nat := 243311
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }
def rhsRaw : List Term := Proof.Events950.exact243308RawTerms
def group : MergeGroup := .relation 243310
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243310) (rhsResult := 243308)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 243309 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51429⟩⟩]⟩) (none) 243308) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243311

namespace LeftMerge243312
def owner : Owner := ⟨.program ⟨257⟩, ⟨51432⟩⟩
def mergeEvent : Nat := 243312
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52497⟩⟩] } }
def rhsRaw : List Term := Proof.Events950.exact243308RawTerms
def group : MergeGroup := .relation 243310
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243310) (rhsResult := 243308)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 243309 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51429⟩⟩]⟩) (none) 243308) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52497⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52497⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243312

namespace LeftMerge243313
def owner : Owner := ⟨.program ⟨257⟩, ⟨51432⟩⟩
def mergeEvent : Nat := 243313
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24506⟩⟩, ⟨.program ⟨257⟩, ⟨50491⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51997⟩⟩] } }
def rhsRaw : List Term := Proof.Events950.exact243308RawTerms
def group : MergeGroup := .relation 243310
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243310) (rhsResult := 243308)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 243309 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51429⟩⟩]⟩) (none) 243308) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24506⟩⟩, ⟨.program ⟨257⟩, ⟨50491⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51997⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24506⟩⟩, ⟨.program ⟨257⟩, ⟨50491⟩⟩], [⟨.program ⟨257⟩, ⟨51997⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243313

namespace LeftMerge243314
def owner : Owner := ⟨.program ⟨257⟩, ⟨51432⟩⟩
def mergeEvent : Nat := 243314
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events950.exact243308RawTerms
def group : MergeGroup := .relation 243310
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243310) (rhsResult := 243308)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 243309 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51429⟩⟩]⟩) (none) 243308) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨50872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243314

namespace LeftMerge243319
def owner : Owner := ⟨.program ⟨257⟩, ⟨52499⟩⟩
def mergeEvent : Nat := 243319
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24506⟩⟩, ⟨.program ⟨257⟩, ⟨50491⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51997⟩⟩] } }
def leftRaw : List Term := Proof.Events950.exact243315RawTerms
def rightRaw : List Term := Proof.Events949.exact243129RawTerms
def group : MergeGroup := .operator 243315 243129
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243315) (leftOrdinal := 2)
    (rightResult := 243129) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24506⟩⟩, ⟨.program ⟨257⟩, ⟨50491⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51997⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24506⟩⟩, ⟨.program ⟨257⟩, ⟨50491⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51997⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24506⟩⟩, ⟨.program ⟨257⟩, ⟨50491⟩⟩], [⟨.program ⟨257⟩, ⟨51997⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243319

namespace LeftMerge243320
def owner : Owner := ⟨.program ⟨257⟩, ⟨52499⟩⟩
def mergeEvent : Nat := 243320
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52497⟩⟩] } }
def leftRaw : List Term := Proof.Events950.exact243315RawTerms
def rightRaw : List Term := Proof.Events949.exact243129RawTerms
def group : MergeGroup := .operator 243315 243129
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243315) (leftOrdinal := 1)
    (rightResult := 243129) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52497⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52497⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52497⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243320

namespace LeftMerge243328
def owner : Owner := ⟨.program ⟨257⟩, ⟨52892⟩⟩
def mergeEvent : Nat := 243328
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52890⟩⟩] } }
def leftRaw : List Term := Proof.Events950.exact243322RawTerms
def rightRaw : List Term := Proof.Events949.exact243045RawTerms
def group : MergeGroup := .operator 243322 243045
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243322) (leftOrdinal := 0)
    (rightResult := 243045) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52890⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52890⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243328

namespace LeftMerge243329
def owner : Owner := ⟨.program ⟨257⟩, ⟨52892⟩⟩
def mergeEvent : Nat := 243329
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52890⟩⟩] } }
def leftRaw : List Term := Proof.Events950.exact243322RawTerms
def rightRaw : List Term := Proof.Events949.exact243045RawTerms
def group : MergeGroup := .operator 243322 243045
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243322) (leftOrdinal := 1)
    (rightResult := 243045) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52890⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨50872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52890⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243329

namespace LeftMerge243331
def owner : Owner := ⟨.program ⟨257⟩, ⟨52892⟩⟩
def mergeEvent : Nat := 243331
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52143⟩⟩] } }
def rhsRaw : List Term := Proof.Events949.exact243042RawTerms
def group : MergeGroup := .relation 243330
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243330) (rhsResult := 243042)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨50872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52890⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52890⟩⟩) ⟨52143⟩ 243042) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52143⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨50872⟩⟩], [⟨.program ⟨257⟩, ⟨52143⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243331

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
