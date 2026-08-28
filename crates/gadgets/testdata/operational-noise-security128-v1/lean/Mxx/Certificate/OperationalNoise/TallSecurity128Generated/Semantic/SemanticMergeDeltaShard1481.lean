import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge240233
def owner : Owner := ⟨.program ⟨257⟩, ⟨27898⟩⟩
def mergeEvent : Nat := 240233
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27897⟩⟩] } }
def leftRaw : List Term := Proof.Events938.exact240227RawTerms
def rightRaw : List Term := Proof.Events938.exact240163RawTerms
def group : MergeGroup := .operator 240227 240163
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240227) (leftOrdinal := 1)
    (rightResult := 240163) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27897⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27897⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge240233

namespace LeftMerge240235
def owner : Owner := ⟨.program ⟨257⟩, ⟨27898⟩⟩
def mergeEvent : Nat := 240235
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27397⟩⟩] } }
def rhsRaw : List Term := Proof.Events938.exact240160RawTerms
def group : MergeGroup := .relation 240234
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 240234) (rhsResult := 240160)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27897⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27897⟩⟩) ⟨27397⟩ 240160) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27397⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], [⟨.program ⟨257⟩, ⟨27397⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge240235

namespace LeftMerge240236
def owner : Owner := ⟨.program ⟨257⟩, ⟨27898⟩⟩
def mergeEvent : Nat := 240236
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27897⟩⟩] } }
def leftRaw : List Term := Proof.Events938.exact240227RawTerms
def rightRaw : List Term := Proof.Events938.exact240163RawTerms
def group : MergeGroup := .operator 240227 240163
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240227) (leftOrdinal := 0)
    (rightResult := 240163) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27897⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27897⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge240236

namespace LeftMerge240250
def owner : Owner := ⟨.program ⟨257⟩, ⟨26832⟩⟩
def mergeEvent : Nat := 240250
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26829⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236870RawTerms
def rightRaw : List Term := Proof.Events938.exact240244RawTerms
def group : MergeGroup := .operator 236870 240244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236870) (leftOrdinal := 0)
    (rightResult := 240244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨26829⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26829⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge240250

namespace LeftMerge240329
def owner : Owner := ⟨.program ⟨257⟩, ⟨26047⟩⟩
def mergeEvent : Nat := 240329
def frameStart : Nat := 240299
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events938.exact240325RawTerms
def rightRaw : List Term := Proof.Events938.exact240322RawTerms
def group : MergeGroup := .operator 240325 240322
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240325) (leftOrdinal := 0)
    (rightResult := 240322) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12951⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26046⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge240329

namespace LeftMerge240359
def owner : Owner := ⟨.program ⟨257⟩, ⟨27680⟩⟩
def mergeEvent : Nat := 240359
def frameStart : Nat := 240299
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events938.exact240355RawTerms
def rightRaw : List Term := Proof.Events938.exact240353RawTerms
def group : MergeGroup := .operator 240355 240353
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240355) (leftOrdinal := 0)
    (rightResult := 240353) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge240359

namespace LeftMerge240382
def owner : Owner := ⟨.program ⟨257⟩, ⟨9546⟩⟩
def mergeEvent : Nat := 240382
def frameStart : Nat := 240299
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }
def leftRaw : List Term := Proof.Events938.exact240378RawTerms
def rightRaw : List Term := Proof.Events938.exact240375RawTerms
def group : MergeGroup := .operator 240378 240375
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240378) (leftOrdinal := 0)
    (rightResult := 240375) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9544⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge240382

namespace LeftMerge240391
def owner : Owner := ⟨.program ⟨257⟩, ⟨27900⟩⟩
def mergeEvent : Nat := 240391
def frameStart : Nat := 240299
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27897⟩⟩] } }
def leftRaw : List Term := Proof.Events939.exact240387RawTerms
def rightRaw : List Term := Proof.Events938.exact240344RawTerms
def group : MergeGroup := .operator 240387 240344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240387) (leftOrdinal := 0)
    (rightResult := 240344) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27897⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27897⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge240391

namespace LeftMerge240392
def owner : Owner := ⟨.program ⟨257⟩, ⟨27900⟩⟩
def mergeEvent : Nat := 240392
def frameStart : Nat := 240299
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27897⟩⟩] } }
def leftRaw : List Term := Proof.Events939.exact240387RawTerms
def rightRaw : List Term := Proof.Events938.exact240344RawTerms
def group : MergeGroup := .operator 240387 240344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240387) (leftOrdinal := 1)
    (rightResult := 240344) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27897⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27897⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge240392

namespace LeftMerge240394
def owner : Owner := ⟨.program ⟨257⟩, ⟨27900⟩⟩
def mergeEvent : Nat := 240394
def frameStart : Nat := 240299
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27397⟩⟩] } }
def rhsRaw : List Term := Proof.Events938.exact240341RawTerms
def group : MergeGroup := .relation 240393
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 240393) (rhsResult := 240341)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27897⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27897⟩⟩) ⟨27397⟩ 240341) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27397⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], [⟨.program ⟨257⟩, ⟨27397⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge240394

namespace LeftMerge240402
def owner : Owner := ⟨.program ⟨257⟩, ⟨26394⟩⟩
def mergeEvent : Nat := 240402
def frameStart : Nat := 240299
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26392⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events938.exact240355RawTerms
def rightRaw : List Term := Proof.Events939.exact240398RawTerms
def group : MergeGroup := .operator 240355 240398
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240355) (leftOrdinal := 0)
    (rightResult := 240398) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26392⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26392⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge240402

namespace LeftMerge240419
def owner : Owner := ⟨.program ⟨257⟩, ⟨26832⟩⟩
def mergeEvent : Nat := 240419
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }
def rhsRaw : List Term := Proof.Events939.exact240416RawTerms
def group : MergeGroup := .relation 240418
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 240418) (rhsResult := 240416)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26829⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 240417 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26829⟩⟩]⟩) (none) 240416) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge240419

namespace LeftMerge240420
def owner : Owner := ⟨.program ⟨257⟩, ⟨26832⟩⟩
def mergeEvent : Nat := 240420
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27897⟩⟩] } }
def rhsRaw : List Term := Proof.Events939.exact240416RawTerms
def group : MergeGroup := .relation 240418
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 240418) (rhsResult := 240416)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26829⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 240417 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26829⟩⟩]⟩) (none) 240416) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27897⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27897⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge240420

namespace LeftMerge240421
def owner : Owner := ⟨.program ⟨257⟩, ⟨26832⟩⟩
def mergeEvent : Nat := 240421
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27397⟩⟩] } }
def rhsRaw : List Term := Proof.Events939.exact240416RawTerms
def group : MergeGroup := .relation 240418
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 240418) (rhsResult := 240416)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26829⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 240417 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26829⟩⟩]⟩) (none) 240416) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27397⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], [⟨.program ⟨257⟩, ⟨27397⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge240421

namespace LeftMerge240422
def owner : Owner := ⟨.program ⟨257⟩, ⟨26832⟩⟩
def mergeEvent : Nat := 240422
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨26392⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events939.exact240416RawTerms
def group : MergeGroup := .relation 240418
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 240418) (rhsResult := 240416)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26829⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 240417 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26829⟩⟩]⟩) (none) 240416) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26392⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨26392⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge240422

namespace LeftMerge240427
def owner : Owner := ⟨.program ⟨257⟩, ⟨27899⟩⟩
def mergeEvent : Nat := 240427
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27397⟩⟩] } }
def leftRaw : List Term := Proof.Events939.exact240423RawTerms
def rightRaw : List Term := Proof.Events938.exact240237RawTerms
def group : MergeGroup := .operator 240423 240237
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240423) (leftOrdinal := 2)
    (rightResult := 240237) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27397⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27397⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], [⟨.program ⟨257⟩, ⟨27397⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge240427

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
