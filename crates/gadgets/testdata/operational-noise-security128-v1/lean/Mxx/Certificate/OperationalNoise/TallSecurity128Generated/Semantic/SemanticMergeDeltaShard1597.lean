import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge259357
def owner : Owner := ⟨.program ⟨257⟩, ⟨20167⟩⟩
def mergeEvent : Nat := 259357
def frameStart : Nat := 259262
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12606⟩⟩, ⟨.program ⟨257⟩, ⟨18154⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19679⟩⟩] } }
def rhsRaw : List Term := Proof.Events1012.exact259304RawTerms
def group : MergeGroup := .relation 259356
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 259356) (rhsResult := 259304)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12606⟩⟩, ⟨.program ⟨257⟩, ⟨18154⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20164⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20164⟩⟩) ⟨19679⟩ 259304) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19679⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12606⟩⟩, ⟨.program ⟨257⟩, ⟨18154⟩⟩], [⟨.program ⟨257⟩, ⟨19679⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge259357

namespace LeftMerge259365
def owner : Owner := ⟨.program ⟨257⟩, ⟨18550⟩⟩
def mergeEvent : Nat := 259365
def frameStart : Nat := 259262
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18548⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1012.exact259318RawTerms
def rightRaw : List Term := Proof.Events1013.exact259361RawTerms
def group : MergeGroup := .operator 259318 259361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259318) (leftOrdinal := 0)
    (rightResult := 259361) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18548⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18548⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259365

namespace LeftMerge259382
def owner : Owner := ⟨.program ⟨257⟩, ⟨19102⟩⟩
def mergeEvent : Nat := 259382
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }
def rhsRaw : List Term := Proof.Events1013.exact259379RawTerms
def group : MergeGroup := .relation 259381
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 259381) (rhsResult := 259379)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19099⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 259380 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19099⟩⟩]⟩) (none) 259379) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259382

namespace LeftMerge259383
def owner : Owner := ⟨.program ⟨257⟩, ⟨19102⟩⟩
def mergeEvent : Nat := 259383
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20164⟩⟩] } }
def rhsRaw : List Term := Proof.Events1013.exact259379RawTerms
def group : MergeGroup := .relation 259381
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 259381) (rhsResult := 259379)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19099⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 259380 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19099⟩⟩]⟩) (none) 259379) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20164⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20164⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge259383

namespace LeftMerge259384
def owner : Owner := ⟨.program ⟨257⟩, ⟨19102⟩⟩
def mergeEvent : Nat := 259384
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12606⟩⟩, ⟨.program ⟨257⟩, ⟨18154⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19679⟩⟩] } }
def rhsRaw : List Term := Proof.Events1013.exact259379RawTerms
def group : MergeGroup := .relation 259381
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 259381) (rhsResult := 259379)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19099⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 259380 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19099⟩⟩]⟩) (none) 259379) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12606⟩⟩, ⟨.program ⟨257⟩, ⟨18154⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19679⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12606⟩⟩, ⟨.program ⟨257⟩, ⟨18154⟩⟩], [⟨.program ⟨257⟩, ⟨19679⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259384

namespace LeftMerge259385
def owner : Owner := ⟨.program ⟨257⟩, ⟨19102⟩⟩
def mergeEvent : Nat := 259385
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18548⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1013.exact259379RawTerms
def group : MergeGroup := .relation 259381
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 259381) (rhsResult := 259379)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19099⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 259380 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19099⟩⟩]⟩) (none) 259379) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18548⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18548⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge259385

namespace LeftMerge259390
def owner : Owner := ⟨.program ⟨257⟩, ⟨20166⟩⟩
def mergeEvent : Nat := 259390
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12606⟩⟩, ⟨.program ⟨257⟩, ⟨18154⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19679⟩⟩] } }
def leftRaw : List Term := Proof.Events1013.exact259386RawTerms
def rightRaw : List Term := Proof.Events1012.exact259200RawTerms
def group : MergeGroup := .operator 259386 259200
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259386) (leftOrdinal := 2)
    (rightResult := 259200) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12606⟩⟩, ⟨.program ⟨257⟩, ⟨18154⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19679⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12606⟩⟩, ⟨.program ⟨257⟩, ⟨18154⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19679⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12606⟩⟩, ⟨.program ⟨257⟩, ⟨18154⟩⟩], [⟨.program ⟨257⟩, ⟨19679⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge259390

namespace LeftMerge259391
def owner : Owner := ⟨.program ⟨257⟩, ⟨20166⟩⟩
def mergeEvent : Nat := 259391
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20164⟩⟩] } }
def leftRaw : List Term := Proof.Events1013.exact259386RawTerms
def rightRaw : List Term := Proof.Events1012.exact259200RawTerms
def group : MergeGroup := .operator 259386 259200
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259386) (leftOrdinal := 1)
    (rightResult := 259200) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20164⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20164⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20164⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259391

namespace LeftMerge259399
def owner : Owner := ⟨.program ⟨257⟩, ⟨20499⟩⟩
def mergeEvent : Nat := 259399
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20497⟩⟩] } }
def leftRaw : List Term := Proof.Events1013.exact259393RawTerms
def rightRaw : List Term := Proof.Events1012.exact259116RawTerms
def group : MergeGroup := .operator 259393 259116
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259393) (leftOrdinal := 0)
    (rightResult := 259116) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20497⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20497⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259399

namespace LeftMerge259400
def owner : Owner := ⟨.program ⟨257⟩, ⟨20499⟩⟩
def mergeEvent : Nat := 259400
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18548⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20497⟩⟩] } }
def leftRaw : List Term := Proof.Events1013.exact259393RawTerms
def rightRaw : List Term := Proof.Events1012.exact259116RawTerms
def group : MergeGroup := .operator 259393 259116
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259393) (leftOrdinal := 1)
    (rightResult := 259116) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18548⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20497⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18548⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20497⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge259400

namespace LeftMerge259402
def owner : Owner := ⟨.program ⟨257⟩, ⟨20499⟩⟩
def mergeEvent : Nat := 259402
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18548⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19816⟩⟩] } }
def rhsRaw : List Term := Proof.Events1012.exact259113RawTerms
def group : MergeGroup := .relation 259401
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 259401) (rhsResult := 259113)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18548⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20497⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20497⟩⟩) ⟨19816⟩ 259113) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19816⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18548⟩⟩], [⟨.program ⟨257⟩, ⟨19816⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge259402

namespace LeftMerge259416
def owner : Owner := ⟨.program ⟨257⟩, ⟨19359⟩⟩
def mergeEvent : Nat := 259416
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19356⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events1013.exact259410RawTerms
def group : MergeGroup := .operator 251495 259410
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 259410) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19356⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19356⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259416

namespace LeftMerge259537
def owner : Owner := ⟨.program ⟨257⟩, ⟨20048⟩⟩
def mergeEvent : Nat := 259537
def frameStart : Nat := 259471
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18548⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1013.exact259533RawTerms
def rightRaw : List Term := Proof.Events1013.exact259531RawTerms
def group : MergeGroup := .operator 259533 259531
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259533) (leftOrdinal := 0)
    (rightResult := 259531) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18548⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18548⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259537

namespace LeftMerge259549
def owner : Owner := ⟨.program ⟨257⟩, ⟨20498⟩⟩
def mergeEvent : Nat := 259549
def frameStart : Nat := 259471
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20497⟩⟩] } }
def leftRaw : List Term := Proof.Events1013.exact259545RawTerms
def rightRaw : List Term := Proof.Events1013.exact259522RawTerms
def group : MergeGroup := .operator 259545 259522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259545) (leftOrdinal := 0)
    (rightResult := 259522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20497⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20497⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259549

namespace LeftMerge259550
def owner : Owner := ⟨.program ⟨257⟩, ⟨20498⟩⟩
def mergeEvent : Nat := 259550
def frameStart : Nat := 259471
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18548⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20497⟩⟩] } }
def leftRaw : List Term := Proof.Events1013.exact259545RawTerms
def rightRaw : List Term := Proof.Events1013.exact259522RawTerms
def group : MergeGroup := .operator 259545 259522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259545) (leftOrdinal := 1)
    (rightResult := 259522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18548⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20497⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18548⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20497⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge259550

namespace LeftMerge259552
def owner : Owner := ⟨.program ⟨257⟩, ⟨20498⟩⟩
def mergeEvent : Nat := 259552
def frameStart : Nat := 259471
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18548⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19816⟩⟩] } }
def rhsRaw : List Term := Proof.Events1013.exact259519RawTerms
def group : MergeGroup := .relation 259551
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 259551) (rhsResult := 259519)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨18548⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20497⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20497⟩⟩) ⟨19816⟩ 259519) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19816⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨18548⟩⟩], [⟨.program ⟨257⟩, ⟨19816⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge259552

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
