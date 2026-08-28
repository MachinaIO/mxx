import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge142198
def owner : Owner := ⟨.program ⟨257⟩, ⟨20143⟩⟩
def mergeEvent : Nat := 142198
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19667⟩⟩] } }
def rhsRaw : List Term := Proof.Events555.exact142123RawTerms
def group : MergeGroup := .relation 142197
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 142197) (rhsResult := 142123)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20142⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20142⟩⟩) ⟨19667⟩ 142123) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19667⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], [⟨.program ⟨257⟩, ⟨19667⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge142198

namespace LeftMerge142199
def owner : Owner := ⟨.program ⟨257⟩, ⟨20143⟩⟩
def mergeEvent : Nat := 142199
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20142⟩⟩] } }
def leftRaw : List Term := Proof.Events555.exact142190RawTerms
def rightRaw : List Term := Proof.Events555.exact142126RawTerms
def group : MergeGroup := .operator 142190 142126
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142190) (leftOrdinal := 0)
    (rightResult := 142126) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20142⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142199

namespace LeftMerge142213
def owner : Owner := ⟨.program ⟨257⟩, ⟨19082⟩⟩
def mergeEvent : Nat := 142213
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19079⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134495RawTerms
def rightRaw : List Term := Proof.Events555.exact142207RawTerms
def group : MergeGroup := .operator 134495 142207
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134495) (leftOrdinal := 0)
    (rightResult := 142207) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19079⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19079⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142213

namespace LeftMerge142292
def owner : Owner := ⟨.program ⟨257⟩, ⟨18107⟩⟩
def mergeEvent : Nat := 142292
def frameStart : Nat := 142262
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events555.exact142288RawTerms
def rightRaw : List Term := Proof.Events555.exact142285RawTerms
def group : MergeGroup := .operator 142288 142285
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142288) (leftOrdinal := 0)
    (rightResult := 142285) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12576⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18106⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142292

namespace LeftMerge142322
def owner : Owner := ⟨.program ⟨257⟩, ⟨19960⟩⟩
def mergeEvent : Nat := 142322
def frameStart : Nat := 142262
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events555.exact142318RawTerms
def rightRaw : List Term := Proof.Events555.exact142316RawTerms
def group : MergeGroup := .operator 142318 142316
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142318) (leftOrdinal := 0)
    (rightResult := 142316) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142322

namespace LeftMerge142345
def owner : Owner := ⟨.program ⟨257⟩, ⟨9573⟩⟩
def mergeEvent : Nat := 142345
def frameStart : Nat := 142262
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }
def leftRaw : List Term := Proof.Events556.exact142341RawTerms
def rightRaw : List Term := Proof.Events556.exact142338RawTerms
def group : MergeGroup := .operator 142341 142338
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142341) (leftOrdinal := 0)
    (rightResult := 142338) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9571⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142345

namespace LeftMerge142354
def owner : Owner := ⟨.program ⟨257⟩, ⟨20145⟩⟩
def mergeEvent : Nat := 142354
def frameStart : Nat := 142262
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20142⟩⟩] } }
def leftRaw : List Term := Proof.Events556.exact142350RawTerms
def rightRaw : List Term := Proof.Events555.exact142307RawTerms
def group : MergeGroup := .operator 142350 142307
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142350) (leftOrdinal := 0)
    (rightResult := 142307) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20142⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142354

namespace LeftMerge142355
def owner : Owner := ⟨.program ⟨257⟩, ⟨20145⟩⟩
def mergeEvent : Nat := 142355
def frameStart : Nat := 142262
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20142⟩⟩] } }
def leftRaw : List Term := Proof.Events556.exact142350RawTerms
def rightRaw : List Term := Proof.Events555.exact142307RawTerms
def group : MergeGroup := .operator 142350 142307
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142350) (leftOrdinal := 1)
    (rightResult := 142307) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20142⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20142⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge142355

namespace LeftMerge142357
def owner : Owner := ⟨.program ⟨257⟩, ⟨20145⟩⟩
def mergeEvent : Nat := 142357
def frameStart : Nat := 142262
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19667⟩⟩] } }
def rhsRaw : List Term := Proof.Events555.exact142304RawTerms
def group : MergeGroup := .relation 142356
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 142356) (rhsResult := 142304)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20142⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20142⟩⟩) ⟨19667⟩ 142304) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19667⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], [⟨.program ⟨257⟩, ⟨19667⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge142357

namespace LeftMerge142365
def owner : Owner := ⟨.program ⟨257⟩, ⟨18534⟩⟩
def mergeEvent : Nat := 142365
def frameStart : Nat := 142262
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events555.exact142318RawTerms
def rightRaw : List Term := Proof.Events556.exact142361RawTerms
def group : MergeGroup := .operator 142318 142361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142318) (leftOrdinal := 0)
    (rightResult := 142361) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18532⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142365

namespace LeftMerge142382
def owner : Owner := ⟨.program ⟨257⟩, ⟨19082⟩⟩
def mergeEvent : Nat := 142382
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }
def rhsRaw : List Term := Proof.Events556.exact142379RawTerms
def group : MergeGroup := .relation 142381
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 142381) (rhsResult := 142379)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19079⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 142380 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19079⟩⟩]⟩) (none) 142379) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142382

namespace LeftMerge142383
def owner : Owner := ⟨.program ⟨257⟩, ⟨19082⟩⟩
def mergeEvent : Nat := 142383
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20142⟩⟩] } }
def rhsRaw : List Term := Proof.Events556.exact142379RawTerms
def group : MergeGroup := .relation 142381
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 142381) (rhsResult := 142379)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19079⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 142380 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19079⟩⟩]⟩) (none) 142379) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20142⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20142⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge142383

namespace LeftMerge142384
def owner : Owner := ⟨.program ⟨257⟩, ⟨19082⟩⟩
def mergeEvent : Nat := 142384
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19667⟩⟩] } }
def rhsRaw : List Term := Proof.Events556.exact142379RawTerms
def group : MergeGroup := .relation 142381
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 142381) (rhsResult := 142379)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19079⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 142380 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19079⟩⟩]⟩) (none) 142379) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19667⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], [⟨.program ⟨257⟩, ⟨19667⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142384

namespace LeftMerge142385
def owner : Owner := ⟨.program ⟨257⟩, ⟨19082⟩⟩
def mergeEvent : Nat := 142385
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events556.exact142379RawTerms
def group : MergeGroup := .relation 142381
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 142381) (rhsResult := 142379)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19079⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 142380 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19079⟩⟩]⟩) (none) 142379) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge142385

namespace LeftMerge142390
def owner : Owner := ⟨.program ⟨257⟩, ⟨20144⟩⟩
def mergeEvent : Nat := 142390
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19667⟩⟩] } }
def leftRaw : List Term := Proof.Events556.exact142386RawTerms
def rightRaw : List Term := Proof.Events555.exact142200RawTerms
def group : MergeGroup := .operator 142386 142200
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142386) (leftOrdinal := 2)
    (rightResult := 142200) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19667⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19667⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], [⟨.program ⟨257⟩, ⟨19667⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge142390

namespace LeftMerge142391
def owner : Owner := ⟨.program ⟨257⟩, ⟨20144⟩⟩
def mergeEvent : Nat := 142391
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20142⟩⟩] } }
def leftRaw : List Term := Proof.Events556.exact142386RawTerms
def rightRaw : List Term := Proof.Events555.exact142200RawTerms
def group : MergeGroup := .operator 142386 142200
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142386) (leftOrdinal := 1)
    (rightResult := 142200) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20142⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20142⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142391

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
