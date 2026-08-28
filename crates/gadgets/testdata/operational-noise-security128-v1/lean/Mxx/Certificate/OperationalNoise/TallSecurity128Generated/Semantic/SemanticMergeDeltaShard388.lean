import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge66368
def owner : Owner := ⟨.program ⟨257⟩, ⟨60462⟩⟩
def mergeEvent : Nat := 66368
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events259.exact66362RawTerms
def group : MergeGroup := .relation 66364
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66364) (rhsResult := 66362)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60459⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 66363 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60459⟩⟩]⟩) (none) 66362) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66368

namespace LeftMerge66373
def owner : Owner := ⟨.program ⟨257⟩, ⟨61538⟩⟩
def mergeEvent : Nat := 66373
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60991⟩⟩] } }
def leftRaw : List Term := Proof.Events259.exact66369RawTerms
def rightRaw : List Term := Proof.Events258.exact66183RawTerms
def group : MergeGroup := .operator 66369 66183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66369) (leftOrdinal := 2)
    (rightResult := 66183) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60991⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60991⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], [⟨.program ⟨257⟩, ⟨60991⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66373

namespace LeftMerge66374
def owner : Owner := ⟨.program ⟨257⟩, ⟨61538⟩⟩
def mergeEvent : Nat := 66374
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61536⟩⟩] } }
def leftRaw : List Term := Proof.Events259.exact66369RawTerms
def rightRaw : List Term := Proof.Events258.exact66183RawTerms
def group : MergeGroup := .operator 66369 66183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66369) (leftOrdinal := 1)
    (rightResult := 66183) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61536⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61536⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61536⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66374

namespace LeftMerge66382
def owner : Owner := ⟨.program ⟨257⟩, ⟨62111⟩⟩
def mergeEvent : Nat := 66382
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62109⟩⟩] } }
def leftRaw : List Term := Proof.Events259.exact66376RawTerms
def rightRaw : List Term := Proof.Events258.exact66099RawTerms
def group : MergeGroup := .operator 66376 66099
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66376) (leftOrdinal := 0)
    (rightResult := 66099) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨62109⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62109⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66382

namespace LeftMerge66383
def owner : Owner := ⟨.program ⟨257⟩, ⟨62111⟩⟩
def mergeEvent : Nat := 66383
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62109⟩⟩] } }
def leftRaw : List Term := Proof.Events259.exact66376RawTerms
def rightRaw : List Term := Proof.Events258.exact66099RawTerms
def group : MergeGroup := .operator 66376 66099
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66376) (leftOrdinal := 1)
    (rightResult := 66099) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨62109⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62109⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66383

namespace LeftMerge66385
def owner : Owner := ⟨.program ⟨257⟩, ⟨62111⟩⟩
def mergeEvent : Nat := 66385
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61164⟩⟩] } }
def rhsRaw : List Term := Proof.Events258.exact66096RawTerms
def group : MergeGroup := .relation 66384
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66384) (rhsResult := 66096)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62109⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨62109⟩⟩) ⟨61164⟩ 66096) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61164⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59884⟩⟩], [⟨.program ⟨257⟩, ⟨61164⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66385

namespace LeftMerge66399
def owner : Owner := ⟨.program ⟨257⟩, ⟨60839⟩⟩
def mergeEvent : Nat := 66399
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60836⟩⟩] } }
def leftRaw : List Term := Proof.Events239.exact61370RawTerms
def rightRaw : List Term := Proof.Events259.exact66393RawTerms
def group : MergeGroup := .operator 61370 66393
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61370) (leftOrdinal := 0)
    (rightResult := 66393) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60836⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66399

namespace LeftMerge66520
def owner : Owner := ⟨.program ⟨257⟩, ⟨61336⟩⟩
def mergeEvent : Nat := 66520
def frameStart : Nat := 66454
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events259.exact66516RawTerms
def rightRaw : List Term := Proof.Events259.exact66514RawTerms
def group : MergeGroup := .operator 66516 66514
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66516) (leftOrdinal := 0)
    (rightResult := 66514) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66520

namespace LeftMerge66532
def owner : Owner := ⟨.program ⟨257⟩, ⟨62110⟩⟩
def mergeEvent : Nat := 66532
def frameStart : Nat := 66454
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62109⟩⟩] } }
def leftRaw : List Term := Proof.Events259.exact66528RawTerms
def rightRaw : List Term := Proof.Events259.exact66505RawTerms
def group : MergeGroup := .operator 66528 66505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66528) (leftOrdinal := 0)
    (rightResult := 66505) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨62109⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62109⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66532

namespace LeftMerge66533
def owner : Owner := ⟨.program ⟨257⟩, ⟨62110⟩⟩
def mergeEvent : Nat := 66533
def frameStart : Nat := 66454
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62109⟩⟩] } }
def leftRaw : List Term := Proof.Events259.exact66528RawTerms
def rightRaw : List Term := Proof.Events259.exact66505RawTerms
def group : MergeGroup := .operator 66528 66505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66528) (leftOrdinal := 1)
    (rightResult := 66505) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨62109⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62109⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66533

namespace LeftMerge66535
def owner : Owner := ⟨.program ⟨257⟩, ⟨62110⟩⟩
def mergeEvent : Nat := 66535
def frameStart : Nat := 66454
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61164⟩⟩] } }
def rhsRaw : List Term := Proof.Events259.exact66502RawTerms
def group : MergeGroup := .relation 66534
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66534) (rhsResult := 66502)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨59884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62109⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨62109⟩⟩) ⟨61164⟩ 66502) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61164⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨59884⟩⟩], [⟨.program ⟨257⟩, ⟨61164⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66535

namespace LeftMerge66543
def owner : Owner := ⟨.program ⟨257⟩, ⟨60236⟩⟩
def mergeEvent : Nat := 66543
def frameStart : Nat := 66454
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events259.exact66516RawTerms
def rightRaw : List Term := Proof.Events259.exact66539RawTerms
def group : MergeGroup := .operator 66516 66539
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66516) (leftOrdinal := 0)
    (rightResult := 66539) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60234⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66543

namespace LeftMerge66560
def owner : Owner := ⟨.program ⟨257⟩, ⟨60839⟩⟩
def mergeEvent : Nat := 66560
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }
def rhsRaw : List Term := Proof.Events259.exact66557RawTerms
def group : MergeGroup := .relation 66559
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66559) (rhsResult := 66557)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60836⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 66558 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60836⟩⟩]⟩) (none) 66557) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66560

namespace LeftMerge66561
def owner : Owner := ⟨.program ⟨257⟩, ⟨60839⟩⟩
def mergeEvent : Nat := 66561
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62109⟩⟩] } }
def rhsRaw : List Term := Proof.Events259.exact66557RawTerms
def group : MergeGroup := .relation 66559
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66559) (rhsResult := 66557)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60836⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 66558 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60836⟩⟩]⟩) (none) 66557) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62109⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62109⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66561

namespace LeftMerge66562
def owner : Owner := ⟨.program ⟨257⟩, ⟨60839⟩⟩
def mergeEvent : Nat := 66562
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61164⟩⟩] } }
def rhsRaw : List Term := Proof.Events259.exact66557RawTerms
def group : MergeGroup := .relation 66559
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66559) (rhsResult := 66557)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60836⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 66558 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60836⟩⟩]⟩) (none) 66557) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61164⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59884⟩⟩], [⟨.program ⟨257⟩, ⟨61164⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66562

namespace LeftMerge66563
def owner : Owner := ⟨.program ⟨257⟩, ⟨60839⟩⟩
def mergeEvent : Nat := 66563
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events259.exact66557RawTerms
def group : MergeGroup := .relation 66559
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66559) (rhsResult := 66557)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60836⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 66558 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60836⟩⟩]⟩) (none) 66557) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66563

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
