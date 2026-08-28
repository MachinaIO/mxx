import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge215321
def owner : Owner := ⟨.program ⟨257⟩, ⟨20220⟩⟩
def mergeEvent : Nat := 215321
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20219⟩⟩] } }
def leftRaw : List Term := Proof.Events841.exact215315RawTerms
def rightRaw : List Term := Proof.Events840.exact215251RawTerms
def group : MergeGroup := .operator 215315 215251
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215315) (leftOrdinal := 1)
    (rightResult := 215251) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20219⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20219⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge215321

namespace LeftMerge215323
def owner : Owner := ⟨.program ⟨257⟩, ⟨20220⟩⟩
def mergeEvent : Nat := 215323
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19709⟩⟩] } }
def rhsRaw : List Term := Proof.Events840.exact215248RawTerms
def group : MergeGroup := .relation 215322
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 215322) (rhsResult := 215248)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20219⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20219⟩⟩) ⟨19709⟩ 215248) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19709⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], [⟨.program ⟨257⟩, ⟨19709⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge215323

namespace LeftMerge215324
def owner : Owner := ⟨.program ⟨257⟩, ⟨20220⟩⟩
def mergeEvent : Nat := 215324
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20219⟩⟩] } }
def leftRaw : List Term := Proof.Events841.exact215315RawTerms
def rightRaw : List Term := Proof.Events840.exact215251RawTerms
def group : MergeGroup := .operator 215315 215251
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215315) (leftOrdinal := 0)
    (rightResult := 215251) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20219⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20219⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215324

namespace LeftMerge215338
def owner : Owner := ⟨.program ⟨257⟩, ⟨19152⟩⟩
def mergeEvent : Nat := 215338
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19149⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207620RawTerms
def rightRaw : List Term := Proof.Events841.exact215332RawTerms
def group : MergeGroup := .operator 207620 215332
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207620) (leftOrdinal := 0)
    (rightResult := 215332) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19149⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19149⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215338

namespace LeftMerge215417
def owner : Owner := ⟨.program ⟨257⟩, ⟨18275⟩⟩
def mergeEvent : Nat := 215417
def frameStart : Nat := 215387
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events841.exact215413RawTerms
def rightRaw : List Term := Proof.Events841.exact215410RawTerms
def group : MergeGroup := .operator 215413 215410
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215413) (leftOrdinal := 0)
    (rightResult := 215410) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12681⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18274⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215417

namespace LeftMerge215447
def owner : Owner := ⟨.program ⟨257⟩, ⟨19988⟩⟩
def mergeEvent : Nat := 215447
def frameStart : Nat := 215387
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events841.exact215443RawTerms
def rightRaw : List Term := Proof.Events841.exact215441RawTerms
def group : MergeGroup := .operator 215443 215441
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215443) (leftOrdinal := 0)
    (rightResult := 215441) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215447

namespace LeftMerge215470
def owner : Owner := ⟨.program ⟨257⟩, ⟨9573⟩⟩
def mergeEvent : Nat := 215470
def frameStart : Nat := 215387
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }
def leftRaw : List Term := Proof.Events841.exact215466RawTerms
def rightRaw : List Term := Proof.Events841.exact215463RawTerms
def group : MergeGroup := .operator 215466 215463
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215466) (leftOrdinal := 0)
    (rightResult := 215463) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9571⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215470

namespace LeftMerge215479
def owner : Owner := ⟨.program ⟨257⟩, ⟨20222⟩⟩
def mergeEvent : Nat := 215479
def frameStart : Nat := 215387
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20219⟩⟩] } }
def leftRaw : List Term := Proof.Events841.exact215475RawTerms
def rightRaw : List Term := Proof.Events841.exact215432RawTerms
def group : MergeGroup := .operator 215475 215432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215475) (leftOrdinal := 0)
    (rightResult := 215432) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20219⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20219⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215479

namespace LeftMerge215480
def owner : Owner := ⟨.program ⟨257⟩, ⟨20222⟩⟩
def mergeEvent : Nat := 215480
def frameStart : Nat := 215387
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20219⟩⟩] } }
def leftRaw : List Term := Proof.Events841.exact215475RawTerms
def rightRaw : List Term := Proof.Events841.exact215432RawTerms
def group : MergeGroup := .operator 215475 215432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215475) (leftOrdinal := 1)
    (rightResult := 215432) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20219⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20219⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge215480

namespace LeftMerge215482
def owner : Owner := ⟨.program ⟨257⟩, ⟨20222⟩⟩
def mergeEvent : Nat := 215482
def frameStart : Nat := 215387
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19709⟩⟩] } }
def rhsRaw : List Term := Proof.Events841.exact215429RawTerms
def group : MergeGroup := .relation 215481
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 215481) (rhsResult := 215429)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20219⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20219⟩⟩) ⟨19709⟩ 215429) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19709⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], [⟨.program ⟨257⟩, ⟨19709⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge215482

namespace LeftMerge215490
def owner : Owner := ⟨.program ⟨257⟩, ⟨18590⟩⟩
def mergeEvent : Nat := 215490
def frameStart : Nat := 215387
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18588⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events841.exact215443RawTerms
def rightRaw : List Term := Proof.Events841.exact215486RawTerms
def group : MergeGroup := .operator 215443 215486
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215443) (leftOrdinal := 0)
    (rightResult := 215486) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18588⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18588⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215490

namespace LeftMerge215507
def owner : Owner := ⟨.program ⟨257⟩, ⟨19152⟩⟩
def mergeEvent : Nat := 215507
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }
def rhsRaw : List Term := Proof.Events841.exact215504RawTerms
def group : MergeGroup := .relation 215506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 215506) (rhsResult := 215504)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19149⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 215505 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19149⟩⟩]⟩) (none) 215504) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215507

namespace LeftMerge215508
def owner : Owner := ⟨.program ⟨257⟩, ⟨19152⟩⟩
def mergeEvent : Nat := 215508
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20219⟩⟩] } }
def rhsRaw : List Term := Proof.Events841.exact215504RawTerms
def group : MergeGroup := .relation 215506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 215506) (rhsResult := 215504)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19149⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 215505 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19149⟩⟩]⟩) (none) 215504) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20219⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20219⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge215508

namespace LeftMerge215509
def owner : Owner := ⟨.program ⟨257⟩, ⟨19152⟩⟩
def mergeEvent : Nat := 215509
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19709⟩⟩] } }
def rhsRaw : List Term := Proof.Events841.exact215504RawTerms
def group : MergeGroup := .relation 215506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 215506) (rhsResult := 215504)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19149⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 215505 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19149⟩⟩]⟩) (none) 215504) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19709⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], [⟨.program ⟨257⟩, ⟨19709⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215509

namespace LeftMerge215510
def owner : Owner := ⟨.program ⟨257⟩, ⟨19152⟩⟩
def mergeEvent : Nat := 215510
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18588⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events841.exact215504RawTerms
def group : MergeGroup := .relation 215506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 215506) (rhsResult := 215504)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19149⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 215505 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19149⟩⟩]⟩) (none) 215504) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18588⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18588⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge215510

namespace LeftMerge215515
def owner : Owner := ⟨.program ⟨257⟩, ⟨20221⟩⟩
def mergeEvent : Nat := 215515
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19709⟩⟩] } }
def leftRaw : List Term := Proof.Events841.exact215511RawTerms
def rightRaw : List Term := Proof.Events841.exact215325RawTerms
def group : MergeGroup := .operator 215511 215325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215511) (leftOrdinal := 2)
    (rightResult := 215325) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19709⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19709⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], [⟨.program ⟨257⟩, ⟨19709⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge215515

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
