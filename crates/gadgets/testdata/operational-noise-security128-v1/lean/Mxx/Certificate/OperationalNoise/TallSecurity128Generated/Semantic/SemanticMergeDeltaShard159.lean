import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge30433
def owner : Owner := ⟨.program ⟨257⟩, ⟨54561⟩⟩
def mergeEvent : Nat := 30433
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53798⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55062⟩⟩] } }
def rhsRaw : List Term := Proof.Events118.exact30429RawTerms
def group : MergeGroup := .relation 30431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 30431) (rhsResult := 30429)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54558⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 30430 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54558⟩⟩]⟩) (none) 30429) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53798⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55062⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53798⟩⟩], [⟨.program ⟨257⟩, ⟨55062⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge30433

namespace LeftMerge30434
def owner : Owner := ⟨.program ⟨257⟩, ⟨54561⟩⟩
def mergeEvent : Nat := 30434
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55655⟩⟩] } }
def rhsRaw : List Term := Proof.Events118.exact30429RawTerms
def group : MergeGroup := .relation 30431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 30431) (rhsResult := 30429)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54558⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 30430 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54558⟩⟩]⟩) (none) 30429) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55655⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55655⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30434

namespace LeftMerge30435
def owner : Owner := ⟨.program ⟨257⟩, ⟨54561⟩⟩
def mergeEvent : Nat := 30435
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events118.exact30429RawTerms
def group : MergeGroup := .relation 30431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 30431) (rhsResult := 30429)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54558⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 30430 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54558⟩⟩]⟩) (none) 30429) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53979⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30435

namespace LeftMerge30440
def owner : Owner := ⟨.program ⟨257⟩, ⟨55658⟩⟩
def mergeEvent : Nat := 30440
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53798⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55062⟩⟩] } }
def leftRaw : List Term := Proof.Events118.exact30436RawTerms
def rightRaw : List Term := Proof.Events118.exact30258RawTerms
def group : MergeGroup := .operator 30436 30258
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30436) (leftOrdinal := 2)
    (rightResult := 30258) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53798⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55062⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53798⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55062⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53798⟩⟩], [⟨.program ⟨257⟩, ⟨55062⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30440

namespace LeftMerge30441
def owner : Owner := ⟨.program ⟨257⟩, ⟨55658⟩⟩
def mergeEvent : Nat := 30441
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55655⟩⟩] } }
def leftRaw : List Term := Proof.Events118.exact30436RawTerms
def rightRaw : List Term := Proof.Events118.exact30258RawTerms
def group : MergeGroup := .operator 30436 30258
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30436) (leftOrdinal := 0)
    (rightResult := 30258) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55655⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55655⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55655⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge30441

namespace LeftMerge30449
def owner : Owner := ⟨.program ⟨257⟩, ⟨55659⟩⟩
def mergeEvent : Nat := 30449
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩] } }
def leftRaw : List Term := Proof.Events118.exact30443RawTerms
def rightRaw : List Term := Proof.Events061.exact15782RawTerms
def group : MergeGroup := .operator 30443 15782
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30443) (leftOrdinal := 0)
    (rightResult := 15782) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7207⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7125⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge30449

namespace LeftMerge30450
def owner : Owner := ⟨.program ⟨257⟩, ⟨55659⟩⟩
def mergeEvent : Nat := 30450
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩] } }
def leftRaw : List Term := Proof.Events118.exact30443RawTerms
def rightRaw : List Term := Proof.Events061.exact15782RawTerms
def group : MergeGroup := .operator 30443 15782
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30443) (leftOrdinal := 1)
    (rightResult := 15782) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7125⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30450

namespace LeftMerge30452
def owner : Owner := ⟨.program ⟨257⟩, ⟨55659⟩⟩
def mergeEvent : Nat := 30452
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15775RawTerms
def group : MergeGroup := .relation 30451
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 30451) (rhsResult := 15775)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7125⟩⟩) ⟨7028⟩ 15775) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30452

namespace LeftMerge30466
def owner : Owner := ⟨.program ⟨257⟩, ⟨52677⟩⟩
def mergeEvent : Nat := 30466
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52675⟩⟩] } }
def leftRaw : List Term := Proof.Events093.exact23868RawTerms
def rightRaw : List Term := Proof.Events118.exact30460RawTerms
def group : MergeGroup := .operator 23868 30460
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23868) (leftOrdinal := 1)
    (rightResult := 30460) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52675⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52675⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30466

namespace LeftMerge30468
def owner : Owner := ⟨.program ⟨257⟩, ⟨52677⟩⟩
def mergeEvent : Nat := 30468
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52082⟩⟩] } }
def rhsRaw : List Term := Proof.Events118.exact30457RawTerms
def group : MergeGroup := .relation 30467
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 30467) (rhsResult := 30457)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52675⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52675⟩⟩) ⟨52082⟩ 30457) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52082⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50818⟩⟩], [⟨.program ⟨257⟩, ⟨52082⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30468

namespace LeftMerge30469
def owner : Owner := ⟨.program ⟨257⟩, ⟨52677⟩⟩
def mergeEvent : Nat := 30469
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52675⟩⟩] } }
def leftRaw : List Term := Proof.Events093.exact23868RawTerms
def rightRaw : List Term := Proof.Events118.exact30460RawTerms
def group : MergeGroup := .operator 23868 30460
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23868) (leftOrdinal := 0)
    (rightResult := 30460) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52675⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52675⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge30469

namespace LeftMerge30483
def owner : Owner := ⟨.program ⟨257⟩, ⟨51581⟩⟩
def mergeEvent : Nat := 30483
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51578⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events119.exact30477RawTerms
def group : MergeGroup := .operator 17169 30477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 30477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51578⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51578⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge30483

namespace LeftMerge30604
def owner : Owner := ⟨.program ⟨257⟩, ⟨52332⟩⟩
def mergeEvent : Nat := 30604
def frameStart : Nat := 30538
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events119.exact30600RawTerms
def rightRaw : List Term := Proof.Events119.exact30598RawTerms
def group : MergeGroup := .operator 30600 30598
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30600) (leftOrdinal := 0)
    (rightResult := 30598) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge30604

namespace LeftMerge30616
def owner : Owner := ⟨.program ⟨257⟩, ⟨52676⟩⟩
def mergeEvent : Nat := 30616
def frameStart : Nat := 30538
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52675⟩⟩] } }
def leftRaw : List Term := Proof.Events119.exact30612RawTerms
def rightRaw : List Term := Proof.Events119.exact30589RawTerms
def group : MergeGroup := .operator 30612 30589
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30612) (leftOrdinal := 1)
    (rightResult := 30589) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52675⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52675⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30616

namespace LeftMerge30618
def owner : Owner := ⟨.program ⟨257⟩, ⟨52676⟩⟩
def mergeEvent : Nat := 30618
def frameStart : Nat := 30538
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52082⟩⟩] } }
def rhsRaw : List Term := Proof.Events119.exact30586RawTerms
def group : MergeGroup := .relation 30617
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 30617) (rhsResult := 30586)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨50818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52675⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52675⟩⟩) ⟨52082⟩ 30586) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52082⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨50818⟩⟩], [⟨.program ⟨257⟩, ⟨52082⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30618

namespace LeftMerge30619
def owner : Owner := ⟨.program ⟨257⟩, ⟨52676⟩⟩
def mergeEvent : Nat := 30619
def frameStart : Nat := 30538
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52675⟩⟩] } }
def leftRaw : List Term := Proof.Events119.exact30612RawTerms
def rightRaw : List Term := Proof.Events119.exact30589RawTerms
def group : MergeGroup := .operator 30612 30589
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30612) (leftOrdinal := 0)
    (rightResult := 30589) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52675⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52675⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge30619

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
