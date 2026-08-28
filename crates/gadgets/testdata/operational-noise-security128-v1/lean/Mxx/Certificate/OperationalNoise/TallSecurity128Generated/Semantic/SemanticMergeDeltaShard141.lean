import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge27623
def owner : Owner := ⟨.program ⟨257⟩, ⟨68286⟩⟩
def mergeEvent : Nat := 27623
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26505⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27605RawTerms
def group : MergeGroup := .relation 27607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27607) (rhsResult := 27605)
    (sourceTermOrdinal := 22) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27606 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩) (none) 27605) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26505⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26505⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27623

namespace LeftMerge27624
def owner : Owner := ⟨.program ⟨257⟩, ⟨68286⟩⟩
def mergeEvent : Nat := 27624
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27605RawTerms
def group : MergeGroup := .relation 27607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27607) (rhsResult := 27605)
    (sourceTermOrdinal := 10) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27606 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩) (none) 27605) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27624

namespace LeftMerge27625
def owner : Owner := ⟨.program ⟨257⟩, ⟨68286⟩⟩
def mergeEvent : Nat := 27625
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27605RawTerms
def group : MergeGroup := .relation 27607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27607) (rhsResult := 27605)
    (sourceTermOrdinal := 36) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27606 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩) (none) 27605) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65993⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27625

namespace LeftMerge27626
def owner : Owner := ⟨.program ⟨257⟩, ⟨68286⟩⟩
def mergeEvent : Nat := 27626
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27605RawTerms
def group : MergeGroup := .relation 27607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27607) (rhsResult := 27605)
    (sourceTermOrdinal := 9) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27606 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩) (none) 27605) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27626

namespace LeftMerge27627
def owner : Owner := ⟨.program ⟨257⟩, ⟨68286⟩⟩
def mergeEvent : Nat := 27627
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62915⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27605RawTerms
def group : MergeGroup := .relation 27607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27607) (rhsResult := 27605)
    (sourceTermOrdinal := 35) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27606 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩) (none) 27605) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62915⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62915⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27627

namespace LeftMerge27628
def owner : Owner := ⟨.program ⟨257⟩, ⟨68286⟩⟩
def mergeEvent : Nat := 27628
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27605RawTerms
def group : MergeGroup := .relation 27607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27607) (rhsResult := 27605)
    (sourceTermOrdinal := 8) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27606 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩) (none) 27605) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27628

namespace LeftMerge27629
def owner : Owner := ⟨.program ⟨257⟩, ⟨68286⟩⟩
def mergeEvent : Nat := 27629
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59935⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27605RawTerms
def group : MergeGroup := .relation 27607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27607) (rhsResult := 27605)
    (sourceTermOrdinal := 34) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27606 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩) (none) 27605) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59935⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59935⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27629

namespace LeftMerge27630
def owner : Owner := ⟨.program ⟨257⟩, ⟨68286⟩⟩
def mergeEvent : Nat := 27630
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27605RawTerms
def group : MergeGroup := .relation 27607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27607) (rhsResult := 27605)
    (sourceTermOrdinal := 7) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27606 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩) (none) 27605) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27630

namespace LeftMerge27631
def owner : Owner := ⟨.program ⟨257⟩, ⟨68286⟩⟩
def mergeEvent : Nat := 27631
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56955⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27605RawTerms
def group : MergeGroup := .relation 27607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27607) (rhsResult := 27605)
    (sourceTermOrdinal := 33) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27606 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩) (none) 27605) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56955⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56955⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27631

namespace LeftMerge27632
def owner : Owner := ⟨.program ⟨257⟩, ⟨68286⟩⟩
def mergeEvent : Nat := 27632
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27605RawTerms
def group : MergeGroup := .relation 27607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27607) (rhsResult := 27605)
    (sourceTermOrdinal := 6) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27606 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩) (none) 27605) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27632

namespace LeftMerge27633
def owner : Owner := ⟨.program ⟨257⟩, ⟨68286⟩⟩
def mergeEvent : Nat := 27633
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53975⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27605RawTerms
def group : MergeGroup := .relation 27607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27607) (rhsResult := 27605)
    (sourceTermOrdinal := 32) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27606 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩) (none) 27605) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53975⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53975⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27633

namespace LeftMerge27634
def owner : Owner := ⟨.program ⟨257⟩, ⟨68286⟩⟩
def mergeEvent : Nat := 27634
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27605RawTerms
def group : MergeGroup := .relation 27607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27607) (rhsResult := 27605)
    (sourceTermOrdinal := 5) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27606 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩) (none) 27605) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27634

namespace LeftMerge27635
def owner : Owner := ⟨.program ⟨257⟩, ⟨68286⟩⟩
def mergeEvent : Nat := 27635
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50995⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27605RawTerms
def group : MergeGroup := .relation 27607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27607) (rhsResult := 27605)
    (sourceTermOrdinal := 31) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27606 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩) (none) 27605) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50995⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27635

namespace LeftMerge27636
def owner : Owner := ⟨.program ⟨257⟩, ⟨68286⟩⟩
def mergeEvent : Nat := 27636
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27605RawTerms
def group : MergeGroup := .relation 27607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27607) (rhsResult := 27605)
    (sourceTermOrdinal := 4) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27606 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩) (none) 27605) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27636

namespace LeftMerge27637
def owner : Owner := ⟨.program ⟨257⟩, ⟨68286⟩⟩
def mergeEvent : Nat := 27637
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27605RawTerms
def group : MergeGroup := .relation 27607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27607) (rhsResult := 27605)
    (sourceTermOrdinal := 24) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27606 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩) (none) 27605) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31940⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27637

namespace LeftMerge27638
def owner : Owner := ⟨.program ⟨257⟩, ⟨68286⟩⟩
def mergeEvent : Nat := 27638
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def rhsRaw : List Term := Proof.Events107.exact27605RawTerms
def group : MergeGroup := .relation 27607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27607) (rhsResult := 27605)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 27606 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩) (none) 27605) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27638

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
