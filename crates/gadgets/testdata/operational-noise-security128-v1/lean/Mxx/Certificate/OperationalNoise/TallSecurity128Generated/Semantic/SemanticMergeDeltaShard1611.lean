import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge261642
def owner : Owner := ⟨.program ⟨257⟩, ⟨68323⟩⟩
def mergeEvent : Nat := 261642
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨54046⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def rhsRaw : List Term := Proof.Events1021.exact261608RawTerms
def group : MergeGroup := .relation 261610
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 261610) (rhsResult := 261608)
    (sourceTermOrdinal := 32) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68320⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 261609 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68320⟩⟩]⟩) (none) 261608) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54046⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨54046⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261642

namespace LeftMerge261643
def owner : Owner := ⟨.program ⟨257⟩, ⟨68323⟩⟩
def mergeEvent : Nat := 261643
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨51066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def rhsRaw : List Term := Proof.Events1021.exact261608RawTerms
def group : MergeGroup := .relation 261610
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 261610) (rhsResult := 261608)
    (sourceTermOrdinal := 31) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68320⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 261609 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68320⟩⟩]⟩) (none) 261608) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨51066⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261643

namespace LeftMerge261644
def owner : Owner := ⟨.program ⟨257⟩, ⟨68323⟩⟩
def mergeEvent : Nat := 261644
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨32011⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def rhsRaw : List Term := Proof.Events1021.exact261608RawTerms
def group : MergeGroup := .relation 261610
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 261610) (rhsResult := 261608)
    (sourceTermOrdinal := 24) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68320⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 261609 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68320⟩⟩]⟩) (none) 261608) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32011⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨32011⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261644

namespace LeftMerge261645
def owner : Owner := ⟨.program ⟨257⟩, ⟨68323⟩⟩
def mergeEvent : Nat := 261645
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨21991⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def rhsRaw : List Term := Proof.Events1021.exact261608RawTerms
def group : MergeGroup := .relation 261610
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 261610) (rhsResult := 261608)
    (sourceTermOrdinal := 21) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68320⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 261609 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68320⟩⟩]⟩) (none) 261608) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21991⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨21991⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261645

namespace LeftMerge261646
def owner : Owner := ⟨.program ⟨257⟩, ⟨68323⟩⟩
def mergeEvent : Nat := 261646
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18771⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def rhsRaw : List Term := Proof.Events1021.exact261608RawTerms
def group : MergeGroup := .relation 261610
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 261610) (rhsResult := 261608)
    (sourceTermOrdinal := 20) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68320⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 261609 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68320⟩⟩]⟩) (none) 261608) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18771⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18771⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261646

namespace LeftMerge261647
def owner : Owner := ⟨.program ⟨257⟩, ⟨68323⟩⟩
def mergeEvent : Nat := 261647
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15955⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def rhsRaw : List Term := Proof.Events1021.exact261608RawTerms
def group : MergeGroup := .relation 261610
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 261610) (rhsResult := 261608)
    (sourceTermOrdinal := 19) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68320⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 261609 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68320⟩⟩]⟩) (none) 261608) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15955⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261647

namespace LeftMerge261648
def owner : Owner := ⟨.program ⟨257⟩, ⟨68323⟩⟩
def mergeEvent : Nat := 261648
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨67363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1021.exact261608RawTerms
def group : MergeGroup := .relation 261610
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 261610) (rhsResult := 261608)
    (sourceTermOrdinal := 37) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68320⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 261609 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68320⟩⟩]⟩) (none) 261608) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨67363⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261648

namespace LeftMerge261653
def owner : Owner := ⟨.program ⟨257⟩, ⟨71085⟩⟩
def mergeEvent : Nat := 261653
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1022.exact261649RawTerms
def rightRaw : List Term := Proof.Events1016.exact260233RawTerms
def group : MergeGroup := .operator 261649 260233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261649) (leftOrdinal := 17)
    (rightResult := 260233) (rightOrdinal := 17) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261653

namespace LeftMerge261654
def owner : Owner := ⟨.program ⟨257⟩, ⟨71085⟩⟩
def mergeEvent : Nat := 261654
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨48298⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def leftRaw : List Term := Proof.Events1022.exact261649RawTerms
def rightRaw : List Term := Proof.Events1016.exact260233RawTerms
def group : MergeGroup := .operator 261649 260233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261649) (leftOrdinal := 30)
    (rightResult := 260233) (rightOrdinal := 29) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨48298⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨48298⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨48298⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261654

namespace LeftMerge261655
def owner : Owner := ⟨.program ⟨257⟩, ⟨71085⟩⟩
def mergeEvent : Nat := 261655
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1022.exact261649RawTerms
def rightRaw : List Term := Proof.Events1016.exact260233RawTerms
def group : MergeGroup := .operator 261649 260233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261649) (leftOrdinal := 16)
    (rightResult := 260233) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261655

namespace LeftMerge261656
def owner : Owner := ⟨.program ⟨257⟩, ⟨71085⟩⟩
def mergeEvent : Nat := 261656
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def leftRaw : List Term := Proof.Events1022.exact261649RawTerms
def rightRaw : List Term := Proof.Events1016.exact260233RawTerms
def group : MergeGroup := .operator 261649 260233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261649) (leftOrdinal := 29)
    (rightResult := 260233) (rightOrdinal := 28) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45618⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261656

namespace LeftMerge261657
def owner : Owner := ⟨.program ⟨257⟩, ⟨71085⟩⟩
def mergeEvent : Nat := 261657
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1022.exact261649RawTerms
def rightRaw : List Term := Proof.Events1016.exact260233RawTerms
def group : MergeGroup := .operator 261649 260233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261649) (leftOrdinal := 15)
    (rightResult := 260233) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261657

namespace LeftMerge261658
def owner : Owner := ⟨.program ⟨257⟩, ⟨71085⟩⟩
def mergeEvent : Nat := 261658
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨42934⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def leftRaw : List Term := Proof.Events1022.exact261649RawTerms
def rightRaw : List Term := Proof.Events1016.exact260233RawTerms
def group : MergeGroup := .operator 261649 260233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261649) (leftOrdinal := 28)
    (rightResult := 260233) (rightOrdinal := 27) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨42934⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨42934⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨42934⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261658

namespace LeftMerge261659
def owner : Owner := ⟨.program ⟨257⟩, ⟨71085⟩⟩
def mergeEvent : Nat := 261659
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1022.exact261649RawTerms
def rightRaw : List Term := Proof.Events1016.exact260233RawTerms
def group : MergeGroup := .operator 261649 260233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261649) (leftOrdinal := 14)
    (rightResult := 260233) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261659

namespace LeftMerge261660
def owner : Owner := ⟨.program ⟨257⟩, ⟨71085⟩⟩
def mergeEvent : Nat := 261660
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨40254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def leftRaw : List Term := Proof.Events1022.exact261649RawTerms
def rightRaw : List Term := Proof.Events1016.exact260233RawTerms
def group : MergeGroup := .operator 261649 260233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261649) (leftOrdinal := 27)
    (rightResult := 260233) (rightOrdinal := 26) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨40254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨40254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨40254⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261660

namespace LeftMerge261661
def owner : Owner := ⟨.program ⟨257⟩, ⟨71085⟩⟩
def mergeEvent : Nat := 261661
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1022.exact261649RawTerms
def rightRaw : List Term := Proof.Events1016.exact260233RawTerms
def group : MergeGroup := .operator 261649 260233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261649) (leftOrdinal := 13)
    (rightResult := 260233) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261661

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
