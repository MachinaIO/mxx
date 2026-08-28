import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge122456
def owner : Owner := ⟨.program ⟨257⟩, ⟨35152⟩⟩
def mergeEvent : Nat := 122456
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36215⟩⟩] } }
def rhsRaw : List Term := Proof.Events478.exact122452RawTerms
def group : MergeGroup := .relation 122454
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 122454) (rhsResult := 122452)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35149⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 122453 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35149⟩⟩]⟩) (none) 122452) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36215⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36215⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge122456

namespace LeftMerge122457
def owner : Owner := ⟨.program ⟨257⟩, ⟨35152⟩⟩
def mergeEvent : Nat := 122457
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13521⟩⟩, ⟨.program ⟨257⟩, ⟨34338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35725⟩⟩] } }
def rhsRaw : List Term := Proof.Events478.exact122452RawTerms
def group : MergeGroup := .relation 122454
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 122454) (rhsResult := 122452)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35149⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 122453 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35149⟩⟩]⟩) (none) 122452) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13521⟩⟩, ⟨.program ⟨257⟩, ⟨34338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35725⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13521⟩⟩, ⟨.program ⟨257⟩, ⟨34338⟩⟩], [⟨.program ⟨257⟩, ⟨35725⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122457

namespace LeftMerge122458
def owner : Owner := ⟨.program ⟨257⟩, ⟨35152⟩⟩
def mergeEvent : Nat := 122458
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨34716⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events478.exact122452RawTerms
def group : MergeGroup := .relation 122454
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 122454) (rhsResult := 122452)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35149⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 122453 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35149⟩⟩]⟩) (none) 122452) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34716⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨34716⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge122458

namespace LeftMerge122463
def owner : Owner := ⟨.program ⟨257⟩, ⟨36217⟩⟩
def mergeEvent : Nat := 122463
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13521⟩⟩, ⟨.program ⟨257⟩, ⟨34338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35725⟩⟩] } }
def leftRaw : List Term := Proof.Events478.exact122459RawTerms
def rightRaw : List Term := Proof.Events477.exact122273RawTerms
def group : MergeGroup := .operator 122459 122273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 122459) (leftOrdinal := 2)
    (rightResult := 122273) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13521⟩⟩, ⟨.program ⟨257⟩, ⟨34338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35725⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13521⟩⟩, ⟨.program ⟨257⟩, ⟨34338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35725⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13521⟩⟩, ⟨.program ⟨257⟩, ⟨34338⟩⟩], [⟨.program ⟨257⟩, ⟨35725⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge122463

namespace LeftMerge122464
def owner : Owner := ⟨.program ⟨257⟩, ⟨36217⟩⟩
def mergeEvent : Nat := 122464
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36215⟩⟩] } }
def leftRaw : List Term := Proof.Events478.exact122459RawTerms
def rightRaw : List Term := Proof.Events477.exact122273RawTerms
def group : MergeGroup := .operator 122459 122273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 122459) (leftOrdinal := 1)
    (rightResult := 122273) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36215⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36215⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36215⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122464

namespace LeftMerge122472
def owner : Owner := ⟨.program ⟨257⟩, ⟨36531⟩⟩
def mergeEvent : Nat := 122472
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36529⟩⟩] } }
def leftRaw : List Term := Proof.Events478.exact122466RawTerms
def rightRaw : List Term := Proof.Events477.exact122189RawTerms
def group : MergeGroup := .operator 122466 122189
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 122466) (leftOrdinal := 0)
    (rightResult := 122189) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36529⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122472

namespace LeftMerge122473
def owner : Owner := ⟨.program ⟨257⟩, ⟨36531⟩⟩
def mergeEvent : Nat := 122473
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨34716⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36529⟩⟩] } }
def leftRaw : List Term := Proof.Events478.exact122466RawTerms
def rightRaw : List Term := Proof.Events477.exact122189RawTerms
def group : MergeGroup := .operator 122466 122189
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 122466) (leftOrdinal := 1)
    (rightResult := 122189) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨34716⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36529⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨34716⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36529⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge122473

namespace LeftMerge122475
def owner : Owner := ⟨.program ⟨257⟩, ⟨36531⟩⟩
def mergeEvent : Nat := 122475
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨34716⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35865⟩⟩] } }
def rhsRaw : List Term := Proof.Events477.exact122186RawTerms
def group : MergeGroup := .relation 122474
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 122474) (rhsResult := 122186)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨34716⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36529⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36529⟩⟩) ⟨35865⟩ 122186) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35865⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨34716⟩⟩], [⟨.program ⟨257⟩, ⟨35865⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge122475

namespace LeftMerge122489
def owner : Owner := ⟨.program ⟨257⟩, ⟨35419⟩⟩
def mergeEvent : Nat := 122489
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35416⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119870RawTerms
def rightRaw : List Term := Proof.Events478.exact122483RawTerms
def group : MergeGroup := .operator 119870 122483
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119870) (leftOrdinal := 0)
    (rightResult := 122483) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35416⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35416⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122489

namespace LeftMerge122610
def owner : Owner := ⟨.program ⟨257⟩, ⟨36092⟩⟩
def mergeEvent : Nat := 122610
def frameStart : Nat := 122544
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34716⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events478.exact122606RawTerms
def rightRaw : List Term := Proof.Events478.exact122604RawTerms
def group : MergeGroup := .operator 122606 122604
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 122606) (leftOrdinal := 0)
    (rightResult := 122604) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34716⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34716⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122610

namespace LeftMerge122622
def owner : Owner := ⟨.program ⟨257⟩, ⟨36530⟩⟩
def mergeEvent : Nat := 122622
def frameStart : Nat := 122544
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36529⟩⟩] } }
def leftRaw : List Term := Proof.Events478.exact122618RawTerms
def rightRaw : List Term := Proof.Events478.exact122595RawTerms
def group : MergeGroup := .operator 122618 122595
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 122618) (leftOrdinal := 0)
    (rightResult := 122595) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36529⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122622

namespace LeftMerge122623
def owner : Owner := ⟨.program ⟨257⟩, ⟨36530⟩⟩
def mergeEvent : Nat := 122623
def frameStart : Nat := 122544
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34716⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36529⟩⟩] } }
def leftRaw : List Term := Proof.Events478.exact122618RawTerms
def rightRaw : List Term := Proof.Events478.exact122595RawTerms
def group : MergeGroup := .operator 122618 122595
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 122618) (leftOrdinal := 1)
    (rightResult := 122595) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34716⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36529⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34716⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36529⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge122623

namespace LeftMerge122625
def owner : Owner := ⟨.program ⟨257⟩, ⟨36530⟩⟩
def mergeEvent : Nat := 122625
def frameStart : Nat := 122544
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34716⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35865⟩⟩] } }
def rhsRaw : List Term := Proof.Events478.exact122592RawTerms
def group : MergeGroup := .relation 122624
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 122624) (rhsResult := 122592)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨34716⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36529⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36529⟩⟩) ⟨35865⟩ 122592) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35865⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨34716⟩⟩], [⟨.program ⟨257⟩, ⟨35865⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge122625

namespace LeftMerge122633
def owner : Owner := ⟨.program ⟨257⟩, ⟨34912⟩⟩
def mergeEvent : Nat := 122633
def frameStart : Nat := 122544
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34911⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events478.exact122606RawTerms
def rightRaw : List Term := Proof.Events479.exact122629RawTerms
def group : MergeGroup := .operator 122606 122629
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 122606) (leftOrdinal := 0)
    (rightResult := 122629) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34911⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122633

namespace LeftMerge122650
def owner : Owner := ⟨.program ⟨257⟩, ⟨35419⟩⟩
def mergeEvent : Nat := 122650
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }
def rhsRaw : List Term := Proof.Events479.exact122647RawTerms
def group : MergeGroup := .relation 122649
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 122649) (rhsResult := 122647)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35416⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 122648 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35416⟩⟩]⟩) (none) 122647) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122650

namespace LeftMerge122651
def owner : Owner := ⟨.program ⟨257⟩, ⟨35419⟩⟩
def mergeEvent : Nat := 122651
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36529⟩⟩] } }
def rhsRaw : List Term := Proof.Events479.exact122647RawTerms
def group : MergeGroup := .relation 122649
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 122649) (rhsResult := 122647)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35416⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 122648 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35416⟩⟩]⟩) (none) 122647) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36529⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36529⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge122651

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
