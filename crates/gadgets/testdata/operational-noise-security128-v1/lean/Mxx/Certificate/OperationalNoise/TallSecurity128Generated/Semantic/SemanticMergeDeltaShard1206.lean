import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge196516
def owner : Owner := ⟨.program ⟨257⟩, ⟨27944⟩⟩
def mergeEvent : Nat := 196516
def frameStart : Nat := 196424
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27941⟩⟩] } }
def leftRaw : List Term := Proof.Events767.exact196512RawTerms
def rightRaw : List Term := Proof.Events767.exact196469RawTerms
def group : MergeGroup := .operator 196512 196469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 196512) (leftOrdinal := 0)
    (rightResult := 196469) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27941⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27941⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge196516

namespace LeftMerge196517
def owner : Owner := ⟨.program ⟨257⟩, ⟨27944⟩⟩
def mergeEvent : Nat := 196517
def frameStart : Nat := 196424
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13011⟩⟩, ⟨.program ⟨257⟩, ⟨26142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27941⟩⟩] } }
def leftRaw : List Term := Proof.Events767.exact196512RawTerms
def rightRaw : List Term := Proof.Events767.exact196469RawTerms
def group : MergeGroup := .operator 196512 196469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 196512) (leftOrdinal := 1)
    (rightResult := 196469) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13011⟩⟩, ⟨.program ⟨257⟩, ⟨26142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27941⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13011⟩⟩, ⟨.program ⟨257⟩, ⟨26142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27941⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge196517

namespace LeftMerge196519
def owner : Owner := ⟨.program ⟨257⟩, ⟨27944⟩⟩
def mergeEvent : Nat := 196519
def frameStart : Nat := 196424
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13011⟩⟩, ⟨.program ⟨257⟩, ⟨26142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27421⟩⟩] } }
def rhsRaw : List Term := Proof.Events767.exact196466RawTerms
def group : MergeGroup := .relation 196518
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 196518) (rhsResult := 196466)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13011⟩⟩, ⟨.program ⟨257⟩, ⟨26142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27941⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27941⟩⟩) ⟨27421⟩ 196466) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27421⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13011⟩⟩, ⟨.program ⟨257⟩, ⟨26142⟩⟩], [⟨.program ⟨257⟩, ⟨27421⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge196519

namespace LeftMerge196527
def owner : Owner := ⟨.program ⟨257⟩, ⟨26426⟩⟩
def mergeEvent : Nat := 196527
def frameStart : Nat := 196424
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events767.exact196480RawTerms
def rightRaw : List Term := Proof.Events767.exact196523RawTerms
def group : MergeGroup := .operator 196480 196523
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 196480) (leftOrdinal := 0)
    (rightResult := 196523) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26424⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge196527

namespace LeftMerge196544
def owner : Owner := ⟨.program ⟨257⟩, ⟨26872⟩⟩
def mergeEvent : Nat := 196544
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }
def rhsRaw : List Term := Proof.Events767.exact196541RawTerms
def group : MergeGroup := .relation 196543
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 196543) (rhsResult := 196541)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26869⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 196542 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26869⟩⟩]⟩) (none) 196541) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge196544

namespace LeftMerge196545
def owner : Owner := ⟨.program ⟨257⟩, ⟨26872⟩⟩
def mergeEvent : Nat := 196545
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27941⟩⟩] } }
def rhsRaw : List Term := Proof.Events767.exact196541RawTerms
def group : MergeGroup := .relation 196543
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 196543) (rhsResult := 196541)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26869⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 196542 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26869⟩⟩]⟩) (none) 196541) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27941⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27941⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge196545

namespace LeftMerge196546
def owner : Owner := ⟨.program ⟨257⟩, ⟨26872⟩⟩
def mergeEvent : Nat := 196546
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13011⟩⟩, ⟨.program ⟨257⟩, ⟨26142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27421⟩⟩] } }
def rhsRaw : List Term := Proof.Events767.exact196541RawTerms
def group : MergeGroup := .relation 196543
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 196543) (rhsResult := 196541)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26869⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 196542 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26869⟩⟩]⟩) (none) 196541) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13011⟩⟩, ⟨.program ⟨257⟩, ⟨26142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27421⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13011⟩⟩, ⟨.program ⟨257⟩, ⟨26142⟩⟩], [⟨.program ⟨257⟩, ⟨27421⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge196546

namespace LeftMerge196547
def owner : Owner := ⟨.program ⟨257⟩, ⟨26872⟩⟩
def mergeEvent : Nat := 196547
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events767.exact196541RawTerms
def group : MergeGroup := .relation 196543
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 196543) (rhsResult := 196541)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26869⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 196542 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26869⟩⟩]⟩) (none) 196541) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge196547

namespace LeftMerge196552
def owner : Owner := ⟨.program ⟨257⟩, ⟨27943⟩⟩
def mergeEvent : Nat := 196552
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13011⟩⟩, ⟨.program ⟨257⟩, ⟨26142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27421⟩⟩] } }
def leftRaw : List Term := Proof.Events767.exact196548RawTerms
def rightRaw : List Term := Proof.Events767.exact196362RawTerms
def group : MergeGroup := .operator 196548 196362
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 196548) (leftOrdinal := 2)
    (rightResult := 196362) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13011⟩⟩, ⟨.program ⟨257⟩, ⟨26142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27421⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13011⟩⟩, ⟨.program ⟨257⟩, ⟨26142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27421⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13011⟩⟩, ⟨.program ⟨257⟩, ⟨26142⟩⟩], [⟨.program ⟨257⟩, ⟨27421⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge196552

namespace LeftMerge196553
def owner : Owner := ⟨.program ⟨257⟩, ⟨27943⟩⟩
def mergeEvent : Nat := 196553
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27941⟩⟩] } }
def leftRaw : List Term := Proof.Events767.exact196548RawTerms
def rightRaw : List Term := Proof.Events767.exact196362RawTerms
def group : MergeGroup := .operator 196548 196362
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 196548) (leftOrdinal := 1)
    (rightResult := 196362) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27941⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27941⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27941⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge196553

namespace LeftMerge196561
def owner : Owner := ⟨.program ⟨257⟩, ⟨28341⟩⟩
def mergeEvent : Nat := 196561
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28339⟩⟩] } }
def leftRaw : List Term := Proof.Events767.exact196555RawTerms
def rightRaw : List Term := Proof.Events766.exact196278RawTerms
def group : MergeGroup := .operator 196555 196278
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 196555) (leftOrdinal := 0)
    (rightResult := 196278) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28339⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28339⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge196561

namespace LeftMerge196562
def owner : Owner := ⟨.program ⟨257⟩, ⟨28341⟩⟩
def mergeEvent : Nat := 196562
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28339⟩⟩] } }
def leftRaw : List Term := Proof.Events767.exact196555RawTerms
def rightRaw : List Term := Proof.Events766.exact196278RawTerms
def group : MergeGroup := .operator 196555 196278
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 196555) (leftOrdinal := 1)
    (rightResult := 196278) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28339⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28339⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge196562

namespace LeftMerge196564
def owner : Owner := ⟨.program ⟨257⟩, ⟨28341⟩⟩
def mergeEvent : Nat := 196564
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27579⟩⟩] } }
def rhsRaw : List Term := Proof.Events766.exact196275RawTerms
def group : MergeGroup := .relation 196563
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 196563) (rhsResult := 196275)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28339⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28339⟩⟩) ⟨27579⟩ 196275) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27579⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨27579⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge196564

namespace LeftMerge196578
def owner : Owner := ⟨.program ⟨257⟩, ⟨27199⟩⟩
def mergeEvent : Nat := 196578
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27196⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events767.exact196572RawTerms
def group : MergeGroup := .operator 192995 196572
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 196572) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27196⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge196578

namespace LeftMerge196699
def owner : Owner := ⟨.program ⟨257⟩, ⟨27776⟩⟩
def mergeEvent : Nat := 196699
def frameStart : Nat := 196633
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events768.exact196695RawTerms
def rightRaw : List Term := Proof.Events768.exact196693RawTerms
def group : MergeGroup := .operator 196695 196693
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 196695) (leftOrdinal := 0)
    (rightResult := 196693) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26424⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge196699

namespace LeftMerge196711
def owner : Owner := ⟨.program ⟨257⟩, ⟨28340⟩⟩
def mergeEvent : Nat := 196711
def frameStart : Nat := 196633
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28339⟩⟩] } }
def leftRaw : List Term := Proof.Events768.exact196707RawTerms
def rightRaw : List Term := Proof.Events768.exact196684RawTerms
def group : MergeGroup := .operator 196707 196684
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 196707) (leftOrdinal := 0)
    (rightResult := 196684) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28339⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28339⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge196711

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
