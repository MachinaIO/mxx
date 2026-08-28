import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge171604
def owner : Owner := ⟨.program ⟨257⟩, ⟨20266⟩⟩
def mergeEvent : Nat := 171604
def frameStart : Nat := 171512
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20263⟩⟩] } }
def leftRaw : List Term := Proof.Events670.exact171600RawTerms
def rightRaw : List Term := Proof.Events670.exact171557RawTerms
def group : MergeGroup := .operator 171600 171557
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171600) (leftOrdinal := 0)
    (rightResult := 171557) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20263⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20263⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171604

namespace LeftMerge171605
def owner : Owner := ⟨.program ⟨257⟩, ⟨20266⟩⟩
def mergeEvent : Nat := 171605
def frameStart : Nat := 171512
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12741⟩⟩, ⟨.program ⟨257⟩, ⟨18370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20263⟩⟩] } }
def leftRaw : List Term := Proof.Events670.exact171600RawTerms
def rightRaw : List Term := Proof.Events670.exact171557RawTerms
def group : MergeGroup := .operator 171600 171557
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171600) (leftOrdinal := 1)
    (rightResult := 171557) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12741⟩⟩, ⟨.program ⟨257⟩, ⟨18370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20263⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12741⟩⟩, ⟨.program ⟨257⟩, ⟨18370⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20263⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge171605

namespace LeftMerge171607
def owner : Owner := ⟨.program ⟨257⟩, ⟨20266⟩⟩
def mergeEvent : Nat := 171607
def frameStart : Nat := 171512
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12741⟩⟩, ⟨.program ⟨257⟩, ⟨18370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19733⟩⟩] } }
def rhsRaw : List Term := Proof.Events670.exact171554RawTerms
def group : MergeGroup := .relation 171606
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 171606) (rhsResult := 171554)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12741⟩⟩, ⟨.program ⟨257⟩, ⟨18370⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20263⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20263⟩⟩) ⟨19733⟩ 171554) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19733⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12741⟩⟩, ⟨.program ⟨257⟩, ⟨18370⟩⟩], [⟨.program ⟨257⟩, ⟨19733⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge171607

namespace LeftMerge171615
def owner : Owner := ⟨.program ⟨257⟩, ⟨18622⟩⟩
def mergeEvent : Nat := 171615
def frameStart : Nat := 171512
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events670.exact171568RawTerms
def rightRaw : List Term := Proof.Events670.exact171611RawTerms
def group : MergeGroup := .operator 171568 171611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171568) (leftOrdinal := 0)
    (rightResult := 171611) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18620⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171615

namespace LeftMerge171632
def owner : Owner := ⟨.program ⟨257⟩, ⟨19192⟩⟩
def mergeEvent : Nat := 171632
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }
def rhsRaw : List Term := Proof.Events670.exact171629RawTerms
def group : MergeGroup := .relation 171631
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 171631) (rhsResult := 171629)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19189⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 171630 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19189⟩⟩]⟩) (none) 171629) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171632

namespace LeftMerge171633
def owner : Owner := ⟨.program ⟨257⟩, ⟨19192⟩⟩
def mergeEvent : Nat := 171633
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20263⟩⟩] } }
def rhsRaw : List Term := Proof.Events670.exact171629RawTerms
def group : MergeGroup := .relation 171631
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 171631) (rhsResult := 171629)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19189⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 171630 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19189⟩⟩]⟩) (none) 171629) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20263⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20263⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge171633

namespace LeftMerge171634
def owner : Owner := ⟨.program ⟨257⟩, ⟨19192⟩⟩
def mergeEvent : Nat := 171634
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨12741⟩⟩, ⟨.program ⟨257⟩, ⟨18370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19733⟩⟩] } }
def rhsRaw : List Term := Proof.Events670.exact171629RawTerms
def group : MergeGroup := .relation 171631
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 171631) (rhsResult := 171629)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19189⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 171630 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19189⟩⟩]⟩) (none) 171629) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12741⟩⟩, ⟨.program ⟨257⟩, ⟨18370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19733⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨12741⟩⟩, ⟨.program ⟨257⟩, ⟨18370⟩⟩], [⟨.program ⟨257⟩, ⟨19733⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171634

namespace LeftMerge171635
def owner : Owner := ⟨.program ⟨257⟩, ⟨19192⟩⟩
def mergeEvent : Nat := 171635
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events670.exact171629RawTerms
def group : MergeGroup := .relation 171631
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 171631) (rhsResult := 171629)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19189⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 171630 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19189⟩⟩]⟩) (none) 171629) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge171635

namespace LeftMerge171640
def owner : Owner := ⟨.program ⟨257⟩, ⟨20265⟩⟩
def mergeEvent : Nat := 171640
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨12741⟩⟩, ⟨.program ⟨257⟩, ⟨18370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19733⟩⟩] } }
def leftRaw : List Term := Proof.Events670.exact171636RawTerms
def rightRaw : List Term := Proof.Events669.exact171450RawTerms
def group : MergeGroup := .operator 171636 171450
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171636) (leftOrdinal := 2)
    (rightResult := 171450) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨12741⟩⟩, ⟨.program ⟨257⟩, ⟨18370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19733⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨12741⟩⟩, ⟨.program ⟨257⟩, ⟨18370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19733⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨12741⟩⟩, ⟨.program ⟨257⟩, ⟨18370⟩⟩], [⟨.program ⟨257⟩, ⟨19733⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge171640

namespace LeftMerge171641
def owner : Owner := ⟨.program ⟨257⟩, ⟨20265⟩⟩
def mergeEvent : Nat := 171641
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20263⟩⟩] } }
def leftRaw : List Term := Proof.Events670.exact171636RawTerms
def rightRaw : List Term := Proof.Events669.exact171450RawTerms
def group : MergeGroup := .operator 171636 171450
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171636) (leftOrdinal := 1)
    (rightResult := 171450) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20263⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20263⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20263⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171641

namespace LeftMerge171649
def owner : Owner := ⟨.program ⟨257⟩, ⟨20778⟩⟩
def mergeEvent : Nat := 171649
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20776⟩⟩] } }
def leftRaw : List Term := Proof.Events670.exact171643RawTerms
def rightRaw : List Term := Proof.Events669.exact171366RawTerms
def group : MergeGroup := .operator 171643 171366
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171643) (leftOrdinal := 0)
    (rightResult := 171366) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20776⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20776⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171649

namespace LeftMerge171650
def owner : Owner := ⟨.program ⟨257⟩, ⟨20778⟩⟩
def mergeEvent : Nat := 171650
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20776⟩⟩] } }
def leftRaw : List Term := Proof.Events670.exact171643RawTerms
def rightRaw : List Term := Proof.Events669.exact171366RawTerms
def group : MergeGroup := .operator 171643 171366
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171643) (leftOrdinal := 1)
    (rightResult := 171366) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20776⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20776⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge171650

namespace LeftMerge171652
def owner : Owner := ⟨.program ⟨257⟩, ⟨20778⟩⟩
def mergeEvent : Nat := 171652
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19897⟩⟩] } }
def rhsRaw : List Term := Proof.Events669.exact171363RawTerms
def group : MergeGroup := .relation 171651
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 171651) (rhsResult := 171363)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20776⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20776⟩⟩) ⟨19897⟩ 171363) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19897⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18620⟩⟩], [⟨.program ⟨257⟩, ⟨19897⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge171652

namespace LeftMerge171666
def owner : Owner := ⟨.program ⟨257⟩, ⟨19539⟩⟩
def mergeEvent : Nat := 171666
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19536⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events670.exact171660RawTerms
def group : MergeGroup := .operator 163745 171660
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 171660) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19536⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19536⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171666

namespace LeftMerge171787
def owner : Owner := ⟨.program ⟨257⟩, ⟨20084⟩⟩
def mergeEvent : Nat := 171787
def frameStart : Nat := 171721
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events671.exact171783RawTerms
def rightRaw : List Term := Proof.Events671.exact171781RawTerms
def group : MergeGroup := .operator 171783 171781
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171783) (leftOrdinal := 0)
    (rightResult := 171781) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18620⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171787

namespace LeftMerge171799
def owner : Owner := ⟨.program ⟨257⟩, ⟨20777⟩⟩
def mergeEvent : Nat := 171799
def frameStart : Nat := 171721
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20776⟩⟩] } }
def leftRaw : List Term := Proof.Events671.exact171795RawTerms
def rightRaw : List Term := Proof.Events670.exact171772RawTerms
def group : MergeGroup := .operator 171795 171772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171795) (leftOrdinal := 0)
    (rightResult := 171772) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20776⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20776⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171799

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
