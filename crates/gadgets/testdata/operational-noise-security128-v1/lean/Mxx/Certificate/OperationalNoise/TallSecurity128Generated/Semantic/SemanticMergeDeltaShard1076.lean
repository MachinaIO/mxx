import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge176474
def owner : Owner := ⟨.program ⟨257⟩, ⟨57795⟩⟩
def mergeEvent : Nat := 176474
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59029⟩⟩] } }
def rhsRaw : List Term := Proof.Events689.exact176470RawTerms
def group : MergeGroup := .relation 176472
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 176472) (rhsResult := 176470)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57792⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 176471 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57792⟩⟩]⟩) (none) 176470) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59029⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59029⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176474

namespace LeftMerge176475
def owner : Owner := ⟨.program ⟨257⟩, ⟨57795⟩⟩
def mergeEvent : Nat := 176475
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58156⟩⟩] } }
def rhsRaw : List Term := Proof.Events689.exact176470RawTerms
def group : MergeGroup := .relation 176472
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 176472) (rhsResult := 176470)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57792⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 176471 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57792⟩⟩]⟩) (none) 176470) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58156⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨56880⟩⟩], [⟨.program ⟨257⟩, ⟨58156⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176475

namespace LeftMerge176476
def owner : Owner := ⟨.program ⟨257⟩, ⟨57795⟩⟩
def mergeEvent : Nat := 176476
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events689.exact176470RawTerms
def group : MergeGroup := .relation 176472
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 176472) (rhsResult := 176470)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57792⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 176471 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57792⟩⟩]⟩) (none) 176470) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57201⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176476

namespace LeftMerge176481
def owner : Owner := ⟨.program ⟨257⟩, ⟨59032⟩⟩
def mergeEvent : Nat := 176481
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59029⟩⟩] } }
def leftRaw : List Term := Proof.Events689.exact176477RawTerms
def rightRaw : List Term := Proof.Events688.exact176299RawTerms
def group : MergeGroup := .operator 176477 176299
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 176477) (leftOrdinal := 0)
    (rightResult := 176299) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59029⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59029⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59029⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176481

namespace LeftMerge176482
def owner : Owner := ⟨.program ⟨257⟩, ⟨59032⟩⟩
def mergeEvent : Nat := 176482
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58156⟩⟩] } }
def leftRaw : List Term := Proof.Events689.exact176477RawTerms
def rightRaw : List Term := Proof.Events688.exact176299RawTerms
def group : MergeGroup := .operator 176477 176299
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 176477) (leftOrdinal := 2)
    (rightResult := 176299) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58156⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58156⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨56880⟩⟩], [⟨.program ⟨257⟩, ⟨58156⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176482

namespace LeftMerge176490
def owner : Owner := ⟨.program ⟨257⟩, ⟨59033⟩⟩
def mergeEvent : Nat := 176490
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩] } }
def leftRaw : List Term := Proof.Events689.exact176484RawTerms
def rightRaw : List Term := Proof.Events061.exact15762RawTerms
def group : MergeGroup := .operator 176484 15762
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 176484) (leftOrdinal := 0)
    (rightResult := 15762) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7209⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7107⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176490

namespace LeftMerge176491
def owner : Owner := ⟨.program ⟨257⟩, ⟨59033⟩⟩
def mergeEvent : Nat := 176491
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩] } }
def leftRaw : List Term := Proof.Events689.exact176484RawTerms
def rightRaw : List Term := Proof.Events061.exact15762RawTerms
def group : MergeGroup := .operator 176484 15762
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 176484) (leftOrdinal := 1)
    (rightResult := 15762) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7107⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176491

namespace LeftMerge176493
def owner : Owner := ⟨.program ⟨257⟩, ⟨59033⟩⟩
def mergeEvent : Nat := 176493
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15755RawTerms
def group : MergeGroup := .relation 176492
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 176492) (rhsResult := 15755)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7107⟩⟩) ⟨7019⟩ 15755) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176493

namespace LeftMerge176507
def owner : Owner := ⟨.program ⟨257⟩, ⟨56051⟩⟩
def mergeEvent : Nat := 176507
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56049⟩⟩] } }
def leftRaw : List Term := Proof.Events662.exact169715RawTerms
def rightRaw : List Term := Proof.Events689.exact176501RawTerms
def group : MergeGroup := .operator 169715 176501
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169715) (leftOrdinal := 0)
    (rightResult := 176501) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56049⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56049⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176507

namespace LeftMerge176508
def owner : Owner := ⟨.program ⟨257⟩, ⟨56051⟩⟩
def mergeEvent : Nat := 176508
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨53900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56049⟩⟩] } }
def leftRaw : List Term := Proof.Events662.exact169715RawTerms
def rightRaw : List Term := Proof.Events689.exact176501RawTerms
def group : MergeGroup := .operator 169715 176501
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169715) (leftOrdinal := 1)
    (rightResult := 176501) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨53900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56049⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨53900⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56049⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176508

namespace LeftMerge176510
def owner : Owner := ⟨.program ⟨257⟩, ⟨56051⟩⟩
def mergeEvent : Nat := 176510
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨53900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55176⟩⟩] } }
def rhsRaw : List Term := Proof.Events689.exact176498RawTerms
def group : MergeGroup := .relation 176509
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 176509) (rhsResult := 176498)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨53900⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56049⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨56049⟩⟩) ⟨55176⟩ 176498) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55176⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨53900⟩⟩], [⟨.program ⟨257⟩, ⟨55176⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176510

namespace LeftMerge176524
def owner : Owner := ⟨.program ⟨257⟩, ⟨54815⟩⟩
def mergeEvent : Nat := 176524
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54812⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events689.exact176518RawTerms
def group : MergeGroup := .operator 163745 176518
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 176518) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54812⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54812⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176524

namespace LeftMerge176645
def owner : Owner := ⟨.program ⟨257⟩, ⟨55364⟩⟩
def mergeEvent : Nat := 176645
def frameStart : Nat := 176579
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events690.exact176641RawTerms
def rightRaw : List Term := Proof.Events689.exact176639RawTerms
def group : MergeGroup := .operator 176641 176639
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 176641) (leftOrdinal := 0)
    (rightResult := 176639) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53900⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53900⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176645

namespace LeftMerge176657
def owner : Owner := ⟨.program ⟨257⟩, ⟨56050⟩⟩
def mergeEvent : Nat := 176657
def frameStart : Nat := 176579
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56049⟩⟩] } }
def leftRaw : List Term := Proof.Events690.exact176653RawTerms
def rightRaw : List Term := Proof.Events689.exact176630RawTerms
def group : MergeGroup := .operator 176653 176630
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 176653) (leftOrdinal := 0)
    (rightResult := 176630) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56049⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56049⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176657

namespace LeftMerge176658
def owner : Owner := ⟨.program ⟨257⟩, ⟨56050⟩⟩
def mergeEvent : Nat := 176658
def frameStart : Nat := 176579
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56049⟩⟩] } }
def leftRaw : List Term := Proof.Events690.exact176653RawTerms
def rightRaw : List Term := Proof.Events689.exact176630RawTerms
def group : MergeGroup := .operator 176653 176630
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 176653) (leftOrdinal := 1)
    (rightResult := 176630) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56049⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53900⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56049⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176658

namespace LeftMerge176660
def owner : Owner := ⟨.program ⟨257⟩, ⟨56050⟩⟩
def mergeEvent : Nat := 176660
def frameStart : Nat := 176579
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55176⟩⟩] } }
def rhsRaw : List Term := Proof.Events689.exact176627RawTerms
def group : MergeGroup := .relation 176659
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 176659) (rhsResult := 176627)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨53900⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56049⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨56049⟩⟩) ⟨55176⟩ 176627) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55176⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨53900⟩⟩], [⟨.program ⟨257⟩, ⟨55176⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176660

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
