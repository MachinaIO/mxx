import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge144551
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144551
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29208⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events564.exact144511RawTerms
def rightRaw : List Term := Proof.Events563.exact144352RawTerms
def group : MergeGroup := .operator 144511 144352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144511) (leftOrdinal := 22)
    (rightResult := 144352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29208⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144551

namespace LeftMerge144553
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144553
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29208⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events563.exact144349RawTerms
def group : MergeGroup := .relation 144552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 144552) (rhsResult := 144349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144553

namespace LeftMerge144554
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144554
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26528⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events564.exact144511RawTerms
def rightRaw : List Term := Proof.Events563.exact144352RawTerms
def group : MergeGroup := .operator 144511 144352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144511) (leftOrdinal := 21)
    (rightResult := 144352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26528⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144554

namespace LeftMerge144556
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144556
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26528⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events563.exact144349RawTerms
def group : MergeGroup := .relation 144555
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 144555) (rhsResult := 144349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144556

namespace LeftMerge144557
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144557
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66111⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events564.exact144511RawTerms
def rightRaw : List Term := Proof.Events563.exact144352RawTerms
def group : MergeGroup := .operator 144511 144352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144511) (leftOrdinal := 35)
    (rightResult := 144352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66111⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144557

namespace LeftMerge144559
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144559
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66111⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events563.exact144349RawTerms
def group : MergeGroup := .relation 144558
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 144558) (rhsResult := 144349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144559

namespace LeftMerge144560
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144560
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62948⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events564.exact144511RawTerms
def rightRaw : List Term := Proof.Events563.exact144352RawTerms
def group : MergeGroup := .operator 144511 144352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144511) (leftOrdinal := 34)
    (rightResult := 144352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62948⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144560

namespace LeftMerge144562
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144562
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62948⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events563.exact144349RawTerms
def group : MergeGroup := .relation 144561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 144561) (rhsResult := 144349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144562

namespace LeftMerge144563
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144563
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59968⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events564.exact144511RawTerms
def rightRaw : List Term := Proof.Events563.exact144352RawTerms
def group : MergeGroup := .operator 144511 144352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144511) (leftOrdinal := 33)
    (rightResult := 144352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59968⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144563

namespace LeftMerge144565
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144565
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59968⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events563.exact144349RawTerms
def group : MergeGroup := .relation 144564
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 144564) (rhsResult := 144349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144565

namespace LeftMerge144566
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144566
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56988⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events564.exact144511RawTerms
def rightRaw : List Term := Proof.Events563.exact144352RawTerms
def group : MergeGroup := .operator 144511 144352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144511) (leftOrdinal := 32)
    (rightResult := 144352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56988⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144566

namespace LeftMerge144568
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144568
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56988⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events563.exact144349RawTerms
def group : MergeGroup := .relation 144567
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 144567) (rhsResult := 144349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144568

namespace LeftMerge144569
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144569
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54008⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events564.exact144511RawTerms
def rightRaw : List Term := Proof.Events563.exact144352RawTerms
def group : MergeGroup := .operator 144511 144352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144511) (leftOrdinal := 31)
    (rightResult := 144352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54008⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144569

namespace LeftMerge144571
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144571
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54008⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events563.exact144349RawTerms
def group : MergeGroup := .relation 144570
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 144570) (rhsResult := 144349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144571

namespace LeftMerge144572
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144572
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events564.exact144511RawTerms
def rightRaw : List Term := Proof.Events563.exact144352RawTerms
def group : MergeGroup := .operator 144511 144352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144511) (leftOrdinal := 30)
    (rightResult := 144352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144572

namespace LeftMerge144574
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144574
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events563.exact144349RawTerms
def group : MergeGroup := .relation 144573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 144573) (rhsResult := 144349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144574

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
