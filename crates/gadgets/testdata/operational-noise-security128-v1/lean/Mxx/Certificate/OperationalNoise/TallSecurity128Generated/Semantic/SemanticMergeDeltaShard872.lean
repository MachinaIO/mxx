import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge144529
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144529
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events564.exact144511RawTerms
def rightRaw : List Term := Proof.Events563.exact144352RawTerms
def group : MergeGroup := .operator 144511 144352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144511) (leftOrdinal := 3)
    (rightResult := 144352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge144529

namespace LeftMerge144530
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144530
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events564.exact144511RawTerms
def rightRaw : List Term := Proof.Events563.exact144352RawTerms
def group : MergeGroup := .operator 144511 144352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144511) (leftOrdinal := 2)
    (rightResult := 144352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge144530

namespace LeftMerge144531
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144531
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events564.exact144511RawTerms
def rightRaw : List Term := Proof.Events563.exact144352RawTerms
def group : MergeGroup := .operator 144511 144352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144511) (leftOrdinal := 1)
    (rightResult := 144352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge144531

namespace LeftMerge144532
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144532
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events564.exact144511RawTerms
def rightRaw : List Term := Proof.Events563.exact144352RawTerms
def group : MergeGroup := .operator 144511 144352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144511) (leftOrdinal := 0)
    (rightResult := 144352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge144532

namespace LeftMerge144533
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144533
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48272⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events564.exact144511RawTerms
def rightRaw : List Term := Proof.Events563.exact144352RawTerms
def group : MergeGroup := .operator 144511 144352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144511) (leftOrdinal := 29)
    (rightResult := 144352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48272⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144533

namespace LeftMerge144535
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144535
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48272⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events563.exact144349RawTerms
def group : MergeGroup := .relation 144534
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 144534) (rhsResult := 144349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144535

namespace LeftMerge144536
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144536
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45592⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events564.exact144511RawTerms
def rightRaw : List Term := Proof.Events563.exact144352RawTerms
def group : MergeGroup := .operator 144511 144352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144511) (leftOrdinal := 28)
    (rightResult := 144352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45592⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45592⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144536

namespace LeftMerge144538
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144538
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45592⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events563.exact144349RawTerms
def group : MergeGroup := .relation 144537
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 144537) (rhsResult := 144349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45592⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45592⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144538

namespace LeftMerge144539
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144539
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42908⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events564.exact144511RawTerms
def rightRaw : List Term := Proof.Events563.exact144352RawTerms
def group : MergeGroup := .operator 144511 144352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144511) (leftOrdinal := 27)
    (rightResult := 144352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42908⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144539

namespace LeftMerge144541
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144541
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42908⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events563.exact144349RawTerms
def group : MergeGroup := .relation 144540
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 144540) (rhsResult := 144349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144541

namespace LeftMerge144542
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144542
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40228⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events564.exact144511RawTerms
def rightRaw : List Term := Proof.Events563.exact144352RawTerms
def group : MergeGroup := .operator 144511 144352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144511) (leftOrdinal := 26)
    (rightResult := 144352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40228⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144542

namespace LeftMerge144544
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144544
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40228⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events563.exact144349RawTerms
def group : MergeGroup := .relation 144543
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 144543) (rhsResult := 144349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144544

namespace LeftMerge144545
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144545
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events564.exact144511RawTerms
def rightRaw : List Term := Proof.Events563.exact144352RawTerms
def group : MergeGroup := .operator 144511 144352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144511) (leftOrdinal := 25)
    (rightResult := 144352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144545

namespace LeftMerge144547
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144547
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events563.exact144349RawTerms
def group : MergeGroup := .relation 144546
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 144546) (rhsResult := 144349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144547

namespace LeftMerge144548
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144548
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events564.exact144511RawTerms
def rightRaw : List Term := Proof.Events563.exact144352RawTerms
def group : MergeGroup := .operator 144511 144352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144511) (leftOrdinal := 24)
    (rightResult := 144352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144548

namespace LeftMerge144550
def owner : Owner := ⟨.program ⟨257⟩, ⟨71018⟩⟩
def mergeEvent : Nat := 144550
def frameStart : Nat := 143836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events563.exact144349RawTerms
def group : MergeGroup := .relation 144549
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 144549) (rhsResult := 144349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144550

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
