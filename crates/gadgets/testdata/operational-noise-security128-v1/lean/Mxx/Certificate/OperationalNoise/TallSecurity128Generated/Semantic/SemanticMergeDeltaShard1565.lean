import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge253414
def owner : Owner := ⟨.program ⟨257⟩, ⟨38885⟩⟩
def mergeEvent : Nat := 253414
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38399⟩⟩] } }
def rhsRaw : List Term := Proof.Events989.exact253339RawTerms
def group : MergeGroup := .relation 253413
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 253413) (rhsResult := 253339)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38884⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38884⟩⟩) ⟨38399⟩ 253339) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38399⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], [⟨.program ⟨257⟩, ⟨38399⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge253414

namespace LeftMerge253415
def owner : Owner := ⟨.program ⟨257⟩, ⟨38885⟩⟩
def mergeEvent : Nat := 253415
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38884⟩⟩] } }
def leftRaw : List Term := Proof.Events989.exact253406RawTerms
def rightRaw : List Term := Proof.Events989.exact253342RawTerms
def group : MergeGroup := .operator 253406 253342
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 253406) (leftOrdinal := 0)
    (rightResult := 253342) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38884⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38884⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge253415

namespace LeftMerge253429
def owner : Owner := ⟨.program ⟨257⟩, ⟨37822⟩⟩
def mergeEvent : Nat := 253429
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37819⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events989.exact253423RawTerms
def group : MergeGroup := .operator 251495 253423
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 253423) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨37819⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37819⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge253429

namespace LeftMerge253508
def owner : Owner := ⟨.program ⟨257⟩, ⟨36995⟩⟩
def mergeEvent : Nat := 253508
def frameStart : Nat := 253478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events990.exact253504RawTerms
def rightRaw : List Term := Proof.Events990.exact253501RawTerms
def group : MergeGroup := .operator 253504 253501
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 253504) (leftOrdinal := 0)
    (rightResult := 253501) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13806⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨36994⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge253508

namespace LeftMerge253538
def owner : Owner := ⟨.program ⟨257⟩, ⟨38688⟩⟩
def mergeEvent : Nat := 253538
def frameStart : Nat := 253478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events990.exact253534RawTerms
def rightRaw : List Term := Proof.Events990.exact253532RawTerms
def group : MergeGroup := .operator 253534 253532
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 253534) (leftOrdinal := 0)
    (rightResult := 253532) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge253538

namespace LeftMerge253561
def owner : Owner := ⟨.program ⟨257⟩, ⟨9555⟩⟩
def mergeEvent : Nat := 253561
def frameStart : Nat := 253478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }
def leftRaw : List Term := Proof.Events990.exact253557RawTerms
def rightRaw : List Term := Proof.Events990.exact253554RawTerms
def group : MergeGroup := .operator 253557 253554
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 253557) (leftOrdinal := 0)
    (rightResult := 253554) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9553⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge253561

namespace LeftMerge253570
def owner : Owner := ⟨.program ⟨257⟩, ⟨38887⟩⟩
def mergeEvent : Nat := 253570
def frameStart : Nat := 253478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38884⟩⟩] } }
def leftRaw : List Term := Proof.Events990.exact253566RawTerms
def rightRaw : List Term := Proof.Events990.exact253523RawTerms
def group : MergeGroup := .operator 253566 253523
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 253566) (leftOrdinal := 0)
    (rightResult := 253523) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38884⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38884⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge253570

namespace LeftMerge253571
def owner : Owner := ⟨.program ⟨257⟩, ⟨38887⟩⟩
def mergeEvent : Nat := 253571
def frameStart : Nat := 253478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38884⟩⟩] } }
def leftRaw : List Term := Proof.Events990.exact253566RawTerms
def rightRaw : List Term := Proof.Events990.exact253523RawTerms
def group : MergeGroup := .operator 253566 253523
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 253566) (leftOrdinal := 1)
    (rightResult := 253523) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38884⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38884⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge253571

namespace LeftMerge253573
def owner : Owner := ⟨.program ⟨257⟩, ⟨38887⟩⟩
def mergeEvent : Nat := 253573
def frameStart : Nat := 253478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38399⟩⟩] } }
def rhsRaw : List Term := Proof.Events990.exact253520RawTerms
def group : MergeGroup := .relation 253572
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 253572) (rhsResult := 253520)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38884⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38884⟩⟩) ⟨38399⟩ 253520) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38399⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], [⟨.program ⟨257⟩, ⟨38399⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge253573

namespace LeftMerge253581
def owner : Owner := ⟨.program ⟨257⟩, ⟨37390⟩⟩
def mergeEvent : Nat := 253581
def frameStart : Nat := 253478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37388⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events990.exact253534RawTerms
def rightRaw : List Term := Proof.Events990.exact253577RawTerms
def group : MergeGroup := .operator 253534 253577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 253534) (leftOrdinal := 0)
    (rightResult := 253577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37388⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37388⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge253581

namespace LeftMerge253598
def owner : Owner := ⟨.program ⟨257⟩, ⟨37822⟩⟩
def mergeEvent : Nat := 253598
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }
def rhsRaw : List Term := Proof.Events990.exact253595RawTerms
def group : MergeGroup := .relation 253597
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 253597) (rhsResult := 253595)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37819⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 253596 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37819⟩⟩]⟩) (none) 253595) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge253598

namespace LeftMerge253599
def owner : Owner := ⟨.program ⟨257⟩, ⟨37822⟩⟩
def mergeEvent : Nat := 253599
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38884⟩⟩] } }
def rhsRaw : List Term := Proof.Events990.exact253595RawTerms
def group : MergeGroup := .relation 253597
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 253597) (rhsResult := 253595)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37819⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 253596 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37819⟩⟩]⟩) (none) 253595) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38884⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38884⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge253599

namespace LeftMerge253600
def owner : Owner := ⟨.program ⟨257⟩, ⟨37822⟩⟩
def mergeEvent : Nat := 253600
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38399⟩⟩] } }
def rhsRaw : List Term := Proof.Events990.exact253595RawTerms
def group : MergeGroup := .relation 253597
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 253597) (rhsResult := 253595)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37819⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 253596 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37819⟩⟩]⟩) (none) 253595) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38399⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], [⟨.program ⟨257⟩, ⟨38399⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge253600

namespace LeftMerge253601
def owner : Owner := ⟨.program ⟨257⟩, ⟨37822⟩⟩
def mergeEvent : Nat := 253601
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨37388⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events990.exact253595RawTerms
def group : MergeGroup := .relation 253597
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 253597) (rhsResult := 253595)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37819⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 253596 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37819⟩⟩]⟩) (none) 253595) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37388⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨37388⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge253601

namespace LeftMerge253606
def owner : Owner := ⟨.program ⟨257⟩, ⟨38886⟩⟩
def mergeEvent : Nat := 253606
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38399⟩⟩] } }
def leftRaw : List Term := Proof.Events990.exact253602RawTerms
def rightRaw : List Term := Proof.Events989.exact253416RawTerms
def group : MergeGroup := .operator 253602 253416
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 253602) (leftOrdinal := 2)
    (rightResult := 253416) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38399⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38399⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13806⟩⟩, ⟨.program ⟨257⟩, ⟨36994⟩⟩], [⟨.program ⟨257⟩, ⟨38399⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge253606

namespace LeftMerge253607
def owner : Owner := ⟨.program ⟨257⟩, ⟨38886⟩⟩
def mergeEvent : Nat := 253607
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38884⟩⟩] } }
def leftRaw : List Term := Proof.Events990.exact253602RawTerms
def rightRaw : List Term := Proof.Events989.exact253416RawTerms
def group : MergeGroup := .operator 253602 253416
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 253602) (leftOrdinal := 1)
    (rightResult := 253416) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38884⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38884⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38884⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge253607

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
