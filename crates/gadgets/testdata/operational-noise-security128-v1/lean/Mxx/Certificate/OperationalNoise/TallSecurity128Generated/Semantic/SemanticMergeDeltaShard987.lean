import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge162307
def owner : Owner := ⟨.program ⟨257⟩, ⟨33794⟩⟩
def mergeEvent : Nat := 162307
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33792⟩⟩] } }
def leftRaw : List Term := Proof.Events609.exact156054RawTerms
def rightRaw : List Term := Proof.Events633.exact162300RawTerms
def group : MergeGroup := .operator 156054 162300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 156054) (leftOrdinal := 1)
    (rightResult := 162300) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33792⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨31804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33792⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162307

namespace LeftMerge162309
def owner : Owner := ⟨.program ⟨257⟩, ⟨33794⟩⟩
def mergeEvent : Nat := 162309
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33073⟩⟩] } }
def rhsRaw : List Term := Proof.Events633.exact162297RawTerms
def group : MergeGroup := .relation 162308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 162308) (rhsResult := 162297)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨31804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33792⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33792⟩⟩) ⟨33073⟩ 162297) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33073⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨31804⟩⟩], [⟨.program ⟨257⟩, ⟨33073⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162309

namespace LeftMerge162323
def owner : Owner := ⟨.program ⟨257⟩, ⟨32635⟩⟩
def mergeEvent : Nat := 162323
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32632⟩⟩] } }
def leftRaw : List Term := Proof.Events582.exact149120RawTerms
def rightRaw : List Term := Proof.Events634.exact162317RawTerms
def group : MergeGroup := .operator 149120 162317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149120) (leftOrdinal := 0)
    (rightResult := 162317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32632⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32632⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge162323

namespace LeftMerge162444
def owner : Owner := ⟨.program ⟨257⟩, ⟨33296⟩⟩
def mergeEvent : Nat := 162444
def frameStart : Nat := 162378
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events634.exact162440RawTerms
def rightRaw : List Term := Proof.Events634.exact162438RawTerms
def group : MergeGroup := .operator 162440 162438
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 162440) (leftOrdinal := 0)
    (rightResult := 162438) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge162444

namespace LeftMerge162456
def owner : Owner := ⟨.program ⟨257⟩, ⟨33793⟩⟩
def mergeEvent : Nat := 162456
def frameStart : Nat := 162378
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33792⟩⟩] } }
def leftRaw : List Term := Proof.Events634.exact162452RawTerms
def rightRaw : List Term := Proof.Events634.exact162429RawTerms
def group : MergeGroup := .operator 162452 162429
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 162452) (leftOrdinal := 0)
    (rightResult := 162429) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33792⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33792⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge162456

namespace LeftMerge162457
def owner : Owner := ⟨.program ⟨257⟩, ⟨33793⟩⟩
def mergeEvent : Nat := 162457
def frameStart : Nat := 162378
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33792⟩⟩] } }
def leftRaw : List Term := Proof.Events634.exact162452RawTerms
def rightRaw : List Term := Proof.Events634.exact162429RawTerms
def group : MergeGroup := .operator 162452 162429
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 162452) (leftOrdinal := 1)
    (rightResult := 162429) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33792⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33792⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162457

namespace LeftMerge162459
def owner : Owner := ⟨.program ⟨257⟩, ⟨33793⟩⟩
def mergeEvent : Nat := 162459
def frameStart : Nat := 162378
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33073⟩⟩] } }
def rhsRaw : List Term := Proof.Events634.exact162426RawTerms
def group : MergeGroup := .relation 162458
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 162458) (rhsResult := 162426)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨31804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33792⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33792⟩⟩) ⟨33073⟩ 162426) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33073⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨31804⟩⟩], [⟨.program ⟨257⟩, ⟨33073⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162459

namespace LeftMerge162467
def owner : Owner := ⟨.program ⟨257⟩, ⟨32047⟩⟩
def mergeEvent : Nat := 162467
def frameStart : Nat := 162378
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32044⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events634.exact162440RawTerms
def rightRaw : List Term := Proof.Events634.exact162463RawTerms
def group : MergeGroup := .operator 162440 162463
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 162440) (leftOrdinal := 0)
    (rightResult := 162463) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32044⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge162467

namespace LeftMerge162484
def owner : Owner := ⟨.program ⟨257⟩, ⟨32635⟩⟩
def mergeEvent : Nat := 162484
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7203⟩⟩] } }
def rhsRaw : List Term := Proof.Events634.exact162481RawTerms
def group : MergeGroup := .relation 162483
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 162483) (rhsResult := 162481)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32632⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 162482 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32632⟩⟩]⟩) (none) 162481) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7203⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge162484

namespace LeftMerge162485
def owner : Owner := ⟨.program ⟨257⟩, ⟨32635⟩⟩
def mergeEvent : Nat := 162485
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33792⟩⟩] } }
def rhsRaw : List Term := Proof.Events634.exact162481RawTerms
def group : MergeGroup := .relation 162483
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 162483) (rhsResult := 162481)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32632⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 162482 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32632⟩⟩]⟩) (none) 162481) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33792⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33792⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162485

namespace LeftMerge162486
def owner : Owner := ⟨.program ⟨257⟩, ⟨32635⟩⟩
def mergeEvent : Nat := 162486
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33073⟩⟩] } }
def rhsRaw : List Term := Proof.Events634.exact162481RawTerms
def group : MergeGroup := .relation 162483
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 162483) (rhsResult := 162481)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32632⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 162482 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32632⟩⟩]⟩) (none) 162481) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33073⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨31804⟩⟩], [⟨.program ⟨257⟩, ⟨33073⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge162486

namespace LeftMerge162487
def owner : Owner := ⟨.program ⟨257⟩, ⟨32635⟩⟩
def mergeEvent : Nat := 162487
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events634.exact162481RawTerms
def group : MergeGroup := .relation 162483
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 162483) (rhsResult := 162481)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32632⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 162482 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32632⟩⟩]⟩) (none) 162481) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32044⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162487

namespace LeftMerge162492
def owner : Owner := ⟨.program ⟨257⟩, ⟨33795⟩⟩
def mergeEvent : Nat := 162492
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33792⟩⟩] } }
def leftRaw : List Term := Proof.Events634.exact162488RawTerms
def rightRaw : List Term := Proof.Events634.exact162310RawTerms
def group : MergeGroup := .operator 162488 162310
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 162488) (leftOrdinal := 0)
    (rightResult := 162310) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33792⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33792⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33792⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge162492

namespace LeftMerge162493
def owner : Owner := ⟨.program ⟨257⟩, ⟨33795⟩⟩
def mergeEvent : Nat := 162493
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33073⟩⟩] } }
def leftRaw : List Term := Proof.Events634.exact162488RawTerms
def rightRaw : List Term := Proof.Events634.exact162310RawTerms
def group : MergeGroup := .operator 162488 162310
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 162488) (leftOrdinal := 2)
    (rightResult := 162310) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33073⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33073⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨31804⟩⟩], [⟨.program ⟨257⟩, ⟨33073⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162493

namespace LeftMerge162501
def owner : Owner := ⟨.program ⟨257⟩, ⟨33796⟩⟩
def mergeEvent : Nat := 162501
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩] } }
def leftRaw : List Term := Proof.Events634.exact162495RawTerms
def rightRaw : List Term := Proof.Events061.exact15822RawTerms
def group : MergeGroup := .operator 162495 15822
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 162495) (leftOrdinal := 0)
    (rightResult := 15822) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7203⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7145⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge162501

namespace LeftMerge162502
def owner : Owner := ⟨.program ⟨257⟩, ⟨33796⟩⟩
def mergeEvent : Nat := 162502
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩] } }
def leftRaw : List Term := Proof.Events634.exact162495RawTerms
def rightRaw : List Term := Proof.Events061.exact15822RawTerms
def group : MergeGroup := .operator 162495 15822
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 162495) (leftOrdinal := 1)
    (rightResult := 15822) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7145⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162502

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
