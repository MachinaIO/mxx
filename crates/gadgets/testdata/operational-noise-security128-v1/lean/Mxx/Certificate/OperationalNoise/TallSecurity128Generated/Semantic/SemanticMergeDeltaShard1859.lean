import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge300409
def owner : Owner := ⟨.program ⟨257⟩, ⟨54332⟩⟩
def mergeEvent : Nat := 300409
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54329⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1173.exact300403RawTerms
def group : MergeGroup := .operator 295195 300403
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 300403) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge300409

namespace LeftMerge300464
def owner : Owner := ⟨.program ⟨257⟩, ⟨53256⟩⟩
def mergeEvent : Nat := 300464
def frameStart : Nat := 300446
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24650⟩⟩, ⟨.program ⟨257⟩, ⟨53255⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1173.exact300460RawTerms
def rightRaw : List Term := Proof.Events1173.exact300457RawTerms
def group : MergeGroup := .operator 300460 300457
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300460) (leftOrdinal := 0)
    (rightResult := 300457) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53255⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24650⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24650⟩⟩, ⟨.program ⟨257⟩, ⟨53255⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge300464

namespace LeftMerge300494
def owner : Owner := ⟨.program ⟨257⟩, ⟨55228⟩⟩
def mergeEvent : Nat := 300494
def frameStart : Nat := 300446
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24650⟩⟩, ⟨.program ⟨257⟩, ⟨53255⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1173.exact300490RawTerms
def rightRaw : List Term := Proof.Events1173.exact300488RawTerms
def group : MergeGroup := .operator 300490 300488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300490) (leftOrdinal := 0)
    (rightResult := 300488) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24650⟩⟩, ⟨.program ⟨257⟩, ⟨53255⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24650⟩⟩, ⟨.program ⟨257⟩, ⟨53255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge300494

namespace LeftMerge300517
def owner : Owner := ⟨.program ⟨257⟩, ⟨9531⟩⟩
def mergeEvent : Nat := 300517
def frameStart : Nat := 300446
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }
def leftRaw : List Term := Proof.Events1173.exact300513RawTerms
def rightRaw : List Term := Proof.Events1173.exact300510RawTerms
def group : MergeGroup := .operator 300513 300510
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300513) (leftOrdinal := 0)
    (rightResult := 300510) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9529⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge300517

namespace LeftMerge300526
def owner : Owner := ⟨.program ⟨257⟩, ⟨55392⟩⟩
def mergeEvent : Nat := 300526
def frameStart : Nat := 300446
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55389⟩⟩] } }
def leftRaw : List Term := Proof.Events1173.exact300522RawTerms
def rightRaw : List Term := Proof.Events1173.exact300479RawTerms
def group : MergeGroup := .operator 300522 300479
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300522) (leftOrdinal := 0)
    (rightResult := 300479) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55389⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55389⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge300526

namespace LeftMerge300527
def owner : Owner := ⟨.program ⟨257⟩, ⟨55392⟩⟩
def mergeEvent : Nat := 300527
def frameStart : Nat := 300446
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24650⟩⟩, ⟨.program ⟨257⟩, ⟨53255⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55389⟩⟩] } }
def leftRaw : List Term := Proof.Events1173.exact300522RawTerms
def rightRaw : List Term := Proof.Events1173.exact300479RawTerms
def group : MergeGroup := .operator 300522 300479
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300522) (leftOrdinal := 1)
    (rightResult := 300479) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24650⟩⟩, ⟨.program ⟨257⟩, ⟨53255⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55389⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24650⟩⟩, ⟨.program ⟨257⟩, ⟨53255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55389⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge300527

namespace LeftMerge300529
def owner : Owner := ⟨.program ⟨257⟩, ⟨55392⟩⟩
def mergeEvent : Nat := 300529
def frameStart : Nat := 300446
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24650⟩⟩, ⟨.program ⟨257⟩, ⟨53255⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54929⟩⟩] } }
def rhsRaw : List Term := Proof.Events1173.exact300476RawTerms
def group : MergeGroup := .relation 300528
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 300528) (rhsResult := 300476)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24650⟩⟩, ⟨.program ⟨257⟩, ⟨53255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55389⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55389⟩⟩) ⟨54929⟩ 300476) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54929⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24650⟩⟩, ⟨.program ⟨257⟩, ⟨53255⟩⟩], [⟨.program ⟨257⟩, ⟨54929⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge300529

namespace LeftMerge300537
def owner : Owner := ⟨.program ⟨257⟩, ⟨53790⟩⟩
def mergeEvent : Nat := 300537
def frameStart : Nat := 300446
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1173.exact300490RawTerms
def rightRaw : List Term := Proof.Events1173.exact300533RawTerms
def group : MergeGroup := .operator 300490 300533
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300490) (leftOrdinal := 0)
    (rightResult := 300533) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53788⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge300537

namespace LeftMerge300554
def owner : Owner := ⟨.program ⟨257⟩, ⟨54332⟩⟩
def mergeEvent : Nat := 300554
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }
def rhsRaw : List Term := Proof.Events1174.exact300551RawTerms
def group : MergeGroup := .relation 300553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 300553) (rhsResult := 300551)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54329⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 300552 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54329⟩⟩]⟩) (none) 300551) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge300554

namespace LeftMerge300555
def owner : Owner := ⟨.program ⟨257⟩, ⟨54332⟩⟩
def mergeEvent : Nat := 300555
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55389⟩⟩] } }
def rhsRaw : List Term := Proof.Events1174.exact300551RawTerms
def group : MergeGroup := .relation 300553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 300553) (rhsResult := 300551)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54329⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 300552 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54329⟩⟩]⟩) (none) 300551) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55389⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55389⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge300555

namespace LeftMerge300556
def owner : Owner := ⟨.program ⟨257⟩, ⟨54332⟩⟩
def mergeEvent : Nat := 300556
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24650⟩⟩, ⟨.program ⟨257⟩, ⟨53255⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54929⟩⟩] } }
def rhsRaw : List Term := Proof.Events1174.exact300551RawTerms
def group : MergeGroup := .relation 300553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 300553) (rhsResult := 300551)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54329⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 300552 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54329⟩⟩]⟩) (none) 300551) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24650⟩⟩, ⟨.program ⟨257⟩, ⟨53255⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54929⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24650⟩⟩, ⟨.program ⟨257⟩, ⟨53255⟩⟩], [⟨.program ⟨257⟩, ⟨54929⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge300556

namespace LeftMerge300557
def owner : Owner := ⟨.program ⟨257⟩, ⟨54332⟩⟩
def mergeEvent : Nat := 300557
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1174.exact300551RawTerms
def group : MergeGroup := .relation 300553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 300553) (rhsResult := 300551)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54329⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 300552 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54329⟩⟩]⟩) (none) 300551) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge300557

namespace LeftMerge300562
def owner : Owner := ⟨.program ⟨257⟩, ⟨55391⟩⟩
def mergeEvent : Nat := 300562
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24650⟩⟩, ⟨.program ⟨257⟩, ⟨53255⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54929⟩⟩] } }
def leftRaw : List Term := Proof.Events1174.exact300558RawTerms
def rightRaw : List Term := Proof.Events1173.exact300396RawTerms
def group : MergeGroup := .operator 300558 300396
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300558) (leftOrdinal := 2)
    (rightResult := 300396) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24650⟩⟩, ⟨.program ⟨257⟩, ⟨53255⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54929⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24650⟩⟩, ⟨.program ⟨257⟩, ⟨53255⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54929⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24650⟩⟩, ⟨.program ⟨257⟩, ⟨53255⟩⟩], [⟨.program ⟨257⟩, ⟨54929⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge300562

namespace LeftMerge300563
def owner : Owner := ⟨.program ⟨257⟩, ⟨55391⟩⟩
def mergeEvent : Nat := 300563
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55389⟩⟩] } }
def leftRaw : List Term := Proof.Events1174.exact300558RawTerms
def rightRaw : List Term := Proof.Events1173.exact300396RawTerms
def group : MergeGroup := .operator 300558 300396
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300558) (leftOrdinal := 1)
    (rightResult := 300396) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55389⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55389⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55389⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge300563

namespace LeftMerge300571
def owner : Owner := ⟨.program ⟨257⟩, ⟨55624⟩⟩
def mergeEvent : Nat := 300571
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55622⟩⟩] } }
def leftRaw : List Term := Proof.Events1174.exact300565RawTerms
def rightRaw : List Term := Proof.Events1173.exact300312RawTerms
def group : MergeGroup := .operator 300565 300312
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300565) (leftOrdinal := 0)
    (rightResult := 300312) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55622⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55622⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge300571

namespace LeftMerge300572
def owner : Owner := ⟨.program ⟨257⟩, ⟨55624⟩⟩
def mergeEvent : Nat := 300572
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55622⟩⟩] } }
def leftRaw : List Term := Proof.Events1174.exact300565RawTerms
def rightRaw : List Term := Proof.Events1173.exact300312RawTerms
def group : MergeGroup := .operator 300565 300312
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300565) (leftOrdinal := 1)
    (rightResult := 300312) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55622⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55622⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge300572

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
