import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge197339
def owner : Owner := ⟨.program ⟨257⟩, ⟨63392⟩⟩
def mergeEvent : Nat := 197339
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63389⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events770.exact197333RawTerms
def group : MergeGroup := .operator 192995 197333
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 197333) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63389⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63389⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197339

namespace LeftMerge197418
def owner : Owner := ⟨.program ⟨257⟩, ⟨62520⟩⟩
def mergeEvent : Nat := 197418
def frameStart : Nat := 197388
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25514⟩⟩, ⟨.program ⟨257⟩, ⟨62519⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events771.exact197414RawTerms
def rightRaw : List Term := Proof.Events771.exact197411RawTerms
def group : MergeGroup := .operator 197414 197411
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197414) (leftOrdinal := 0)
    (rightResult := 197411) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62519⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25514⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25514⟩⟩, ⟨.program ⟨257⟩, ⟨62519⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197418

namespace LeftMerge197448
def owner : Owner := ⟨.program ⟨257⟩, ⟨64216⟩⟩
def mergeEvent : Nat := 197448
def frameStart : Nat := 197388
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25514⟩⟩, ⟨.program ⟨257⟩, ⟨62519⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events771.exact197444RawTerms
def rightRaw : List Term := Proof.Events771.exact197442RawTerms
def group : MergeGroup := .operator 197444 197442
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197444) (leftOrdinal := 0)
    (rightResult := 197442) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25514⟩⟩, ⟨.program ⟨257⟩, ⟨62519⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25514⟩⟩, ⟨.program ⟨257⟩, ⟨62519⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197448

namespace LeftMerge197471
def owner : Owner := ⟨.program ⟨257⟩, ⟨9540⟩⟩
def mergeEvent : Nat := 197471
def frameStart : Nat := 197388
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }
def leftRaw : List Term := Proof.Events771.exact197467RawTerms
def rightRaw : List Term := Proof.Events771.exact197464RawTerms
def group : MergeGroup := .operator 197467 197464
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197467) (leftOrdinal := 0)
    (rightResult := 197464) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9538⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197471

namespace LeftMerge197480
def owner : Owner := ⟨.program ⟨257⟩, ⟨64464⟩⟩
def mergeEvent : Nat := 197480
def frameStart : Nat := 197388
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64461⟩⟩] } }
def leftRaw : List Term := Proof.Events771.exact197476RawTerms
def rightRaw : List Term := Proof.Events771.exact197433RawTerms
def group : MergeGroup := .operator 197476 197433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197476) (leftOrdinal := 0)
    (rightResult := 197433) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64461⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64461⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197480

namespace LeftMerge197481
def owner : Owner := ⟨.program ⟨257⟩, ⟨64464⟩⟩
def mergeEvent : Nat := 197481
def frameStart : Nat := 197388
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25514⟩⟩, ⟨.program ⟨257⟩, ⟨62519⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64461⟩⟩] } }
def leftRaw : List Term := Proof.Events771.exact197476RawTerms
def rightRaw : List Term := Proof.Events771.exact197433RawTerms
def group : MergeGroup := .operator 197476 197433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197476) (leftOrdinal := 1)
    (rightResult := 197433) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25514⟩⟩, ⟨.program ⟨257⟩, ⟨62519⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64461⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25514⟩⟩, ⟨.program ⟨257⟩, ⟨62519⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64461⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge197481

namespace LeftMerge197483
def owner : Owner := ⟨.program ⟨257⟩, ⟨64464⟩⟩
def mergeEvent : Nat := 197483
def frameStart : Nat := 197388
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25514⟩⟩, ⟨.program ⟨257⟩, ⟨62519⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63941⟩⟩] } }
def rhsRaw : List Term := Proof.Events771.exact197430RawTerms
def group : MergeGroup := .relation 197482
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 197482) (rhsResult := 197430)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25514⟩⟩, ⟨.program ⟨257⟩, ⟨62519⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64461⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64461⟩⟩) ⟨63941⟩ 197430) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63941⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25514⟩⟩, ⟨.program ⟨257⟩, ⟨62519⟩⟩], [⟨.program ⟨257⟩, ⟨63941⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge197483

namespace LeftMerge197491
def owner : Owner := ⟨.program ⟨257⟩, ⟨62826⟩⟩
def mergeEvent : Nat := 197491
def frameStart : Nat := 197388
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events771.exact197444RawTerms
def rightRaw : List Term := Proof.Events771.exact197487RawTerms
def group : MergeGroup := .operator 197444 197487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197444) (leftOrdinal := 0)
    (rightResult := 197487) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197491

namespace LeftMerge197508
def owner : Owner := ⟨.program ⟨257⟩, ⟨63392⟩⟩
def mergeEvent : Nat := 197508
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }
def rhsRaw : List Term := Proof.Events771.exact197505RawTerms
def group : MergeGroup := .relation 197507
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 197507) (rhsResult := 197505)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 197506 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63389⟩⟩]⟩) (none) 197505) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197508

namespace LeftMerge197509
def owner : Owner := ⟨.program ⟨257⟩, ⟨63392⟩⟩
def mergeEvent : Nat := 197509
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64461⟩⟩] } }
def rhsRaw : List Term := Proof.Events771.exact197505RawTerms
def group : MergeGroup := .relation 197507
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 197507) (rhsResult := 197505)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 197506 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63389⟩⟩]⟩) (none) 197505) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64461⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64461⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge197509

namespace LeftMerge197510
def owner : Owner := ⟨.program ⟨257⟩, ⟨63392⟩⟩
def mergeEvent : Nat := 197510
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25514⟩⟩, ⟨.program ⟨257⟩, ⟨62519⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63941⟩⟩] } }
def rhsRaw : List Term := Proof.Events771.exact197505RawTerms
def group : MergeGroup := .relation 197507
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 197507) (rhsResult := 197505)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 197506 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63389⟩⟩]⟩) (none) 197505) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25514⟩⟩, ⟨.program ⟨257⟩, ⟨62519⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63941⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25514⟩⟩, ⟨.program ⟨257⟩, ⟨62519⟩⟩], [⟨.program ⟨257⟩, ⟨63941⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197510

namespace LeftMerge197511
def owner : Owner := ⟨.program ⟨257⟩, ⟨63392⟩⟩
def mergeEvent : Nat := 197511
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events771.exact197505RawTerms
def group : MergeGroup := .relation 197507
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 197507) (rhsResult := 197505)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 197506 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63389⟩⟩]⟩) (none) 197505) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge197511

namespace LeftMerge197516
def owner : Owner := ⟨.program ⟨257⟩, ⟨64463⟩⟩
def mergeEvent : Nat := 197516
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25514⟩⟩, ⟨.program ⟨257⟩, ⟨62519⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63941⟩⟩] } }
def leftRaw : List Term := Proof.Events771.exact197512RawTerms
def rightRaw : List Term := Proof.Events770.exact197326RawTerms
def group : MergeGroup := .operator 197512 197326
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197512) (leftOrdinal := 2)
    (rightResult := 197326) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25514⟩⟩, ⟨.program ⟨257⟩, ⟨62519⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63941⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25514⟩⟩, ⟨.program ⟨257⟩, ⟨62519⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63941⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25514⟩⟩, ⟨.program ⟨257⟩, ⟨62519⟩⟩], [⟨.program ⟨257⟩, ⟨63941⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge197516

namespace LeftMerge197517
def owner : Owner := ⟨.program ⟨257⟩, ⟨64463⟩⟩
def mergeEvent : Nat := 197517
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64461⟩⟩] } }
def leftRaw : List Term := Proof.Events771.exact197512RawTerms
def rightRaw : List Term := Proof.Events770.exact197326RawTerms
def group : MergeGroup := .operator 197512 197326
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197512) (leftOrdinal := 1)
    (rightResult := 197326) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64461⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64461⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64461⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197517

namespace LeftMerge197525
def owner : Owner := ⟨.program ⟨257⟩, ⟨64936⟩⟩
def mergeEvent : Nat := 197525
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64934⟩⟩] } }
def leftRaw : List Term := Proof.Events771.exact197519RawTerms
def rightRaw : List Term := Proof.Events770.exact197242RawTerms
def group : MergeGroup := .operator 197519 197242
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197519) (leftOrdinal := 0)
    (rightResult := 197242) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197525

namespace LeftMerge197526
def owner : Owner := ⟨.program ⟨257⟩, ⟨64936⟩⟩
def mergeEvent : Nat := 197526
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64934⟩⟩] } }
def leftRaw : List Term := Proof.Events771.exact197519RawTerms
def rightRaw : List Term := Proof.Events770.exact197242RawTerms
def group : MergeGroup := .operator 197519 197242
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197519) (leftOrdinal := 1)
    (rightResult := 197242) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge197526

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
