import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge82267
def owner : Owner := ⟨.program ⟨257⟩, ⟨51512⟩⟩
def mergeEvent : Nat := 82267
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51509⟩⟩] } }
def leftRaw : List Term := Proof.Events296.exact75995RawTerms
def rightRaw : List Term := Proof.Events321.exact82261RawTerms
def group : MergeGroup := .operator 75995 82261
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75995) (leftOrdinal := 0)
    (rightResult := 82261) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51509⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51509⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82267

namespace LeftMerge82346
def owner : Owner := ⟨.program ⟨257⟩, ⟨50708⟩⟩
def mergeEvent : Nat := 82346
def frameStart : Nat := 82316
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events321.exact82342RawTerms
def rightRaw : List Term := Proof.Events321.exact82339RawTerms
def group : MergeGroup := .operator 82342 82339
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 82342) (leftOrdinal := 0)
    (rightResult := 82339) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50707⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24602⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82346

namespace LeftMerge82376
def owner : Owner := ⟨.program ⟨257⟩, ⟨52312⟩⟩
def mergeEvent : Nat := 82376
def frameStart : Nat := 82316
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events321.exact82372RawTerms
def rightRaw : List Term := Proof.Events321.exact82370RawTerms
def group : MergeGroup := .operator 82372 82370
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 82372) (leftOrdinal := 0)
    (rightResult := 82370) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82376

namespace LeftMerge82399
def owner : Owner := ⟨.program ⟨257⟩, ⟨9582⟩⟩
def mergeEvent : Nat := 82399
def frameStart : Nat := 82316
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }
def leftRaw : List Term := Proof.Events321.exact82395RawTerms
def rightRaw : List Term := Proof.Events321.exact82392RawTerms
def group : MergeGroup := .operator 82395 82392
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 82395) (leftOrdinal := 0)
    (rightResult := 82392) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9580⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82399

namespace LeftMerge82408
def owner : Owner := ⟨.program ⟨257⟩, ⟨52588⟩⟩
def mergeEvent : Nat := 82408
def frameStart : Nat := 82316
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52585⟩⟩] } }
def leftRaw : List Term := Proof.Events321.exact82404RawTerms
def rightRaw : List Term := Proof.Events321.exact82361RawTerms
def group : MergeGroup := .operator 82404 82361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 82404) (leftOrdinal := 0)
    (rightResult := 82361) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52585⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52585⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82408

namespace LeftMerge82409
def owner : Owner := ⟨.program ⟨257⟩, ⟨52588⟩⟩
def mergeEvent : Nat := 82409
def frameStart : Nat := 82316
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52585⟩⟩] } }
def leftRaw : List Term := Proof.Events321.exact82404RawTerms
def rightRaw : List Term := Proof.Events321.exact82361RawTerms
def group : MergeGroup := .operator 82404 82361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 82404) (leftOrdinal := 1)
    (rightResult := 82361) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52585⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52585⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge82409

namespace LeftMerge82411
def owner : Owner := ⟨.program ⟨257⟩, ⟨52588⟩⟩
def mergeEvent : Nat := 82411
def frameStart : Nat := 82316
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52045⟩⟩] } }
def rhsRaw : List Term := Proof.Events321.exact82358RawTerms
def group : MergeGroup := .relation 82410
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 82410) (rhsResult := 82358)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52585⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52585⟩⟩) ⟨52045⟩ 82358) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52045⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], [⟨.program ⟨257⟩, ⟨52045⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge82411

namespace LeftMerge82419
def owner : Owner := ⟨.program ⟨257⟩, ⟨50938⟩⟩
def mergeEvent : Nat := 82419
def frameStart : Nat := 82316
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50936⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events321.exact82372RawTerms
def rightRaw : List Term := Proof.Events321.exact82415RawTerms
def group : MergeGroup := .operator 82372 82415
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 82372) (leftOrdinal := 0)
    (rightResult := 82415) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50936⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82419

namespace LeftMerge82436
def owner : Owner := ⟨.program ⟨257⟩, ⟨51512⟩⟩
def mergeEvent : Nat := 82436
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }
def rhsRaw : List Term := Proof.Events322.exact82433RawTerms
def group : MergeGroup := .relation 82435
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 82435) (rhsResult := 82433)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51509⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 82434 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51509⟩⟩]⟩) (none) 82433) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82436

namespace LeftMerge82437
def owner : Owner := ⟨.program ⟨257⟩, ⟨51512⟩⟩
def mergeEvent : Nat := 82437
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52585⟩⟩] } }
def rhsRaw : List Term := Proof.Events322.exact82433RawTerms
def group : MergeGroup := .relation 82435
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 82435) (rhsResult := 82433)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51509⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 82434 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51509⟩⟩]⟩) (none) 82433) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52585⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52585⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge82437

namespace LeftMerge82438
def owner : Owner := ⟨.program ⟨257⟩, ⟨51512⟩⟩
def mergeEvent : Nat := 82438
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52045⟩⟩] } }
def rhsRaw : List Term := Proof.Events322.exact82433RawTerms
def group : MergeGroup := .relation 82435
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 82435) (rhsResult := 82433)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51509⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 82434 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51509⟩⟩]⟩) (none) 82433) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52045⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], [⟨.program ⟨257⟩, ⟨52045⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82438

namespace LeftMerge82439
def owner : Owner := ⟨.program ⟨257⟩, ⟨51512⟩⟩
def mergeEvent : Nat := 82439
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨50936⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events322.exact82433RawTerms
def group : MergeGroup := .relation 82435
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 82435) (rhsResult := 82433)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51509⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 82434 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51509⟩⟩]⟩) (none) 82433) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50936⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge82439

namespace LeftMerge82444
def owner : Owner := ⟨.program ⟨257⟩, ⟨52587⟩⟩
def mergeEvent : Nat := 82444
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52045⟩⟩] } }
def leftRaw : List Term := Proof.Events322.exact82440RawTerms
def rightRaw : List Term := Proof.Events321.exact82254RawTerms
def group : MergeGroup := .operator 82440 82254
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 82440) (leftOrdinal := 2)
    (rightResult := 82254) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52045⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52045⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], [⟨.program ⟨257⟩, ⟨52045⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge82444

namespace LeftMerge82445
def owner : Owner := ⟨.program ⟨257⟩, ⟨52587⟩⟩
def mergeEvent : Nat := 82445
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52585⟩⟩] } }
def leftRaw : List Term := Proof.Events322.exact82440RawTerms
def rightRaw : List Term := Proof.Events321.exact82254RawTerms
def group : MergeGroup := .operator 82440 82254
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 82440) (leftOrdinal := 1)
    (rightResult := 82254) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52585⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52585⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52585⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82445

namespace LeftMerge82453
def owner : Owner := ⟨.program ⟨257⟩, ⟨53140⟩⟩
def mergeEvent : Nat := 82453
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53138⟩⟩] } }
def leftRaw : List Term := Proof.Events322.exact82447RawTerms
def rightRaw : List Term := Proof.Events320.exact82170RawTerms
def group : MergeGroup := .operator 82447 82170
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 82447) (leftOrdinal := 0)
    (rightResult := 82170) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53138⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53138⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82453

namespace LeftMerge82454
def owner : Owner := ⟨.program ⟨257⟩, ⟨53140⟩⟩
def mergeEvent : Nat := 82454
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨50936⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53138⟩⟩] } }
def leftRaw : List Term := Proof.Events322.exact82447RawTerms
def rightRaw : List Term := Proof.Events320.exact82170RawTerms
def group : MergeGroup := .operator 82447 82170
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 82447) (leftOrdinal := 1)
    (rightResult := 82170) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨50936⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53138⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53138⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge82454

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
