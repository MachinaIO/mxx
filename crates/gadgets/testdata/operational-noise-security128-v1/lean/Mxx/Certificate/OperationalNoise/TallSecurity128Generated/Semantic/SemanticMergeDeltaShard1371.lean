import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge222251
def owner : Owner := ⟨.program ⟨257⟩, ⟨48582⟩⟩
def mergeEvent : Nat := 222251
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48579⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events868.exact222234RawTerms
def group : MergeGroup := .operator 222245 222234
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 222234) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨48579⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48579⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222251

namespace LeftMerge222330
def owner : Owner := ⟨.program ⟨257⟩, ⟨47811⟩⟩
def mergeEvent : Nat := 222330
def frameStart : Nat := 222300
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15066⟩⟩, ⟨.program ⟨257⟩, ⟨47810⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events868.exact222326RawTerms
def rightRaw : List Term := Proof.Events868.exact222323RawTerms
def group : MergeGroup := .operator 222326 222323
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222326) (leftOrdinal := 0)
    (rightResult := 222323) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15066⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨47810⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15066⟩⟩, ⟨.program ⟨257⟩, ⟨47810⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222330

namespace LeftMerge222360
def owner : Owner := ⟨.program ⟨257⟩, ⟨49424⟩⟩
def mergeEvent : Nat := 222360
def frameStart : Nat := 222300
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15066⟩⟩, ⟨.program ⟨257⟩, ⟨47810⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222356RawTerms
def rightRaw : List Term := Proof.Events868.exact222354RawTerms
def group : MergeGroup := .operator 222356 222354
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222356) (leftOrdinal := 0)
    (rightResult := 222354) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15066⟩⟩, ⟨.program ⟨257⟩, ⟨47810⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15066⟩⟩, ⟨.program ⟨257⟩, ⟨47810⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222360

namespace LeftMerge222383
def owner : Owner := ⟨.program ⟨257⟩, ⟨9567⟩⟩
def mergeEvent : Nat := 222383
def frameStart : Nat := 222300
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222379RawTerms
def rightRaw : List Term := Proof.Events868.exact222376RawTerms
def group : MergeGroup := .operator 222379 222376
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222379) (leftOrdinal := 0)
    (rightResult := 222376) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9565⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222383

namespace LeftMerge222392
def owner : Owner := ⟨.program ⟨257⟩, ⟨49651⟩⟩
def mergeEvent : Nat := 222392
def frameStart : Nat := 222300
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49648⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222388RawTerms
def rightRaw : List Term := Proof.Events868.exact222345RawTerms
def group : MergeGroup := .operator 222388 222345
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222388) (leftOrdinal := 0)
    (rightResult := 222345) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49648⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49648⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222392

namespace LeftMerge222393
def owner : Owner := ⟨.program ⟨257⟩, ⟨49651⟩⟩
def mergeEvent : Nat := 222393
def frameStart : Nat := 222300
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15066⟩⟩, ⟨.program ⟨257⟩, ⟨47810⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49648⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222388RawTerms
def rightRaw : List Term := Proof.Events868.exact222345RawTerms
def group : MergeGroup := .operator 222388 222345
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222388) (leftOrdinal := 1)
    (rightResult := 222345) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15066⟩⟩, ⟨.program ⟨257⟩, ⟨47810⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49648⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15066⟩⟩, ⟨.program ⟨257⟩, ⟨47810⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49648⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge222393

namespace LeftMerge222395
def owner : Owner := ⟨.program ⟨257⟩, ⟨49651⟩⟩
def mergeEvent : Nat := 222395
def frameStart : Nat := 222300
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15066⟩⟩, ⟨.program ⟨257⟩, ⟨47810⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49143⟩⟩] } }
def rhsRaw : List Term := Proof.Events868.exact222342RawTerms
def group : MergeGroup := .relation 222394
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 222394) (rhsResult := 222342)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15066⟩⟩, ⟨.program ⟨257⟩, ⟨47810⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49648⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49648⟩⟩) ⟨49143⟩ 222342) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49143⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15066⟩⟩, ⟨.program ⟨257⟩, ⟨47810⟩⟩], [⟨.program ⟨257⟩, ⟨49143⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge222395

namespace LeftMerge222403
def owner : Owner := ⟨.program ⟨257⟩, ⟨48142⟩⟩
def mergeEvent : Nat := 222403
def frameStart : Nat := 222300
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48140⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222356RawTerms
def rightRaw : List Term := Proof.Events868.exact222399RawTerms
def group : MergeGroup := .operator 222356 222399
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222356) (leftOrdinal := 0)
    (rightResult := 222399) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48140⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222403

namespace LeftMerge222420
def owner : Owner := ⟨.program ⟨257⟩, ⟨48582⟩⟩
def mergeEvent : Nat := 222420
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }
def rhsRaw : List Term := Proof.Events868.exact222417RawTerms
def group : MergeGroup := .relation 222419
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 222419) (rhsResult := 222417)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48579⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 222418 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48579⟩⟩]⟩) (none) 222417) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222420

namespace LeftMerge222421
def owner : Owner := ⟨.program ⟨257⟩, ⟨48582⟩⟩
def mergeEvent : Nat := 222421
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49648⟩⟩] } }
def rhsRaw : List Term := Proof.Events868.exact222417RawTerms
def group : MergeGroup := .relation 222419
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 222419) (rhsResult := 222417)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48579⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 222418 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48579⟩⟩]⟩) (none) 222417) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49648⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49648⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge222421

namespace LeftMerge222422
def owner : Owner := ⟨.program ⟨257⟩, ⟨48582⟩⟩
def mergeEvent : Nat := 222422
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15066⟩⟩, ⟨.program ⟨257⟩, ⟨47810⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49143⟩⟩] } }
def rhsRaw : List Term := Proof.Events868.exact222417RawTerms
def group : MergeGroup := .relation 222419
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 222419) (rhsResult := 222417)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48579⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 222418 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48579⟩⟩]⟩) (none) 222417) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15066⟩⟩, ⟨.program ⟨257⟩, ⟨47810⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49143⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15066⟩⟩, ⟨.program ⟨257⟩, ⟨47810⟩⟩], [⟨.program ⟨257⟩, ⟨49143⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222422

namespace LeftMerge222423
def owner : Owner := ⟨.program ⟨257⟩, ⟨48582⟩⟩
def mergeEvent : Nat := 222423
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨48140⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events868.exact222417RawTerms
def group : MergeGroup := .relation 222419
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 222419) (rhsResult := 222417)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48579⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 222418 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48579⟩⟩]⟩) (none) 222417) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48140⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨48140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge222423

namespace LeftMerge222428
def owner : Owner := ⟨.program ⟨257⟩, ⟨49650⟩⟩
def mergeEvent : Nat := 222428
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15066⟩⟩, ⟨.program ⟨257⟩, ⟨47810⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49143⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222424RawTerms
def rightRaw : List Term := Proof.Events868.exact222227RawTerms
def group : MergeGroup := .operator 222424 222227
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222424) (leftOrdinal := 2)
    (rightResult := 222227) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15066⟩⟩, ⟨.program ⟨257⟩, ⟨47810⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49143⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15066⟩⟩, ⟨.program ⟨257⟩, ⟨47810⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49143⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15066⟩⟩, ⟨.program ⟨257⟩, ⟨47810⟩⟩], [⟨.program ⟨257⟩, ⟨49143⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge222428

namespace LeftMerge222429
def owner : Owner := ⟨.program ⟨257⟩, ⟨49650⟩⟩
def mergeEvent : Nat := 222429
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49648⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222424RawTerms
def rightRaw : List Term := Proof.Events868.exact222227RawTerms
def group : MergeGroup := .operator 222424 222227
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222424) (leftOrdinal := 1)
    (rightResult := 222227) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49648⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49648⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49648⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222429

namespace LeftMerge222437
def owner : Owner := ⟨.program ⟨257⟩, ⟨50006⟩⟩
def mergeEvent : Nat := 222437
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50004⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222431RawTerms
def rightRaw : List Term := Proof.Events867.exact222138RawTerms
def group : MergeGroup := .operator 222431 222138
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222431) (leftOrdinal := 0)
    (rightResult := 222138) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50004⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50004⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222437

namespace LeftMerge222438
def owner : Owner := ⟨.program ⟨257⟩, ⟨50006⟩⟩
def mergeEvent : Nat := 222438
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨48140⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50004⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222431RawTerms
def rightRaw : List Term := Proof.Events867.exact222138RawTerms
def group : MergeGroup := .operator 222431 222138
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222431) (leftOrdinal := 1)
    (rightResult := 222138) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨48140⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50004⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨48140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50004⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge222438

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
