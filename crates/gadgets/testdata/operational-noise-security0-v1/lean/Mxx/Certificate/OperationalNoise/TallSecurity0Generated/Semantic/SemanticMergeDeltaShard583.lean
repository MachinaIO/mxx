import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge94407
def owner : Owner := ⟨.program ⟨214⟩, ⟨7107⟩⟩
def mergeEvent : Nat := 94407
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact27RawTerms
def rightRaw : List Term := Proof.Events025.exact6498RawTerms
def group : MergeGroup := .operator 27 6498
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27) (leftOrdinal := 0)
    (rightResult := 6498) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94407

namespace LeftMerge94424
def owner : Owner := ⟨.program ⟨214⟩, ⟨10334⟩⟩
def mergeEvent : Nat := 94424
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10330⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94418RawTerms
def rightRaw : List Term := Proof.Events025.exact6487RawTerms
def group : MergeGroup := .operator 94418 6487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94418) (leftOrdinal := 1)
    (rightResult := 6487) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10330⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7882⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10330⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94424

namespace LeftMerge94426
def owner : Owner := ⟨.program ⟨214⟩, ⟨10334⟩⟩
def mergeEvent : Nat := 94426
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10330⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6790⟩⟩] } }
def rhsRaw : List Term := Proof.Events025.exact6457RawTerms
def group : MergeGroup := .relation 94425
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 94425) (rhsResult := 6457)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10330⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7882⟩⟩) ⟨6790⟩ 6457) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6790⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10330⟩⟩], [⟨.program ⟨214⟩, ⟨6790⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94426

namespace LeftMerge94427
def owner : Owner := ⟨.program ⟨214⟩, ⟨10334⟩⟩
def mergeEvent : Nat := 94427
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94418RawTerms
def rightRaw : List Term := Proof.Events025.exact6487RawTerms
def group : MergeGroup := .operator 94418 6487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94418) (leftOrdinal := 0)
    (rightResult := 6487) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7882⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94427

namespace LeftMerge94432
def owner : Owner := ⟨.program ⟨214⟩, ⟨13333⟩⟩
def mergeEvent : Nat := 94432
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10330⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6790⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94428RawTerms
def rightRaw : List Term := Proof.Events368.exact94398RawTerms
def group : MergeGroup := .operator 94428 94398
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94428) (leftOrdinal := 1)
    (rightResult := 94398) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10330⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6790⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10330⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6790⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10330⟩⟩], [⟨.program ⟨214⟩, ⟨6790⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94432

namespace LeftMerge94440
def owner : Owner := ⟨.program ⟨214⟩, ⟨25746⟩⟩
def mergeEvent : Nat := 94440
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25745⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94434RawTerms
def rightRaw : List Term := Proof.Events368.exact94370RawTerms
def group : MergeGroup := .operator 94434 94370
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94434) (leftOrdinal := 1)
    (rightResult := 94370) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25745⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25745⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94440

namespace LeftMerge94442
def owner : Owner := ⟨.program ⟨214⟩, ⟨25746⟩⟩
def mergeEvent : Nat := 94442
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23410⟩⟩] } }
def rhsRaw : List Term := Proof.Events368.exact94367RawTerms
def group : MergeGroup := .relation 94441
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 94441) (rhsResult := 94367)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25745⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25745⟩⟩) ⟨23410⟩ 94367) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23410⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], [⟨.program ⟨214⟩, ⟨23410⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94442

namespace LeftMerge94443
def owner : Owner := ⟨.program ⟨214⟩, ⟨25746⟩⟩
def mergeEvent : Nat := 94443
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25745⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94434RawTerms
def rightRaw : List Term := Proof.Events368.exact94370RawTerms
def group : MergeGroup := .operator 94434 94370
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94434) (leftOrdinal := 0)
    (rightResult := 94370) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25745⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25745⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94443

namespace LeftMerge94455
def owner : Owner := ⟨.program ⟨214⟩, ⟨5508⟩⟩
def mergeEvent : Nat := 94455
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact27RawTerms
def rightRaw : List Term := Proof.Events025.exact6550RawTerms
def group : MergeGroup := .operator 27 6550
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27) (leftOrdinal := 0)
    (rightResult := 6550) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94455

namespace LeftMerge94468
def owner : Owner := ⟨.program ⟨214⟩, ⟨20240⟩⟩
def mergeEvent : Nat := 94468
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20237⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94462RawTerms
def rightRaw : List Term := Proof.Events368.exact94451RawTerms
def group : MergeGroup := .operator 94462 94451
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94462) (leftOrdinal := 0)
    (rightResult := 94451) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20237⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20237⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94468

namespace LeftMerge94523
def owner : Owner := ⟨.program ⟨214⟩, ⟨13327⟩⟩
def mergeEvent : Nat := 94523
def frameStart : Nat := 94505
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events369.exact94519RawTerms
def rightRaw : List Term := Proof.Events369.exact94516RawTerms
def group : MergeGroup := .operator 94519 94516
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94519) (leftOrdinal := 0)
    (rightResult := 94516) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10330⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13326⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94523

namespace LeftMerge94553
def owner : Owner := ⟨.program ⟨214⟩, ⟨13440⟩⟩
def mergeEvent : Nat := 94553
def frameStart : Nat := 94505
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events369.exact94549RawTerms
def rightRaw : List Term := Proof.Events369.exact94547RawTerms
def group : MergeGroup := .operator 94549 94547
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94549) (leftOrdinal := 0)
    (rightResult := 94547) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94553

namespace LeftMerge94576
def owner : Owner := ⟨.program ⟨214⟩, ⟨7884⟩⟩
def mergeEvent : Nat := 94576
def frameStart : Nat := 94505
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }
def leftRaw : List Term := Proof.Events369.exact94572RawTerms
def rightRaw : List Term := Proof.Events369.exact94569RawTerms
def group : MergeGroup := .operator 94572 94569
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94572) (leftOrdinal := 0)
    (rightResult := 94569) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7882⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94576

namespace LeftMerge94585
def owner : Owner := ⟨.program ⟨214⟩, ⟨25748⟩⟩
def mergeEvent : Nat := 94585
def frameStart : Nat := 94505
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25745⟩⟩] } }
def leftRaw : List Term := Proof.Events369.exact94581RawTerms
def rightRaw : List Term := Proof.Events369.exact94538RawTerms
def group : MergeGroup := .operator 94581 94538
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94581) (leftOrdinal := 0)
    (rightResult := 94538) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25745⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25745⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94585

namespace LeftMerge94586
def owner : Owner := ⟨.program ⟨214⟩, ⟨25748⟩⟩
def mergeEvent : Nat := 94586
def frameStart : Nat := 94505
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25745⟩⟩] } }
def leftRaw : List Term := Proof.Events369.exact94581RawTerms
def rightRaw : List Term := Proof.Events369.exact94538RawTerms
def group : MergeGroup := .operator 94581 94538
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94581) (leftOrdinal := 1)
    (rightResult := 94538) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25745⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25745⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94586

namespace LeftMerge94588
def owner : Owner := ⟨.program ⟨214⟩, ⟨25748⟩⟩
def mergeEvent : Nat := 94588
def frameStart : Nat := 94505
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23410⟩⟩] } }
def rhsRaw : List Term := Proof.Events369.exact94535RawTerms
def group : MergeGroup := .relation 94587
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 94587) (rhsResult := 94535)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25745⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25745⟩⟩) ⟨23410⟩ 94535) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23410⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], [⟨.program ⟨214⟩, ⟨23410⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94588

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
