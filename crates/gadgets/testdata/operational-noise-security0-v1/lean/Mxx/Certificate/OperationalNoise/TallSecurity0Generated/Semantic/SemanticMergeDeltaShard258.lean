import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge43348
def owner : Owner := ⟨.program ⟨214⟩, ⟨11000⟩⟩
def mergeEvent : Nat := 43348
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10852⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } }
def leftRaw : List Term := Proof.Events169.exact43344RawTerms
def rightRaw : List Term := Proof.Events169.exact43314RawTerms
def group : MergeGroup := .operator 43344 43314
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43344) (leftOrdinal := 1)
    (rightResult := 43314) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10852⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10852⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10852⟩⟩], [⟨.program ⟨214⟩, ⟨6774⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43348

namespace LeftMerge43356
def owner : Owner := ⟨.program ⟨214⟩, ⟨25076⟩⟩
def mergeEvent : Nat := 43356
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25075⟩⟩] } }
def leftRaw : List Term := Proof.Events169.exact43350RawTerms
def rightRaw : List Term := Proof.Events169.exact43286RawTerms
def group : MergeGroup := .operator 43350 43286
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43350) (leftOrdinal := 1)
    (rightResult := 43286) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25075⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25075⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43356

namespace LeftMerge43358
def owner : Owner := ⟨.program ⟨214⟩, ⟨25076⟩⟩
def mergeEvent : Nat := 43358
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23042⟩⟩] } }
def rhsRaw : List Term := Proof.Events169.exact43283RawTerms
def group : MergeGroup := .relation 43357
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43357) (rhsResult := 43283)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25075⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25075⟩⟩) ⟨23042⟩ 43283) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23042⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], [⟨.program ⟨214⟩, ⟨23042⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43358

namespace LeftMerge43359
def owner : Owner := ⟨.program ⟨214⟩, ⟨25076⟩⟩
def mergeEvent : Nat := 43359
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25075⟩⟩] } }
def leftRaw : List Term := Proof.Events169.exact43350RawTerms
def rightRaw : List Term := Proof.Events169.exact43286RawTerms
def group : MergeGroup := .operator 43350 43286
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43350) (leftOrdinal := 0)
    (rightResult := 43286) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25075⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25075⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43359

namespace LeftMerge43373
def owner : Owner := ⟨.program ⟨214⟩, ⟨19179⟩⟩
def mergeEvent : Nat := 43373
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19176⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events169.exact43367RawTerms
def group : MergeGroup := .operator 36137 43367
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 43367) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19176⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19176⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43373

namespace LeftMerge43452
def owner : Owner := ⟨.program ⟨214⟩, ⟨10994⟩⟩
def mergeEvent : Nat := 43452
def frameStart : Nat := 43422
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events169.exact43448RawTerms
def rightRaw : List Term := Proof.Events169.exact43445RawTerms
def group : MergeGroup := .operator 43448 43445
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43448) (leftOrdinal := 0)
    (rightResult := 43445) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10852⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10993⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43452

namespace LeftMerge43482
def owner : Owner := ⟨.program ⟨214⟩, ⟨11083⟩⟩
def mergeEvent : Nat := 43482
def frameStart : Nat := 43422
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events169.exact43478RawTerms
def rightRaw : List Term := Proof.Events169.exact43476RawTerms
def group : MergeGroup := .operator 43478 43476
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43478) (leftOrdinal := 0)
    (rightResult := 43476) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43482

namespace LeftMerge43505
def owner : Owner := ⟨.program ⟨214⟩, ⟨7839⟩⟩
def mergeEvent : Nat := 43505
def frameStart : Nat := 43422
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩] } }
def leftRaw : List Term := Proof.Events169.exact43501RawTerms
def rightRaw : List Term := Proof.Events169.exact43498RawTerms
def group : MergeGroup := .operator 43501 43498
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43501) (leftOrdinal := 0)
    (rightResult := 43498) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7837⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43505

namespace LeftMerge43514
def owner : Owner := ⟨.program ⟨214⟩, ⟨25078⟩⟩
def mergeEvent : Nat := 43514
def frameStart : Nat := 43422
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25075⟩⟩] } }
def leftRaw : List Term := Proof.Events169.exact43510RawTerms
def rightRaw : List Term := Proof.Events169.exact43467RawTerms
def group : MergeGroup := .operator 43510 43467
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43510) (leftOrdinal := 0)
    (rightResult := 43467) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25075⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25075⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43514

namespace LeftMerge43515
def owner : Owner := ⟨.program ⟨214⟩, ⟨25078⟩⟩
def mergeEvent : Nat := 43515
def frameStart : Nat := 43422
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25075⟩⟩] } }
def leftRaw : List Term := Proof.Events169.exact43510RawTerms
def rightRaw : List Term := Proof.Events169.exact43467RawTerms
def group : MergeGroup := .operator 43510 43467
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43510) (leftOrdinal := 1)
    (rightResult := 43467) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25075⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25075⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43515

namespace LeftMerge43517
def owner : Owner := ⟨.program ⟨214⟩, ⟨25078⟩⟩
def mergeEvent : Nat := 43517
def frameStart : Nat := 43422
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23042⟩⟩] } }
def rhsRaw : List Term := Proof.Events169.exact43464RawTerms
def group : MergeGroup := .relation 43516
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43516) (rhsResult := 43464)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25075⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25075⟩⟩) ⟨23042⟩ 43464) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23042⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], [⟨.program ⟨214⟩, ⟨23042⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43517

namespace LeftMerge43525
def owner : Owner := ⟨.program ⟨214⟩, ⟨15124⟩⟩
def mergeEvent : Nat := 43525
def frameStart : Nat := 43422
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15122⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events169.exact43478RawTerms
def rightRaw : List Term := Proof.Events170.exact43521RawTerms
def group : MergeGroup := .operator 43478 43521
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43478) (leftOrdinal := 0)
    (rightResult := 43521) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15122⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15122⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43525

namespace LeftMerge43542
def owner : Owner := ⟨.program ⟨214⟩, ⟨19179⟩⟩
def mergeEvent : Nat := 43542
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩] } }
def rhsRaw : List Term := Proof.Events170.exact43539RawTerms
def group : MergeGroup := .relation 43541
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43541) (rhsResult := 43539)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19176⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 43540 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19176⟩⟩]⟩) (none) 43539) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43542

namespace LeftMerge43543
def owner : Owner := ⟨.program ⟨214⟩, ⟨19179⟩⟩
def mergeEvent : Nat := 43543
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25075⟩⟩] } }
def rhsRaw : List Term := Proof.Events170.exact43539RawTerms
def group : MergeGroup := .relation 43541
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43541) (rhsResult := 43539)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19176⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 43540 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19176⟩⟩]⟩) (none) 43539) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25075⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25075⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43543

namespace LeftMerge43544
def owner : Owner := ⟨.program ⟨214⟩, ⟨19179⟩⟩
def mergeEvent : Nat := 43544
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23042⟩⟩] } }
def rhsRaw : List Term := Proof.Events170.exact43539RawTerms
def group : MergeGroup := .relation 43541
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43541) (rhsResult := 43539)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19176⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 43540 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19176⟩⟩]⟩) (none) 43539) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23042⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], [⟨.program ⟨214⟩, ⟨23042⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43544

namespace LeftMerge43545
def owner : Owner := ⟨.program ⟨214⟩, ⟨19179⟩⟩
def mergeEvent : Nat := 43545
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15122⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events170.exact43539RawTerms
def group : MergeGroup := .relation 43541
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43541) (rhsResult := 43539)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19176⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 43540 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19176⟩⟩]⟩) (none) 43539) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15122⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15122⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43545

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
