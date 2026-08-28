import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge96367
def owner : Owner := ⟨.program ⟨214⟩, ⟨29135⟩⟩
def mergeEvent : Nat := 96367
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩] } }
def leftRaw : List Term := Proof.Events376.exact96360RawTerms
def rightRaw : List Term := Proof.Events375.exact96107RawTerms
def group : MergeGroup := .operator 96360 96107
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96360) (leftOrdinal := 1)
    (rightResult := 96107) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29133⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96367

namespace LeftMerge96369
def owner : Owner := ⟨.program ⟨214⟩, ⟨29135⟩⟩
def mergeEvent : Nat := 96369
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24531⟩⟩] } }
def rhsRaw : List Term := Proof.Events375.exact96104RawTerms
def group : MergeGroup := .relation 96368
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96368) (rhsResult := 96104)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29133⟩⟩) ⟨24531⟩ 96104) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24531⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨24531⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96369

namespace LeftMerge96383
def owner : Owner := ⟨.program ⟨214⟩, ⟨22256⟩⟩
def mergeEvent : Nat := 96383
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22253⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94462RawTerms
def rightRaw : List Term := Proof.Events376.exact96377RawTerms
def group : MergeGroup := .operator 94462 96377
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94462) (leftOrdinal := 0)
    (rightResult := 96377) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22253⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22253⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96383

namespace LeftMerge96480
def owner : Owner := ⟨.program ⟨214⟩, ⟨16583⟩⟩
def mergeEvent : Nat := 96480
def frameStart : Nat := 96426
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events376.exact96476RawTerms
def rightRaw : List Term := Proof.Events376.exact96474RawTerms
def group : MergeGroup := .operator 96476 96474
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96476) (leftOrdinal := 0)
    (rightResult := 96474) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96480

namespace LeftMerge96492
def owner : Owner := ⟨.program ⟨214⟩, ⟨29134⟩⟩
def mergeEvent : Nat := 96492
def frameStart : Nat := 96426
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩] } }
def leftRaw : List Term := Proof.Events376.exact96488RawTerms
def rightRaw : List Term := Proof.Events376.exact96465RawTerms
def group : MergeGroup := .operator 96488 96465
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96488) (leftOrdinal := 0)
    (rightResult := 96465) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29133⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96492

namespace LeftMerge96493
def owner : Owner := ⟨.program ⟨214⟩, ⟨29134⟩⟩
def mergeEvent : Nat := 96493
def frameStart : Nat := 96426
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩] } }
def leftRaw : List Term := Proof.Events376.exact96488RawTerms
def rightRaw : List Term := Proof.Events376.exact96465RawTerms
def group : MergeGroup := .operator 96488 96465
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96488) (leftOrdinal := 1)
    (rightResult := 96465) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29133⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96493

namespace LeftMerge96495
def owner : Owner := ⟨.program ⟨214⟩, ⟨29134⟩⟩
def mergeEvent : Nat := 96495
def frameStart : Nat := 96426
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24531⟩⟩] } }
def rhsRaw : List Term := Proof.Events376.exact96462RawTerms
def group : MergeGroup := .relation 96494
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96494) (rhsResult := 96462)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29133⟩⟩) ⟨24531⟩ 96462) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24531⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨24531⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96495

namespace LeftMerge96503
def owner : Owner := ⟨.program ⟨214⟩, ⟨18199⟩⟩
def mergeEvent : Nat := 96503
def frameStart : Nat := 96426
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18198⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events376.exact96476RawTerms
def rightRaw : List Term := Proof.Events376.exact96499RawTerms
def group : MergeGroup := .operator 96476 96499
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96476) (leftOrdinal := 0)
    (rightResult := 96499) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18198⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96503

namespace LeftMerge96520
def owner : Owner := ⟨.program ⟨214⟩, ⟨22256⟩⟩
def mergeEvent : Nat := 96520
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }
def rhsRaw : List Term := Proof.Events377.exact96517RawTerms
def group : MergeGroup := .relation 96519
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96519) (rhsResult := 96517)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22253⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 96518 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22253⟩⟩]⟩) (none) 96517) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96520

namespace LeftMerge96521
def owner : Owner := ⟨.program ⟨214⟩, ⟨22256⟩⟩
def mergeEvent : Nat := 96521
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩] } }
def rhsRaw : List Term := Proof.Events377.exact96517RawTerms
def group : MergeGroup := .relation 96519
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96519) (rhsResult := 96517)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22253⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 96518 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22253⟩⟩]⟩) (none) 96517) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96521

namespace LeftMerge96522
def owner : Owner := ⟨.program ⟨214⟩, ⟨22256⟩⟩
def mergeEvent : Nat := 96522
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24531⟩⟩] } }
def rhsRaw : List Term := Proof.Events377.exact96517RawTerms
def group : MergeGroup := .relation 96519
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96519) (rhsResult := 96517)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22253⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 96518 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22253⟩⟩]⟩) (none) 96517) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24531⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨24531⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96522

namespace LeftMerge96523
def owner : Owner := ⟨.program ⟨214⟩, ⟨22256⟩⟩
def mergeEvent : Nat := 96523
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18198⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events377.exact96517RawTerms
def group : MergeGroup := .relation 96519
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96519) (rhsResult := 96517)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22253⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 96518 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22253⟩⟩]⟩) (none) 96517) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18198⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96523

namespace LeftMerge96528
def owner : Owner := ⟨.program ⟨214⟩, ⟨29136⟩⟩
def mergeEvent : Nat := 96528
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩] } }
def leftRaw : List Term := Proof.Events377.exact96524RawTerms
def rightRaw : List Term := Proof.Events376.exact96370RawTerms
def group : MergeGroup := .operator 96524 96370
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96524) (leftOrdinal := 0)
    (rightResult := 96370) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96528

namespace LeftMerge96529
def owner : Owner := ⟨.program ⟨214⟩, ⟨29136⟩⟩
def mergeEvent : Nat := 96529
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24531⟩⟩] } }
def leftRaw : List Term := Proof.Events377.exact96524RawTerms
def rightRaw : List Term := Proof.Events376.exact96370RawTerms
def group : MergeGroup := .operator 96524 96370
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96524) (leftOrdinal := 2)
    (rightResult := 96370) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24531⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24531⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨24531⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96529

namespace LeftMerge96555
def owner : Owner := ⟨.program ⟨214⟩, ⟨12349⟩⟩
def mergeEvent : Nat := 96555
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events018.exact4681RawTerms
def rightRaw : List Term := Proof.Events000.exact32RawTerms
def group : MergeGroup := .operator 4681 32
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4681) (leftOrdinal := 0)
    (rightResult := 32) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12346⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96555

namespace LeftMerge96560
def owner : Owner := ⟨.program ⟨214⟩, ⟨7122⟩⟩
def mergeEvent : Nat := 96560
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6785⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact27RawTerms
def rightRaw : List Term := Proof.Events035.exact8977RawTerms
def group : MergeGroup := .operator 27 8977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27) (leftOrdinal := 0)
    (rightResult := 8977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6785⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6785⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96560

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
