import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge54329
def owner : Owner := ⟨.program ⟨214⟩, ⟨28532⟩⟩
def mergeEvent : Nat := 54329
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16266⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28530⟩⟩] } }
def leftRaw : List Term := Proof.Events212.exact54322RawTerms
def rightRaw : List Term := Proof.Events211.exact54045RawTerms
def group : MergeGroup := .operator 54322 54045
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54322) (leftOrdinal := 1)
    (rightResult := 54045) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16266⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28530⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16266⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28530⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54329

namespace LeftMerge54331
def owner : Owner := ⟨.program ⟨214⟩, ⟨28532⟩⟩
def mergeEvent : Nat := 54331
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16266⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24354⟩⟩] } }
def rhsRaw : List Term := Proof.Events211.exact54042RawTerms
def group : MergeGroup := .relation 54330
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54330) (rhsResult := 54042)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16266⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28530⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28530⟩⟩) ⟨24354⟩ 54042) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24354⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16266⟩⟩], [⟨.program ⟨214⟩, ⟨24354⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54331

namespace LeftMerge54345
def owner : Owner := ⟨.program ⟨214⟩, ⟨21839⟩⟩
def mergeEvent : Nat := 54345
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21836⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events212.exact54339RawTerms
def group : MergeGroup := .operator 50762 54339
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 54339) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21836⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54345

namespace LeftMerge54466
def owner : Owner := ⟨.program ⟨214⟩, ⟨16343⟩⟩
def mergeEvent : Nat := 54466
def frameStart : Nat := 54400
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16266⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events212.exact54462RawTerms
def rightRaw : List Term := Proof.Events212.exact54460RawTerms
def group : MergeGroup := .operator 54462 54460
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54462) (leftOrdinal := 0)
    (rightResult := 54460) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16266⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16266⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54466

namespace LeftMerge54478
def owner : Owner := ⟨.program ⟨214⟩, ⟨28531⟩⟩
def mergeEvent : Nat := 54478
def frameStart : Nat := 54400
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28530⟩⟩] } }
def leftRaw : List Term := Proof.Events212.exact54474RawTerms
def rightRaw : List Term := Proof.Events212.exact54451RawTerms
def group : MergeGroup := .operator 54474 54451
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54474) (leftOrdinal := 0)
    (rightResult := 54451) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28530⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28530⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54478

namespace LeftMerge54479
def owner : Owner := ⟨.program ⟨214⟩, ⟨28531⟩⟩
def mergeEvent : Nat := 54479
def frameStart : Nat := 54400
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16266⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28530⟩⟩] } }
def leftRaw : List Term := Proof.Events212.exact54474RawTerms
def rightRaw : List Term := Proof.Events212.exact54451RawTerms
def group : MergeGroup := .operator 54474 54451
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54474) (leftOrdinal := 1)
    (rightResult := 54451) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16266⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28530⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16266⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28530⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54479

namespace LeftMerge54481
def owner : Owner := ⟨.program ⟨214⟩, ⟨28531⟩⟩
def mergeEvent : Nat := 54481
def frameStart : Nat := 54400
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16266⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24354⟩⟩] } }
def rhsRaw : List Term := Proof.Events212.exact54448RawTerms
def group : MergeGroup := .relation 54480
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54480) (rhsResult := 54448)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16266⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28530⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28530⟩⟩) ⟨24354⟩ 54448) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24354⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16266⟩⟩], [⟨.program ⟨214⟩, ⟨24354⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54481

namespace LeftMerge54489
def owner : Owner := ⟨.program ⟨214⟩, ⟨16312⟩⟩
def mergeEvent : Nat := 54489
def frameStart : Nat := 54400
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16311⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events212.exact54462RawTerms
def rightRaw : List Term := Proof.Events212.exact54485RawTerms
def group : MergeGroup := .operator 54462 54485
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54462) (leftOrdinal := 0)
    (rightResult := 54485) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16311⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16311⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54489

namespace LeftMerge54506
def owner : Owner := ⟨.program ⟨214⟩, ⟨21839⟩⟩
def mergeEvent : Nat := 54506
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }
def rhsRaw : List Term := Proof.Events212.exact54503RawTerms
def group : MergeGroup := .relation 54505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54505) (rhsResult := 54503)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21836⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 54504 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21836⟩⟩]⟩) (none) 54503) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54506

namespace LeftMerge54507
def owner : Owner := ⟨.program ⟨214⟩, ⟨21839⟩⟩
def mergeEvent : Nat := 54507
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28530⟩⟩] } }
def rhsRaw : List Term := Proof.Events212.exact54503RawTerms
def group : MergeGroup := .relation 54505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54505) (rhsResult := 54503)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21836⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 54504 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21836⟩⟩]⟩) (none) 54503) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28530⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28530⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54507

namespace LeftMerge54508
def owner : Owner := ⟨.program ⟨214⟩, ⟨21839⟩⟩
def mergeEvent : Nat := 54508
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16266⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24354⟩⟩] } }
def rhsRaw : List Term := Proof.Events212.exact54503RawTerms
def group : MergeGroup := .relation 54505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54505) (rhsResult := 54503)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21836⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 54504 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21836⟩⟩]⟩) (none) 54503) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16266⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24354⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16266⟩⟩], [⟨.program ⟨214⟩, ⟨24354⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54508

namespace LeftMerge54509
def owner : Owner := ⟨.program ⟨214⟩, ⟨21839⟩⟩
def mergeEvent : Nat := 54509
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16311⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events212.exact54503RawTerms
def group : MergeGroup := .relation 54505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54505) (rhsResult := 54503)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21836⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 54504 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21836⟩⟩]⟩) (none) 54503) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16311⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16311⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54509

namespace LeftMerge54514
def owner : Owner := ⟨.program ⟨214⟩, ⟨28533⟩⟩
def mergeEvent : Nat := 54514
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28530⟩⟩] } }
def leftRaw : List Term := Proof.Events212.exact54510RawTerms
def rightRaw : List Term := Proof.Events212.exact54332RawTerms
def group : MergeGroup := .operator 54510 54332
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54510) (leftOrdinal := 0)
    (rightResult := 54332) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28530⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28530⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28530⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54514

namespace LeftMerge54515
def owner : Owner := ⟨.program ⟨214⟩, ⟨28533⟩⟩
def mergeEvent : Nat := 54515
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16266⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24354⟩⟩] } }
def leftRaw : List Term := Proof.Events212.exact54510RawTerms
def rightRaw : List Term := Proof.Events212.exact54332RawTerms
def group : MergeGroup := .operator 54510 54332
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54510) (leftOrdinal := 2)
    (rightResult := 54332) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16266⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24354⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16266⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24354⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16266⟩⟩], [⟨.program ⟨214⟩, ⟨24354⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54515

namespace LeftMerge54541
def owner : Owner := ⟨.program ⟨214⟩, ⟨11642⟩⟩
def mergeEvent : Nat := 54541
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11641⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events009.exact2522RawTerms
def rightRaw : List Term := Proof.Events197.exact50670RawTerms
def group : MergeGroup := .operator 2522 50670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2522) (leftOrdinal := 0)
    (rightResult := 50670) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11641⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54541

namespace LeftMerge54546
def owner : Owner := ⟨.program ⟨214⟩, ⟨7275⟩⟩
def mergeEvent : Nat := 54546
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }
def leftRaw : List Term := Proof.Events197.exact50540RawTerms
def rightRaw : List Term := Proof.Events040.exact10480RawTerms
def group : MergeGroup := .operator 50540 10480
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50540) (leftOrdinal := 0)
    (rightResult := 10480) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54546

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
