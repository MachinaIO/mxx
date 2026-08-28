import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge68460
def owner : Owner := ⟨.program ⟨257⟩, ⟨34110⟩⟩
def mergeEvent : Nat := 68460
def frameStart : Nat := 68382
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩] } }
def leftRaw : List Term := Proof.Events267.exact68456RawTerms
def rightRaw : List Term := Proof.Events267.exact68433RawTerms
def group : MergeGroup := .operator 68456 68433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 68456) (leftOrdinal := 0)
    (rightResult := 68433) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨34109⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68460

namespace LeftMerge68461
def owner : Owner := ⟨.program ⟨257⟩, ⟨34110⟩⟩
def mergeEvent : Nat := 68461
def frameStart : Nat := 68382
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩] } }
def leftRaw : List Term := Proof.Events267.exact68456RawTerms
def rightRaw : List Term := Proof.Events267.exact68433RawTerms
def group : MergeGroup := .operator 68456 68433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 68456) (leftOrdinal := 1)
    (rightResult := 68433) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨34109⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge68461

namespace LeftMerge68463
def owner : Owner := ⟨.program ⟨257⟩, ⟨34110⟩⟩
def mergeEvent : Nat := 68463
def frameStart : Nat := 68382
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33164⟩⟩] } }
def rhsRaw : List Term := Proof.Events267.exact68430RawTerms
def group : MergeGroup := .relation 68462
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 68462) (rhsResult := 68430)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨34109⟩⟩) ⟨33164⟩ 68430) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33164⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨33164⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge68463

namespace LeftMerge68471
def owner : Owner := ⟨.program ⟨257⟩, ⟨32241⟩⟩
def mergeEvent : Nat := 68471
def frameStart : Nat := 68382
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32239⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events267.exact68444RawTerms
def rightRaw : List Term := Proof.Events267.exact68467RawTerms
def group : MergeGroup := .operator 68444 68467
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 68444) (leftOrdinal := 0)
    (rightResult := 68467) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32239⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68471

namespace LeftMerge68488
def owner : Owner := ⟨.program ⟨257⟩, ⟨32839⟩⟩
def mergeEvent : Nat := 68488
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }
def rhsRaw : List Term := Proof.Events267.exact68485RawTerms
def group : MergeGroup := .relation 68487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 68487) (rhsResult := 68485)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32836⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 68486 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32836⟩⟩]⟩) (none) 68485) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68488

namespace LeftMerge68489
def owner : Owner := ⟨.program ⟨257⟩, ⟨32839⟩⟩
def mergeEvent : Nat := 68489
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩] } }
def rhsRaw : List Term := Proof.Events267.exact68485RawTerms
def group : MergeGroup := .relation 68487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 68487) (rhsResult := 68485)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32836⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 68486 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32836⟩⟩]⟩) (none) 68485) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge68489

namespace LeftMerge68490
def owner : Owner := ⟨.program ⟨257⟩, ⟨32839⟩⟩
def mergeEvent : Nat := 68490
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33164⟩⟩] } }
def rhsRaw : List Term := Proof.Events267.exact68485RawTerms
def group : MergeGroup := .relation 68487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 68487) (rhsResult := 68485)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32836⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 68486 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32836⟩⟩]⟩) (none) 68485) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33164⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨33164⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68490

namespace LeftMerge68491
def owner : Owner := ⟨.program ⟨257⟩, ⟨32839⟩⟩
def mergeEvent : Nat := 68491
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events267.exact68485RawTerms
def group : MergeGroup := .relation 68487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 68487) (rhsResult := 68485)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32836⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 68486 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32836⟩⟩]⟩) (none) 68485) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32239⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge68491

namespace LeftMerge68496
def owner : Owner := ⟨.program ⟨257⟩, ⟨34112⟩⟩
def mergeEvent : Nat := 68496
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩] } }
def leftRaw : List Term := Proof.Events267.exact68492RawTerms
def rightRaw : List Term := Proof.Events266.exact68314RawTerms
def group : MergeGroup := .operator 68492 68314
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 68492) (leftOrdinal := 0)
    (rightResult := 68314) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68496

namespace LeftMerge68497
def owner : Owner := ⟨.program ⟨257⟩, ⟨34112⟩⟩
def mergeEvent : Nat := 68497
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33164⟩⟩] } }
def leftRaw : List Term := Proof.Events267.exact68492RawTerms
def rightRaw : List Term := Proof.Events266.exact68314RawTerms
def group : MergeGroup := .operator 68492 68314
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 68492) (leftOrdinal := 2)
    (rightResult := 68314) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33164⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33164⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨33164⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge68497

namespace LeftMerge68523
def owner : Owner := ⟨.program ⟨257⟩, ⟨21665⟩⟩
def mergeEvent : Nat := 68523
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events010.exact2683RawTerms
def rightRaw : List Term := Proof.Events239.exact61278RawTerms
def group : MergeGroup := .operator 2683 61278
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2683) (leftOrdinal := 0)
    (rightResult := 61278) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21662⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68523

namespace LeftMerge68528
def owner : Owner := ⟨.program ⟨257⟩, ⟨10788⟩⟩
def mergeEvent : Nat := 68528
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61148RawTerms
def rightRaw : List Term := Proof.Events096.exact24595RawTerms
def group : MergeGroup := .operator 61148 24595
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61148) (leftOrdinal := 0)
    (rightResult := 24595) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68528

namespace LeftMerge68545
def owner : Owner := ⟨.program ⟨257⟩, ⟨21668⟩⟩
def mergeEvent : Nat := 68545
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events267.exact68539RawTerms
def rightRaw : List Term := Proof.Events010.exact2686RawTerms
def group : MergeGroup := .operator 68539 2686
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 68539) (leftOrdinal := 1)
    (rightResult := 2686) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21206⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge68545

namespace LeftMerge68546
def owner : Owner := ⟨.program ⟨257⟩, ⟨21668⟩⟩
def mergeEvent : Nat := 68546
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }
def leftRaw : List Term := Proof.Events267.exact68539RawTerms
def rightRaw : List Term := Proof.Events010.exact2686RawTerms
def group : MergeGroup := .operator 68539 2686
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 68539) (leftOrdinal := 0)
    (rightResult := 2686) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21206⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68546

namespace LeftMerge68551
def owner : Owner := ⟨.program ⟨257⟩, ⟨21207⟩⟩
def mergeEvent : Nat := 68551
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events010.exact2686RawTerms
def rightRaw : List Term := Proof.Events239.exact61278RawTerms
def group : MergeGroup := .operator 2686 61278
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2686) (leftOrdinal := 0)
    (rightResult := 61278) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21206⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68551

namespace LeftMerge68556
def owner : Owner := ⟨.program ⟨257⟩, ⟨10768⟩⟩
def mergeEvent : Nat := 68556
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61148RawTerms
def rightRaw : List Term := Proof.Events096.exact24636RawTerms
def group : MergeGroup := .operator 61148 24636
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61148) (leftOrdinal := 0)
    (rightResult := 24636) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68556

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
