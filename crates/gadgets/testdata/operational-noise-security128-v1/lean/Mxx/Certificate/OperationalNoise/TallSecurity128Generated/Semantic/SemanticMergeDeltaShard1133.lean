import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge185296
def owner : Owner := ⟨.program ⟨257⟩, ⟨32422⟩⟩
def mergeEvent : Nat := 185296
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨31852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events723.exact185290RawTerms
def group : MergeGroup := .relation 185292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185292) (rhsResult := 185290)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 185291 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32419⟩⟩]⟩) (none) 185290) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨31852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185296

namespace LeftMerge185301
def owner : Owner := ⟨.program ⟨257⟩, ⟨33494⟩⟩
def mergeEvent : Nat := 185301
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32967⟩⟩] } }
def leftRaw : List Term := Proof.Events723.exact185297RawTerms
def rightRaw : List Term := Proof.Events723.exact185111RawTerms
def group : MergeGroup := .operator 185297 185111
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185297) (leftOrdinal := 2)
    (rightResult := 185111) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32967⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32967⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], [⟨.program ⟨257⟩, ⟨32967⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185301

namespace LeftMerge185302
def owner : Owner := ⟨.program ⟨257⟩, ⟨33494⟩⟩
def mergeEvent : Nat := 185302
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33492⟩⟩] } }
def leftRaw : List Term := Proof.Events723.exact185297RawTerms
def rightRaw : List Term := Proof.Events723.exact185111RawTerms
def group : MergeGroup := .operator 185297 185111
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185297) (leftOrdinal := 1)
    (rightResult := 185111) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33492⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33492⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33492⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185302

namespace LeftMerge185310
def owner : Owner := ⟨.program ⟨257⟩, ⟨33987⟩⟩
def mergeEvent : Nat := 185310
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33985⟩⟩] } }
def leftRaw : List Term := Proof.Events723.exact185304RawTerms
def rightRaw : List Term := Proof.Events722.exact185027RawTerms
def group : MergeGroup := .operator 185304 185027
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185304) (leftOrdinal := 0)
    (rightResult := 185027) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33985⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33985⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185310

namespace LeftMerge185311
def owner : Owner := ⟨.program ⟨257⟩, ⟨33987⟩⟩
def mergeEvent : Nat := 185311
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨31852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33985⟩⟩] } }
def leftRaw : List Term := Proof.Events723.exact185304RawTerms
def rightRaw : List Term := Proof.Events722.exact185027RawTerms
def group : MergeGroup := .operator 185304 185027
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185304) (leftOrdinal := 1)
    (rightResult := 185027) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨31852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33985⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨31852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33985⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185311

namespace LeftMerge185313
def owner : Owner := ⟨.program ⟨257⟩, ⟨33987⟩⟩
def mergeEvent : Nat := 185313
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨31852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33128⟩⟩] } }
def rhsRaw : List Term := Proof.Events722.exact185024RawTerms
def group : MergeGroup := .relation 185312
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185312) (rhsResult := 185024)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨31852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33985⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33985⟩⟩) ⟨33128⟩ 185024) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33128⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨31852⟩⟩], [⟨.program ⟨257⟩, ⟨33128⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185313

namespace LeftMerge185327
def owner : Owner := ⟨.program ⟨257⟩, ⟨32759⟩⟩
def mergeEvent : Nat := 185327
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32756⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events723.exact185321RawTerms
def group : MergeGroup := .operator 178370 185321
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 185321) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32756⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185327

namespace LeftMerge185448
def owner : Owner := ⟨.program ⟨257⟩, ⟨33320⟩⟩
def mergeEvent : Nat := 185448
def frameStart : Nat := 185382
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events724.exact185444RawTerms
def rightRaw : List Term := Proof.Events724.exact185442RawTerms
def group : MergeGroup := .operator 185444 185442
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185444) (leftOrdinal := 0)
    (rightResult := 185442) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31852⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185448

namespace LeftMerge185460
def owner : Owner := ⟨.program ⟨257⟩, ⟨33986⟩⟩
def mergeEvent : Nat := 185460
def frameStart : Nat := 185382
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33985⟩⟩] } }
def leftRaw : List Term := Proof.Events724.exact185456RawTerms
def rightRaw : List Term := Proof.Events724.exact185433RawTerms
def group : MergeGroup := .operator 185456 185433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185456) (leftOrdinal := 0)
    (rightResult := 185433) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33985⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33985⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185460

namespace LeftMerge185461
def owner : Owner := ⟨.program ⟨257⟩, ⟨33986⟩⟩
def mergeEvent : Nat := 185461
def frameStart : Nat := 185382
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33985⟩⟩] } }
def leftRaw : List Term := Proof.Events724.exact185456RawTerms
def rightRaw : List Term := Proof.Events724.exact185433RawTerms
def group : MergeGroup := .operator 185456 185433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185456) (leftOrdinal := 1)
    (rightResult := 185433) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33985⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33985⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185461

namespace LeftMerge185463
def owner : Owner := ⟨.program ⟨257⟩, ⟨33986⟩⟩
def mergeEvent : Nat := 185463
def frameStart : Nat := 185382
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33128⟩⟩] } }
def rhsRaw : List Term := Proof.Events724.exact185430RawTerms
def group : MergeGroup := .relation 185462
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185462) (rhsResult := 185430)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨31852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33985⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33985⟩⟩) ⟨33128⟩ 185430) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33128⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨31852⟩⟩], [⟨.program ⟨257⟩, ⟨33128⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185463

namespace LeftMerge185471
def owner : Owner := ⟨.program ⟨257⟩, ⟨32165⟩⟩
def mergeEvent : Nat := 185471
def frameStart : Nat := 185382
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32163⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events724.exact185444RawTerms
def rightRaw : List Term := Proof.Events724.exact185467RawTerms
def group : MergeGroup := .operator 185444 185467
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185444) (leftOrdinal := 0)
    (rightResult := 185467) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32163⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185471

namespace LeftMerge185488
def owner : Owner := ⟨.program ⟨257⟩, ⟨32759⟩⟩
def mergeEvent : Nat := 185488
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }
def rhsRaw : List Term := Proof.Events724.exact185485RawTerms
def group : MergeGroup := .relation 185487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185487) (rhsResult := 185485)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32756⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 185486 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32756⟩⟩]⟩) (none) 185485) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185488

namespace LeftMerge185489
def owner : Owner := ⟨.program ⟨257⟩, ⟨32759⟩⟩
def mergeEvent : Nat := 185489
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33985⟩⟩] } }
def rhsRaw : List Term := Proof.Events724.exact185485RawTerms
def group : MergeGroup := .relation 185487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185487) (rhsResult := 185485)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32756⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 185486 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32756⟩⟩]⟩) (none) 185485) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33985⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33985⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185489

namespace LeftMerge185490
def owner : Owner := ⟨.program ⟨257⟩, ⟨32759⟩⟩
def mergeEvent : Nat := 185490
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨31852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33128⟩⟩] } }
def rhsRaw : List Term := Proof.Events724.exact185485RawTerms
def group : MergeGroup := .relation 185487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185487) (rhsResult := 185485)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32756⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 185486 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32756⟩⟩]⟩) (none) 185485) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33128⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨31852⟩⟩], [⟨.program ⟨257⟩, ⟨33128⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185490

namespace LeftMerge185491
def owner : Owner := ⟨.program ⟨257⟩, ⟨32759⟩⟩
def mergeEvent : Nat := 185491
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events724.exact185485RawTerms
def group : MergeGroup := .relation 185487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185487) (rhsResult := 185485)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32756⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 185486 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32756⟩⟩]⟩) (none) 185485) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32163⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185491

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
