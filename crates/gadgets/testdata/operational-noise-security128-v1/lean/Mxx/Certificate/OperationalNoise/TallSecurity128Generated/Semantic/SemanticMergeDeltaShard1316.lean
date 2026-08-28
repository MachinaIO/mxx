import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge214453
def owner : Owner := ⟨.program ⟨257⟩, ⟨31486⟩⟩
def mergeEvent : Nat := 214453
def frameStart : Nat := 214423
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24290⟩⟩, ⟨.program ⟨257⟩, ⟨31485⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events837.exact214449RawTerms
def rightRaw : List Term := Proof.Events837.exact214446RawTerms
def group : MergeGroup := .operator 214449 214446
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214449) (leftOrdinal := 0)
    (rightResult := 214446) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31485⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24290⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24290⟩⟩, ⟨.program ⟨257⟩, ⟨31485⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214453

namespace LeftMerge214483
def owner : Owner := ⟨.program ⟨257⟩, ⟨33228⟩⟩
def mergeEvent : Nat := 214483
def frameStart : Nat := 214423
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24290⟩⟩, ⟨.program ⟨257⟩, ⟨31485⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events837.exact214479RawTerms
def rightRaw : List Term := Proof.Events837.exact214477RawTerms
def group : MergeGroup := .operator 214479 214477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214479) (leftOrdinal := 0)
    (rightResult := 214477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24290⟩⟩, ⟨.program ⟨257⟩, ⟨31485⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24290⟩⟩, ⟨.program ⟨257⟩, ⟨31485⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214483

namespace LeftMerge214506
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def mergeEvent : Nat := 214506
def frameStart : Nat := 214423
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }
def leftRaw : List Term := Proof.Events837.exact214502RawTerms
def rightRaw : List Term := Proof.Events837.exact214499RawTerms
def group : MergeGroup := .operator 214502 214499
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214502) (leftOrdinal := 0)
    (rightResult := 214499) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9577⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214506

namespace LeftMerge214515
def owner : Owner := ⟨.program ⟨257⟩, ⟨33462⟩⟩
def mergeEvent : Nat := 214515
def frameStart : Nat := 214423
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33459⟩⟩] } }
def leftRaw : List Term := Proof.Events837.exact214511RawTerms
def rightRaw : List Term := Proof.Events837.exact214468RawTerms
def group : MergeGroup := .operator 214511 214468
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214511) (leftOrdinal := 0)
    (rightResult := 214468) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33459⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33459⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214515

namespace LeftMerge214516
def owner : Owner := ⟨.program ⟨257⟩, ⟨33462⟩⟩
def mergeEvent : Nat := 214516
def frameStart : Nat := 214423
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24290⟩⟩, ⟨.program ⟨257⟩, ⟨31485⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33459⟩⟩] } }
def leftRaw : List Term := Proof.Events837.exact214511RawTerms
def rightRaw : List Term := Proof.Events837.exact214468RawTerms
def group : MergeGroup := .operator 214511 214468
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214511) (leftOrdinal := 1)
    (rightResult := 214468) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24290⟩⟩, ⟨.program ⟨257⟩, ⟨31485⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33459⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24290⟩⟩, ⟨.program ⟨257⟩, ⟨31485⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33459⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge214516

namespace LeftMerge214518
def owner : Owner := ⟨.program ⟨257⟩, ⟨33462⟩⟩
def mergeEvent : Nat := 214518
def frameStart : Nat := 214423
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24290⟩⟩, ⟨.program ⟨257⟩, ⟨31485⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32949⟩⟩] } }
def rhsRaw : List Term := Proof.Events837.exact214465RawTerms
def group : MergeGroup := .relation 214517
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 214517) (rhsResult := 214465)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24290⟩⟩, ⟨.program ⟨257⟩, ⟨31485⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33459⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33459⟩⟩) ⟨32949⟩ 214465) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32949⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24290⟩⟩, ⟨.program ⟨257⟩, ⟨31485⟩⟩], [⟨.program ⟨257⟩, ⟨32949⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge214518

namespace LeftMerge214526
def owner : Owner := ⟨.program ⟨257⟩, ⟨31830⟩⟩
def mergeEvent : Nat := 214526
def frameStart : Nat := 214423
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events837.exact214479RawTerms
def rightRaw : List Term := Proof.Events837.exact214522RawTerms
def group : MergeGroup := .operator 214479 214522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214479) (leftOrdinal := 0)
    (rightResult := 214522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31828⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214526

namespace LeftMerge214543
def owner : Owner := ⟨.program ⟨257⟩, ⟨32392⟩⟩
def mergeEvent : Nat := 214543
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }
def rhsRaw : List Term := Proof.Events838.exact214540RawTerms
def group : MergeGroup := .relation 214542
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 214542) (rhsResult := 214540)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 214541 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32389⟩⟩]⟩) (none) 214540) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214543

namespace LeftMerge214544
def owner : Owner := ⟨.program ⟨257⟩, ⟨32392⟩⟩
def mergeEvent : Nat := 214544
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33459⟩⟩] } }
def rhsRaw : List Term := Proof.Events838.exact214540RawTerms
def group : MergeGroup := .relation 214542
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 214542) (rhsResult := 214540)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 214541 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32389⟩⟩]⟩) (none) 214540) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33459⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33459⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge214544

namespace LeftMerge214545
def owner : Owner := ⟨.program ⟨257⟩, ⟨32392⟩⟩
def mergeEvent : Nat := 214545
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24290⟩⟩, ⟨.program ⟨257⟩, ⟨31485⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32949⟩⟩] } }
def rhsRaw : List Term := Proof.Events838.exact214540RawTerms
def group : MergeGroup := .relation 214542
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 214542) (rhsResult := 214540)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 214541 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32389⟩⟩]⟩) (none) 214540) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24290⟩⟩, ⟨.program ⟨257⟩, ⟨31485⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32949⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24290⟩⟩, ⟨.program ⟨257⟩, ⟨31485⟩⟩], [⟨.program ⟨257⟩, ⟨32949⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214545

namespace LeftMerge214546
def owner : Owner := ⟨.program ⟨257⟩, ⟨32392⟩⟩
def mergeEvent : Nat := 214546
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨31828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events838.exact214540RawTerms
def group : MergeGroup := .relation 214542
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 214542) (rhsResult := 214540)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 214541 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32389⟩⟩]⟩) (none) 214540) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨31828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge214546

namespace LeftMerge214551
def owner : Owner := ⟨.program ⟨257⟩, ⟨33461⟩⟩
def mergeEvent : Nat := 214551
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24290⟩⟩, ⟨.program ⟨257⟩, ⟨31485⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32949⟩⟩] } }
def leftRaw : List Term := Proof.Events838.exact214547RawTerms
def rightRaw : List Term := Proof.Events837.exact214361RawTerms
def group : MergeGroup := .operator 214547 214361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214547) (leftOrdinal := 2)
    (rightResult := 214361) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24290⟩⟩, ⟨.program ⟨257⟩, ⟨31485⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32949⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24290⟩⟩, ⟨.program ⟨257⟩, ⟨31485⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32949⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24290⟩⟩, ⟨.program ⟨257⟩, ⟨31485⟩⟩], [⟨.program ⟨257⟩, ⟨32949⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge214551

namespace LeftMerge214552
def owner : Owner := ⟨.program ⟨257⟩, ⟨33461⟩⟩
def mergeEvent : Nat := 214552
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33459⟩⟩] } }
def leftRaw : List Term := Proof.Events838.exact214547RawTerms
def rightRaw : List Term := Proof.Events837.exact214361RawTerms
def group : MergeGroup := .operator 214547 214361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214547) (leftOrdinal := 1)
    (rightResult := 214361) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33459⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33459⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33459⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214552

namespace LeftMerge214560
def owner : Owner := ⟨.program ⟨257⟩, ⟨33894⟩⟩
def mergeEvent : Nat := 214560
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33892⟩⟩] } }
def leftRaw : List Term := Proof.Events838.exact214554RawTerms
def rightRaw : List Term := Proof.Events837.exact214277RawTerms
def group : MergeGroup := .operator 214554 214277
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214554) (leftOrdinal := 0)
    (rightResult := 214277) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33892⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33892⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214560

namespace LeftMerge214561
def owner : Owner := ⟨.program ⟨257⟩, ⟨33894⟩⟩
def mergeEvent : Nat := 214561
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨31828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33892⟩⟩] } }
def leftRaw : List Term := Proof.Events838.exact214554RawTerms
def rightRaw : List Term := Proof.Events837.exact214277RawTerms
def group : MergeGroup := .operator 214554 214277
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214554) (leftOrdinal := 1)
    (rightResult := 214277) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨31828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33892⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨31828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33892⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge214561

namespace LeftMerge214563
def owner : Owner := ⟨.program ⟨257⟩, ⟨33894⟩⟩
def mergeEvent : Nat := 214563
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨31828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33101⟩⟩] } }
def rhsRaw : List Term := Proof.Events837.exact214274RawTerms
def group : MergeGroup := .relation 214562
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 214562) (rhsResult := 214274)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨31828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33892⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33892⟩⟩) ⟨33101⟩ 214274) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33101⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨31828⟩⟩], [⟨.program ⟨257⟩, ⟨33101⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge214563

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
