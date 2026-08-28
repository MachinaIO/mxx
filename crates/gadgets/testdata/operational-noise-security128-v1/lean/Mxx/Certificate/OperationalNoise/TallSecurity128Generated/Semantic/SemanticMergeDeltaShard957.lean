import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge157472
def owner : Owner := ⟨.program ⟨257⟩, ⟨15766⟩⟩
def mergeEvent : Nat := 157472
def frameStart : Nat := 157369
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events614.exact157425RawTerms
def rightRaw : List Term := Proof.Events615.exact157468RawTerms
def group : MergeGroup := .operator 157425 157468
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 157425) (leftOrdinal := 0)
    (rightResult := 157468) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15764⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge157472

namespace LeftMerge157489
def owner : Owner := ⟨.program ⟨257⟩, ⟨16262⟩⟩
def mergeEvent : Nat := 157489
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }
def rhsRaw : List Term := Proof.Events615.exact157486RawTerms
def group : MergeGroup := .relation 157488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 157488) (rhsResult := 157486)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16259⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 157487 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16259⟩⟩]⟩) (none) 157486) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge157489

namespace LeftMerge157490
def owner : Owner := ⟨.program ⟨257⟩, ⟨16262⟩⟩
def mergeEvent : Nat := 157490
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17326⟩⟩] } }
def rhsRaw : List Term := Proof.Events615.exact157486RawTerms
def group : MergeGroup := .relation 157488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 157488) (rhsResult := 157486)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16259⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 157487 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16259⟩⟩]⟩) (none) 157486) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17326⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17326⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge157490

namespace LeftMerge157491
def owner : Owner := ⟨.program ⟨257⟩, ⟨16262⟩⟩
def mergeEvent : Nat := 157491
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12336⟩⟩, ⟨.program ⟨257⟩, ⟨15402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16831⟩⟩] } }
def rhsRaw : List Term := Proof.Events615.exact157486RawTerms
def group : MergeGroup := .relation 157488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 157488) (rhsResult := 157486)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16259⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 157487 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16259⟩⟩]⟩) (none) 157486) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12336⟩⟩, ⟨.program ⟨257⟩, ⟨15402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16831⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12336⟩⟩, ⟨.program ⟨257⟩, ⟨15402⟩⟩], [⟨.program ⟨257⟩, ⟨16831⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge157491

namespace LeftMerge157492
def owner : Owner := ⟨.program ⟨257⟩, ⟨16262⟩⟩
def mergeEvent : Nat := 157492
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events615.exact157486RawTerms
def group : MergeGroup := .relation 157488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 157488) (rhsResult := 157486)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16259⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 157487 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16259⟩⟩]⟩) (none) 157486) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge157492

namespace LeftMerge157497
def owner : Owner := ⟨.program ⟨257⟩, ⟨17328⟩⟩
def mergeEvent : Nat := 157497
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12336⟩⟩, ⟨.program ⟨257⟩, ⟨15402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16831⟩⟩] } }
def leftRaw : List Term := Proof.Events615.exact157493RawTerms
def rightRaw : List Term := Proof.Events614.exact157307RawTerms
def group : MergeGroup := .operator 157493 157307
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 157493) (leftOrdinal := 2)
    (rightResult := 157307) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12336⟩⟩, ⟨.program ⟨257⟩, ⟨15402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16831⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12336⟩⟩, ⟨.program ⟨257⟩, ⟨15402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16831⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12336⟩⟩, ⟨.program ⟨257⟩, ⟨15402⟩⟩], [⟨.program ⟨257⟩, ⟨16831⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge157497

namespace LeftMerge157498
def owner : Owner := ⟨.program ⟨257⟩, ⟨17328⟩⟩
def mergeEvent : Nat := 157498
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17326⟩⟩] } }
def leftRaw : List Term := Proof.Events615.exact157493RawTerms
def rightRaw : List Term := Proof.Events614.exact157307RawTerms
def group : MergeGroup := .operator 157493 157307
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 157493) (leftOrdinal := 1)
    (rightResult := 157307) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17326⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17326⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17326⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge157498

namespace LeftMerge157506
def owner : Owner := ⟨.program ⟨257⟩, ⟨17679⟩⟩
def mergeEvent : Nat := 157506
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17677⟩⟩] } }
def leftRaw : List Term := Proof.Events615.exact157500RawTerms
def rightRaw : List Term := Proof.Events614.exact157223RawTerms
def group : MergeGroup := .operator 157500 157223
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 157500) (leftOrdinal := 0)
    (rightResult := 157223) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17677⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17677⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge157506

namespace LeftMerge157507
def owner : Owner := ⟨.program ⟨257⟩, ⟨17679⟩⟩
def mergeEvent : Nat := 157507
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17677⟩⟩] } }
def leftRaw : List Term := Proof.Events615.exact157500RawTerms
def rightRaw : List Term := Proof.Events614.exact157223RawTerms
def group : MergeGroup := .operator 157500 157223
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 157500) (leftOrdinal := 1)
    (rightResult := 157223) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17677⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17677⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge157507

namespace LeftMerge157509
def owner : Owner := ⟨.program ⟨257⟩, ⟨17679⟩⟩
def mergeEvent : Nat := 157509
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16974⟩⟩] } }
def rhsRaw : List Term := Proof.Events614.exact157220RawTerms
def group : MergeGroup := .relation 157508
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 157508) (rhsResult := 157220)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17677⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17677⟩⟩) ⟨16974⟩ 157220) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16974⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨16974⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge157509

namespace LeftMerge157523
def owner : Owner := ⟨.program ⟨257⟩, ⟨16539⟩⟩
def mergeEvent : Nat := 157523
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16536⟩⟩] } }
def leftRaw : List Term := Proof.Events582.exact149120RawTerms
def rightRaw : List Term := Proof.Events615.exact157517RawTerms
def group : MergeGroup := .operator 149120 157517
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149120) (leftOrdinal := 0)
    (rightResult := 157517) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16536⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16536⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge157523

namespace LeftMerge157644
def owner : Owner := ⟨.program ⟨257⟩, ⟨17196⟩⟩
def mergeEvent : Nat := 157644
def frameStart : Nat := 157578
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events615.exact157640RawTerms
def rightRaw : List Term := Proof.Events615.exact157638RawTerms
def group : MergeGroup := .operator 157640 157638
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 157640) (leftOrdinal := 0)
    (rightResult := 157638) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15764⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge157644

namespace LeftMerge157656
def owner : Owner := ⟨.program ⟨257⟩, ⟨17678⟩⟩
def mergeEvent : Nat := 157656
def frameStart : Nat := 157578
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17677⟩⟩] } }
def leftRaw : List Term := Proof.Events615.exact157652RawTerms
def rightRaw : List Term := Proof.Events615.exact157629RawTerms
def group : MergeGroup := .operator 157652 157629
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 157652) (leftOrdinal := 0)
    (rightResult := 157629) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17677⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17677⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge157656

namespace LeftMerge157657
def owner : Owner := ⟨.program ⟨257⟩, ⟨17678⟩⟩
def mergeEvent : Nat := 157657
def frameStart : Nat := 157578
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17677⟩⟩] } }
def leftRaw : List Term := Proof.Events615.exact157652RawTerms
def rightRaw : List Term := Proof.Events615.exact157629RawTerms
def group : MergeGroup := .operator 157652 157629
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 157652) (leftOrdinal := 1)
    (rightResult := 157629) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17677⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17677⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge157657

namespace LeftMerge157659
def owner : Owner := ⟨.program ⟨257⟩, ⟨17678⟩⟩
def mergeEvent : Nat := 157659
def frameStart : Nat := 157578
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16974⟩⟩] } }
def rhsRaw : List Term := Proof.Events615.exact157626RawTerms
def group : MergeGroup := .relation 157658
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 157658) (rhsResult := 157626)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17677⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17677⟩⟩) ⟨16974⟩ 157626) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16974⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨16974⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge157659

namespace LeftMerge157667
def owner : Owner := ⟨.program ⟨257⟩, ⟨15988⟩⟩
def mergeEvent : Nat := 157667
def frameStart : Nat := 157578
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15987⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events615.exact157640RawTerms
def rightRaw : List Term := Proof.Events615.exact157663RawTerms
def group : MergeGroup := .operator 157640 157663
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 157640) (leftOrdinal := 0)
    (rightResult := 157663) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15987⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge157667

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
