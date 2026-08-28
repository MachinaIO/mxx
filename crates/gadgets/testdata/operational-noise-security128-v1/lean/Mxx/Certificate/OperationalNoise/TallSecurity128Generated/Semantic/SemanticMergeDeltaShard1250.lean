import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge203849
def owner : Owner := ⟨.program ⟨257⟩, ⟨42035⟩⟩
def mergeEvent : Nat := 203849
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42033⟩⟩] } }
def leftRaw : List Term := Proof.Events760.exact194627RawTerms
def rightRaw : List Term := Proof.Events796.exact203843RawTerms
def group : MergeGroup := .operator 194627 203843
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 194627) (leftOrdinal := 0)
    (rightResult := 203843) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42033⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42033⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203849

namespace LeftMerge203850
def owner : Owner := ⟨.program ⟨257⟩, ⟨42035⟩⟩
def mergeEvent : Nat := 203850
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42033⟩⟩] } }
def leftRaw : List Term := Proof.Events760.exact194627RawTerms
def rightRaw : List Term := Proof.Events796.exact203843RawTerms
def group : MergeGroup := .operator 194627 203843
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 194627) (leftOrdinal := 1)
    (rightResult := 203843) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42033⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42033⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203850

namespace LeftMerge203852
def owner : Owner := ⟨.program ⟨257⟩, ⟨42035⟩⟩
def mergeEvent : Nat := 203852
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41278⟩⟩] } }
def rhsRaw : List Term := Proof.Events796.exact203840RawTerms
def group : MergeGroup := .relation 203851
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 203851) (rhsResult := 203840)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42033⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42033⟩⟩) ⟨41278⟩ 203840) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41278⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40124⟩⟩], [⟨.program ⟨257⟩, ⟨41278⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203852

namespace LeftMerge203866
def owner : Owner := ⟨.program ⟨257⟩, ⟨40895⟩⟩
def mergeEvent : Nat := 203866
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40892⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events796.exact203860RawTerms
def group : MergeGroup := .operator 192995 203860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 203860) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40892⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40892⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203866

namespace LeftMerge203987
def owner : Owner := ⟨.program ⟨257⟩, ⟨41476⟩⟩
def mergeEvent : Nat := 203987
def frameStart : Nat := 203921
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events796.exact203983RawTerms
def rightRaw : List Term := Proof.Events796.exact203981RawTerms
def group : MergeGroup := .operator 203983 203981
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203983) (leftOrdinal := 0)
    (rightResult := 203981) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40124⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203987

namespace LeftMerge203999
def owner : Owner := ⟨.program ⟨257⟩, ⟨42034⟩⟩
def mergeEvent : Nat := 203999
def frameStart : Nat := 203921
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42033⟩⟩] } }
def leftRaw : List Term := Proof.Events796.exact203995RawTerms
def rightRaw : List Term := Proof.Events796.exact203972RawTerms
def group : MergeGroup := .operator 203995 203972
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203995) (leftOrdinal := 0)
    (rightResult := 203972) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42033⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42033⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203999

namespace LeftMerge204000
def owner : Owner := ⟨.program ⟨257⟩, ⟨42034⟩⟩
def mergeEvent : Nat := 204000
def frameStart : Nat := 203921
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42033⟩⟩] } }
def leftRaw : List Term := Proof.Events796.exact203995RawTerms
def rightRaw : List Term := Proof.Events796.exact203972RawTerms
def group : MergeGroup := .operator 203995 203972
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203995) (leftOrdinal := 1)
    (rightResult := 203972) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42033⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42033⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204000

namespace LeftMerge204002
def owner : Owner := ⟨.program ⟨257⟩, ⟨42034⟩⟩
def mergeEvent : Nat := 204002
def frameStart : Nat := 203921
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41278⟩⟩] } }
def rhsRaw : List Term := Proof.Events796.exact203969RawTerms
def group : MergeGroup := .relation 204001
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 204001) (rhsResult := 203969)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42033⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42033⟩⟩) ⟨41278⟩ 203969) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41278⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40124⟩⟩], [⟨.program ⟨257⟩, ⟨41278⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204002

namespace LeftMerge204010
def owner : Owner := ⟨.program ⟨257⟩, ⟨40350⟩⟩
def mergeEvent : Nat := 204010
def frameStart : Nat := 203921
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events796.exact203983RawTerms
def rightRaw : List Term := Proof.Events796.exact204006RawTerms
def group : MergeGroup := .operator 203983 204006
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203983) (leftOrdinal := 0)
    (rightResult := 204006) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40348⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40348⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge204010

namespace LeftMerge204027
def owner : Owner := ⟨.program ⟨257⟩, ⟨40895⟩⟩
def mergeEvent : Nat := 204027
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩] } }
def rhsRaw : List Term := Proof.Events796.exact204024RawTerms
def group : MergeGroup := .relation 204026
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 204026) (rhsResult := 204024)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40892⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 204025 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40892⟩⟩]⟩) (none) 204024) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge204027

namespace LeftMerge204028
def owner : Owner := ⟨.program ⟨257⟩, ⟨40895⟩⟩
def mergeEvent : Nat := 204028
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42033⟩⟩] } }
def rhsRaw : List Term := Proof.Events796.exact204024RawTerms
def group : MergeGroup := .relation 204026
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 204026) (rhsResult := 204024)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40892⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 204025 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40892⟩⟩]⟩) (none) 204024) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42033⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42033⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204028

namespace LeftMerge204029
def owner : Owner := ⟨.program ⟨257⟩, ⟨40895⟩⟩
def mergeEvent : Nat := 204029
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41278⟩⟩] } }
def rhsRaw : List Term := Proof.Events796.exact204024RawTerms
def group : MergeGroup := .relation 204026
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 204026) (rhsResult := 204024)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40892⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 204025 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40892⟩⟩]⟩) (none) 204024) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41278⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40124⟩⟩], [⟨.program ⟨257⟩, ⟨41278⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge204029

namespace LeftMerge204030
def owner : Owner := ⟨.program ⟨257⟩, ⟨40895⟩⟩
def mergeEvent : Nat := 204030
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events796.exact204024RawTerms
def group : MergeGroup := .relation 204026
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 204026) (rhsResult := 204024)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40892⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 204025 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40892⟩⟩]⟩) (none) 204024) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40348⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204030

namespace LeftMerge204035
def owner : Owner := ⟨.program ⟨257⟩, ⟨42036⟩⟩
def mergeEvent : Nat := 204035
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42033⟩⟩] } }
def leftRaw : List Term := Proof.Events796.exact204031RawTerms
def rightRaw : List Term := Proof.Events796.exact203853RawTerms
def group : MergeGroup := .operator 204031 203853
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 204031) (leftOrdinal := 0)
    (rightResult := 203853) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42033⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42033⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42033⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge204035

namespace LeftMerge204036
def owner : Owner := ⟨.program ⟨257⟩, ⟨42036⟩⟩
def mergeEvent : Nat := 204036
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41278⟩⟩] } }
def leftRaw : List Term := Proof.Events796.exact204031RawTerms
def rightRaw : List Term := Proof.Events796.exact203853RawTerms
def group : MergeGroup := .operator 204031 203853
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 204031) (leftOrdinal := 2)
    (rightResult := 203853) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41278⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41278⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40124⟩⟩], [⟨.program ⟨257⟩, ⟨41278⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204036

namespace LeftMerge204044
def owner : Owner := ⟨.program ⟨257⟩, ⟨42037⟩⟩
def mergeEvent : Nat := 204044
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩] } }
def leftRaw : List Term := Proof.Events797.exact204038RawTerms
def rightRaw : List Term := Proof.Events060.exact15602RawTerms
def group : MergeGroup := .operator 204038 15602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 204038) (leftOrdinal := 0)
    (rightResult := 15602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7159⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge204044

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
