import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge108949
def owner : Owner := ⟨.program ⟨257⟩, ⟨27772⟩⟩
def mergeEvent : Nat := 108949
def frameStart : Nat := 108883
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events425.exact108945RawTerms
def rightRaw : List Term := Proof.Events425.exact108943RawTerms
def group : MergeGroup := .operator 108945 108943
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108945) (leftOrdinal := 0)
    (rightResult := 108943) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26416⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108949

namespace LeftMerge108961
def owner : Owner := ⟨.program ⟨257⟩, ⟨28315⟩⟩
def mergeEvent : Nat := 108961
def frameStart : Nat := 108883
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩] } }
def leftRaw : List Term := Proof.Events425.exact108957RawTerms
def rightRaw : List Term := Proof.Events425.exact108934RawTerms
def group : MergeGroup := .operator 108957 108934
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108957) (leftOrdinal := 0)
    (rightResult := 108934) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28314⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108961

namespace LeftMerge108962
def owner : Owner := ⟨.program ⟨257⟩, ⟨28315⟩⟩
def mergeEvent : Nat := 108962
def frameStart : Nat := 108883
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩] } }
def leftRaw : List Term := Proof.Events425.exact108957RawTerms
def rightRaw : List Term := Proof.Events425.exact108934RawTerms
def group : MergeGroup := .operator 108957 108934
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108957) (leftOrdinal := 1)
    (rightResult := 108934) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28314⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108962

namespace LeftMerge108964
def owner : Owner := ⟨.program ⟨257⟩, ⟨28315⟩⟩
def mergeEvent : Nat := 108964
def frameStart : Nat := 108883
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27570⟩⟩] } }
def rhsRaw : List Term := Proof.Events425.exact108931RawTerms
def group : MergeGroup := .relation 108963
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108963) (rhsResult := 108931)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28314⟩⟩) ⟨27570⟩ 108931) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27570⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨27570⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108964

namespace LeftMerge108972
def owner : Owner := ⟨.program ⟨257⟩, ⟨26633⟩⟩
def mergeEvent : Nat := 108972
def frameStart : Nat := 108883
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26632⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events425.exact108945RawTerms
def rightRaw : List Term := Proof.Events425.exact108968RawTerms
def group : MergeGroup := .operator 108945 108968
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108945) (leftOrdinal := 0)
    (rightResult := 108968) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26632⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108972

namespace LeftMerge108989
def owner : Owner := ⟨.program ⟨257⟩, ⟨27179⟩⟩
def mergeEvent : Nat := 108989
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩] } }
def rhsRaw : List Term := Proof.Events425.exact108986RawTerms
def group : MergeGroup := .relation 108988
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108988) (rhsResult := 108986)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27176⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 108987 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27176⟩⟩]⟩) (none) 108986) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108989

namespace LeftMerge108990
def owner : Owner := ⟨.program ⟨257⟩, ⟨27179⟩⟩
def mergeEvent : Nat := 108990
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩] } }
def rhsRaw : List Term := Proof.Events425.exact108986RawTerms
def group : MergeGroup := .relation 108988
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108988) (rhsResult := 108986)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27176⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 108987 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27176⟩⟩]⟩) (none) 108986) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108990

namespace LeftMerge108991
def owner : Owner := ⟨.program ⟨257⟩, ⟨27179⟩⟩
def mergeEvent : Nat := 108991
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27570⟩⟩] } }
def rhsRaw : List Term := Proof.Events425.exact108986RawTerms
def group : MergeGroup := .relation 108988
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108988) (rhsResult := 108986)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27176⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 108987 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27176⟩⟩]⟩) (none) 108986) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27570⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨27570⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108991

namespace LeftMerge108992
def owner : Owner := ⟨.program ⟨257⟩, ⟨27179⟩⟩
def mergeEvent : Nat := 108992
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events425.exact108986RawTerms
def group : MergeGroup := .relation 108988
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108988) (rhsResult := 108986)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27176⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 108987 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27176⟩⟩]⟩) (none) 108986) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26632⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108992

namespace LeftMerge108997
def owner : Owner := ⟨.program ⟨257⟩, ⟨28317⟩⟩
def mergeEvent : Nat := 108997
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩] } }
def leftRaw : List Term := Proof.Events425.exact108993RawTerms
def rightRaw : List Term := Proof.Events425.exact108815RawTerms
def group : MergeGroup := .operator 108993 108815
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108993) (leftOrdinal := 0)
    (rightResult := 108815) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108997

namespace LeftMerge108998
def owner : Owner := ⟨.program ⟨257⟩, ⟨28317⟩⟩
def mergeEvent : Nat := 108998
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27570⟩⟩] } }
def leftRaw : List Term := Proof.Events425.exact108993RawTerms
def rightRaw : List Term := Proof.Events425.exact108815RawTerms
def group : MergeGroup := .operator 108993 108815
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108993) (leftOrdinal := 2)
    (rightResult := 108815) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27570⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27570⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨27570⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108998

namespace LeftMerge109024
def owner : Owner := ⟨.program ⟨257⟩, ⟨25743⟩⟩
def mergeEvent : Nat := 109024
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25742⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events018.exact4766RawTerms
def rightRaw : List Term := Proof.Events410.exact105153RawTerms
def group : MergeGroup := .operator 4766 105153
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4766) (leftOrdinal := 0)
    (rightResult := 105153) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25742⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25742⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge109024

namespace LeftMerge109029
def owner : Owner := ⟨.program ⟨257⟩, ⟨8696⟩⟩
def mergeEvent : Nat := 109029
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }
def leftRaw : List Term := Proof.Events410.exact105023RawTerms
def rightRaw : List Term := Proof.Events082.exact21088RawTerms
def group : MergeGroup := .operator 105023 21088
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105023) (leftOrdinal := 0)
    (rightResult := 21088) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge109029

namespace LeftMerge109046
def owner : Owner := ⟨.program ⟨257⟩, ⟨65475⟩⟩
def mergeEvent : Nat := 109046
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25742⟩⟩, ⟨.program ⟨257⟩, ⟨65472⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events425.exact109040RawTerms
def rightRaw : List Term := Proof.Events018.exact4769RawTerms
def group : MergeGroup := .operator 109040 4769
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 109040) (leftOrdinal := 1)
    (rightResult := 4769) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25742⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65472⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25742⟩⟩, ⟨.program ⟨257⟩, ⟨65472⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge109046

namespace LeftMerge109047
def owner : Owner := ⟨.program ⟨257⟩, ⟨65475⟩⟩
def mergeEvent : Nat := 109047
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨65472⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }
def leftRaw : List Term := Proof.Events425.exact109040RawTerms
def rightRaw : List Term := Proof.Events018.exact4769RawTerms
def group : MergeGroup := .operator 109040 4769
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 109040) (leftOrdinal := 0)
    (rightResult := 4769) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65472⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨65472⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge109047

namespace LeftMerge109052
def owner : Owner := ⟨.program ⟨257⟩, ⟨65476⟩⟩
def mergeEvent : Nat := 109052
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨65472⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events018.exact4769RawTerms
def rightRaw : List Term := Proof.Events410.exact105153RawTerms
def group : MergeGroup := .operator 4769 105153
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4769) (leftOrdinal := 0)
    (rightResult := 105153) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65472⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨65472⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge109052

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
