import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge267944
def owner : Owner := ⟨.program ⟨257⟩, ⟨41785⟩⟩
def mergeEvent : Nat := 267944
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41782⟩⟩] } }
def leftRaw : List Term := Proof.Events1046.exact267940RawTerms
def rightRaw : List Term := Proof.Events1045.exact267762RawTerms
def group : MergeGroup := .operator 267940 267762
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267940) (leftOrdinal := 0)
    (rightResult := 267762) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41782⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41782⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41782⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267944

namespace LeftMerge267945
def owner : Owner := ⟨.program ⟨257⟩, ⟨41785⟩⟩
def mergeEvent : Nat := 267945
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41186⟩⟩] } }
def leftRaw : List Term := Proof.Events1046.exact267940RawTerms
def rightRaw : List Term := Proof.Events1045.exact267762RawTerms
def group : MergeGroup := .operator 267940 267762
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267940) (leftOrdinal := 2)
    (rightResult := 267762) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41186⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41186⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40042⟩⟩], [⟨.program ⟨257⟩, ⟨41186⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge267945

namespace LeftMerge267971
def owner : Owner := ⟨.program ⟨257⟩, ⟨36917⟩⟩
def mergeEvent : Nat := 267971
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events050.exact12902RawTerms
def rightRaw : List Term := Proof.Events1039.exact266028RawTerms
def group : MergeGroup := .operator 12902 266028
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12902) (leftOrdinal := 0)
    (rightResult := 266028) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨36914⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267971

namespace LeftMerge267976
def owner : Owner := ⟨.program ⟨257⟩, ⟨7637⟩⟩
def mergeEvent : Nat := 267976
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } }
def leftRaw : List Term := Proof.Events1038.exact265898RawTerms
def rightRaw : List Term := Proof.Events074.exact19084RawTerms
def group : MergeGroup := .operator 265898 19084
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 265898) (leftOrdinal := 0)
    (rightResult := 19084) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267976

namespace LeftMerge267993
def owner : Owner := ⟨.program ⟨257⟩, ⟨36920⟩⟩
def mergeEvent : Nat := 267993
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1046.exact267987RawTerms
def rightRaw : List Term := Proof.Events050.exact12905RawTerms
def group : MergeGroup := .operator 267987 12905
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267987) (leftOrdinal := 1)
    (rightResult := 12905) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13756⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge267993

namespace LeftMerge267994
def owner : Owner := ⟨.program ⟨257⟩, ⟨36920⟩⟩
def mergeEvent : Nat := 267994
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } }
def leftRaw : List Term := Proof.Events1046.exact267987RawTerms
def rightRaw : List Term := Proof.Events050.exact12905RawTerms
def group : MergeGroup := .operator 267987 12905
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267987) (leftOrdinal := 0)
    (rightResult := 12905) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13756⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267994

namespace LeftMerge267999
def owner : Owner := ⟨.program ⟨257⟩, ⟨13757⟩⟩
def mergeEvent : Nat := 267999
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events050.exact12905RawTerms
def rightRaw : List Term := Proof.Events1039.exact266028RawTerms
def group : MergeGroup := .operator 12905 266028
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12905) (leftOrdinal := 0)
    (rightResult := 266028) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13756⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267999

namespace LeftMerge268004
def owner : Owner := ⟨.program ⟨257⟩, ⟨7654⟩⟩
def mergeEvent : Nat := 268004
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }
def leftRaw : List Term := Proof.Events1038.exact265898RawTerms
def rightRaw : List Term := Proof.Events074.exact19125RawTerms
def group : MergeGroup := .operator 265898 19125
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 265898) (leftOrdinal := 0)
    (rightResult := 19125) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge268004

namespace LeftMerge268021
def owner : Owner := ⟨.program ⟨257⟩, ⟨13760⟩⟩
def mergeEvent : Nat := 268021
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }
def leftRaw : List Term := Proof.Events1046.exact268015RawTerms
def rightRaw : List Term := Proof.Events074.exact19114RawTerms
def group : MergeGroup := .operator 268015 19114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 268015) (leftOrdinal := 1)
    (rightResult := 19114) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9553⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge268021

namespace LeftMerge268023
def owner : Owner := ⟨.program ⟨257⟩, ⟨13760⟩⟩
def mergeEvent : Nat := 268023
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } }
def rhsRaw : List Term := Proof.Events074.exact19084RawTerms
def group : MergeGroup := .relation 268022
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 268022) (rhsResult := 19084)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9553⟩⟩) ⟨7281⟩ 19084) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge268023

namespace LeftMerge268024
def owner : Owner := ⟨.program ⟨257⟩, ⟨13760⟩⟩
def mergeEvent : Nat := 268024
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }
def leftRaw : List Term := Proof.Events1046.exact268015RawTerms
def rightRaw : List Term := Proof.Events074.exact19114RawTerms
def group : MergeGroup := .operator 268015 19114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 268015) (leftOrdinal := 0)
    (rightResult := 19114) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9553⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge268024

namespace LeftMerge268029
def owner : Owner := ⟨.program ⟨257⟩, ⟨36921⟩⟩
def mergeEvent : Nat := 268029
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } }
def leftRaw : List Term := Proof.Events1046.exact268025RawTerms
def rightRaw : List Term := Proof.Events1046.exact267995RawTerms
def group : MergeGroup := .operator 268025 267995
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 268025) (leftOrdinal := 1)
    (rightResult := 267995) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge268029

namespace LeftMerge268037
def owner : Owner := ⟨.program ⟨257⟩, ⟨38849⟩⟩
def mergeEvent : Nat := 268037
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38848⟩⟩] } }
def leftRaw : List Term := Proof.Events1046.exact268031RawTerms
def rightRaw : List Term := Proof.Events1046.exact267967RawTerms
def group : MergeGroup := .operator 268031 267967
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 268031) (leftOrdinal := 1)
    (rightResult := 267967) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38848⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge268037

namespace LeftMerge268039
def owner : Owner := ⟨.program ⟨257⟩, ⟨38849⟩⟩
def mergeEvent : Nat := 268039
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38379⟩⟩] } }
def rhsRaw : List Term := Proof.Events1046.exact267964RawTerms
def group : MergeGroup := .relation 268038
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 268038) (rhsResult := 267964)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38848⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38848⟩⟩) ⟨38379⟩ 267964) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38379⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], [⟨.program ⟨257⟩, ⟨38379⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge268039

namespace LeftMerge268040
def owner : Owner := ⟨.program ⟨257⟩, ⟨38849⟩⟩
def mergeEvent : Nat := 268040
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38848⟩⟩] } }
def leftRaw : List Term := Proof.Events1046.exact268031RawTerms
def rightRaw : List Term := Proof.Events1046.exact267967RawTerms
def group : MergeGroup := .operator 268031 267967
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 268031) (leftOrdinal := 0)
    (rightResult := 267967) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38848⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38848⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge268040

namespace LeftMerge268054
def owner : Owner := ⟨.program ⟨257⟩, ⟨37789⟩⟩
def mergeEvent : Nat := 268054
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37786⟩⟩] } }
def leftRaw : List Term := Proof.Events1039.exact266120RawTerms
def rightRaw : List Term := Proof.Events1047.exact268048RawTerms
def group : MergeGroup := .operator 266120 268048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266120) (leftOrdinal := 0)
    (rightResult := 268048) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨37786⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37786⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge268054

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
