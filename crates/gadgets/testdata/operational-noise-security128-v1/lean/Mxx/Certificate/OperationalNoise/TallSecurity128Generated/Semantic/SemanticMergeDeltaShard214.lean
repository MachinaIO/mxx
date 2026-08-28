import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge38857
def owner : Owner := ⟨.program ⟨257⟩, ⟨33559⟩⟩
def mergeEvent : Nat := 38857
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33558⟩⟩] } }
def leftRaw : List Term := Proof.Events151.exact38851RawTerms
def rightRaw : List Term := Proof.Events151.exact38787RawTerms
def group : MergeGroup := .operator 38851 38787
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38851) (leftOrdinal := 1)
    (rightResult := 38787) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33558⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33558⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38857

namespace LeftMerge38859
def owner : Owner := ⟨.program ⟨257⟩, ⟨33559⟩⟩
def mergeEvent : Nat := 38859
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33003⟩⟩] } }
def rhsRaw : List Term := Proof.Events151.exact38784RawTerms
def group : MergeGroup := .relation 38858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38858) (rhsResult := 38784)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33558⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33558⟩⟩) ⟨33003⟩ 38784) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33003⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], [⟨.program ⟨257⟩, ⟨33003⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38859

namespace LeftMerge38860
def owner : Owner := ⟨.program ⟨257⟩, ⟨33559⟩⟩
def mergeEvent : Nat := 38860
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33558⟩⟩] } }
def leftRaw : List Term := Proof.Events151.exact38851RawTerms
def rightRaw : List Term := Proof.Events151.exact38787RawTerms
def group : MergeGroup := .operator 38851 38787
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38851) (leftOrdinal := 0)
    (rightResult := 38787) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33558⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33558⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38860

namespace LeftMerge38874
def owner : Owner := ⟨.program ⟨257⟩, ⟨32482⟩⟩
def mergeEvent : Nat := 38874
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32479⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32120RawTerms
def rightRaw : List Term := Proof.Events151.exact38868RawTerms
def group : MergeGroup := .operator 32120 38868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32120) (leftOrdinal := 0)
    (rightResult := 38868) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32479⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32479⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38874

namespace LeftMerge38953
def owner : Owner := ⟨.program ⟨257⟩, ⟨31729⟩⟩
def mergeEvent : Nat := 38953
def frameStart : Nat := 38923
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events152.exact38949RawTerms
def rightRaw : List Term := Proof.Events152.exact38946RawTerms
def group : MergeGroup := .operator 38949 38946
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38949) (leftOrdinal := 0)
    (rightResult := 38946) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31728⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24398⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38953

namespace LeftMerge38983
def owner : Owner := ⟨.program ⟨257⟩, ⟨33264⟩⟩
def mergeEvent : Nat := 38983
def frameStart : Nat := 38923
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events152.exact38979RawTerms
def rightRaw : List Term := Proof.Events152.exact38977RawTerms
def group : MergeGroup := .operator 38979 38977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38979) (leftOrdinal := 0)
    (rightResult := 38977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38983

namespace LeftMerge39006
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def mergeEvent : Nat := 39006
def frameStart : Nat := 38923
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }
def leftRaw : List Term := Proof.Events152.exact39002RawTerms
def rightRaw : List Term := Proof.Events152.exact38999RawTerms
def group : MergeGroup := .operator 39002 38999
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39002) (leftOrdinal := 0)
    (rightResult := 38999) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9577⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39006

namespace LeftMerge39015
def owner : Owner := ⟨.program ⟨257⟩, ⟨33561⟩⟩
def mergeEvent : Nat := 39015
def frameStart : Nat := 38923
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33558⟩⟩] } }
def leftRaw : List Term := Proof.Events152.exact39011RawTerms
def rightRaw : List Term := Proof.Events152.exact38968RawTerms
def group : MergeGroup := .operator 39011 38968
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39011) (leftOrdinal := 0)
    (rightResult := 38968) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33558⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33558⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39015

namespace LeftMerge39016
def owner : Owner := ⟨.program ⟨257⟩, ⟨33561⟩⟩
def mergeEvent : Nat := 39016
def frameStart : Nat := 38923
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33558⟩⟩] } }
def leftRaw : List Term := Proof.Events152.exact39011RawTerms
def rightRaw : List Term := Proof.Events152.exact38968RawTerms
def group : MergeGroup := .operator 39011 38968
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39011) (leftOrdinal := 1)
    (rightResult := 38968) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33558⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33558⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39016

namespace LeftMerge39018
def owner : Owner := ⟨.program ⟨257⟩, ⟨33561⟩⟩
def mergeEvent : Nat := 39018
def frameStart : Nat := 38923
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33003⟩⟩] } }
def rhsRaw : List Term := Proof.Events152.exact38965RawTerms
def group : MergeGroup := .relation 39017
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39017) (rhsResult := 38965)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33558⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33558⟩⟩) ⟨33003⟩ 38965) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33003⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], [⟨.program ⟨257⟩, ⟨33003⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39018

namespace LeftMerge39026
def owner : Owner := ⟨.program ⟨257⟩, ⟨31902⟩⟩
def mergeEvent : Nat := 39026
def frameStart : Nat := 38923
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events152.exact38979RawTerms
def rightRaw : List Term := Proof.Events152.exact39022RawTerms
def group : MergeGroup := .operator 38979 39022
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38979) (leftOrdinal := 0)
    (rightResult := 39022) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31900⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31900⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39026

namespace LeftMerge39043
def owner : Owner := ⟨.program ⟨257⟩, ⟨32482⟩⟩
def mergeEvent : Nat := 39043
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }
def rhsRaw : List Term := Proof.Events152.exact39040RawTerms
def group : MergeGroup := .relation 39042
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39042) (rhsResult := 39040)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 39041 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32479⟩⟩]⟩) (none) 39040) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39043

namespace LeftMerge39044
def owner : Owner := ⟨.program ⟨257⟩, ⟨32482⟩⟩
def mergeEvent : Nat := 39044
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33558⟩⟩] } }
def rhsRaw : List Term := Proof.Events152.exact39040RawTerms
def group : MergeGroup := .relation 39042
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39042) (rhsResult := 39040)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 39041 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32479⟩⟩]⟩) (none) 39040) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33558⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33558⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39044

namespace LeftMerge39045
def owner : Owner := ⟨.program ⟨257⟩, ⟨32482⟩⟩
def mergeEvent : Nat := 39045
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33003⟩⟩] } }
def rhsRaw : List Term := Proof.Events152.exact39040RawTerms
def group : MergeGroup := .relation 39042
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39042) (rhsResult := 39040)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 39041 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32479⟩⟩]⟩) (none) 39040) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33003⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], [⟨.program ⟨257⟩, ⟨33003⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39045

namespace LeftMerge39046
def owner : Owner := ⟨.program ⟨257⟩, ⟨32482⟩⟩
def mergeEvent : Nat := 39046
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨31900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events152.exact39040RawTerms
def group : MergeGroup := .relation 39042
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39042) (rhsResult := 39040)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 39041 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32479⟩⟩]⟩) (none) 39040) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨31900⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39046

namespace LeftMerge39051
def owner : Owner := ⟨.program ⟨257⟩, ⟨33560⟩⟩
def mergeEvent : Nat := 39051
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33003⟩⟩] } }
def leftRaw : List Term := Proof.Events152.exact39047RawTerms
def rightRaw : List Term := Proof.Events151.exact38861RawTerms
def group : MergeGroup := .operator 39047 38861
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39047) (leftOrdinal := 2)
    (rightResult := 38861) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33003⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33003⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], [⟨.program ⟨257⟩, ⟨33003⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39051

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
