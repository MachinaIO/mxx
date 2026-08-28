import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge111974
def owner : Owner := ⟨.program ⟨257⟩, ⟨31520⟩⟩
def mergeEvent : Nat := 111974
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }
def leftRaw : List Term := Proof.Events437.exact111970RawTerms
def rightRaw : List Term := Proof.Events437.exact111940RawTerms
def group : MergeGroup := .operator 111970 111940
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111970) (leftOrdinal := 1)
    (rightResult := 111940) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111974

namespace LeftMerge111982
def owner : Owner := ⟨.program ⟨257⟩, ⟨33471⟩⟩
def mergeEvent : Nat := 111982
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24302⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33470⟩⟩] } }
def leftRaw : List Term := Proof.Events437.exact111976RawTerms
def rightRaw : List Term := Proof.Events437.exact111912RawTerms
def group : MergeGroup := .operator 111976 111912
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111976) (leftOrdinal := 1)
    (rightResult := 111912) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24302⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33470⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24302⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33470⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge111982

namespace LeftMerge111984
def owner : Owner := ⟨.program ⟨257⟩, ⟨33471⟩⟩
def mergeEvent : Nat := 111984
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24302⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32955⟩⟩] } }
def rhsRaw : List Term := Proof.Events437.exact111909RawTerms
def group : MergeGroup := .relation 111983
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 111983) (rhsResult := 111909)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24302⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33470⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33470⟩⟩) ⟨32955⟩ 111909) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32955⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24302⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], [⟨.program ⟨257⟩, ⟨32955⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge111984

namespace LeftMerge111985
def owner : Owner := ⟨.program ⟨257⟩, ⟨33471⟩⟩
def mergeEvent : Nat := 111985
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33470⟩⟩] } }
def leftRaw : List Term := Proof.Events437.exact111976RawTerms
def rightRaw : List Term := Proof.Events437.exact111912RawTerms
def group : MergeGroup := .operator 111976 111912
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111976) (leftOrdinal := 0)
    (rightResult := 111912) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33470⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33470⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111985

namespace LeftMerge111999
def owner : Owner := ⟨.program ⟨257⟩, ⟨32402⟩⟩
def mergeEvent : Nat := 111999
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32399⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105245RawTerms
def rightRaw : List Term := Proof.Events437.exact111993RawTerms
def group : MergeGroup := .operator 105245 111993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105245) (leftOrdinal := 0)
    (rightResult := 111993) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32399⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32399⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111999

namespace LeftMerge112078
def owner : Owner := ⟨.program ⟨257⟩, ⟨31513⟩⟩
def mergeEvent : Nat := 112078
def frameStart : Nat := 112048
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24302⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events437.exact112074RawTerms
def rightRaw : List Term := Proof.Events437.exact112071RawTerms
def group : MergeGroup := .operator 112074 112071
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 112074) (leftOrdinal := 0)
    (rightResult := 112071) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31512⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24302⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24302⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge112078

namespace LeftMerge112108
def owner : Owner := ⟨.program ⟨257⟩, ⟨33232⟩⟩
def mergeEvent : Nat := 112108
def frameStart : Nat := 112048
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24302⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events437.exact112104RawTerms
def rightRaw : List Term := Proof.Events437.exact112102RawTerms
def group : MergeGroup := .operator 112104 112102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 112104) (leftOrdinal := 0)
    (rightResult := 112102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24302⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24302⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge112108

namespace LeftMerge112131
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def mergeEvent : Nat := 112131
def frameStart : Nat := 112048
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }
def leftRaw : List Term := Proof.Events437.exact112127RawTerms
def rightRaw : List Term := Proof.Events437.exact112124RawTerms
def group : MergeGroup := .operator 112127 112124
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 112127) (leftOrdinal := 0)
    (rightResult := 112124) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9577⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge112131

namespace LeftMerge112140
def owner : Owner := ⟨.program ⟨257⟩, ⟨33473⟩⟩
def mergeEvent : Nat := 112140
def frameStart : Nat := 112048
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33470⟩⟩] } }
def leftRaw : List Term := Proof.Events438.exact112136RawTerms
def rightRaw : List Term := Proof.Events437.exact112093RawTerms
def group : MergeGroup := .operator 112136 112093
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 112136) (leftOrdinal := 0)
    (rightResult := 112093) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33470⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33470⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge112140

namespace LeftMerge112141
def owner : Owner := ⟨.program ⟨257⟩, ⟨33473⟩⟩
def mergeEvent : Nat := 112141
def frameStart : Nat := 112048
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24302⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33470⟩⟩] } }
def leftRaw : List Term := Proof.Events438.exact112136RawTerms
def rightRaw : List Term := Proof.Events437.exact112093RawTerms
def group : MergeGroup := .operator 112136 112093
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 112136) (leftOrdinal := 1)
    (rightResult := 112093) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24302⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33470⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24302⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33470⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge112141

namespace LeftMerge112143
def owner : Owner := ⟨.program ⟨257⟩, ⟨33473⟩⟩
def mergeEvent : Nat := 112143
def frameStart : Nat := 112048
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24302⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32955⟩⟩] } }
def rhsRaw : List Term := Proof.Events437.exact112090RawTerms
def group : MergeGroup := .relation 112142
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 112142) (rhsResult := 112090)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24302⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33470⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33470⟩⟩) ⟨32955⟩ 112090) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32955⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24302⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], [⟨.program ⟨257⟩, ⟨32955⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge112143

namespace LeftMerge112151
def owner : Owner := ⟨.program ⟨257⟩, ⟨31838⟩⟩
def mergeEvent : Nat := 112151
def frameStart : Nat := 112048
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events437.exact112104RawTerms
def rightRaw : List Term := Proof.Events438.exact112147RawTerms
def group : MergeGroup := .operator 112104 112147
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 112104) (leftOrdinal := 0)
    (rightResult := 112147) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31836⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge112151

namespace LeftMerge112168
def owner : Owner := ⟨.program ⟨257⟩, ⟨32402⟩⟩
def mergeEvent : Nat := 112168
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }
def rhsRaw : List Term := Proof.Events438.exact112165RawTerms
def group : MergeGroup := .relation 112167
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 112167) (rhsResult := 112165)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32399⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 112166 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32399⟩⟩]⟩) (none) 112165) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge112168

namespace LeftMerge112169
def owner : Owner := ⟨.program ⟨257⟩, ⟨32402⟩⟩
def mergeEvent : Nat := 112169
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33470⟩⟩] } }
def rhsRaw : List Term := Proof.Events438.exact112165RawTerms
def group : MergeGroup := .relation 112167
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 112167) (rhsResult := 112165)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32399⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 112166 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32399⟩⟩]⟩) (none) 112165) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33470⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33470⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge112169

namespace LeftMerge112170
def owner : Owner := ⟨.program ⟨257⟩, ⟨32402⟩⟩
def mergeEvent : Nat := 112170
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24302⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32955⟩⟩] } }
def rhsRaw : List Term := Proof.Events438.exact112165RawTerms
def group : MergeGroup := .relation 112167
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 112167) (rhsResult := 112165)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32399⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 112166 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32399⟩⟩]⟩) (none) 112165) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24302⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32955⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24302⟩⟩, ⟨.program ⟨257⟩, ⟨31512⟩⟩], [⟨.program ⟨257⟩, ⟨32955⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge112170

namespace LeftMerge112171
def owner : Owner := ⟨.program ⟨257⟩, ⟨32402⟩⟩
def mergeEvent : Nat := 112171
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨31836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events438.exact112165RawTerms
def group : MergeGroup := .relation 112167
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 112167) (rhsResult := 112165)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32399⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 112166 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32399⟩⟩]⟩) (none) 112165) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨31836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge112171

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
