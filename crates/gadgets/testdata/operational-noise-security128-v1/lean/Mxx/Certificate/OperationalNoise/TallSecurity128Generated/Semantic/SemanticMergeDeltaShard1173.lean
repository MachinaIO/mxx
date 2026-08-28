import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge191997
def owner : Owner := ⟨.program ⟨257⟩, ⟨19515⟩⟩
def mergeEvent : Nat := 191997
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19512⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events749.exact191991RawTerms
def group : MergeGroup := .operator 178370 191991
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 191991) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19512⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19512⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge191997

namespace LeftMerge192118
def owner : Owner := ⟨.program ⟨257⟩, ⟨20080⟩⟩
def mergeEvent : Nat := 192118
def frameStart : Nat := 192052
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18612⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events750.exact192114RawTerms
def rightRaw : List Term := Proof.Events750.exact192112RawTerms
def group : MergeGroup := .operator 192114 192112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192114) (leftOrdinal := 0)
    (rightResult := 192112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18612⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18612⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192118

namespace LeftMerge192130
def owner : Owner := ⟨.program ⟨257⟩, ⟨20739⟩⟩
def mergeEvent : Nat := 192130
def frameStart : Nat := 192052
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20738⟩⟩] } }
def leftRaw : List Term := Proof.Events750.exact192126RawTerms
def rightRaw : List Term := Proof.Events750.exact192103RawTerms
def group : MergeGroup := .operator 192126 192103
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192126) (leftOrdinal := 0)
    (rightResult := 192103) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20738⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20738⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192130

namespace LeftMerge192131
def owner : Owner := ⟨.program ⟨257⟩, ⟨20739⟩⟩
def mergeEvent : Nat := 192131
def frameStart : Nat := 192052
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18612⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20738⟩⟩] } }
def leftRaw : List Term := Proof.Events750.exact192126RawTerms
def rightRaw : List Term := Proof.Events750.exact192103RawTerms
def group : MergeGroup := .operator 192126 192103
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192126) (leftOrdinal := 1)
    (rightResult := 192103) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18612⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20738⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18612⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20738⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge192131

namespace LeftMerge192133
def owner : Owner := ⟨.program ⟨257⟩, ⟨20739⟩⟩
def mergeEvent : Nat := 192133
def frameStart : Nat := 192052
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18612⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19887⟩⟩] } }
def rhsRaw : List Term := Proof.Events750.exact192100RawTerms
def group : MergeGroup := .relation 192132
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 192132) (rhsResult := 192100)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨18612⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20738⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20738⟩⟩) ⟨19887⟩ 192100) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19887⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨18612⟩⟩], [⟨.program ⟨257⟩, ⟨19887⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge192133

namespace LeftMerge192141
def owner : Owner := ⟨.program ⟨257⟩, ⟨18921⟩⟩
def mergeEvent : Nat := 192141
def frameStart : Nat := 192052
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18918⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events750.exact192114RawTerms
def rightRaw : List Term := Proof.Events750.exact192137RawTerms
def group : MergeGroup := .operator 192114 192137
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192114) (leftOrdinal := 0)
    (rightResult := 192137) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18918⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192141

namespace LeftMerge192158
def owner : Owner := ⟨.program ⟨257⟩, ⟨19515⟩⟩
def mergeEvent : Nat := 192158
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7199⟩⟩] } }
def rhsRaw : List Term := Proof.Events750.exact192155RawTerms
def group : MergeGroup := .relation 192157
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 192157) (rhsResult := 192155)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19512⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 192156 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19512⟩⟩]⟩) (none) 192155) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7199⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192158

namespace LeftMerge192159
def owner : Owner := ⟨.program ⟨257⟩, ⟨19515⟩⟩
def mergeEvent : Nat := 192159
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20738⟩⟩] } }
def rhsRaw : List Term := Proof.Events750.exact192155RawTerms
def group : MergeGroup := .relation 192157
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 192157) (rhsResult := 192155)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19512⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 192156 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19512⟩⟩]⟩) (none) 192155) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20738⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20738⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge192159

namespace LeftMerge192160
def owner : Owner := ⟨.program ⟨257⟩, ⟨19515⟩⟩
def mergeEvent : Nat := 192160
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18612⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19887⟩⟩] } }
def rhsRaw : List Term := Proof.Events750.exact192155RawTerms
def group : MergeGroup := .relation 192157
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 192157) (rhsResult := 192155)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19512⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 192156 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19512⟩⟩]⟩) (none) 192155) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18612⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19887⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18612⟩⟩], [⟨.program ⟨257⟩, ⟨19887⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192160

namespace LeftMerge192161
def owner : Owner := ⟨.program ⟨257⟩, ⟨19515⟩⟩
def mergeEvent : Nat := 192161
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events750.exact192155RawTerms
def group : MergeGroup := .relation 192157
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 192157) (rhsResult := 192155)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19512⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 192156 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19512⟩⟩]⟩) (none) 192155) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18918⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge192161

namespace LeftMerge192166
def owner : Owner := ⟨.program ⟨257⟩, ⟨20741⟩⟩
def mergeEvent : Nat := 192166
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20738⟩⟩] } }
def leftRaw : List Term := Proof.Events750.exact192162RawTerms
def rightRaw : List Term := Proof.Events749.exact191984RawTerms
def group : MergeGroup := .operator 192162 191984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192162) (leftOrdinal := 0)
    (rightResult := 191984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20738⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20738⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20738⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192166

namespace LeftMerge192167
def owner : Owner := ⟨.program ⟨257⟩, ⟨20741⟩⟩
def mergeEvent : Nat := 192167
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18612⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19887⟩⟩] } }
def leftRaw : List Term := Proof.Events750.exact192162RawTerms
def rightRaw : List Term := Proof.Events749.exact191984RawTerms
def group : MergeGroup := .operator 192162 191984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192162) (leftOrdinal := 2)
    (rightResult := 191984) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18612⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19887⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18612⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19887⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18612⟩⟩], [⟨.program ⟨257⟩, ⟨19887⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge192167

namespace LeftMerge192175
def owner : Owner := ⟨.program ⟨257⟩, ⟨20742⟩⟩
def mergeEvent : Nat := 192175
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩] } }
def leftRaw : List Term := Proof.Events750.exact192169RawTerms
def rightRaw : List Term := Proof.Events061.exact15862RawTerms
def group : MergeGroup := .operator 192169 15862
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192169) (leftOrdinal := 0)
    (rightResult := 15862) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7199⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7165⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192175

namespace LeftMerge192176
def owner : Owner := ⟨.program ⟨257⟩, ⟨20742⟩⟩
def mergeEvent : Nat := 192176
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩] } }
def leftRaw : List Term := Proof.Events750.exact192169RawTerms
def rightRaw : List Term := Proof.Events061.exact15862RawTerms
def group : MergeGroup := .operator 192169 15862
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192169) (leftOrdinal := 1)
    (rightResult := 15862) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7165⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge192176

namespace LeftMerge192178
def owner : Owner := ⟨.program ⟨257⟩, ⟨20742⟩⟩
def mergeEvent : Nat := 192178
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15855RawTerms
def group : MergeGroup := .relation 192177
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 192177) (rhsResult := 15855)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7165⟩⟩) ⟨7048⟩ 15855) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge192178

namespace LeftMerge192192
def owner : Owner := ⟨.program ⟨257⟩, ⟨17840⟩⟩
def mergeEvent : Nat := 192192
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17838⟩⟩] } }
def leftRaw : List Term := Proof.Events729.exact186750RawTerms
def rightRaw : List Term := Proof.Events750.exact192186RawTerms
def group : MergeGroup := .operator 186750 192186
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186750) (leftOrdinal := 0)
    (rightResult := 192186) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17838⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17838⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192192

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
