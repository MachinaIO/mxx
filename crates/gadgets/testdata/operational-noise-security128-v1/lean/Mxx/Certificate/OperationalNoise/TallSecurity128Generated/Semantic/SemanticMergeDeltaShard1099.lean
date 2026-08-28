import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge179036
def owner : Owner := ⟨.program ⟨257⟩, ⟨47014⟩⟩
def mergeEvent : Nat := 179036
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47012⟩⟩] } }
def leftRaw : List Term := Proof.Events699.exact179031RawTerms
def rightRaw : List Term := Proof.Events698.exact178845RawTerms
def group : MergeGroup := .operator 179031 178845
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 179031) (leftOrdinal := 1)
    (rightResult := 178845) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47012⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47012⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47012⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge179036

namespace LeftMerge179044
def owner : Owner := ⟨.program ⟨257⟩, ⟨47426⟩⟩
def mergeEvent : Nat := 179044
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47424⟩⟩] } }
def leftRaw : List Term := Proof.Events699.exact179038RawTerms
def rightRaw : List Term := Proof.Events698.exact178761RawTerms
def group : MergeGroup := .operator 179038 178761
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 179038) (leftOrdinal := 0)
    (rightResult := 178761) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47424⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47424⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge179044

namespace LeftMerge179045
def owner : Owner := ⟨.program ⟨257⟩, ⟨47426⟩⟩
def mergeEvent : Nat := 179045
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47424⟩⟩] } }
def leftRaw : List Term := Proof.Events699.exact179038RawTerms
def rightRaw : List Term := Proof.Events698.exact178761RawTerms
def group : MergeGroup := .operator 179038 178761
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 179038) (leftOrdinal := 1)
    (rightResult := 178761) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47424⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47424⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge179045

namespace LeftMerge179047
def owner : Owner := ⟨.program ⟨257⟩, ⟨47426⟩⟩
def mergeEvent : Nat := 179047
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46648⟩⟩] } }
def rhsRaw : List Term := Proof.Events698.exact178758RawTerms
def group : MergeGroup := .relation 179046
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 179046) (rhsResult := 178758)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47424⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47424⟩⟩) ⟨46648⟩ 178758) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46648⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], [⟨.program ⟨257⟩, ⟨46648⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge179047

namespace LeftMerge179061
def owner : Owner := ⟨.program ⟨257⟩, ⟨46279⟩⟩
def mergeEvent : Nat := 179061
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46276⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events699.exact179055RawTerms
def group : MergeGroup := .operator 178370 179055
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 179055) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46276⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46276⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge179061

namespace LeftMerge179182
def owner : Owner := ⟨.program ⟨257⟩, ⟨46840⟩⟩
def mergeEvent : Nat := 179182
def frameStart : Nat := 179116
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events699.exact179178RawTerms
def rightRaw : List Term := Proof.Events699.exact179176RawTerms
def group : MergeGroup := .operator 179178 179176
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 179178) (leftOrdinal := 0)
    (rightResult := 179176) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge179182

namespace LeftMerge179194
def owner : Owner := ⟨.program ⟨257⟩, ⟨47425⟩⟩
def mergeEvent : Nat := 179194
def frameStart : Nat := 179116
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47424⟩⟩] } }
def leftRaw : List Term := Proof.Events699.exact179190RawTerms
def rightRaw : List Term := Proof.Events699.exact179167RawTerms
def group : MergeGroup := .operator 179190 179167
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 179190) (leftOrdinal := 0)
    (rightResult := 179167) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47424⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47424⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge179194

namespace LeftMerge179195
def owner : Owner := ⟨.program ⟨257⟩, ⟨47425⟩⟩
def mergeEvent : Nat := 179195
def frameStart : Nat := 179116
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47424⟩⟩] } }
def leftRaw : List Term := Proof.Events699.exact179190RawTerms
def rightRaw : List Term := Proof.Events699.exact179167RawTerms
def group : MergeGroup := .operator 179190 179167
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 179190) (leftOrdinal := 1)
    (rightResult := 179167) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47424⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47424⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge179195

namespace LeftMerge179197
def owner : Owner := ⟨.program ⟨257⟩, ⟨47425⟩⟩
def mergeEvent : Nat := 179197
def frameStart : Nat := 179116
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46648⟩⟩] } }
def rhsRaw : List Term := Proof.Events699.exact179164RawTerms
def group : MergeGroup := .relation 179196
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 179196) (rhsResult := 179164)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47424⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47424⟩⟩) ⟨46648⟩ 179164) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46648⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45492⟩⟩], [⟨.program ⟨257⟩, ⟨46648⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge179197

namespace LeftMerge179205
def owner : Owner := ⟨.program ⟨257⟩, ⟨45723⟩⟩
def mergeEvent : Nat := 179205
def frameStart : Nat := 179116
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events699.exact179178RawTerms
def rightRaw : List Term := Proof.Events700.exact179201RawTerms
def group : MergeGroup := .operator 179178 179201
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 179178) (leftOrdinal := 0)
    (rightResult := 179201) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45722⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge179205

namespace LeftMerge179222
def owner : Owner := ⟨.program ⟨257⟩, ⟨46279⟩⟩
def mergeEvent : Nat := 179222
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }
def rhsRaw : List Term := Proof.Events700.exact179219RawTerms
def group : MergeGroup := .relation 179221
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 179221) (rhsResult := 179219)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46276⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 179220 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46276⟩⟩]⟩) (none) 179219) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge179222

namespace LeftMerge179223
def owner : Owner := ⟨.program ⟨257⟩, ⟨46279⟩⟩
def mergeEvent : Nat := 179223
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47424⟩⟩] } }
def rhsRaw : List Term := Proof.Events700.exact179219RawTerms
def group : MergeGroup := .relation 179221
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 179221) (rhsResult := 179219)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46276⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 179220 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46276⟩⟩]⟩) (none) 179219) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47424⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47424⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge179223

namespace LeftMerge179224
def owner : Owner := ⟨.program ⟨257⟩, ⟨46279⟩⟩
def mergeEvent : Nat := 179224
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46648⟩⟩] } }
def rhsRaw : List Term := Proof.Events700.exact179219RawTerms
def group : MergeGroup := .relation 179221
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 179221) (rhsResult := 179219)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46276⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 179220 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46276⟩⟩]⟩) (none) 179219) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46648⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], [⟨.program ⟨257⟩, ⟨46648⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge179224

namespace LeftMerge179225
def owner : Owner := ⟨.program ⟨257⟩, ⟨46279⟩⟩
def mergeEvent : Nat := 179225
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events700.exact179219RawTerms
def group : MergeGroup := .relation 179221
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 179221) (rhsResult := 179219)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46276⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 179220 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46276⟩⟩]⟩) (none) 179219) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge179225

namespace LeftMerge179230
def owner : Owner := ⟨.program ⟨257⟩, ⟨47427⟩⟩
def mergeEvent : Nat := 179230
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47424⟩⟩] } }
def leftRaw : List Term := Proof.Events700.exact179226RawTerms
def rightRaw : List Term := Proof.Events699.exact179048RawTerms
def group : MergeGroup := .operator 179226 179048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 179226) (leftOrdinal := 0)
    (rightResult := 179048) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47424⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47424⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47424⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge179230

namespace LeftMerge179231
def owner : Owner := ⟨.program ⟨257⟩, ⟨47427⟩⟩
def mergeEvent : Nat := 179231
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46648⟩⟩] } }
def leftRaw : List Term := Proof.Events700.exact179226RawTerms
def rightRaw : List Term := Proof.Events699.exact179048RawTerms
def group : MergeGroup := .operator 179226 179048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 179226) (leftOrdinal := 2)
    (rightResult := 179048) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46648⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46648⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], [⟨.program ⟨257⟩, ⟨46648⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge179231

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
