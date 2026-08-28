import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge285180
def owner : Owner := ⟨.program ⟨257⟩, ⟨64184⟩⟩
def mergeEvent : Nat := 285180
def frameStart : Nat := 285120
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25418⟩⟩, ⟨.program ⟨257⟩, ⟨62303⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1113.exact285176RawTerms
def rightRaw : List Term := Proof.Events1113.exact285174RawTerms
def group : MergeGroup := .operator 285176 285174
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285176) (leftOrdinal := 0)
    (rightResult := 285174) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25418⟩⟩, ⟨.program ⟨257⟩, ⟨62303⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25418⟩⟩, ⟨.program ⟨257⟩, ⟨62303⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285180

namespace LeftMerge285201
def owner : Owner := ⟨.program ⟨257⟩, ⟨9540⟩⟩
def mergeEvent : Nat := 285201
def frameStart : Nat := 285120
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }
def leftRaw : List Term := Proof.Events1114.exact285197RawTerms
def rightRaw : List Term := Proof.Events1114.exact285194RawTerms
def group : MergeGroup := .operator 285197 285194
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285197) (leftOrdinal := 0)
    (rightResult := 285194) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9538⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285201

namespace LeftMerge285210
def owner : Owner := ⟨.program ⟨257⟩, ⟨64376⟩⟩
def mergeEvent : Nat := 285210
def frameStart : Nat := 285120
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64373⟩⟩] } }
def leftRaw : List Term := Proof.Events1114.exact285206RawTerms
def rightRaw : List Term := Proof.Events1113.exact285165RawTerms
def group : MergeGroup := .operator 285206 285165
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285206) (leftOrdinal := 0)
    (rightResult := 285165) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64373⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64373⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285210

namespace LeftMerge285211
def owner : Owner := ⟨.program ⟨257⟩, ⟨64376⟩⟩
def mergeEvent : Nat := 285211
def frameStart : Nat := 285120
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25418⟩⟩, ⟨.program ⟨257⟩, ⟨62303⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64373⟩⟩] } }
def leftRaw : List Term := Proof.Events1114.exact285206RawTerms
def rightRaw : List Term := Proof.Events1113.exact285165RawTerms
def group : MergeGroup := .operator 285206 285165
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285206) (leftOrdinal := 1)
    (rightResult := 285165) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25418⟩⟩, ⟨.program ⟨257⟩, ⟨62303⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64373⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25418⟩⟩, ⟨.program ⟨257⟩, ⟨62303⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64373⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge285211

namespace LeftMerge285213
def owner : Owner := ⟨.program ⟨257⟩, ⟨64376⟩⟩
def mergeEvent : Nat := 285213
def frameStart : Nat := 285120
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25418⟩⟩, ⟨.program ⟨257⟩, ⟨62303⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63893⟩⟩] } }
def rhsRaw : List Term := Proof.Events1113.exact285162RawTerms
def group : MergeGroup := .relation 285212
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 285212) (rhsResult := 285162)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25418⟩⟩, ⟨.program ⟨257⟩, ⟨62303⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64373⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64373⟩⟩) ⟨63893⟩ 285162) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63893⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25418⟩⟩, ⟨.program ⟨257⟩, ⟨62303⟩⟩], [⟨.program ⟨257⟩, ⟨63893⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge285213

namespace LeftMerge285221
def owner : Owner := ⟨.program ⟨257⟩, ⟨62762⟩⟩
def mergeEvent : Nat := 285221
def frameStart : Nat := 285120
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1113.exact285176RawTerms
def rightRaw : List Term := Proof.Events1114.exact285217RawTerms
def group : MergeGroup := .operator 285176 285217
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285176) (leftOrdinal := 0)
    (rightResult := 285217) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62760⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285221

namespace LeftMerge285238
def owner : Owner := ⟨.program ⟨257⟩, ⟨63312⟩⟩
def mergeEvent : Nat := 285238
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }
def rhsRaw : List Term := Proof.Events1114.exact285235RawTerms
def group : MergeGroup := .relation 285237
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 285237) (rhsResult := 285235)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63309⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 285236 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63309⟩⟩]⟩) (none) 285235) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285238

namespace LeftMerge285239
def owner : Owner := ⟨.program ⟨257⟩, ⟨63312⟩⟩
def mergeEvent : Nat := 285239
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64373⟩⟩] } }
def rhsRaw : List Term := Proof.Events1114.exact285235RawTerms
def group : MergeGroup := .relation 285237
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 285237) (rhsResult := 285235)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63309⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 285236 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63309⟩⟩]⟩) (none) 285235) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64373⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64373⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge285239

namespace LeftMerge285240
def owner : Owner := ⟨.program ⟨257⟩, ⟨63312⟩⟩
def mergeEvent : Nat := 285240
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25418⟩⟩, ⟨.program ⟨257⟩, ⟨62303⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63893⟩⟩] } }
def rhsRaw : List Term := Proof.Events1114.exact285235RawTerms
def group : MergeGroup := .relation 285237
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 285237) (rhsResult := 285235)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63309⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 285236 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63309⟩⟩]⟩) (none) 285235) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25418⟩⟩, ⟨.program ⟨257⟩, ⟨62303⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63893⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25418⟩⟩, ⟨.program ⟨257⟩, ⟨62303⟩⟩], [⟨.program ⟨257⟩, ⟨63893⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285240

namespace LeftMerge285241
def owner : Owner := ⟨.program ⟨257⟩, ⟨63312⟩⟩
def mergeEvent : Nat := 285241
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1114.exact285235RawTerms
def group : MergeGroup := .relation 285237
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 285237) (rhsResult := 285235)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63309⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 285236 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63309⟩⟩]⟩) (none) 285235) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge285241

namespace LeftMerge285246
def owner : Owner := ⟨.program ⟨257⟩, ⟨64375⟩⟩
def mergeEvent : Nat := 285246
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25418⟩⟩, ⟨.program ⟨257⟩, ⟨62303⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63893⟩⟩] } }
def leftRaw : List Term := Proof.Events1114.exact285242RawTerms
def rightRaw : List Term := Proof.Events1113.exact285058RawTerms
def group : MergeGroup := .operator 285242 285058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285242) (leftOrdinal := 2)
    (rightResult := 285058) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25418⟩⟩, ⟨.program ⟨257⟩, ⟨62303⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63893⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25418⟩⟩, ⟨.program ⟨257⟩, ⟨62303⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63893⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25418⟩⟩, ⟨.program ⟨257⟩, ⟨62303⟩⟩], [⟨.program ⟨257⟩, ⟨63893⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge285246

namespace LeftMerge285247
def owner : Owner := ⟨.program ⟨257⟩, ⟨64375⟩⟩
def mergeEvent : Nat := 285247
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64373⟩⟩] } }
def leftRaw : List Term := Proof.Events1114.exact285242RawTerms
def rightRaw : List Term := Proof.Events1113.exact285058RawTerms
def group : MergeGroup := .operator 285242 285058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285242) (leftOrdinal := 1)
    (rightResult := 285058) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64373⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64373⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64373⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285247

namespace LeftMerge285255
def owner : Owner := ⟨.program ⟨257⟩, ⟨64688⟩⟩
def mergeEvent : Nat := 285255
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64686⟩⟩] } }
def leftRaw : List Term := Proof.Events1114.exact285249RawTerms
def rightRaw : List Term := Proof.Events1113.exact284974RawTerms
def group : MergeGroup := .operator 285249 284974
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285249) (leftOrdinal := 0)
    (rightResult := 284974) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64686⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64686⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285255

namespace LeftMerge285256
def owner : Owner := ⟨.program ⟨257⟩, ⟨64688⟩⟩
def mergeEvent : Nat := 285256
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64686⟩⟩] } }
def leftRaw : List Term := Proof.Events1114.exact285249RawTerms
def rightRaw : List Term := Proof.Events1113.exact284974RawTerms
def group : MergeGroup := .operator 285249 284974
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285249) (leftOrdinal := 1)
    (rightResult := 284974) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64686⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64686⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge285256

namespace LeftMerge285258
def owner : Owner := ⟨.program ⟨257⟩, ⟨64688⟩⟩
def mergeEvent : Nat := 285258
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64027⟩⟩] } }
def rhsRaw : List Term := Proof.Events1113.exact284971RawTerms
def group : MergeGroup := .relation 285257
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 285257) (rhsResult := 284971)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64686⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64686⟩⟩) ⟨64027⟩ 284971) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64027⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62760⟩⟩], [⟨.program ⟨257⟩, ⟨64027⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge285258

namespace LeftMerge285272
def owner : Owner := ⟨.program ⟨257⟩, ⟨63559⟩⟩
def mergeEvent : Nat := 285272
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63556⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280745RawTerms
def rightRaw : List Term := Proof.Events1114.exact285266RawTerms
def group : MergeGroup := .operator 280745 285266
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280745) (leftOrdinal := 0)
    (rightResult := 285266) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63556⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285272

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
