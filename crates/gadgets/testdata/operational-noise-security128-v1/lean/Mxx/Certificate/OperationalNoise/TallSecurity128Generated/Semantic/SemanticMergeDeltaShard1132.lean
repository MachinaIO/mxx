import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge185094
def owner : Owner := ⟨.program ⟨257⟩, ⟨31573⟩⟩
def mergeEvent : Nat := 185094
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }
def leftRaw : List Term := Proof.Events722.exact185085RawTerms
def rightRaw : List Term := Proof.Events094.exact24124RawTerms
def group : MergeGroup := .operator 185085 24124
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185085) (leftOrdinal := 0)
    (rightResult := 24124) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9577⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185094

namespace LeftMerge185099
def owner : Owner := ⟨.program ⟨257⟩, ⟨31574⟩⟩
def mergeEvent : Nat := 185099
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }
def leftRaw : List Term := Proof.Events723.exact185095RawTerms
def rightRaw : List Term := Proof.Events722.exact185065RawTerms
def group : MergeGroup := .operator 185095 185065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185095) (leftOrdinal := 1)
    (rightResult := 185065) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185099

namespace LeftMerge185107
def owner : Owner := ⟨.program ⟨257⟩, ⟨33493⟩⟩
def mergeEvent : Nat := 185107
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33492⟩⟩] } }
def leftRaw : List Term := Proof.Events723.exact185101RawTerms
def rightRaw : List Term := Proof.Events722.exact185037RawTerms
def group : MergeGroup := .operator 185101 185037
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185101) (leftOrdinal := 1)
    (rightResult := 185037) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33492⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33492⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185107

namespace LeftMerge185109
def owner : Owner := ⟨.program ⟨257⟩, ⟨33493⟩⟩
def mergeEvent : Nat := 185109
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32967⟩⟩] } }
def rhsRaw : List Term := Proof.Events722.exact185034RawTerms
def group : MergeGroup := .relation 185108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185108) (rhsResult := 185034)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33492⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33492⟩⟩) ⟨32967⟩ 185034) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32967⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], [⟨.program ⟨257⟩, ⟨32967⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185109

namespace LeftMerge185110
def owner : Owner := ⟨.program ⟨257⟩, ⟨33493⟩⟩
def mergeEvent : Nat := 185110
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33492⟩⟩] } }
def leftRaw : List Term := Proof.Events723.exact185101RawTerms
def rightRaw : List Term := Proof.Events722.exact185037RawTerms
def group : MergeGroup := .operator 185101 185037
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185101) (leftOrdinal := 0)
    (rightResult := 185037) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33492⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33492⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185110

namespace LeftMerge185124
def owner : Owner := ⟨.program ⟨257⟩, ⟨32422⟩⟩
def mergeEvent : Nat := 185124
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32419⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events723.exact185118RawTerms
def group : MergeGroup := .operator 178370 185118
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 185118) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32419⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32419⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185124

namespace LeftMerge185203
def owner : Owner := ⟨.program ⟨257⟩, ⟨31567⟩⟩
def mergeEvent : Nat := 185203
def frameStart : Nat := 185173
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events723.exact185199RawTerms
def rightRaw : List Term := Proof.Events723.exact185196RawTerms
def group : MergeGroup := .operator 185199 185196
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185199) (leftOrdinal := 0)
    (rightResult := 185196) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31566⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24326⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185203

namespace LeftMerge185233
def owner : Owner := ⟨.program ⟨257⟩, ⟨33240⟩⟩
def mergeEvent : Nat := 185233
def frameStart : Nat := 185173
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events723.exact185229RawTerms
def rightRaw : List Term := Proof.Events723.exact185227RawTerms
def group : MergeGroup := .operator 185229 185227
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185229) (leftOrdinal := 0)
    (rightResult := 185227) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185233

namespace LeftMerge185256
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def mergeEvent : Nat := 185256
def frameStart : Nat := 185173
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }
def leftRaw : List Term := Proof.Events723.exact185252RawTerms
def rightRaw : List Term := Proof.Events723.exact185249RawTerms
def group : MergeGroup := .operator 185252 185249
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185252) (leftOrdinal := 0)
    (rightResult := 185249) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9577⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185256

namespace LeftMerge185265
def owner : Owner := ⟨.program ⟨257⟩, ⟨33495⟩⟩
def mergeEvent : Nat := 185265
def frameStart : Nat := 185173
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33492⟩⟩] } }
def leftRaw : List Term := Proof.Events723.exact185261RawTerms
def rightRaw : List Term := Proof.Events723.exact185218RawTerms
def group : MergeGroup := .operator 185261 185218
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185261) (leftOrdinal := 0)
    (rightResult := 185218) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33492⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33492⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185265

namespace LeftMerge185266
def owner : Owner := ⟨.program ⟨257⟩, ⟨33495⟩⟩
def mergeEvent : Nat := 185266
def frameStart : Nat := 185173
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33492⟩⟩] } }
def leftRaw : List Term := Proof.Events723.exact185261RawTerms
def rightRaw : List Term := Proof.Events723.exact185218RawTerms
def group : MergeGroup := .operator 185261 185218
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185261) (leftOrdinal := 1)
    (rightResult := 185218) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33492⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33492⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185266

namespace LeftMerge185268
def owner : Owner := ⟨.program ⟨257⟩, ⟨33495⟩⟩
def mergeEvent : Nat := 185268
def frameStart : Nat := 185173
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32967⟩⟩] } }
def rhsRaw : List Term := Proof.Events723.exact185215RawTerms
def group : MergeGroup := .relation 185267
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185267) (rhsResult := 185215)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33492⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33492⟩⟩) ⟨32967⟩ 185215) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32967⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], [⟨.program ⟨257⟩, ⟨32967⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185268

namespace LeftMerge185276
def owner : Owner := ⟨.program ⟨257⟩, ⟨31854⟩⟩
def mergeEvent : Nat := 185276
def frameStart : Nat := 185173
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events723.exact185229RawTerms
def rightRaw : List Term := Proof.Events723.exact185272RawTerms
def group : MergeGroup := .operator 185229 185272
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185229) (leftOrdinal := 0)
    (rightResult := 185272) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31852⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185276

namespace LeftMerge185293
def owner : Owner := ⟨.program ⟨257⟩, ⟨32422⟩⟩
def mergeEvent : Nat := 185293
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }
def rhsRaw : List Term := Proof.Events723.exact185290RawTerms
def group : MergeGroup := .relation 185292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185292) (rhsResult := 185290)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 185291 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32419⟩⟩]⟩) (none) 185290) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185293

namespace LeftMerge185294
def owner : Owner := ⟨.program ⟨257⟩, ⟨32422⟩⟩
def mergeEvent : Nat := 185294
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33492⟩⟩] } }
def rhsRaw : List Term := Proof.Events723.exact185290RawTerms
def group : MergeGroup := .relation 185292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185292) (rhsResult := 185290)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 185291 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32419⟩⟩]⟩) (none) 185290) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33492⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33492⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185294

namespace LeftMerge185295
def owner : Owner := ⟨.program ⟨257⟩, ⟨32422⟩⟩
def mergeEvent : Nat := 185295
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32967⟩⟩] } }
def rhsRaw : List Term := Proof.Events723.exact185290RawTerms
def group : MergeGroup := .relation 185292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185292) (rhsResult := 185290)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 185291 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32419⟩⟩]⟩) (none) 185290) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32967⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24326⟩⟩, ⟨.program ⟨257⟩, ⟨31566⟩⟩], [⟨.program ⟨257⟩, ⟨32967⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185295

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
