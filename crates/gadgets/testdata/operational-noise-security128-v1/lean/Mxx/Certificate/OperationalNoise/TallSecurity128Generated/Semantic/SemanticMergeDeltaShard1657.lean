import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge268133
def owner : Owner := ⟨.program ⟨257⟩, ⟨36915⟩⟩
def mergeEvent : Nat := 268133
def frameStart : Nat := 268103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1047.exact268129RawTerms
def rightRaw : List Term := Proof.Events1047.exact268126RawTerms
def group : MergeGroup := .operator 268129 268126
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 268129) (leftOrdinal := 0)
    (rightResult := 268126) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13756⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨36914⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge268133

namespace LeftMerge268163
def owner : Owner := ⟨.program ⟨257⟩, ⟨38676⟩⟩
def mergeEvent : Nat := 268163
def frameStart : Nat := 268103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1047.exact268159RawTerms
def rightRaw : List Term := Proof.Events1047.exact268157RawTerms
def group : MergeGroup := .operator 268159 268157
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 268159) (leftOrdinal := 0)
    (rightResult := 268157) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge268163

namespace LeftMerge268186
def owner : Owner := ⟨.program ⟨257⟩, ⟨9555⟩⟩
def mergeEvent : Nat := 268186
def frameStart : Nat := 268103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }
def leftRaw : List Term := Proof.Events1047.exact268182RawTerms
def rightRaw : List Term := Proof.Events1047.exact268179RawTerms
def group : MergeGroup := .operator 268182 268179
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 268182) (leftOrdinal := 0)
    (rightResult := 268179) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9553⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge268186

namespace LeftMerge268195
def owner : Owner := ⟨.program ⟨257⟩, ⟨38851⟩⟩
def mergeEvent : Nat := 268195
def frameStart : Nat := 268103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38848⟩⟩] } }
def leftRaw : List Term := Proof.Events1047.exact268191RawTerms
def rightRaw : List Term := Proof.Events1047.exact268148RawTerms
def group : MergeGroup := .operator 268191 268148
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 268191) (leftOrdinal := 0)
    (rightResult := 268148) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38848⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38848⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge268195

namespace LeftMerge268196
def owner : Owner := ⟨.program ⟨257⟩, ⟨38851⟩⟩
def mergeEvent : Nat := 268196
def frameStart : Nat := 268103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38848⟩⟩] } }
def leftRaw : List Term := Proof.Events1047.exact268191RawTerms
def rightRaw : List Term := Proof.Events1047.exact268148RawTerms
def group : MergeGroup := .operator 268191 268148
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 268191) (leftOrdinal := 1)
    (rightResult := 268148) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38848⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge268196

namespace LeftMerge268198
def owner : Owner := ⟨.program ⟨257⟩, ⟨38851⟩⟩
def mergeEvent : Nat := 268198
def frameStart : Nat := 268103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38379⟩⟩] } }
def rhsRaw : List Term := Proof.Events1047.exact268145RawTerms
def group : MergeGroup := .relation 268197
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 268197) (rhsResult := 268145)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38848⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38848⟩⟩) ⟨38379⟩ 268145) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38379⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], [⟨.program ⟨257⟩, ⟨38379⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge268198

namespace LeftMerge268206
def owner : Owner := ⟨.program ⟨257⟩, ⟨37364⟩⟩
def mergeEvent : Nat := 268206
def frameStart : Nat := 268103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37362⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1047.exact268159RawTerms
def rightRaw : List Term := Proof.Events1047.exact268202RawTerms
def group : MergeGroup := .operator 268159 268202
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 268159) (leftOrdinal := 0)
    (rightResult := 268202) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37362⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37362⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge268206

namespace LeftMerge268223
def owner : Owner := ⟨.program ⟨257⟩, ⟨37789⟩⟩
def mergeEvent : Nat := 268223
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }
def rhsRaw : List Term := Proof.Events1047.exact268220RawTerms
def group : MergeGroup := .relation 268222
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 268222) (rhsResult := 268220)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37786⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 268221 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37786⟩⟩]⟩) (none) 268220) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge268223

namespace LeftMerge268224
def owner : Owner := ⟨.program ⟨257⟩, ⟨37789⟩⟩
def mergeEvent : Nat := 268224
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38848⟩⟩] } }
def rhsRaw : List Term := Proof.Events1047.exact268220RawTerms
def group : MergeGroup := .relation 268222
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 268222) (rhsResult := 268220)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37786⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 268221 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37786⟩⟩]⟩) (none) 268220) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge268224

namespace LeftMerge268225
def owner : Owner := ⟨.program ⟨257⟩, ⟨37789⟩⟩
def mergeEvent : Nat := 268225
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38379⟩⟩] } }
def rhsRaw : List Term := Proof.Events1047.exact268220RawTerms
def group : MergeGroup := .relation 268222
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 268222) (rhsResult := 268220)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37786⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 268221 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37786⟩⟩]⟩) (none) 268220) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38379⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], [⟨.program ⟨257⟩, ⟨38379⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge268225

namespace LeftMerge268226
def owner : Owner := ⟨.program ⟨257⟩, ⟨37789⟩⟩
def mergeEvent : Nat := 268226
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37362⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1047.exact268220RawTerms
def group : MergeGroup := .relation 268222
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 268222) (rhsResult := 268220)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37786⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 268221 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37786⟩⟩]⟩) (none) 268220) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37362⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37362⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge268226

namespace LeftMerge268231
def owner : Owner := ⟨.program ⟨257⟩, ⟨38850⟩⟩
def mergeEvent : Nat := 268231
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38379⟩⟩] } }
def leftRaw : List Term := Proof.Events1047.exact268227RawTerms
def rightRaw : List Term := Proof.Events1047.exact268041RawTerms
def group : MergeGroup := .operator 268227 268041
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 268227) (leftOrdinal := 2)
    (rightResult := 268041) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38379⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38379⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13756⟩⟩, ⟨.program ⟨257⟩, ⟨36914⟩⟩], [⟨.program ⟨257⟩, ⟨38379⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge268231

namespace LeftMerge268232
def owner : Owner := ⟨.program ⟨257⟩, ⟨38850⟩⟩
def mergeEvent : Nat := 268232
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38848⟩⟩] } }
def leftRaw : List Term := Proof.Events1047.exact268227RawTerms
def rightRaw : List Term := Proof.Events1047.exact268041RawTerms
def group : MergeGroup := .operator 268227 268041
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 268227) (leftOrdinal := 1)
    (rightResult := 268041) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38848⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38848⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38848⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge268232

namespace LeftMerge268240
def owner : Owner := ⟨.program ⟨257⟩, ⟨39104⟩⟩
def mergeEvent : Nat := 268240
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39102⟩⟩] } }
def leftRaw : List Term := Proof.Events1047.exact268234RawTerms
def rightRaw : List Term := Proof.Events1046.exact267957RawTerms
def group : MergeGroup := .operator 268234 267957
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 268234) (leftOrdinal := 0)
    (rightResult := 267957) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39102⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39102⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge268240

namespace LeftMerge268241
def owner : Owner := ⟨.program ⟨257⟩, ⟨39104⟩⟩
def mergeEvent : Nat := 268241
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37362⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39102⟩⟩] } }
def leftRaw : List Term := Proof.Events1047.exact268234RawTerms
def rightRaw : List Term := Proof.Events1046.exact267957RawTerms
def group : MergeGroup := .operator 268234 267957
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 268234) (leftOrdinal := 1)
    (rightResult := 267957) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37362⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39102⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37362⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39102⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge268241

namespace LeftMerge268243
def owner : Owner := ⟨.program ⟨257⟩, ⟨39104⟩⟩
def mergeEvent : Nat := 268243
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37362⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38506⟩⟩] } }
def rhsRaw : List Term := Proof.Events1046.exact267954RawTerms
def group : MergeGroup := .relation 268242
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 268242) (rhsResult := 267954)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37362⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39102⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39102⟩⟩) ⟨38506⟩ 267954) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38506⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37362⟩⟩], [⟨.program ⟨257⟩, ⟨38506⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge268243

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
