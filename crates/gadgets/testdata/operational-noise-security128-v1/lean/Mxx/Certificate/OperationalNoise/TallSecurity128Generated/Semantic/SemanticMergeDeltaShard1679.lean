import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge272113
def owner : Owner := ⟨.program ⟨257⟩, ⟨54573⟩⟩
def mergeEvent : Nat := 272113
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54570⟩⟩] } }
def leftRaw : List Term := Proof.Events1039.exact266120RawTerms
def rightRaw : List Term := Proof.Events1062.exact272107RawTerms
def group : MergeGroup := .operator 266120 272107
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266120) (leftOrdinal := 0)
    (rightResult := 272107) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54570⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54570⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272113

namespace LeftMerge272234
def owner : Owner := ⟨.program ⟨257⟩, ⟨55316⟩⟩
def mergeEvent : Nat := 272234
def frameStart : Nat := 272168
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53802⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1063.exact272230RawTerms
def rightRaw : List Term := Proof.Events1063.exact272228RawTerms
def group : MergeGroup := .operator 272230 272228
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272230) (leftOrdinal := 0)
    (rightResult := 272228) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53802⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53802⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272234

namespace LeftMerge272246
def owner : Owner := ⟨.program ⟨257⟩, ⟨55676⟩⟩
def mergeEvent : Nat := 272246
def frameStart : Nat := 272168
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55675⟩⟩] } }
def leftRaw : List Term := Proof.Events1063.exact272242RawTerms
def rightRaw : List Term := Proof.Events1063.exact272219RawTerms
def group : MergeGroup := .operator 272242 272219
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272242) (leftOrdinal := 0)
    (rightResult := 272219) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55675⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55675⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272246

namespace LeftMerge272247
def owner : Owner := ⟨.program ⟨257⟩, ⟨55676⟩⟩
def mergeEvent : Nat := 272247
def frameStart : Nat := 272168
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53802⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55675⟩⟩] } }
def leftRaw : List Term := Proof.Events1063.exact272242RawTerms
def rightRaw : List Term := Proof.Events1063.exact272219RawTerms
def group : MergeGroup := .operator 272242 272219
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272242) (leftOrdinal := 1)
    (rightResult := 272219) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53802⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55675⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53802⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55675⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272247

namespace LeftMerge272249
def owner : Owner := ⟨.program ⟨257⟩, ⟨55676⟩⟩
def mergeEvent : Nat := 272249
def frameStart : Nat := 272168
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53802⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55066⟩⟩] } }
def rhsRaw : List Term := Proof.Events1063.exact272216RawTerms
def group : MergeGroup := .relation 272248
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272248) (rhsResult := 272216)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨53802⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55675⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55675⟩⟩) ⟨55066⟩ 272216) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55066⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨53802⟩⟩], [⟨.program ⟨257⟩, ⟨55066⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272249

namespace LeftMerge272257
def owner : Owner := ⟨.program ⟨257⟩, ⟨53986⟩⟩
def mergeEvent : Nat := 272257
def frameStart : Nat := 272168
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53984⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1063.exact272230RawTerms
def rightRaw : List Term := Proof.Events1063.exact272253RawTerms
def group : MergeGroup := .operator 272230 272253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272230) (leftOrdinal := 0)
    (rightResult := 272253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53984⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272257

namespace LeftMerge272274
def owner : Owner := ⟨.program ⟨257⟩, ⟨54573⟩⟩
def mergeEvent : Nat := 272274
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }
def rhsRaw : List Term := Proof.Events1063.exact272271RawTerms
def group : MergeGroup := .relation 272273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272273) (rhsResult := 272271)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54570⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 272272 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54570⟩⟩]⟩) (none) 272271) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272274

namespace LeftMerge272275
def owner : Owner := ⟨.program ⟨257⟩, ⟨54573⟩⟩
def mergeEvent : Nat := 272275
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55675⟩⟩] } }
def rhsRaw : List Term := Proof.Events1063.exact272271RawTerms
def group : MergeGroup := .relation 272273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272273) (rhsResult := 272271)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54570⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 272272 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54570⟩⟩]⟩) (none) 272271) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55675⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55675⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272275

namespace LeftMerge272276
def owner : Owner := ⟨.program ⟨257⟩, ⟨54573⟩⟩
def mergeEvent : Nat := 272276
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53802⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55066⟩⟩] } }
def rhsRaw : List Term := Proof.Events1063.exact272271RawTerms
def group : MergeGroup := .relation 272273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272273) (rhsResult := 272271)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54570⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 272272 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54570⟩⟩]⟩) (none) 272271) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53802⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55066⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53802⟩⟩], [⟨.program ⟨257⟩, ⟨55066⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272276

namespace LeftMerge272277
def owner : Owner := ⟨.program ⟨257⟩, ⟨54573⟩⟩
def mergeEvent : Nat := 272277
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1063.exact272271RawTerms
def group : MergeGroup := .relation 272273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272273) (rhsResult := 272271)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54570⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 272272 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54570⟩⟩]⟩) (none) 272271) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53984⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272277

namespace LeftMerge272282
def owner : Owner := ⟨.program ⟨257⟩, ⟨55678⟩⟩
def mergeEvent : Nat := 272282
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55675⟩⟩] } }
def leftRaw : List Term := Proof.Events1063.exact272278RawTerms
def rightRaw : List Term := Proof.Events1062.exact272100RawTerms
def group : MergeGroup := .operator 272278 272100
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272278) (leftOrdinal := 0)
    (rightResult := 272100) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55675⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55675⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55675⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272282

namespace LeftMerge272283
def owner : Owner := ⟨.program ⟨257⟩, ⟨55678⟩⟩
def mergeEvent : Nat := 272283
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53802⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55066⟩⟩] } }
def leftRaw : List Term := Proof.Events1063.exact272278RawTerms
def rightRaw : List Term := Proof.Events1062.exact272100RawTerms
def group : MergeGroup := .operator 272278 272100
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272278) (leftOrdinal := 2)
    (rightResult := 272100) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53802⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55066⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53802⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55066⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53802⟩⟩], [⟨.program ⟨257⟩, ⟨55066⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272283

namespace LeftMerge272309
def owner : Owner := ⟨.program ⟨257⟩, ⟨24431⟩⟩
def mergeEvent : Nat := 272309
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24430⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events051.exact13109RawTerms
def rightRaw : List Term := Proof.Events1039.exact266028RawTerms
def group : MergeGroup := .operator 13109 266028
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13109) (leftOrdinal := 0)
    (rightResult := 266028) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24430⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24430⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272309

namespace LeftMerge272314
def owner : Owner := ⟨.program ⟨257⟩, ⟨7664⟩⟩
def mergeEvent : Nat := 272314
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } }
def leftRaw : List Term := Proof.Events1038.exact265898RawTerms
def rightRaw : List Term := Proof.Events092.exact23593RawTerms
def group : MergeGroup := .operator 265898 23593
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 265898) (leftOrdinal := 0)
    (rightResult := 23593) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272314

namespace LeftMerge272331
def owner : Owner := ⟨.program ⟨257⟩, ⟨50323⟩⟩
def mergeEvent : Nat := 272331
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1063.exact272325RawTerms
def rightRaw : List Term := Proof.Events051.exact13112RawTerms
def group : MergeGroup := .operator 272325 13112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272325) (leftOrdinal := 1)
    (rightResult := 13112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24430⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272331

namespace LeftMerge272332
def owner : Owner := ⟨.program ⟨257⟩, ⟨50323⟩⟩
def mergeEvent : Nat := 272332
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } }
def leftRaw : List Term := Proof.Events1063.exact272325RawTerms
def rightRaw : List Term := Proof.Events051.exact13112RawTerms
def group : MergeGroup := .operator 272325 13112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272325) (leftOrdinal := 0)
    (rightResult := 13112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272332

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
