import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge89184
def owner : Owner := ⟨.program ⟨257⟩, ⟨34073⟩⟩
def mergeEvent : Nat := 89184
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33154⟩⟩] } }
def rhsRaw : List Term := Proof.Events348.exact89172RawTerms
def group : MergeGroup := .relation 89183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 89183) (rhsResult := 89172)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34071⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨34071⟩⟩) ⟨33154⟩ 89172) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33154⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨33154⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge89184

namespace LeftMerge89198
def owner : Owner := ⟨.program ⟨257⟩, ⟨32815⟩⟩
def mergeEvent : Nat := 89198
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32812⟩⟩] } }
def leftRaw : List Term := Proof.Events296.exact75995RawTerms
def rightRaw : List Term := Proof.Events348.exact89192RawTerms
def group : MergeGroup := .operator 75995 89192
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75995) (leftOrdinal := 0)
    (rightResult := 89192) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32812⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32812⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge89198

namespace LeftMerge89319
def owner : Owner := ⟨.program ⟨257⟩, ⟨33332⟩⟩
def mergeEvent : Nat := 89319
def frameStart : Nat := 89253
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events348.exact89315RawTerms
def rightRaw : List Term := Proof.Events348.exact89313RawTerms
def group : MergeGroup := .operator 89315 89313
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89315) (leftOrdinal := 0)
    (rightResult := 89313) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge89319

namespace LeftMerge89331
def owner : Owner := ⟨.program ⟨257⟩, ⟨34072⟩⟩
def mergeEvent : Nat := 89331
def frameStart : Nat := 89253
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34071⟩⟩] } }
def leftRaw : List Term := Proof.Events348.exact89327RawTerms
def rightRaw : List Term := Proof.Events348.exact89304RawTerms
def group : MergeGroup := .operator 89327 89304
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89327) (leftOrdinal := 0)
    (rightResult := 89304) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨34071⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34071⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge89331

namespace LeftMerge89332
def owner : Owner := ⟨.program ⟨257⟩, ⟨34072⟩⟩
def mergeEvent : Nat := 89332
def frameStart : Nat := 89253
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34071⟩⟩] } }
def leftRaw : List Term := Proof.Events348.exact89327RawTerms
def rightRaw : List Term := Proof.Events348.exact89304RawTerms
def group : MergeGroup := .operator 89327 89304
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89327) (leftOrdinal := 1)
    (rightResult := 89304) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨34071⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34071⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge89332

namespace LeftMerge89334
def owner : Owner := ⟨.program ⟨257⟩, ⟨34072⟩⟩
def mergeEvent : Nat := 89334
def frameStart : Nat := 89253
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33154⟩⟩] } }
def rhsRaw : List Term := Proof.Events348.exact89301RawTerms
def group : MergeGroup := .relation 89333
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 89333) (rhsResult := 89301)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34071⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨34071⟩⟩) ⟨33154⟩ 89301) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33154⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨33154⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge89334

namespace LeftMerge89342
def owner : Owner := ⟨.program ⟨257⟩, ⟨32218⟩⟩
def mergeEvent : Nat := 89342
def frameStart : Nat := 89253
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events348.exact89315RawTerms
def rightRaw : List Term := Proof.Events348.exact89338RawTerms
def group : MergeGroup := .operator 89315 89338
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89315) (leftOrdinal := 0)
    (rightResult := 89338) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32215⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge89342

namespace LeftMerge89359
def owner : Owner := ⟨.program ⟨257⟩, ⟨32815⟩⟩
def mergeEvent : Nat := 89359
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7203⟩⟩] } }
def rhsRaw : List Term := Proof.Events349.exact89356RawTerms
def group : MergeGroup := .relation 89358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 89358) (rhsResult := 89356)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32812⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 89357 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32812⟩⟩]⟩) (none) 89356) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7203⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge89359

namespace LeftMerge89360
def owner : Owner := ⟨.program ⟨257⟩, ⟨32815⟩⟩
def mergeEvent : Nat := 89360
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34071⟩⟩] } }
def rhsRaw : List Term := Proof.Events349.exact89356RawTerms
def group : MergeGroup := .relation 89358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 89358) (rhsResult := 89356)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32812⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 89357 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32812⟩⟩]⟩) (none) 89356) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34071⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34071⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge89360

namespace LeftMerge89361
def owner : Owner := ⟨.program ⟨257⟩, ⟨32815⟩⟩
def mergeEvent : Nat := 89361
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33154⟩⟩] } }
def rhsRaw : List Term := Proof.Events349.exact89356RawTerms
def group : MergeGroup := .relation 89358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 89358) (rhsResult := 89356)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32812⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 89357 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32812⟩⟩]⟩) (none) 89356) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33154⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨33154⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge89361

namespace LeftMerge89362
def owner : Owner := ⟨.program ⟨257⟩, ⟨32815⟩⟩
def mergeEvent : Nat := 89362
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events349.exact89356RawTerms
def group : MergeGroup := .relation 89358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 89358) (rhsResult := 89356)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32812⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 89357 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32812⟩⟩]⟩) (none) 89356) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge89362

namespace LeftMerge89367
def owner : Owner := ⟨.program ⟨257⟩, ⟨34074⟩⟩
def mergeEvent : Nat := 89367
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34071⟩⟩] } }
def leftRaw : List Term := Proof.Events349.exact89363RawTerms
def rightRaw : List Term := Proof.Events348.exact89185RawTerms
def group : MergeGroup := .operator 89363 89185
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89363) (leftOrdinal := 0)
    (rightResult := 89185) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34071⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34071⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34071⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge89367

namespace LeftMerge89368
def owner : Owner := ⟨.program ⟨257⟩, ⟨34074⟩⟩
def mergeEvent : Nat := 89368
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33154⟩⟩] } }
def leftRaw : List Term := Proof.Events349.exact89363RawTerms
def rightRaw : List Term := Proof.Events348.exact89185RawTerms
def group : MergeGroup := .operator 89363 89185
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89363) (leftOrdinal := 2)
    (rightResult := 89185) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33154⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33154⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨33154⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge89368

namespace LeftMerge89376
def owner : Owner := ⟨.program ⟨257⟩, ⟨34075⟩⟩
def mergeEvent : Nat := 89376
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩] } }
def leftRaw : List Term := Proof.Events349.exact89370RawTerms
def rightRaw : List Term := Proof.Events061.exact15822RawTerms
def group : MergeGroup := .operator 89370 15822
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89370) (leftOrdinal := 0)
    (rightResult := 15822) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7203⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7145⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge89376

namespace LeftMerge89377
def owner : Owner := ⟨.program ⟨257⟩, ⟨34075⟩⟩
def mergeEvent : Nat := 89377
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩] } }
def leftRaw : List Term := Proof.Events349.exact89370RawTerms
def rightRaw : List Term := Proof.Events061.exact15822RawTerms
def group : MergeGroup := .operator 89370 15822
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89370) (leftOrdinal := 1)
    (rightResult := 15822) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7145⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge89377

namespace LeftMerge89379
def owner : Owner := ⟨.program ⟨257⟩, ⟨34075⟩⟩
def mergeEvent : Nat := 89379
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15815RawTerms
def group : MergeGroup := .relation 89378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 89378) (rhsResult := 15815)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7145⟩⟩) ⟨7038⟩ 15815) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge89379

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
