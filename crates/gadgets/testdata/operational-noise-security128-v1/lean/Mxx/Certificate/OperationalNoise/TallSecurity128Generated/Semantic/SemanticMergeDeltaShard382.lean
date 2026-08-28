import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge65218
def owner : Owner := ⟨.program ⟨257⟩, ⟨69318⟩⟩
def mergeEvent : Nat := 65218
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69317⟩⟩] } }
def leftRaw : List Term := Proof.Events254.exact65209RawTerms
def rightRaw : List Term := Proof.Events254.exact65145RawTerms
def group : MergeGroup := .operator 65209 65145
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65209) (leftOrdinal := 0)
    (rightResult := 65145) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69317⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69317⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65218

namespace LeftMerge65232
def owner : Owner := ⟨.program ⟨257⟩, ⟨67843⟩⟩
def mergeEvent : Nat := 65232
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67840⟩⟩] } }
def leftRaw : List Term := Proof.Events239.exact61370RawTerms
def rightRaw : List Term := Proof.Events254.exact65226RawTerms
def group : MergeGroup := .operator 61370 65226
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61370) (leftOrdinal := 0)
    (rightResult := 65226) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨67840⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67840⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65232

namespace LeftMerge65311
def owner : Owner := ⟨.program ⟨257⟩, ⟨65635⟩⟩
def mergeEvent : Nat := 65311
def frameStart : Nat := 65281
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25814⟩⟩, ⟨.program ⟨257⟩, ⟨65634⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events255.exact65307RawTerms
def rightRaw : List Term := Proof.Events255.exact65304RawTerms
def group : MergeGroup := .operator 65307 65304
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65307) (leftOrdinal := 0)
    (rightResult := 65304) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65634⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25814⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25814⟩⟩, ⟨.program ⟨257⟩, ⟨65634⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65311

namespace LeftMerge65341
def owner : Owner := ⟨.program ⟨257⟩, ⟨68957⟩⟩
def mergeEvent : Nat := 65341
def frameStart : Nat := 65281
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25814⟩⟩, ⟨.program ⟨257⟩, ⟨65634⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65337RawTerms
def rightRaw : List Term := Proof.Events255.exact65335RawTerms
def group : MergeGroup := .operator 65337 65335
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65337) (leftOrdinal := 0)
    (rightResult := 65335) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25814⟩⟩, ⟨.program ⟨257⟩, ⟨65634⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25814⟩⟩, ⟨.program ⟨257⟩, ⟨65634⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65341

namespace LeftMerge65364
def owner : Owner := ⟨.program ⟨257⟩, ⟨9543⟩⟩
def mergeEvent : Nat := 65364
def frameStart : Nat := 65281
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65360RawTerms
def rightRaw : List Term := Proof.Events255.exact65357RawTerms
def group : MergeGroup := .operator 65360 65357
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65360) (leftOrdinal := 0)
    (rightResult := 65357) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9541⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65364

namespace LeftMerge65373
def owner : Owner := ⟨.program ⟨257⟩, ⟨69320⟩⟩
def mergeEvent : Nat := 65373
def frameStart : Nat := 65281
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69317⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65369RawTerms
def rightRaw : List Term := Proof.Events255.exact65326RawTerms
def group : MergeGroup := .operator 65369 65326
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65369) (leftOrdinal := 0)
    (rightResult := 65326) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69317⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69317⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65373

namespace LeftMerge65374
def owner : Owner := ⟨.program ⟨257⟩, ⟨69320⟩⟩
def mergeEvent : Nat := 65374
def frameStart : Nat := 65281
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25814⟩⟩, ⟨.program ⟨257⟩, ⟨65634⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69317⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65369RawTerms
def rightRaw : List Term := Proof.Events255.exact65326RawTerms
def group : MergeGroup := .operator 65369 65326
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65369) (leftOrdinal := 1)
    (rightResult := 65326) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25814⟩⟩, ⟨.program ⟨257⟩, ⟨65634⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69317⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25814⟩⟩, ⟨.program ⟨257⟩, ⟨65634⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69317⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65374

namespace LeftMerge65376
def owner : Owner := ⟨.program ⟨257⟩, ⟨69320⟩⟩
def mergeEvent : Nat := 65376
def frameStart : Nat := 65281
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25814⟩⟩, ⟨.program ⟨257⟩, ⟨65634⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68572⟩⟩] } }
def rhsRaw : List Term := Proof.Events255.exact65323RawTerms
def group : MergeGroup := .relation 65375
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 65375) (rhsResult := 65323)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25814⟩⟩, ⟨.program ⟨257⟩, ⟨65634⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69317⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69317⟩⟩) ⟨68572⟩ 65323) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68572⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25814⟩⟩, ⟨.program ⟨257⟩, ⟨65634⟩⟩], [⟨.program ⟨257⟩, ⟨68572⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65376

namespace LeftMerge65384
def owner : Owner := ⟨.program ⟨257⟩, ⟨65846⟩⟩
def mergeEvent : Nat := 65384
def frameStart : Nat := 65281
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65337RawTerms
def rightRaw : List Term := Proof.Events255.exact65380RawTerms
def group : MergeGroup := .operator 65337 65380
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65337) (leftOrdinal := 0)
    (rightResult := 65380) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65844⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65384

namespace LeftMerge65401
def owner : Owner := ⟨.program ⟨257⟩, ⟨67843⟩⟩
def mergeEvent : Nat := 65401
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }
def rhsRaw : List Term := Proof.Events255.exact65398RawTerms
def group : MergeGroup := .relation 65400
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 65400) (rhsResult := 65398)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67840⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 65399 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67840⟩⟩]⟩) (none) 65398) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65401

namespace LeftMerge65402
def owner : Owner := ⟨.program ⟨257⟩, ⟨67843⟩⟩
def mergeEvent : Nat := 65402
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69317⟩⟩] } }
def rhsRaw : List Term := Proof.Events255.exact65398RawTerms
def group : MergeGroup := .relation 65400
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 65400) (rhsResult := 65398)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67840⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 65399 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67840⟩⟩]⟩) (none) 65398) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69317⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69317⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65402

namespace LeftMerge65403
def owner : Owner := ⟨.program ⟨257⟩, ⟨67843⟩⟩
def mergeEvent : Nat := 65403
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25814⟩⟩, ⟨.program ⟨257⟩, ⟨65634⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68572⟩⟩] } }
def rhsRaw : List Term := Proof.Events255.exact65398RawTerms
def group : MergeGroup := .relation 65400
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 65400) (rhsResult := 65398)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67840⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 65399 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67840⟩⟩]⟩) (none) 65398) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25814⟩⟩, ⟨.program ⟨257⟩, ⟨65634⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68572⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25814⟩⟩, ⟨.program ⟨257⟩, ⟨65634⟩⟩], [⟨.program ⟨257⟩, ⟨68572⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65403

namespace LeftMerge65404
def owner : Owner := ⟨.program ⟨257⟩, ⟨67843⟩⟩
def mergeEvent : Nat := 65404
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨65844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events255.exact65398RawTerms
def group : MergeGroup := .relation 65400
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 65400) (rhsResult := 65398)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67840⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 65399 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67840⟩⟩]⟩) (none) 65398) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨65844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65404

namespace LeftMerge65409
def owner : Owner := ⟨.program ⟨257⟩, ⟨69319⟩⟩
def mergeEvent : Nat := 65409
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25814⟩⟩, ⟨.program ⟨257⟩, ⟨65634⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68572⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65405RawTerms
def rightRaw : List Term := Proof.Events254.exact65219RawTerms
def group : MergeGroup := .operator 65405 65219
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65405) (leftOrdinal := 2)
    (rightResult := 65219) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25814⟩⟩, ⟨.program ⟨257⟩, ⟨65634⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68572⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25814⟩⟩, ⟨.program ⟨257⟩, ⟨65634⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68572⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25814⟩⟩, ⟨.program ⟨257⟩, ⟨65634⟩⟩], [⟨.program ⟨257⟩, ⟨68572⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65409

namespace LeftMerge65410
def owner : Owner := ⟨.program ⟨257⟩, ⟨69319⟩⟩
def mergeEvent : Nat := 65410
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69317⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65405RawTerms
def rightRaw : List Term := Proof.Events254.exact65219RawTerms
def group : MergeGroup := .operator 65405 65219
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65405) (leftOrdinal := 1)
    (rightResult := 65219) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69317⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69317⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69317⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65410

namespace LeftMerge65418
def owner : Owner := ⟨.program ⟨257⟩, ⟨70732⟩⟩
def mergeEvent : Nat := 65418
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70730⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65412RawTerms
def rightRaw : List Term := Proof.Events254.exact65135RawTerms
def group : MergeGroup := .operator 65412 65135
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65412) (leftOrdinal := 0)
    (rightResult := 65135) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70730⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70730⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65418

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
