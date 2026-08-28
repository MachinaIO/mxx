import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge24158
def owner : Owner := ⟨.program ⟨257⟩, ⟨31258⟩⟩
def mergeEvent : Nat := 24158
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }
def rhsRaw : List Term := Proof.Events094.exact24094RawTerms
def group : MergeGroup := .relation 24157
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24157) (rhsResult := 24094)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9577⟩⟩) ⟨7307⟩ 24094) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24158

namespace LeftMerge24159
def owner : Owner := ⟨.program ⟨257⟩, ⟨31258⟩⟩
def mergeEvent : Nat := 24159
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }
def leftRaw : List Term := Proof.Events094.exact24150RawTerms
def rightRaw : List Term := Proof.Events094.exact24124RawTerms
def group : MergeGroup := .operator 24150 24124
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24150) (leftOrdinal := 0)
    (rightResult := 24124) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9577⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24159

namespace LeftMerge24164
def owner : Owner := ⟨.program ⟨257⟩, ⟨31259⟩⟩
def mergeEvent : Nat := 24164
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }
def leftRaw : List Term := Proof.Events094.exact24160RawTerms
def rightRaw : List Term := Proof.Events094.exact24117RawTerms
def group : MergeGroup := .operator 24160 24117
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24160) (leftOrdinal := 1)
    (rightResult := 24117) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24164

namespace LeftMerge24172
def owner : Owner := ⟨.program ⟨257⟩, ⟨33364⟩⟩
def mergeEvent : Nat := 24172
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24186⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33363⟩⟩] } }
def leftRaw : List Term := Proof.Events094.exact24166RawTerms
def rightRaw : List Term := Proof.Events094.exact24083RawTerms
def group : MergeGroup := .operator 24166 24083
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24166) (leftOrdinal := 1)
    (rightResult := 24083) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24186⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33363⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24186⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33363⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24172

namespace LeftMerge24174
def owner : Owner := ⟨.program ⟨257⟩, ⟨33364⟩⟩
def mergeEvent : Nat := 24174
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24186⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32897⟩⟩] } }
def rhsRaw : List Term := Proof.Events094.exact24080RawTerms
def group : MergeGroup := .relation 24173
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24173) (rhsResult := 24080)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24186⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33363⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33363⟩⟩) ⟨32897⟩ 24080) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32897⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24186⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], [⟨.program ⟨257⟩, ⟨32897⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24174

namespace LeftMerge24175
def owner : Owner := ⟨.program ⟨257⟩, ⟨33364⟩⟩
def mergeEvent : Nat := 24175
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33363⟩⟩] } }
def leftRaw : List Term := Proof.Events094.exact24166RawTerms
def rightRaw : List Term := Proof.Events094.exact24083RawTerms
def group : MergeGroup := .operator 24166 24083
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24166) (leftOrdinal := 0)
    (rightResult := 24083) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33363⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33363⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24175

namespace LeftMerge24189
def owner : Owner := ⟨.program ⟨257⟩, ⟨32305⟩⟩
def mergeEvent : Nat := 24189
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32302⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events094.exact24183RawTerms
def group : MergeGroup := .operator 17169 24183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 24183) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32302⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32302⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24189

namespace LeftMerge24268
def owner : Owner := ⟨.program ⟨257⟩, ⟨31252⟩⟩
def mergeEvent : Nat := 24268
def frameStart : Nat := 24238
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24186⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events094.exact24264RawTerms
def rightRaw : List Term := Proof.Events094.exact24261RawTerms
def group : MergeGroup := .operator 24264 24261
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24264) (leftOrdinal := 0)
    (rightResult := 24261) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31251⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24186⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24186⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24268

namespace LeftMerge24298
def owner : Owner := ⟨.program ⟨257⟩, ⟨33192⟩⟩
def mergeEvent : Nat := 24298
def frameStart : Nat := 24238
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24186⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events094.exact24294RawTerms
def rightRaw : List Term := Proof.Events094.exact24292RawTerms
def group : MergeGroup := .operator 24294 24292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24294) (leftOrdinal := 0)
    (rightResult := 24292) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24186⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24186⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24298

namespace LeftMerge24321
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def mergeEvent : Nat := 24321
def frameStart : Nat := 24238
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }
def leftRaw : List Term := Proof.Events094.exact24317RawTerms
def rightRaw : List Term := Proof.Events094.exact24314RawTerms
def group : MergeGroup := .operator 24317 24314
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24317) (leftOrdinal := 0)
    (rightResult := 24314) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9577⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24321

namespace LeftMerge24330
def owner : Owner := ⟨.program ⟨257⟩, ⟨33366⟩⟩
def mergeEvent : Nat := 24330
def frameStart : Nat := 24238
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24186⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33363⟩⟩] } }
def leftRaw : List Term := Proof.Events095.exact24326RawTerms
def rightRaw : List Term := Proof.Events094.exact24283RawTerms
def group : MergeGroup := .operator 24326 24283
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24326) (leftOrdinal := 1)
    (rightResult := 24283) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24186⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33363⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24186⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33363⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24330

namespace LeftMerge24332
def owner : Owner := ⟨.program ⟨257⟩, ⟨33366⟩⟩
def mergeEvent : Nat := 24332
def frameStart : Nat := 24238
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24186⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32897⟩⟩] } }
def rhsRaw : List Term := Proof.Events094.exact24280RawTerms
def group : MergeGroup := .relation 24331
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24331) (rhsResult := 24280)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24186⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33363⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33363⟩⟩) ⟨32897⟩ 24280) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32897⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24186⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], [⟨.program ⟨257⟩, ⟨32897⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24332

namespace LeftMerge24333
def owner : Owner := ⟨.program ⟨257⟩, ⟨33366⟩⟩
def mergeEvent : Nat := 24333
def frameStart : Nat := 24238
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33363⟩⟩] } }
def leftRaw : List Term := Proof.Events095.exact24326RawTerms
def rightRaw : List Term := Proof.Events094.exact24283RawTerms
def group : MergeGroup := .operator 24326 24283
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24326) (leftOrdinal := 0)
    (rightResult := 24283) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33363⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33363⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24333

namespace LeftMerge24341
def owner : Owner := ⟨.program ⟨257⟩, ⟨31760⟩⟩
def mergeEvent : Nat := 24341
def frameStart : Nat := 24238
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31758⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events094.exact24294RawTerms
def rightRaw : List Term := Proof.Events095.exact24337RawTerms
def group : MergeGroup := .operator 24294 24337
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24294) (leftOrdinal := 0)
    (rightResult := 24337) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31758⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31758⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24341

namespace LeftMerge24358
def owner : Owner := ⟨.program ⟨257⟩, ⟨32305⟩⟩
def mergeEvent : Nat := 24358
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24186⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32897⟩⟩] } }
def rhsRaw : List Term := Proof.Events095.exact24355RawTerms
def group : MergeGroup := .relation 24357
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24357) (rhsResult := 24355)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32302⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 24356 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32302⟩⟩]⟩) (none) 24355) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24186⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32897⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24186⟩⟩, ⟨.program ⟨257⟩, ⟨31251⟩⟩], [⟨.program ⟨257⟩, ⟨32897⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24358

namespace LeftMerge24359
def owner : Owner := ⟨.program ⟨257⟩, ⟨32305⟩⟩
def mergeEvent : Nat := 24359
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33363⟩⟩] } }
def rhsRaw : List Term := Proof.Events095.exact24355RawTerms
def group : MergeGroup := .relation 24357
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24357) (rhsResult := 24355)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32302⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 24356 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32302⟩⟩]⟩) (none) 24355) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33363⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33363⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24359

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
