import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge205259
def owner : Owner := ⟨.program ⟨257⟩, ⟨64296⟩⟩
def mergeEvent : Nat := 205259
def frameStart : Nat := 205193
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events801.exact205255RawTerms
def rightRaw : List Term := Proof.Events801.exact205253RawTerms
def group : MergeGroup := .operator 205255 205253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 205255) (leftOrdinal := 0)
    (rightResult := 205253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge205259

namespace LeftMerge205271
def owner : Owner := ⟨.program ⟨257⟩, ⟨64928⟩⟩
def mergeEvent : Nat := 205271
def frameStart : Nat := 205193
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64927⟩⟩] } }
def leftRaw : List Term := Proof.Events801.exact205267RawTerms
def rightRaw : List Term := Proof.Events801.exact205244RawTerms
def group : MergeGroup := .operator 205267 205244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 205267) (leftOrdinal := 0)
    (rightResult := 205244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64927⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64927⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge205271

namespace LeftMerge205272
def owner : Owner := ⟨.program ⟨257⟩, ⟨64928⟩⟩
def mergeEvent : Nat := 205272
def frameStart : Nat := 205193
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64927⟩⟩] } }
def leftRaw : List Term := Proof.Events801.exact205267RawTerms
def rightRaw : List Term := Proof.Events801.exact205244RawTerms
def group : MergeGroup := .operator 205267 205244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 205267) (leftOrdinal := 1)
    (rightResult := 205244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64927⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64927⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge205272

namespace LeftMerge205274
def owner : Owner := ⟨.program ⟨257⟩, ⟨64928⟩⟩
def mergeEvent : Nat := 205274
def frameStart : Nat := 205193
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64098⟩⟩] } }
def rhsRaw : List Term := Proof.Events801.exact205241RawTerms
def group : MergeGroup := .relation 205273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 205273) (rhsResult := 205241)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨62824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64927⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64927⟩⟩) ⟨64098⟩ 205241) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64098⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨62824⟩⟩], [⟨.program ⟨257⟩, ⟨64098⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge205274

namespace LeftMerge205282
def owner : Owner := ⟨.program ⟨257⟩, ⟨63126⟩⟩
def mergeEvent : Nat := 205282
def frameStart : Nat := 205193
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63123⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events801.exact205255RawTerms
def rightRaw : List Term := Proof.Events801.exact205278RawTerms
def group : MergeGroup := .operator 205255 205278
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 205255) (leftOrdinal := 0)
    (rightResult := 205278) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63123⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨63123⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge205282

namespace LeftMerge205299
def owner : Owner := ⟨.program ⟨257⟩, ⟨63715⟩⟩
def mergeEvent : Nat := 205299
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7213⟩⟩] } }
def rhsRaw : List Term := Proof.Events801.exact205296RawTerms
def group : MergeGroup := .relation 205298
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 205298) (rhsResult := 205296)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63712⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 205297 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63712⟩⟩]⟩) (none) 205296) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7213⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge205299

namespace LeftMerge205300
def owner : Owner := ⟨.program ⟨257⟩, ⟨63715⟩⟩
def mergeEvent : Nat := 205300
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64927⟩⟩] } }
def rhsRaw : List Term := Proof.Events801.exact205296RawTerms
def group : MergeGroup := .relation 205298
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 205298) (rhsResult := 205296)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63712⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 205297 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63712⟩⟩]⟩) (none) 205296) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64927⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64927⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge205300

namespace LeftMerge205301
def owner : Owner := ⟨.program ⟨257⟩, ⟨63715⟩⟩
def mergeEvent : Nat := 205301
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64098⟩⟩] } }
def rhsRaw : List Term := Proof.Events801.exact205296RawTerms
def group : MergeGroup := .relation 205298
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 205298) (rhsResult := 205296)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63712⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 205297 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63712⟩⟩]⟩) (none) 205296) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64098⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], [⟨.program ⟨257⟩, ⟨64098⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge205301

namespace LeftMerge205302
def owner : Owner := ⟨.program ⟨257⟩, ⟨63715⟩⟩
def mergeEvent : Nat := 205302
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events801.exact205296RawTerms
def group : MergeGroup := .relation 205298
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 205298) (rhsResult := 205296)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63712⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 205297 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63712⟩⟩]⟩) (none) 205296) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63123⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge205302

namespace LeftMerge205307
def owner : Owner := ⟨.program ⟨257⟩, ⟨64930⟩⟩
def mergeEvent : Nat := 205307
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64927⟩⟩] } }
def leftRaw : List Term := Proof.Events801.exact205303RawTerms
def rightRaw : List Term := Proof.Events801.exact205125RawTerms
def group : MergeGroup := .operator 205303 205125
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 205303) (leftOrdinal := 0)
    (rightResult := 205125) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64927⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64927⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64927⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge205307

namespace LeftMerge205308
def owner : Owner := ⟨.program ⟨257⟩, ⟨64930⟩⟩
def mergeEvent : Nat := 205308
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64098⟩⟩] } }
def leftRaw : List Term := Proof.Events801.exact205303RawTerms
def rightRaw : List Term := Proof.Events801.exact205125RawTerms
def group : MergeGroup := .operator 205303 205125
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 205303) (leftOrdinal := 2)
    (rightResult := 205125) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64098⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64098⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], [⟨.program ⟨257⟩, ⟨64098⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge205308

namespace LeftMerge205316
def owner : Owner := ⟨.program ⟨257⟩, ⟨64931⟩⟩
def mergeEvent : Nat := 205316
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩] } }
def leftRaw : List Term := Proof.Events801.exact205310RawTerms
def rightRaw : List Term := Proof.Events061.exact15722RawTerms
def group : MergeGroup := .operator 205310 15722
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 205310) (leftOrdinal := 0)
    (rightResult := 15722) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7213⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7099⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge205316

namespace LeftMerge205317
def owner : Owner := ⟨.program ⟨257⟩, ⟨64931⟩⟩
def mergeEvent : Nat := 205317
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩] } }
def leftRaw : List Term := Proof.Events801.exact205310RawTerms
def rightRaw : List Term := Proof.Events061.exact15722RawTerms
def group : MergeGroup := .operator 205310 15722
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 205310) (leftOrdinal := 1)
    (rightResult := 15722) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7099⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge205317

namespace LeftMerge205319
def owner : Owner := ⟨.program ⟨257⟩, ⟨64931⟩⟩
def mergeEvent : Nat := 205319
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15715RawTerms
def group : MergeGroup := .relation 205318
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 205318) (rhsResult := 15715)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7099⟩⟩) ⟨7015⟩ 15715) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge205319

namespace LeftMerge205333
def owner : Owner := ⟨.program ⟨257⟩, ⟨61949⟩⟩
def mergeEvent : Nat := 205333
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61947⟩⟩] } }
def leftRaw : List Term := Proof.Events773.exact198001RawTerms
def rightRaw : List Term := Proof.Events802.exact205327RawTerms
def group : MergeGroup := .operator 198001 205327
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198001) (leftOrdinal := 0)
    (rightResult := 205327) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61947⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61947⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge205333

namespace LeftMerge205334
def owner : Owner := ⟨.program ⟨257⟩, ⟨61949⟩⟩
def mergeEvent : Nat := 205334
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨59844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61947⟩⟩] } }
def leftRaw : List Term := Proof.Events773.exact198001RawTerms
def rightRaw : List Term := Proof.Events802.exact205327RawTerms
def group : MergeGroup := .operator 198001 205327
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198001) (leftOrdinal := 1)
    (rightResult := 205327) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨59844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61947⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨59844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61947⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge205334

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
