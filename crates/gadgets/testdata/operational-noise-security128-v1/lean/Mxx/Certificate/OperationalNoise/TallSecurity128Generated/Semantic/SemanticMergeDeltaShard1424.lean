import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge232304
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232304
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 21)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232304

namespace LeftMerge232306
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232306
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }
def rhsRaw : List Term := Proof.Events906.exact232099RawTerms
def group : MergeGroup := .relation 232305
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232305) (rhsResult := 232099)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨26606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71204⟩⟩) ⟨68824⟩ 232099) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨26606⟩⟩], [⟨.program ⟨257⟩, ⟨68824⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232306

namespace LeftMerge232307
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232307
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 35)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232307

namespace LeftMerge232309
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232309
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }
def rhsRaw : List Term := Proof.Events906.exact232099RawTerms
def group : MergeGroup := .relation 232308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232308) (rhsResult := 232099)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71204⟩⟩) ⟨68824⟩ 232099) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨68824⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232309

namespace LeftMerge232310
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232310
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 34)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨63062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232310

namespace LeftMerge232312
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232312
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }
def rhsRaw : List Term := Proof.Events906.exact232099RawTerms
def group : MergeGroup := .relation 232311
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232311) (rhsResult := 232099)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨63062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71204⟩⟩) ⟨68824⟩ 232099) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨63062⟩⟩], [⟨.program ⟨257⟩, ⟨68824⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232312

namespace LeftMerge232313
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232313
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 33)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232313

namespace LeftMerge232315
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232315
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }
def rhsRaw : List Term := Proof.Events906.exact232099RawTerms
def group : MergeGroup := .relation 232314
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232314) (rhsResult := 232099)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71204⟩⟩) ⟨68824⟩ 232099) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨68824⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232315

namespace LeftMerge232316
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232316
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 32)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232316

namespace LeftMerge232318
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232318
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }
def rhsRaw : List Term := Proof.Events906.exact232099RawTerms
def group : MergeGroup := .relation 232317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232317) (rhsResult := 232099)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71204⟩⟩) ⟨68824⟩ 232099) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨68824⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232318

namespace LeftMerge232319
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232319
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 31)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232319

namespace LeftMerge232321
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232321
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }
def rhsRaw : List Term := Proof.Events906.exact232099RawTerms
def group : MergeGroup := .relation 232320
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232320) (rhsResult := 232099)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71204⟩⟩) ⟨68824⟩ 232099) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨68824⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232321

namespace LeftMerge232322
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232322
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 30)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨51142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232322

namespace LeftMerge232324
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232324
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }
def rhsRaw : List Term := Proof.Events906.exact232099RawTerms
def group : MergeGroup := .relation 232323
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232323) (rhsResult := 232099)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨51142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71204⟩⟩) ⟨68824⟩ 232099) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨51142⟩⟩], [⟨.program ⟨257⟩, ⟨68824⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232324

namespace LeftMerge232325
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232325
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 23)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32087⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232325

namespace LeftMerge232327
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232327
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }
def rhsRaw : List Term := Proof.Events906.exact232099RawTerms
def group : MergeGroup := .relation 232326
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232326) (rhsResult := 232099)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨32087⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71204⟩⟩) ⟨68824⟩ 232099) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨32087⟩⟩], [⟨.program ⟨257⟩, ⟨68824⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232327

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
