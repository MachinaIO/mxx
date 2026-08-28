import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge121305
def owner : Owner := ⟨.program ⟨257⟩, ⟨41576⟩⟩
def mergeEvent : Nat := 121305
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41575⟩⟩] } }
def leftRaw : List Term := Proof.Events473.exact121299RawTerms
def rightRaw : List Term := Proof.Events473.exact121235RawTerms
def group : MergeGroup := .operator 121299 121235
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121299) (leftOrdinal := 1)
    (rightResult := 121235) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41575⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41575⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121305

namespace LeftMerge121307
def owner : Owner := ⟨.program ⟨257⟩, ⟨41576⟩⟩
def mergeEvent : Nat := 121307
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41085⟩⟩] } }
def rhsRaw : List Term := Proof.Events473.exact121232RawTerms
def group : MergeGroup := .relation 121306
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121306) (rhsResult := 121232)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41575⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41575⟩⟩) ⟨41085⟩ 121232) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41085⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], [⟨.program ⟨257⟩, ⟨41085⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121307

namespace LeftMerge121308
def owner : Owner := ⟨.program ⟨257⟩, ⟨41576⟩⟩
def mergeEvent : Nat := 121308
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41575⟩⟩] } }
def leftRaw : List Term := Proof.Events473.exact121299RawTerms
def rightRaw : List Term := Proof.Events473.exact121235RawTerms
def group : MergeGroup := .operator 121299 121235
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121299) (leftOrdinal := 0)
    (rightResult := 121235) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41575⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41575⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121308

namespace LeftMerge121322
def owner : Owner := ⟨.program ⟨257⟩, ⟨40512⟩⟩
def mergeEvent : Nat := 121322
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40509⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119870RawTerms
def rightRaw : List Term := Proof.Events473.exact121316RawTerms
def group : MergeGroup := .operator 119870 121316
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119870) (leftOrdinal := 0)
    (rightResult := 121316) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40509⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40509⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121322

namespace LeftMerge121401
def owner : Owner := ⟨.program ⟨257⟩, ⟨39699⟩⟩
def mergeEvent : Nat := 121401
def frameStart : Nat := 121371
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events474.exact121397RawTerms
def rightRaw : List Term := Proof.Events474.exact121394RawTerms
def group : MergeGroup := .operator 121397 121394
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121397) (leftOrdinal := 0)
    (rightResult := 121394) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14121⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨39698⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121401

namespace LeftMerge121431
def owner : Owner := ⟨.program ⟨257⟩, ⟨41372⟩⟩
def mergeEvent : Nat := 121431
def frameStart : Nat := 121371
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events474.exact121427RawTerms
def rightRaw : List Term := Proof.Events474.exact121425RawTerms
def group : MergeGroup := .operator 121427 121425
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121427) (leftOrdinal := 0)
    (rightResult := 121425) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121431

namespace LeftMerge121454
def owner : Owner := ⟨.program ⟨257⟩, ⟨9558⟩⟩
def mergeEvent : Nat := 121454
def frameStart : Nat := 121371
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }
def leftRaw : List Term := Proof.Events474.exact121450RawTerms
def rightRaw : List Term := Proof.Events474.exact121447RawTerms
def group : MergeGroup := .operator 121450 121447
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121450) (leftOrdinal := 0)
    (rightResult := 121447) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9556⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121454

namespace LeftMerge121463
def owner : Owner := ⟨.program ⟨257⟩, ⟨41578⟩⟩
def mergeEvent : Nat := 121463
def frameStart : Nat := 121371
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41575⟩⟩] } }
def leftRaw : List Term := Proof.Events474.exact121459RawTerms
def rightRaw : List Term := Proof.Events474.exact121416RawTerms
def group : MergeGroup := .operator 121459 121416
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121459) (leftOrdinal := 0)
    (rightResult := 121416) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41575⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41575⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121463

namespace LeftMerge121464
def owner : Owner := ⟨.program ⟨257⟩, ⟨41578⟩⟩
def mergeEvent : Nat := 121464
def frameStart : Nat := 121371
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41575⟩⟩] } }
def leftRaw : List Term := Proof.Events474.exact121459RawTerms
def rightRaw : List Term := Proof.Events474.exact121416RawTerms
def group : MergeGroup := .operator 121459 121416
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121459) (leftOrdinal := 1)
    (rightResult := 121416) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41575⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41575⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121464

namespace LeftMerge121466
def owner : Owner := ⟨.program ⟨257⟩, ⟨41578⟩⟩
def mergeEvent : Nat := 121466
def frameStart : Nat := 121371
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41085⟩⟩] } }
def rhsRaw : List Term := Proof.Events474.exact121413RawTerms
def group : MergeGroup := .relation 121465
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121465) (rhsResult := 121413)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41575⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41575⟩⟩) ⟨41085⟩ 121413) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41085⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], [⟨.program ⟨257⟩, ⟨41085⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121466

namespace LeftMerge121474
def owner : Owner := ⟨.program ⟨257⟩, ⟨40078⟩⟩
def mergeEvent : Nat := 121474
def frameStart : Nat := 121371
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40076⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events474.exact121427RawTerms
def rightRaw : List Term := Proof.Events474.exact121470RawTerms
def group : MergeGroup := .operator 121427 121470
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121427) (leftOrdinal := 0)
    (rightResult := 121470) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40076⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121474

namespace LeftMerge121491
def owner : Owner := ⟨.program ⟨257⟩, ⟨40512⟩⟩
def mergeEvent : Nat := 121491
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }
def rhsRaw : List Term := Proof.Events474.exact121488RawTerms
def group : MergeGroup := .relation 121490
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121490) (rhsResult := 121488)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40509⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 121489 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40509⟩⟩]⟩) (none) 121488) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121491

namespace LeftMerge121492
def owner : Owner := ⟨.program ⟨257⟩, ⟨40512⟩⟩
def mergeEvent : Nat := 121492
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41575⟩⟩] } }
def rhsRaw : List Term := Proof.Events474.exact121488RawTerms
def group : MergeGroup := .relation 121490
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121490) (rhsResult := 121488)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40509⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 121489 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40509⟩⟩]⟩) (none) 121488) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41575⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41575⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121492

namespace LeftMerge121493
def owner : Owner := ⟨.program ⟨257⟩, ⟨40512⟩⟩
def mergeEvent : Nat := 121493
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41085⟩⟩] } }
def rhsRaw : List Term := Proof.Events474.exact121488RawTerms
def group : MergeGroup := .relation 121490
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121490) (rhsResult := 121488)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40509⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 121489 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40509⟩⟩]⟩) (none) 121488) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41085⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], [⟨.program ⟨257⟩, ⟨41085⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121493

namespace LeftMerge121494
def owner : Owner := ⟨.program ⟨257⟩, ⟨40512⟩⟩
def mergeEvent : Nat := 121494
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨40076⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events474.exact121488RawTerms
def group : MergeGroup := .relation 121490
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121490) (rhsResult := 121488)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40509⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 121489 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40509⟩⟩]⟩) (none) 121488) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40076⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121494

namespace LeftMerge121499
def owner : Owner := ⟨.program ⟨257⟩, ⟨41577⟩⟩
def mergeEvent : Nat := 121499
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41085⟩⟩] } }
def leftRaw : List Term := Proof.Events474.exact121495RawTerms
def rightRaw : List Term := Proof.Events473.exact121309RawTerms
def group : MergeGroup := .operator 121495 121309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121495) (leftOrdinal := 2)
    (rightResult := 121309) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41085⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41085⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], [⟨.program ⟨257⟩, ⟨41085⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121499

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
