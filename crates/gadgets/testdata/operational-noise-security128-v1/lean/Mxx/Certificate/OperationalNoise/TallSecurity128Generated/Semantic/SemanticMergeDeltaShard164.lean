import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge31314
def owner : Owner := ⟨.program ⟨257⟩, ⟨17512⟩⟩
def mergeEvent : Nat := 31314
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩] } }
def leftRaw : List Term := Proof.Events101.exact25872RawTerms
def rightRaw : List Term := Proof.Events122.exact31308RawTerms
def group : MergeGroup := .operator 25872 31308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25872) (leftOrdinal := 1)
    (rightResult := 31308) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17510⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31314

namespace LeftMerge31316
def owner : Owner := ⟨.program ⟨257⟩, ⟨17512⟩⟩
def mergeEvent : Nat := 31316
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16922⟩⟩] } }
def rhsRaw : List Term := Proof.Events122.exact31305RawTerms
def group : MergeGroup := .relation 31315
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 31315) (rhsResult := 31305)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17510⟩⟩) ⟨16922⟩ 31305) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16922⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16922⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31316

namespace LeftMerge31317
def owner : Owner := ⟨.program ⟨257⟩, ⟨17512⟩⟩
def mergeEvent : Nat := 31317
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩] } }
def leftRaw : List Term := Proof.Events101.exact25872RawTerms
def rightRaw : List Term := Proof.Events122.exact31308RawTerms
def group : MergeGroup := .operator 25872 31308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25872) (leftOrdinal := 0)
    (rightResult := 31308) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17510⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31317

namespace LeftMerge31331
def owner : Owner := ⟨.program ⟨257⟩, ⟨16421⟩⟩
def mergeEvent : Nat := 31331
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16418⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events122.exact31325RawTerms
def group : MergeGroup := .operator 17169 31325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 31325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16418⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16418⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31331

namespace LeftMerge31452
def owner : Owner := ⟨.program ⟨257⟩, ⟨17172⟩⟩
def mergeEvent : Nat := 31452
def frameStart : Nat := 31386
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events122.exact31448RawTerms
def rightRaw : List Term := Proof.Events122.exact31446RawTerms
def group : MergeGroup := .operator 31448 31446
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31448) (leftOrdinal := 0)
    (rightResult := 31446) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31452

namespace LeftMerge31464
def owner : Owner := ⟨.program ⟨257⟩, ⟨17511⟩⟩
def mergeEvent : Nat := 31464
def frameStart : Nat := 31386
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩] } }
def leftRaw : List Term := Proof.Events122.exact31460RawTerms
def rightRaw : List Term := Proof.Events122.exact31437RawTerms
def group : MergeGroup := .operator 31460 31437
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31460) (leftOrdinal := 1)
    (rightResult := 31437) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17510⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31464

namespace LeftMerge31466
def owner : Owner := ⟨.program ⟨257⟩, ⟨17511⟩⟩
def mergeEvent : Nat := 31466
def frameStart : Nat := 31386
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16922⟩⟩] } }
def rhsRaw : List Term := Proof.Events122.exact31434RawTerms
def group : MergeGroup := .relation 31465
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 31465) (rhsResult := 31434)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17510⟩⟩) ⟨16922⟩ 31434) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16922⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16922⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31466

namespace LeftMerge31467
def owner : Owner := ⟨.program ⟨257⟩, ⟨17511⟩⟩
def mergeEvent : Nat := 31467
def frameStart : Nat := 31386
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩] } }
def leftRaw : List Term := Proof.Events122.exact31460RawTerms
def rightRaw : List Term := Proof.Events122.exact31437RawTerms
def group : MergeGroup := .operator 31460 31437
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31460) (leftOrdinal := 0)
    (rightResult := 31437) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17510⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31467

namespace LeftMerge31475
def owner : Owner := ⟨.program ⟨257⟩, ⟨15893⟩⟩
def mergeEvent : Nat := 31475
def frameStart : Nat := 31386
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15890⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events122.exact31448RawTerms
def rightRaw : List Term := Proof.Events122.exact31471RawTerms
def group : MergeGroup := .operator 31448 31471
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31448) (leftOrdinal := 0)
    (rightResult := 31471) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15890⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31475

namespace LeftMerge31492
def owner : Owner := ⟨.program ⟨257⟩, ⟨16421⟩⟩
def mergeEvent : Nat := 31492
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7197⟩⟩] } }
def rhsRaw : List Term := Proof.Events123.exact31489RawTerms
def group : MergeGroup := .relation 31491
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 31491) (rhsResult := 31489)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16418⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 31490 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16418⟩⟩]⟩) (none) 31489) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7197⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31492

namespace LeftMerge31493
def owner : Owner := ⟨.program ⟨257⟩, ⟨16421⟩⟩
def mergeEvent : Nat := 31493
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16922⟩⟩] } }
def rhsRaw : List Term := Proof.Events123.exact31489RawTerms
def group : MergeGroup := .relation 31491
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 31491) (rhsResult := 31489)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16418⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 31490 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16418⟩⟩]⟩) (none) 31489) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16922⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16922⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31493

namespace LeftMerge31494
def owner : Owner := ⟨.program ⟨257⟩, ⟨16421⟩⟩
def mergeEvent : Nat := 31494
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩] } }
def rhsRaw : List Term := Proof.Events123.exact31489RawTerms
def group : MergeGroup := .relation 31491
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 31491) (rhsResult := 31489)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16418⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 31490 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16418⟩⟩]⟩) (none) 31489) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31494

namespace LeftMerge31495
def owner : Owner := ⟨.program ⟨257⟩, ⟨16421⟩⟩
def mergeEvent : Nat := 31495
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events123.exact31489RawTerms
def group : MergeGroup := .relation 31491
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 31491) (rhsResult := 31489)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16418⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 31490 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16418⟩⟩]⟩) (none) 31489) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15890⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31495

namespace LeftMerge31500
def owner : Owner := ⟨.program ⟨257⟩, ⟨17513⟩⟩
def mergeEvent : Nat := 31500
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16922⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31496RawTerms
def rightRaw : List Term := Proof.Events122.exact31318RawTerms
def group : MergeGroup := .operator 31496 31318
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31496) (leftOrdinal := 2)
    (rightResult := 31318) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16922⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16922⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16922⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31500

namespace LeftMerge31501
def owner : Owner := ⟨.program ⟨257⟩, ⟨17513⟩⟩
def mergeEvent : Nat := 31501
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31496RawTerms
def rightRaw : List Term := Proof.Events122.exact31318RawTerms
def group : MergeGroup := .operator 31496 31318
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31496) (leftOrdinal := 0)
    (rightResult := 31318) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31501

namespace LeftMerge31509
def owner : Owner := ⟨.program ⟨257⟩, ⟨17514⟩⟩
def mergeEvent : Nat := 31509
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31503RawTerms
def rightRaw : List Term := Proof.Events062.exact15882RawTerms
def group : MergeGroup := .operator 31503 15882
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31503) (leftOrdinal := 0)
    (rightResult := 15882) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7197⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7171⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31509

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
