import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge29332
def owner : Owner := ⟨.program ⟨257⟩, ⟨27732⟩⟩
def mergeEvent : Nat := 29332
def frameStart : Nat := 29266
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events114.exact29328RawTerms
def rightRaw : List Term := Proof.Events114.exact29326RawTerms
def group : MergeGroup := .operator 29328 29326
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29328) (leftOrdinal := 0)
    (rightResult := 29326) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26338⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29332

namespace LeftMerge29344
def owner : Owner := ⟨.program ⟨257⟩, ⟨28066⟩⟩
def mergeEvent : Nat := 29344
def frameStart : Nat := 29266
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28065⟩⟩] } }
def leftRaw : List Term := Proof.Events114.exact29340RawTerms
def rightRaw : List Term := Proof.Events114.exact29317RawTerms
def group : MergeGroup := .operator 29340 29317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29340) (leftOrdinal := 1)
    (rightResult := 29317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28065⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28065⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29344

namespace LeftMerge29346
def owner : Owner := ⟨.program ⟨257⟩, ⟨28066⟩⟩
def mergeEvent : Nat := 29346
def frameStart : Nat := 29266
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27482⟩⟩] } }
def rhsRaw : List Term := Proof.Events114.exact29314RawTerms
def group : MergeGroup := .relation 29345
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29345) (rhsResult := 29314)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨26338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28065⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28065⟩⟩) ⟨27482⟩ 29314) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27482⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨26338⟩⟩], [⟨.program ⟨257⟩, ⟨27482⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29346

namespace LeftMerge29347
def owner : Owner := ⟨.program ⟨257⟩, ⟨28066⟩⟩
def mergeEvent : Nat := 29347
def frameStart : Nat := 29266
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28065⟩⟩] } }
def leftRaw : List Term := Proof.Events114.exact29340RawTerms
def rightRaw : List Term := Proof.Events114.exact29317RawTerms
def group : MergeGroup := .operator 29340 29317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29340) (leftOrdinal := 0)
    (rightResult := 29317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28065⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28065⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29347

namespace LeftMerge29355
def owner : Owner := ⟨.program ⟨257⟩, ⟨26510⟩⟩
def mergeEvent : Nat := 29355
def frameStart : Nat := 29266
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26508⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events114.exact29328RawTerms
def rightRaw : List Term := Proof.Events114.exact29351RawTerms
def group : MergeGroup := .operator 29328 29351
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29328) (leftOrdinal := 0)
    (rightResult := 29351) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26508⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29355

namespace LeftMerge29372
def owner : Owner := ⟨.program ⟨257⟩, ⟨26981⟩⟩
def mergeEvent : Nat := 29372
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7217⟩⟩] } }
def rhsRaw : List Term := Proof.Events114.exact29369RawTerms
def group : MergeGroup := .relation 29371
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29371) (rhsResult := 29369)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26978⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 29370 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26978⟩⟩]⟩) (none) 29369) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7217⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29372

namespace LeftMerge29373
def owner : Owner := ⟨.program ⟨257⟩, ⟨26981⟩⟩
def mergeEvent : Nat := 29373
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27482⟩⟩] } }
def rhsRaw : List Term := Proof.Events114.exact29369RawTerms
def group : MergeGroup := .relation 29371
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29371) (rhsResult := 29369)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26978⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 29370 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26978⟩⟩]⟩) (none) 29369) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27482⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26338⟩⟩], [⟨.program ⟨257⟩, ⟨27482⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29373

namespace LeftMerge29374
def owner : Owner := ⟨.program ⟨257⟩, ⟨26981⟩⟩
def mergeEvent : Nat := 29374
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28065⟩⟩] } }
def rhsRaw : List Term := Proof.Events114.exact29369RawTerms
def group : MergeGroup := .relation 29371
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29371) (rhsResult := 29369)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26978⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 29370 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26978⟩⟩]⟩) (none) 29369) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28065⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28065⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29374

namespace LeftMerge29375
def owner : Owner := ⟨.program ⟨257⟩, ⟨26981⟩⟩
def mergeEvent : Nat := 29375
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events114.exact29369RawTerms
def group : MergeGroup := .relation 29371
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29371) (rhsResult := 29369)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26978⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 29370 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26978⟩⟩]⟩) (none) 29369) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26508⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29375

namespace LeftMerge29380
def owner : Owner := ⟨.program ⟨257⟩, ⟨28068⟩⟩
def mergeEvent : Nat := 29380
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27482⟩⟩] } }
def leftRaw : List Term := Proof.Events114.exact29376RawTerms
def rightRaw : List Term := Proof.Events114.exact29198RawTerms
def group : MergeGroup := .operator 29376 29198
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29376) (leftOrdinal := 2)
    (rightResult := 29198) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27482⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27482⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26338⟩⟩], [⟨.program ⟨257⟩, ⟨27482⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29380

namespace LeftMerge29381
def owner : Owner := ⟨.program ⟨257⟩, ⟨28068⟩⟩
def mergeEvent : Nat := 29381
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28065⟩⟩] } }
def leftRaw : List Term := Proof.Events114.exact29376RawTerms
def rightRaw : List Term := Proof.Events114.exact29198RawTerms
def group : MergeGroup := .operator 29376 29198
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29376) (leftOrdinal := 0)
    (rightResult := 29198) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28065⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28065⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28065⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29381

namespace LeftMerge29389
def owner : Owner := ⟨.program ⟨257⟩, ⟨28069⟩⟩
def mergeEvent : Nat := 29389
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩] } }
def leftRaw : List Term := Proof.Events114.exact29383RawTerms
def rightRaw : List Term := Proof.Events061.exact15682RawTerms
def group : MergeGroup := .operator 29383 15682
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29383) (leftOrdinal := 0)
    (rightResult := 15682) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7217⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7169⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29389

namespace LeftMerge29390
def owner : Owner := ⟨.program ⟨257⟩, ⟨28069⟩⟩
def mergeEvent : Nat := 29390
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩] } }
def leftRaw : List Term := Proof.Events114.exact29383RawTerms
def rightRaw : List Term := Proof.Events061.exact15682RawTerms
def group : MergeGroup := .operator 29383 15682
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29383) (leftOrdinal := 1)
    (rightResult := 15682) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7169⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29390

namespace LeftMerge29392
def owner : Owner := ⟨.program ⟨257⟩, ⟨28069⟩⟩
def mergeEvent : Nat := 29392
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15675RawTerms
def group : MergeGroup := .relation 29391
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29391) (rhsResult := 15675)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7169⟩⟩) ⟨7050⟩ 15675) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29392

namespace LeftMerge29406
def owner : Owner := ⟨.program ⟨257⟩, ⟨69478⟩⟩
def mergeEvent : Nat := 29406
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69476⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21363RawTerms
def rightRaw : List Term := Proof.Events114.exact29400RawTerms
def group : MergeGroup := .operator 21363 29400
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21363) (leftOrdinal := 1)
    (rightResult := 29400) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69476⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69476⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29406

namespace LeftMerge29408
def owner : Owner := ⟨.program ⟨257⟩, ⟨69478⟩⟩
def mergeEvent : Nat := 29408
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68603⟩⟩] } }
def rhsRaw : List Term := Proof.Events114.exact29397RawTerms
def group : MergeGroup := .relation 29407
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29407) (rhsResult := 29397)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69476⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69476⟩⟩) ⟨68603⟩ 29397) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68603⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65718⟩⟩], [⟨.program ⟨257⟩, ⟨68603⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29408

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
