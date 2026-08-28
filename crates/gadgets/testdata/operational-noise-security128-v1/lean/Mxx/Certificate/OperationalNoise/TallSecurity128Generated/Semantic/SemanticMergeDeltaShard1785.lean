import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge289382
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289382
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨45605⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1096.exact280625RawTerms
def group : MergeGroup := .relation 289381
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289381) (rhsResult := 280625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289382

namespace LeftMerge289383
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289383
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 15)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289383

namespace LeftMerge289384
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289384
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨42921⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 27)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨42921⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289384

namespace LeftMerge289386
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289386
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨42921⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1096.exact280625RawTerms
def group : MergeGroup := .relation 289385
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289385) (rhsResult := 280625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289386

namespace LeftMerge289387
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289387
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 14)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289387

namespace LeftMerge289388
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289388
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40241⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 26)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40241⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289388

namespace LeftMerge289390
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289390
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40241⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1096.exact280625RawTerms
def group : MergeGroup := .relation 289389
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289389) (rhsResult := 280625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289390

namespace LeftMerge289391
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289391
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 13)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289391

namespace LeftMerge289392
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289392
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 25)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289392

namespace LeftMerge289394
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289394
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1096.exact280625RawTerms
def group : MergeGroup := .relation 289393
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289393) (rhsResult := 280625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289394

namespace LeftMerge289395
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289395
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 12)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289395

namespace LeftMerge289396
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289396
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34885⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 24)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34885⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289396

namespace LeftMerge289398
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289398
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34885⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1096.exact280625RawTerms
def group : MergeGroup := .relation 289397
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289397) (rhsResult := 280625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289398

namespace LeftMerge289399
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289399
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 11)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289399

namespace LeftMerge289400
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289400
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 22)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289400

namespace LeftMerge289402
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289402
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1096.exact280625RawTerms
def group : MergeGroup := .relation 289401
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289401) (rhsResult := 280625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289402

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
