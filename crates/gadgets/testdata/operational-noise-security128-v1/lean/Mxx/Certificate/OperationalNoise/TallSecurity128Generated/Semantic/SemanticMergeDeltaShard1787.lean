import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge289424
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289424
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 31)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289424

namespace LeftMerge289426
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289426
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1096.exact280625RawTerms
def group : MergeGroup := .relation 289425
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289425) (rhsResult := 280625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289426

namespace LeftMerge289427
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289427
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 4)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289427

namespace LeftMerge289428
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289428
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 30)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289428

namespace LeftMerge289430
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289430
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1096.exact280625RawTerms
def group : MergeGroup := .relation 289429
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289429) (rhsResult := 280625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289430

namespace LeftMerge289431
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289431
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 3)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289431

namespace LeftMerge289432
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289432
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 23)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289432

namespace LeftMerge289434
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289434
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1096.exact280625RawTerms
def group : MergeGroup := .relation 289433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289433) (rhsResult := 280625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289434

namespace LeftMerge289435
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289435
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 2)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289435

namespace LeftMerge289436
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289436
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 20)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289436

namespace LeftMerge289438
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289438
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1096.exact280625RawTerms
def group : MergeGroup := .relation 289437
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289437) (rhsResult := 280625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289438

namespace LeftMerge289439
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289439
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 1)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289439

namespace LeftMerge289440
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289440
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 19)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289440

namespace LeftMerge289442
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289442
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1096.exact280625RawTerms
def group : MergeGroup := .relation 289441
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289441) (rhsResult := 280625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289442

namespace LeftMerge289443
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289443
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 0)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289443

namespace LeftMerge289444
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289444
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 18)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289444

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
