import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge289403
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289403
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 10)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289403

namespace LeftMerge289404
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289404
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 21)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289404

namespace LeftMerge289406
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289406
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1096.exact280625RawTerms
def group : MergeGroup := .relation 289405
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289405) (rhsResult := 280625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289406

namespace LeftMerge289407
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289407
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 9)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289407

namespace LeftMerge289408
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289408
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 35)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289408

namespace LeftMerge289410
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289410
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1096.exact280625RawTerms
def group : MergeGroup := .relation 289409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289409) (rhsResult := 280625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289410

namespace LeftMerge289411
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289411
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 8)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289411

namespace LeftMerge289412
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289412
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 34)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289412

namespace LeftMerge289414
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289414
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1096.exact280625RawTerms
def group : MergeGroup := .relation 289413
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289413) (rhsResult := 280625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289414

namespace LeftMerge289415
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289415
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 7)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289415

namespace LeftMerge289416
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289416
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 33)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289416

namespace LeftMerge289418
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289418
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1096.exact280625RawTerms
def group : MergeGroup := .relation 289417
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289417) (rhsResult := 280625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289418

namespace LeftMerge289419
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289419
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 6)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289419

namespace LeftMerge289420
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289420
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 32)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289420

namespace LeftMerge289422
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289422
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1096.exact280625RawTerms
def group : MergeGroup := .relation 289421
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289421) (rhsResult := 280625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289422

namespace LeftMerge289423
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def mergeEvent : Nat := 289423
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1130.exact289369RawTerms
def rightRaw : List Term := Proof.Events1096.exact280628RawTerms
def group : MergeGroup := .operator 289369 280628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289369) (leftOrdinal := 5)
    (rightResult := 280628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289423

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
