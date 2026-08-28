import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge108320
def owner : Owner := ⟨.program ⟨257⟩, ⟨30612⟩⟩
def mergeEvent : Nat := 108320
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30095⟩⟩] } }
def leftRaw : List Term := Proof.Events423.exact108316RawTerms
def rightRaw : List Term := Proof.Events422.exact108130RawTerms
def group : MergeGroup := .operator 108316 108130
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108316) (leftOrdinal := 2)
    (rightResult := 108130) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30095⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30095⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13296⟩⟩, ⟨.program ⟨257⟩, ⟨28798⟩⟩], [⟨.program ⟨257⟩, ⟨30095⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108320

namespace LeftMerge108321
def owner : Owner := ⟨.program ⟨257⟩, ⟨30612⟩⟩
def mergeEvent : Nat := 108321
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30610⟩⟩] } }
def leftRaw : List Term := Proof.Events423.exact108316RawTerms
def rightRaw : List Term := Proof.Events422.exact108130RawTerms
def group : MergeGroup := .operator 108316 108130
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108316) (leftOrdinal := 1)
    (rightResult := 108130) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30610⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30610⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30610⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108321

namespace LeftMerge108329
def owner : Owner := ⟨.program ⟨257⟩, ⟨30996⟩⟩
def mergeEvent : Nat := 108329
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30994⟩⟩] } }
def leftRaw : List Term := Proof.Events423.exact108323RawTerms
def rightRaw : List Term := Proof.Events422.exact108046RawTerms
def group : MergeGroup := .operator 108323 108046
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108323) (leftOrdinal := 0)
    (rightResult := 108046) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30994⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30994⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108329

namespace LeftMerge108330
def owner : Owner := ⟨.program ⟨257⟩, ⟨30996⟩⟩
def mergeEvent : Nat := 108330
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29096⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30994⟩⟩] } }
def leftRaw : List Term := Proof.Events423.exact108323RawTerms
def rightRaw : List Term := Proof.Events422.exact108046RawTerms
def group : MergeGroup := .operator 108323 108046
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108323) (leftOrdinal := 1)
    (rightResult := 108046) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29096⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30994⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29096⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30994⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108330

namespace LeftMerge108332
def owner : Owner := ⟨.program ⟨257⟩, ⟨30996⟩⟩
def mergeEvent : Nat := 108332
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29096⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30250⟩⟩] } }
def rhsRaw : List Term := Proof.Events422.exact108043RawTerms
def group : MergeGroup := .relation 108331
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108331) (rhsResult := 108043)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29096⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30994⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30994⟩⟩) ⟨30250⟩ 108043) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30250⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29096⟩⟩], [⟨.program ⟨257⟩, ⟨30250⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108332

namespace LeftMerge108346
def owner : Owner := ⟨.program ⟨257⟩, ⟨29859⟩⟩
def mergeEvent : Nat := 108346
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29856⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105245RawTerms
def rightRaw : List Term := Proof.Events423.exact108340RawTerms
def group : MergeGroup := .operator 105245 108340
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105245) (leftOrdinal := 0)
    (rightResult := 108340) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨29856⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29856⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108346

namespace LeftMerge108467
def owner : Owner := ⟨.program ⟨257⟩, ⟨30452⟩⟩
def mergeEvent : Nat := 108467
def frameStart : Nat := 108401
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29096⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events423.exact108463RawTerms
def rightRaw : List Term := Proof.Events423.exact108461RawTerms
def group : MergeGroup := .operator 108463 108461
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108463) (leftOrdinal := 0)
    (rightResult := 108461) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29096⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29096⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108467

namespace LeftMerge108479
def owner : Owner := ⟨.program ⟨257⟩, ⟨30995⟩⟩
def mergeEvent : Nat := 108479
def frameStart : Nat := 108401
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30994⟩⟩] } }
def leftRaw : List Term := Proof.Events423.exact108475RawTerms
def rightRaw : List Term := Proof.Events423.exact108452RawTerms
def group : MergeGroup := .operator 108475 108452
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108475) (leftOrdinal := 0)
    (rightResult := 108452) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30994⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30994⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108479

namespace LeftMerge108480
def owner : Owner := ⟨.program ⟨257⟩, ⟨30995⟩⟩
def mergeEvent : Nat := 108480
def frameStart : Nat := 108401
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29096⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30994⟩⟩] } }
def leftRaw : List Term := Proof.Events423.exact108475RawTerms
def rightRaw : List Term := Proof.Events423.exact108452RawTerms
def group : MergeGroup := .operator 108475 108452
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108475) (leftOrdinal := 1)
    (rightResult := 108452) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29096⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30994⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29096⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30994⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108480

namespace LeftMerge108482
def owner : Owner := ⟨.program ⟨257⟩, ⟨30995⟩⟩
def mergeEvent : Nat := 108482
def frameStart : Nat := 108401
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29096⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30250⟩⟩] } }
def rhsRaw : List Term := Proof.Events423.exact108449RawTerms
def group : MergeGroup := .relation 108481
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108481) (rhsResult := 108449)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29096⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30994⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30994⟩⟩) ⟨30250⟩ 108449) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30250⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29096⟩⟩], [⟨.program ⟨257⟩, ⟨30250⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108482

namespace LeftMerge108490
def owner : Owner := ⟨.program ⟨257⟩, ⟨29313⟩⟩
def mergeEvent : Nat := 108490
def frameStart : Nat := 108401
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29312⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events423.exact108463RawTerms
def rightRaw : List Term := Proof.Events423.exact108486RawTerms
def group : MergeGroup := .operator 108463 108486
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108463) (leftOrdinal := 0)
    (rightResult := 108486) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29312⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108490

namespace LeftMerge108507
def owner : Owner := ⟨.program ⟨257⟩, ⟨29859⟩⟩
def mergeEvent : Nat := 108507
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }
def rhsRaw : List Term := Proof.Events423.exact108504RawTerms
def group : MergeGroup := .relation 108506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108506) (rhsResult := 108504)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 108505 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29856⟩⟩]⟩) (none) 108504) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108507

namespace LeftMerge108508
def owner : Owner := ⟨.program ⟨257⟩, ⟨29859⟩⟩
def mergeEvent : Nat := 108508
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30994⟩⟩] } }
def rhsRaw : List Term := Proof.Events423.exact108504RawTerms
def group : MergeGroup := .relation 108506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108506) (rhsResult := 108504)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 108505 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29856⟩⟩]⟩) (none) 108504) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30994⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30994⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108508

namespace LeftMerge108509
def owner : Owner := ⟨.program ⟨257⟩, ⟨29859⟩⟩
def mergeEvent : Nat := 108509
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29096⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30250⟩⟩] } }
def rhsRaw : List Term := Proof.Events423.exact108504RawTerms
def group : MergeGroup := .relation 108506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108506) (rhsResult := 108504)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 108505 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29856⟩⟩]⟩) (none) 108504) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29096⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30250⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29096⟩⟩], [⟨.program ⟨257⟩, ⟨30250⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108509

namespace LeftMerge108510
def owner : Owner := ⟨.program ⟨257⟩, ⟨29859⟩⟩
def mergeEvent : Nat := 108510
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29312⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events423.exact108504RawTerms
def group : MergeGroup := .relation 108506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108506) (rhsResult := 108504)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 108505 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29856⟩⟩]⟩) (none) 108504) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29312⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108510

namespace LeftMerge108515
def owner : Owner := ⟨.program ⟨257⟩, ⟨30997⟩⟩
def mergeEvent : Nat := 108515
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30994⟩⟩] } }
def leftRaw : List Term := Proof.Events423.exact108511RawTerms
def rightRaw : List Term := Proof.Events423.exact108333RawTerms
def group : MergeGroup := .operator 108511 108333
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108511) (leftOrdinal := 0)
    (rightResult := 108333) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30994⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30994⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30994⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108515

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
