import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge175373
def owner : Owner := ⟨.program ⟨257⟩, ⟨30464⟩⟩
def mergeEvent : Nat := 175373
def frameStart : Nat := 175307
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events685.exact175369RawTerms
def rightRaw : List Term := Proof.Events685.exact175367RawTerms
def group : MergeGroup := .operator 175369 175367
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 175369) (leftOrdinal := 0)
    (rightResult := 175367) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29120⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge175373

namespace LeftMerge175385
def owner : Owner := ⟨.program ⟨257⟩, ⟨31064⟩⟩
def mergeEvent : Nat := 175385
def frameStart : Nat := 175307
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31063⟩⟩] } }
def leftRaw : List Term := Proof.Events685.exact175381RawTerms
def rightRaw : List Term := Proof.Events684.exact175358RawTerms
def group : MergeGroup := .operator 175381 175358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 175381) (leftOrdinal := 0)
    (rightResult := 175358) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31063⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31063⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge175385

namespace LeftMerge175386
def owner : Owner := ⟨.program ⟨257⟩, ⟨31064⟩⟩
def mergeEvent : Nat := 175386
def frameStart : Nat := 175307
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31063⟩⟩] } }
def leftRaw : List Term := Proof.Events685.exact175381RawTerms
def rightRaw : List Term := Proof.Events684.exact175358RawTerms
def group : MergeGroup := .operator 175381 175358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 175381) (leftOrdinal := 1)
    (rightResult := 175358) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31063⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31063⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge175386

namespace LeftMerge175388
def owner : Owner := ⟨.program ⟨257⟩, ⟨31064⟩⟩
def mergeEvent : Nat := 175388
def frameStart : Nat := 175307
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30276⟩⟩] } }
def rhsRaw : List Term := Proof.Events684.exact175355RawTerms
def group : MergeGroup := .relation 175387
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 175387) (rhsResult := 175355)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31063⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨31063⟩⟩) ⟨30276⟩ 175355) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30276⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29120⟩⟩], [⟨.program ⟨257⟩, ⟨30276⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge175388

namespace LeftMerge175396
def owner : Owner := ⟨.program ⟨257⟩, ⟨29356⟩⟩
def mergeEvent : Nat := 175396
def frameStart : Nat := 175307
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events685.exact175369RawTerms
def rightRaw : List Term := Proof.Events685.exact175392RawTerms
def group : MergeGroup := .operator 175369 175392
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 175369) (leftOrdinal := 0)
    (rightResult := 175392) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29354⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge175396

namespace LeftMerge175413
def owner : Owner := ⟨.program ⟨257⟩, ⟨29915⟩⟩
def mergeEvent : Nat := 175413
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7219⟩⟩] } }
def rhsRaw : List Term := Proof.Events685.exact175410RawTerms
def group : MergeGroup := .relation 175412
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 175412) (rhsResult := 175410)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29912⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 175411 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29912⟩⟩]⟩) (none) 175410) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7219⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge175413

namespace LeftMerge175414
def owner : Owner := ⟨.program ⟨257⟩, ⟨29915⟩⟩
def mergeEvent : Nat := 175414
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31063⟩⟩] } }
def rhsRaw : List Term := Proof.Events685.exact175410RawTerms
def group : MergeGroup := .relation 175412
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 175412) (rhsResult := 175410)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29912⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 175411 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29912⟩⟩]⟩) (none) 175410) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31063⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31063⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge175414

namespace LeftMerge175415
def owner : Owner := ⟨.program ⟨257⟩, ⟨29915⟩⟩
def mergeEvent : Nat := 175415
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30276⟩⟩] } }
def rhsRaw : List Term := Proof.Events685.exact175410RawTerms
def group : MergeGroup := .relation 175412
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 175412) (rhsResult := 175410)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29912⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 175411 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29912⟩⟩]⟩) (none) 175410) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30276⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29120⟩⟩], [⟨.program ⟨257⟩, ⟨30276⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge175415

namespace LeftMerge175416
def owner : Owner := ⟨.program ⟨257⟩, ⟨29915⟩⟩
def mergeEvent : Nat := 175416
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events685.exact175410RawTerms
def group : MergeGroup := .relation 175412
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 175412) (rhsResult := 175410)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29912⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 175411 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29912⟩⟩]⟩) (none) 175410) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge175416

namespace LeftMerge175421
def owner : Owner := ⟨.program ⟨257⟩, ⟨31066⟩⟩
def mergeEvent : Nat := 175421
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31063⟩⟩] } }
def leftRaw : List Term := Proof.Events685.exact175417RawTerms
def rightRaw : List Term := Proof.Events684.exact175239RawTerms
def group : MergeGroup := .operator 175417 175239
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 175417) (leftOrdinal := 0)
    (rightResult := 175239) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31063⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31063⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31063⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge175421

namespace LeftMerge175422
def owner : Owner := ⟨.program ⟨257⟩, ⟨31066⟩⟩
def mergeEvent : Nat := 175422
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30276⟩⟩] } }
def leftRaw : List Term := Proof.Events685.exact175417RawTerms
def rightRaw : List Term := Proof.Events684.exact175239RawTerms
def group : MergeGroup := .operator 175417 175239
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 175417) (leftOrdinal := 2)
    (rightResult := 175239) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30276⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30276⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29120⟩⟩], [⟨.program ⟨257⟩, ⟨30276⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge175422

namespace LeftMerge175430
def owner : Owner := ⟨.program ⟨257⟩, ⟨31067⟩⟩
def mergeEvent : Nat := 175430
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩] } }
def leftRaw : List Term := Proof.Events685.exact175424RawTerms
def rightRaw : List Term := Proof.Events061.exact15662RawTerms
def group : MergeGroup := .operator 175424 15662
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 175424) (leftOrdinal := 0)
    (rightResult := 15662) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7219⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7167⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge175430

namespace LeftMerge175431
def owner : Owner := ⟨.program ⟨257⟩, ⟨31067⟩⟩
def mergeEvent : Nat := 175431
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩] } }
def leftRaw : List Term := Proof.Events685.exact175424RawTerms
def rightRaw : List Term := Proof.Events061.exact15662RawTerms
def group : MergeGroup := .operator 175424 15662
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 175424) (leftOrdinal := 1)
    (rightResult := 15662) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7167⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge175431

namespace LeftMerge175433
def owner : Owner := ⟨.program ⟨257⟩, ⟨31067⟩⟩
def mergeEvent : Nat := 175433
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15655RawTerms
def group : MergeGroup := .relation 175432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 175432) (rhsResult := 15655)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7167⟩⟩) ⟨7049⟩ 15655) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge175433

namespace LeftMerge175447
def owner : Owner := ⟨.program ⟨257⟩, ⟨28385⟩⟩
def mergeEvent : Nat := 175447
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28383⟩⟩] } }
def leftRaw : List Term := Proof.Events653.exact167305RawTerms
def rightRaw : List Term := Proof.Events685.exact175441RawTerms
def group : MergeGroup := .operator 167305 175441
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167305) (leftOrdinal := 0)
    (rightResult := 175441) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28383⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28383⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge175447

namespace LeftMerge175448
def owner : Owner := ⟨.program ⟨257⟩, ⟨28385⟩⟩
def mergeEvent : Nat := 175448
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨26440⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28383⟩⟩] } }
def leftRaw : List Term := Proof.Events653.exact167305RawTerms
def rightRaw : List Term := Proof.Events685.exact175441RawTerms
def group : MergeGroup := .operator 167305 175441
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167305) (leftOrdinal := 1)
    (rightResult := 175441) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨26440⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28383⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨26440⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28383⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge175448

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
