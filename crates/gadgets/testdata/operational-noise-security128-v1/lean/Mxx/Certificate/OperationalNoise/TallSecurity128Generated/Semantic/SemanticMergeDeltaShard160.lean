import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge30627
def owner : Owner := ⟨.program ⟨257⟩, ⟨51002⟩⟩
def mergeEvent : Nat := 30627
def frameStart : Nat := 30538
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events119.exact30600RawTerms
def rightRaw : List Term := Proof.Events119.exact30623RawTerms
def group : MergeGroup := .operator 30600 30623
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30600) (leftOrdinal := 0)
    (rightResult := 30623) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50999⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge30627

namespace LeftMerge30644
def owner : Owner := ⟨.program ⟨257⟩, ⟨51581⟩⟩
def mergeEvent : Nat := 30644
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7205⟩⟩] } }
def rhsRaw : List Term := Proof.Events119.exact30641RawTerms
def group : MergeGroup := .relation 30643
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 30643) (rhsResult := 30641)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51578⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 30642 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51578⟩⟩]⟩) (none) 30641) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7205⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge30644

namespace LeftMerge30645
def owner : Owner := ⟨.program ⟨257⟩, ⟨51581⟩⟩
def mergeEvent : Nat := 30645
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52082⟩⟩] } }
def rhsRaw : List Term := Proof.Events119.exact30641RawTerms
def group : MergeGroup := .relation 30643
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 30643) (rhsResult := 30641)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51578⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 30642 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51578⟩⟩]⟩) (none) 30641) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52082⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50818⟩⟩], [⟨.program ⟨257⟩, ⟨52082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge30645

namespace LeftMerge30646
def owner : Owner := ⟨.program ⟨257⟩, ⟨51581⟩⟩
def mergeEvent : Nat := 30646
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52675⟩⟩] } }
def rhsRaw : List Term := Proof.Events119.exact30641RawTerms
def group : MergeGroup := .relation 30643
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 30643) (rhsResult := 30641)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51578⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 30642 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51578⟩⟩]⟩) (none) 30641) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52675⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52675⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30646

namespace LeftMerge30647
def owner : Owner := ⟨.program ⟨257⟩, ⟨51581⟩⟩
def mergeEvent : Nat := 30647
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events119.exact30641RawTerms
def group : MergeGroup := .relation 30643
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 30643) (rhsResult := 30641)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51578⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 30642 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51578⟩⟩]⟩) (none) 30641) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30647

namespace LeftMerge30652
def owner : Owner := ⟨.program ⟨257⟩, ⟨52678⟩⟩
def mergeEvent : Nat := 30652
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52082⟩⟩] } }
def leftRaw : List Term := Proof.Events119.exact30648RawTerms
def rightRaw : List Term := Proof.Events119.exact30470RawTerms
def group : MergeGroup := .operator 30648 30470
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30648) (leftOrdinal := 2)
    (rightResult := 30470) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52082⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50818⟩⟩], [⟨.program ⟨257⟩, ⟨52082⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30652

namespace LeftMerge30653
def owner : Owner := ⟨.program ⟨257⟩, ⟨52678⟩⟩
def mergeEvent : Nat := 30653
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52675⟩⟩] } }
def leftRaw : List Term := Proof.Events119.exact30648RawTerms
def rightRaw : List Term := Proof.Events119.exact30470RawTerms
def group : MergeGroup := .operator 30648 30470
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30648) (leftOrdinal := 0)
    (rightResult := 30470) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52675⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52675⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52675⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge30653

namespace LeftMerge30661
def owner : Owner := ⟨.program ⟨257⟩, ⟨52679⟩⟩
def mergeEvent : Nat := 30661
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩] } }
def leftRaw : List Term := Proof.Events119.exact30655RawTerms
def rightRaw : List Term := Proof.Events061.exact15802RawTerms
def group : MergeGroup := .operator 30655 15802
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30655) (leftOrdinal := 0)
    (rightResult := 15802) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7205⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7131⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge30661

namespace LeftMerge30662
def owner : Owner := ⟨.program ⟨257⟩, ⟨52679⟩⟩
def mergeEvent : Nat := 30662
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩] } }
def leftRaw : List Term := Proof.Events119.exact30655RawTerms
def rightRaw : List Term := Proof.Events061.exact15802RawTerms
def group : MergeGroup := .operator 30655 15802
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30655) (leftOrdinal := 1)
    (rightResult := 15802) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7131⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30662

namespace LeftMerge30664
def owner : Owner := ⟨.program ⟨257⟩, ⟨52679⟩⟩
def mergeEvent : Nat := 30664
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15795RawTerms
def group : MergeGroup := .relation 30663
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 30663) (rhsResult := 15795)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7131⟩⟩) ⟨7031⟩ 15795) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30664

namespace LeftMerge30678
def owner : Owner := ⟨.program ⟨257⟩, ⟨33617⟩⟩
def mergeEvent : Nat := 30678
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31758⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33615⟩⟩] } }
def leftRaw : List Term := Proof.Events095.exact24369RawTerms
def rightRaw : List Term := Proof.Events119.exact30672RawTerms
def group : MergeGroup := .operator 24369 30672
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24369) (leftOrdinal := 1)
    (rightResult := 30672) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31758⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33615⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31758⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33615⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30678

namespace LeftMerge30680
def owner : Owner := ⟨.program ⟨257⟩, ⟨33617⟩⟩
def mergeEvent : Nat := 30680
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31758⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33022⟩⟩] } }
def rhsRaw : List Term := Proof.Events119.exact30669RawTerms
def group : MergeGroup := .relation 30679
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 30679) (rhsResult := 30669)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31758⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33615⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33615⟩⟩) ⟨33022⟩ 30669) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33022⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31758⟩⟩], [⟨.program ⟨257⟩, ⟨33022⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30680

namespace LeftMerge30681
def owner : Owner := ⟨.program ⟨257⟩, ⟨33617⟩⟩
def mergeEvent : Nat := 30681
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33615⟩⟩] } }
def leftRaw : List Term := Proof.Events095.exact24369RawTerms
def rightRaw : List Term := Proof.Events119.exact30672RawTerms
def group : MergeGroup := .operator 24369 30672
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24369) (leftOrdinal := 0)
    (rightResult := 30672) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33615⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33615⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge30681

namespace LeftMerge30695
def owner : Owner := ⟨.program ⟨257⟩, ⟨32521⟩⟩
def mergeEvent : Nat := 30695
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32518⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events119.exact30689RawTerms
def group : MergeGroup := .operator 17169 30689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 30689) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32518⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32518⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge30695

namespace LeftMerge30816
def owner : Owner := ⟨.program ⟨257⟩, ⟨33272⟩⟩
def mergeEvent : Nat := 30816
def frameStart : Nat := 30750
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31758⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events120.exact30812RawTerms
def rightRaw : List Term := Proof.Events120.exact30810RawTerms
def group : MergeGroup := .operator 30812 30810
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30812) (leftOrdinal := 0)
    (rightResult := 30810) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31758⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31758⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge30816

namespace LeftMerge30828
def owner : Owner := ⟨.program ⟨257⟩, ⟨33616⟩⟩
def mergeEvent : Nat := 30828
def frameStart : Nat := 30750
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31758⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33615⟩⟩] } }
def leftRaw : List Term := Proof.Events120.exact30824RawTerms
def rightRaw : List Term := Proof.Events120.exact30801RawTerms
def group : MergeGroup := .operator 30824 30801
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30824) (leftOrdinal := 1)
    (rightResult := 30801) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31758⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33615⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31758⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33615⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30828

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
