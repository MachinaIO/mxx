import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge142399
def owner : Owner := ⟨.program ⟨257⟩, ⟨20437⟩⟩
def mergeEvent : Nat := 142399
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20435⟩⟩] } }
def leftRaw : List Term := Proof.Events556.exact142393RawTerms
def rightRaw : List Term := Proof.Events555.exact142116RawTerms
def group : MergeGroup := .operator 142393 142116
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142393) (leftOrdinal := 0)
    (rightResult := 142116) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20435⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20435⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142399

namespace LeftMerge142400
def owner : Owner := ⟨.program ⟨257⟩, ⟨20437⟩⟩
def mergeEvent : Nat := 142400
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20435⟩⟩] } }
def leftRaw : List Term := Proof.Events556.exact142393RawTerms
def rightRaw : List Term := Proof.Events555.exact142116RawTerms
def group : MergeGroup := .operator 142393 142116
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142393) (leftOrdinal := 1)
    (rightResult := 142116) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20435⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20435⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge142400

namespace LeftMerge142402
def owner : Owner := ⟨.program ⟨257⟩, ⟨20437⟩⟩
def mergeEvent : Nat := 142402
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19798⟩⟩] } }
def rhsRaw : List Term := Proof.Events555.exact142113RawTerms
def group : MergeGroup := .relation 142401
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 142401) (rhsResult := 142113)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20435⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20435⟩⟩) ⟨19798⟩ 142113) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19798⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18532⟩⟩], [⟨.program ⟨257⟩, ⟨19798⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge142402

namespace LeftMerge142416
def owner : Owner := ⟨.program ⟨257⟩, ⟨19319⟩⟩
def mergeEvent : Nat := 142416
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19316⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134495RawTerms
def rightRaw : List Term := Proof.Events556.exact142410RawTerms
def group : MergeGroup := .operator 134495 142410
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134495) (leftOrdinal := 0)
    (rightResult := 142410) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19316⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19316⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142416

namespace LeftMerge142537
def owner : Owner := ⟨.program ⟨257⟩, ⟨20040⟩⟩
def mergeEvent : Nat := 142537
def frameStart : Nat := 142471
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events556.exact142533RawTerms
def rightRaw : List Term := Proof.Events556.exact142531RawTerms
def group : MergeGroup := .operator 142533 142531
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142533) (leftOrdinal := 0)
    (rightResult := 142531) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18532⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142537

namespace LeftMerge142549
def owner : Owner := ⟨.program ⟨257⟩, ⟨20436⟩⟩
def mergeEvent : Nat := 142549
def frameStart : Nat := 142471
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20435⟩⟩] } }
def leftRaw : List Term := Proof.Events556.exact142545RawTerms
def rightRaw : List Term := Proof.Events556.exact142522RawTerms
def group : MergeGroup := .operator 142545 142522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142545) (leftOrdinal := 0)
    (rightResult := 142522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20435⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20435⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142549

namespace LeftMerge142550
def owner : Owner := ⟨.program ⟨257⟩, ⟨20436⟩⟩
def mergeEvent : Nat := 142550
def frameStart : Nat := 142471
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20435⟩⟩] } }
def leftRaw : List Term := Proof.Events556.exact142545RawTerms
def rightRaw : List Term := Proof.Events556.exact142522RawTerms
def group : MergeGroup := .operator 142545 142522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142545) (leftOrdinal := 1)
    (rightResult := 142522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20435⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20435⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge142550

namespace LeftMerge142552
def owner : Owner := ⟨.program ⟨257⟩, ⟨20436⟩⟩
def mergeEvent : Nat := 142552
def frameStart : Nat := 142471
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19798⟩⟩] } }
def rhsRaw : List Term := Proof.Events556.exact142519RawTerms
def group : MergeGroup := .relation 142551
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 142551) (rhsResult := 142519)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨18532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20435⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20435⟩⟩) ⟨19798⟩ 142519) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19798⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨18532⟩⟩], [⟨.program ⟨257⟩, ⟨19798⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge142552

namespace LeftMerge142560
def owner : Owner := ⟨.program ⟨257⟩, ⟨18735⟩⟩
def mergeEvent : Nat := 142560
def frameStart : Nat := 142471
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18733⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events556.exact142533RawTerms
def rightRaw : List Term := Proof.Events556.exact142556RawTerms
def group : MergeGroup := .operator 142533 142556
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142533) (leftOrdinal := 0)
    (rightResult := 142556) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18733⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142560

namespace LeftMerge142577
def owner : Owner := ⟨.program ⟨257⟩, ⟨19319⟩⟩
def mergeEvent : Nat := 142577
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }
def rhsRaw : List Term := Proof.Events556.exact142574RawTerms
def group : MergeGroup := .relation 142576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 142576) (rhsResult := 142574)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19316⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 142575 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19316⟩⟩]⟩) (none) 142574) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142577

namespace LeftMerge142578
def owner : Owner := ⟨.program ⟨257⟩, ⟨19319⟩⟩
def mergeEvent : Nat := 142578
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20435⟩⟩] } }
def rhsRaw : List Term := Proof.Events556.exact142574RawTerms
def group : MergeGroup := .relation 142576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 142576) (rhsResult := 142574)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19316⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 142575 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19316⟩⟩]⟩) (none) 142574) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20435⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20435⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge142578

namespace LeftMerge142579
def owner : Owner := ⟨.program ⟨257⟩, ⟨19319⟩⟩
def mergeEvent : Nat := 142579
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19798⟩⟩] } }
def rhsRaw : List Term := Proof.Events556.exact142574RawTerms
def group : MergeGroup := .relation 142576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 142576) (rhsResult := 142574)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19316⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 142575 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19316⟩⟩]⟩) (none) 142574) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19798⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18532⟩⟩], [⟨.program ⟨257⟩, ⟨19798⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142579

namespace LeftMerge142580
def owner : Owner := ⟨.program ⟨257⟩, ⟨19319⟩⟩
def mergeEvent : Nat := 142580
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events556.exact142574RawTerms
def group : MergeGroup := .relation 142576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 142576) (rhsResult := 142574)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19316⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 142575 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19316⟩⟩]⟩) (none) 142574) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18733⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge142580

namespace LeftMerge142585
def owner : Owner := ⟨.program ⟨257⟩, ⟨20438⟩⟩
def mergeEvent : Nat := 142585
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20435⟩⟩] } }
def leftRaw : List Term := Proof.Events556.exact142581RawTerms
def rightRaw : List Term := Proof.Events556.exact142403RawTerms
def group : MergeGroup := .operator 142581 142403
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142581) (leftOrdinal := 0)
    (rightResult := 142403) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20435⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20435⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20435⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142585

namespace LeftMerge142586
def owner : Owner := ⟨.program ⟨257⟩, ⟨20438⟩⟩
def mergeEvent : Nat := 142586
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19798⟩⟩] } }
def leftRaw : List Term := Proof.Events556.exact142581RawTerms
def rightRaw : List Term := Proof.Events556.exact142403RawTerms
def group : MergeGroup := .operator 142581 142403
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142581) (leftOrdinal := 2)
    (rightResult := 142403) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19798⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19798⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18532⟩⟩], [⟨.program ⟨257⟩, ⟨19798⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge142586

namespace LeftMerge142612
def owner : Owner := ⟨.program ⟨257⟩, ⟨15309⟩⟩
def mergeEvent : Nat := 142612
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events025.exact6469RawTerms
def rightRaw : List Term := Proof.Events525.exact134403RawTerms
def group : MergeGroup := .operator 6469 134403
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6469) (leftOrdinal := 0)
    (rightResult := 134403) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15306⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142612

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
