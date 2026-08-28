import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge124848
def owner : Owner := ⟨.program ⟨257⟩, ⟨59798⟩⟩
def mergeEvent : Nat := 124848
def frameStart : Nat := 124745
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events487.exact124801RawTerms
def rightRaw : List Term := Proof.Events487.exact124844RawTerms
def group : MergeGroup := .operator 124801 124844
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124801) (leftOrdinal := 0)
    (rightResult := 124844) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59796⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124848

namespace LeftMerge124865
def owner : Owner := ⟨.program ⟨257⟩, ⟨60352⟩⟩
def mergeEvent : Nat := 124865
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }
def rhsRaw : List Term := Proof.Events487.exact124862RawTerms
def group : MergeGroup := .relation 124864
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 124864) (rhsResult := 124862)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 124863 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60349⟩⟩]⟩) (none) 124862) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124865

namespace LeftMerge124866
def owner : Owner := ⟨.program ⟨257⟩, ⟨60352⟩⟩
def mergeEvent : Nat := 124866
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61415⟩⟩] } }
def rhsRaw : List Term := Proof.Events487.exact124862RawTerms
def group : MergeGroup := .relation 124864
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 124864) (rhsResult := 124862)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 124863 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60349⟩⟩]⟩) (none) 124862) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61415⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61415⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124866

namespace LeftMerge124867
def owner : Owner := ⟨.program ⟨257⟩, ⟨60352⟩⟩
def mergeEvent : Nat := 124867
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60925⟩⟩] } }
def rhsRaw : List Term := Proof.Events487.exact124862RawTerms
def group : MergeGroup := .relation 124864
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 124864) (rhsResult := 124862)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 124863 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60349⟩⟩]⟩) (none) 124862) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60925⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], [⟨.program ⟨257⟩, ⟨60925⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124867

namespace LeftMerge124868
def owner : Owner := ⟨.program ⟨257⟩, ⟨60352⟩⟩
def mergeEvent : Nat := 124868
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨59796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events487.exact124862RawTerms
def group : MergeGroup := .relation 124864
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 124864) (rhsResult := 124862)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 124863 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60349⟩⟩]⟩) (none) 124862) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨59796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124868

namespace LeftMerge124873
def owner : Owner := ⟨.program ⟨257⟩, ⟨61417⟩⟩
def mergeEvent : Nat := 124873
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60925⟩⟩] } }
def leftRaw : List Term := Proof.Events487.exact124869RawTerms
def rightRaw : List Term := Proof.Events487.exact124683RawTerms
def group : MergeGroup := .operator 124869 124683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124869) (leftOrdinal := 2)
    (rightResult := 124683) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60925⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60925⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25202⟩⟩, ⟨.program ⟨257⟩, ⟨59377⟩⟩], [⟨.program ⟨257⟩, ⟨60925⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124873

namespace LeftMerge124874
def owner : Owner := ⟨.program ⟨257⟩, ⟨61417⟩⟩
def mergeEvent : Nat := 124874
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61415⟩⟩] } }
def leftRaw : List Term := Proof.Events487.exact124869RawTerms
def rightRaw : List Term := Proof.Events487.exact124683RawTerms
def group : MergeGroup := .operator 124869 124683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124869) (leftOrdinal := 1)
    (rightResult := 124683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61415⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61415⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61415⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124874

namespace LeftMerge124882
def owner : Owner := ⟨.program ⟨257⟩, ⟨61770⟩⟩
def mergeEvent : Nat := 124882
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61768⟩⟩] } }
def leftRaw : List Term := Proof.Events487.exact124876RawTerms
def rightRaw : List Term := Proof.Events486.exact124599RawTerms
def group : MergeGroup := .operator 124876 124599
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124876) (leftOrdinal := 0)
    (rightResult := 124599) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61768⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61768⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124882

namespace LeftMerge124883
def owner : Owner := ⟨.program ⟨257⟩, ⟨61770⟩⟩
def mergeEvent : Nat := 124883
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨59796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61768⟩⟩] } }
def leftRaw : List Term := Proof.Events487.exact124876RawTerms
def rightRaw : List Term := Proof.Events486.exact124599RawTerms
def group : MergeGroup := .operator 124876 124599
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124876) (leftOrdinal := 1)
    (rightResult := 124599) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨59796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61768⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨59796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61768⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124883

namespace LeftMerge124885
def owner : Owner := ⟨.program ⟨257⟩, ⟨61770⟩⟩
def mergeEvent : Nat := 124885
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨59796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61065⟩⟩] } }
def rhsRaw : List Term := Proof.Events486.exact124596RawTerms
def group : MergeGroup := .relation 124884
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 124884) (rhsResult := 124596)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨59796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61768⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61768⟩⟩) ⟨61065⟩ 124596) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61065⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨59796⟩⟩], [⟨.program ⟨257⟩, ⟨61065⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124885

namespace LeftMerge124899
def owner : Owner := ⟨.program ⟨257⟩, ⟨60619⟩⟩
def mergeEvent : Nat := 124899
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60616⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119870RawTerms
def rightRaw : List Term := Proof.Events487.exact124893RawTerms
def group : MergeGroup := .operator 119870 124893
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119870) (leftOrdinal := 0)
    (rightResult := 124893) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60616⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124899

namespace LeftMerge125020
def owner : Owner := ⟨.program ⟨257⟩, ⟨61292⟩⟩
def mergeEvent : Nat := 125020
def frameStart : Nat := 124954
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events488.exact125016RawTerms
def rightRaw : List Term := Proof.Events488.exact125014RawTerms
def group : MergeGroup := .operator 125016 125014
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125016) (leftOrdinal := 0)
    (rightResult := 125014) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59796⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125020

namespace LeftMerge125032
def owner : Owner := ⟨.program ⟨257⟩, ⟨61769⟩⟩
def mergeEvent : Nat := 125032
def frameStart : Nat := 124954
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61768⟩⟩] } }
def leftRaw : List Term := Proof.Events488.exact125028RawTerms
def rightRaw : List Term := Proof.Events488.exact125005RawTerms
def group : MergeGroup := .operator 125028 125005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125028) (leftOrdinal := 0)
    (rightResult := 125005) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61768⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61768⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125032

namespace LeftMerge125033
def owner : Owner := ⟨.program ⟨257⟩, ⟨61769⟩⟩
def mergeEvent : Nat := 125033
def frameStart : Nat := 124954
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61768⟩⟩] } }
def leftRaw : List Term := Proof.Events488.exact125028RawTerms
def rightRaw : List Term := Proof.Events488.exact125005RawTerms
def group : MergeGroup := .operator 125028 125005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125028) (leftOrdinal := 1)
    (rightResult := 125005) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61768⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61768⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge125033

namespace LeftMerge125035
def owner : Owner := ⟨.program ⟨257⟩, ⟨61769⟩⟩
def mergeEvent : Nat := 125035
def frameStart : Nat := 124954
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61065⟩⟩] } }
def rhsRaw : List Term := Proof.Events488.exact125002RawTerms
def group : MergeGroup := .relation 125034
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 125034) (rhsResult := 125002)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨59796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61768⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61768⟩⟩) ⟨61065⟩ 125002) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61065⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨59796⟩⟩], [⟨.program ⟨257⟩, ⟨61065⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge125035

namespace LeftMerge125043
def owner : Owner := ⟨.program ⟨257⟩, ⟨60027⟩⟩
def mergeEvent : Nat := 125043
def frameStart : Nat := 124954
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60025⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events488.exact125016RawTerms
def rightRaw : List Term := Proof.Events488.exact125039RawTerms
def group : MergeGroup := .operator 125016 125039
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125016) (leftOrdinal := 0)
    (rightResult := 125039) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60025⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨60025⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125043

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
