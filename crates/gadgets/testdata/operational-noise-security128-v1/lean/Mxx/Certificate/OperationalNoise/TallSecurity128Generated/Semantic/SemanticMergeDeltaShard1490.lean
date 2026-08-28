import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge241883
def owner : Owner := ⟨.program ⟨257⟩, ⟨61832⟩⟩
def mergeEvent : Nat := 241883
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨59812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61830⟩⟩] } }
def leftRaw : List Term := Proof.Events944.exact241876RawTerms
def rightRaw : List Term := Proof.Events943.exact241599RawTerms
def group : MergeGroup := .operator 241876 241599
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241876) (leftOrdinal := 1)
    (rightResult := 241599) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨59812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61830⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨59812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61830⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241883

namespace LeftMerge241885
def owner : Owner := ⟨.program ⟨257⟩, ⟨61832⟩⟩
def mergeEvent : Nat := 241885
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨59812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61083⟩⟩] } }
def rhsRaw : List Term := Proof.Events943.exact241596RawTerms
def group : MergeGroup := .relation 241884
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 241884) (rhsResult := 241596)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨59812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61830⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61830⟩⟩) ⟨61083⟩ 241596) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61083⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨59812⟩⟩], [⟨.program ⟨257⟩, ⟨61083⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241885

namespace LeftMerge241899
def owner : Owner := ⟨.program ⟨257⟩, ⟨60659⟩⟩
def mergeEvent : Nat := 241899
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60656⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236870RawTerms
def rightRaw : List Term := Proof.Events944.exact241893RawTerms
def group : MergeGroup := .operator 236870 241893
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236870) (leftOrdinal := 0)
    (rightResult := 241893) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60656⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60656⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241899

namespace LeftMerge242020
def owner : Owner := ⟨.program ⟨257⟩, ⟨61300⟩⟩
def mergeEvent : Nat := 242020
def frameStart : Nat := 241954
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events945.exact242016RawTerms
def rightRaw : List Term := Proof.Events945.exact242014RawTerms
def group : MergeGroup := .operator 242016 242014
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242016) (leftOrdinal := 0)
    (rightResult := 242014) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59812⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242020

namespace LeftMerge242032
def owner : Owner := ⟨.program ⟨257⟩, ⟨61831⟩⟩
def mergeEvent : Nat := 242032
def frameStart : Nat := 241954
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61830⟩⟩] } }
def leftRaw : List Term := Proof.Events945.exact242028RawTerms
def rightRaw : List Term := Proof.Events945.exact242005RawTerms
def group : MergeGroup := .operator 242028 242005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242028) (leftOrdinal := 0)
    (rightResult := 242005) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61830⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61830⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242032

namespace LeftMerge242033
def owner : Owner := ⟨.program ⟨257⟩, ⟨61831⟩⟩
def mergeEvent : Nat := 242033
def frameStart : Nat := 241954
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61830⟩⟩] } }
def leftRaw : List Term := Proof.Events945.exact242028RawTerms
def rightRaw : List Term := Proof.Events945.exact242005RawTerms
def group : MergeGroup := .operator 242028 242005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242028) (leftOrdinal := 1)
    (rightResult := 242005) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61830⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61830⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242033

namespace LeftMerge242035
def owner : Owner := ⟨.program ⟨257⟩, ⟨61831⟩⟩
def mergeEvent : Nat := 242035
def frameStart : Nat := 241954
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61083⟩⟩] } }
def rhsRaw : List Term := Proof.Events945.exact242002RawTerms
def group : MergeGroup := .relation 242034
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 242034) (rhsResult := 242002)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨59812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61830⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61830⟩⟩) ⟨61083⟩ 242002) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61083⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨59812⟩⟩], [⟨.program ⟨257⟩, ⟨61083⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242035

namespace LeftMerge242043
def owner : Owner := ⟨.program ⟨257⟩, ⟨60065⟩⟩
def mergeEvent : Nat := 242043
def frameStart : Nat := 241954
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60063⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events945.exact242016RawTerms
def rightRaw : List Term := Proof.Events945.exact242039RawTerms
def group : MergeGroup := .operator 242016 242039
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242016) (leftOrdinal := 0)
    (rightResult := 242039) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60063⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨60063⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242043

namespace LeftMerge242060
def owner : Owner := ⟨.program ⟨257⟩, ⟨60659⟩⟩
def mergeEvent : Nat := 242060
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }
def rhsRaw : List Term := Proof.Events945.exact242057RawTerms
def group : MergeGroup := .relation 242059
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 242059) (rhsResult := 242057)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60656⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 242058 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60656⟩⟩]⟩) (none) 242057) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242060

namespace LeftMerge242061
def owner : Owner := ⟨.program ⟨257⟩, ⟨60659⟩⟩
def mergeEvent : Nat := 242061
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61830⟩⟩] } }
def rhsRaw : List Term := Proof.Events945.exact242057RawTerms
def group : MergeGroup := .relation 242059
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 242059) (rhsResult := 242057)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60656⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 242058 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60656⟩⟩]⟩) (none) 242057) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61830⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61830⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242061

namespace LeftMerge242062
def owner : Owner := ⟨.program ⟨257⟩, ⟨60659⟩⟩
def mergeEvent : Nat := 242062
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨59812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61083⟩⟩] } }
def rhsRaw : List Term := Proof.Events945.exact242057RawTerms
def group : MergeGroup := .relation 242059
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 242059) (rhsResult := 242057)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60656⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 242058 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60656⟩⟩]⟩) (none) 242057) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61083⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨59812⟩⟩], [⟨.program ⟨257⟩, ⟨61083⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242062

namespace LeftMerge242063
def owner : Owner := ⟨.program ⟨257⟩, ⟨60659⟩⟩
def mergeEvent : Nat := 242063
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨60063⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events945.exact242057RawTerms
def group : MergeGroup := .relation 242059
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 242059) (rhsResult := 242057)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60656⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 242058 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60656⟩⟩]⟩) (none) 242057) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60063⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨60063⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242063

namespace LeftMerge242068
def owner : Owner := ⟨.program ⟨257⟩, ⟨61833⟩⟩
def mergeEvent : Nat := 242068
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61830⟩⟩] } }
def leftRaw : List Term := Proof.Events945.exact242064RawTerms
def rightRaw : List Term := Proof.Events944.exact241886RawTerms
def group : MergeGroup := .operator 242064 241886
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242064) (leftOrdinal := 0)
    (rightResult := 241886) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61830⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61830⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61830⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242068

namespace LeftMerge242069
def owner : Owner := ⟨.program ⟨257⟩, ⟨61833⟩⟩
def mergeEvent : Nat := 242069
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨59812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61083⟩⟩] } }
def leftRaw : List Term := Proof.Events945.exact242064RawTerms
def rightRaw : List Term := Proof.Events944.exact241886RawTerms
def group : MergeGroup := .operator 242064 241886
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242064) (leftOrdinal := 2)
    (rightResult := 241886) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨59812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61083⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨59812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61083⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨59812⟩⟩], [⟨.program ⟨257⟩, ⟨61083⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242069

namespace LeftMerge242095
def owner : Owner := ⟨.program ⟨257⟩, ⟨24987⟩⟩
def mergeEvent : Nat := 242095
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events045.exact11567RawTerms
def rightRaw : List Term := Proof.Events924.exact236778RawTerms
def group : MergeGroup := .operator 11567 236778
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11567) (leftOrdinal := 0)
    (rightResult := 236778) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24986⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242095

namespace LeftMerge242100
def owner : Owner := ⟨.program ⟨257⟩, ⟨8351⟩⟩
def mergeEvent : Nat := 242100
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }
def leftRaw : List Term := Proof.Events924.exact236648RawTerms
def rightRaw : List Term := Proof.Events088.exact22591RawTerms
def group : MergeGroup := .operator 236648 22591
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236648) (leftOrdinal := 0)
    (rightResult := 22591) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242100

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
