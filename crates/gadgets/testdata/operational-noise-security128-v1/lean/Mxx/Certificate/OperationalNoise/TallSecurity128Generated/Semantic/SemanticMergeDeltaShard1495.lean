import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge242832
def owner : Owner := ⟨.program ⟨257⟩, ⟨54412⟩⟩
def mergeEvent : Nat := 242832
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events948.exact242826RawTerms
def group : MergeGroup := .relation 242828
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 242828) (rhsResult := 242826)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54409⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 242827 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54409⟩⟩]⟩) (none) 242826) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242832

namespace LeftMerge242837
def owner : Owner := ⟨.program ⟨257⟩, ⟨55479⟩⟩
def mergeEvent : Nat := 242837
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54977⟩⟩] } }
def leftRaw : List Term := Proof.Events948.exact242833RawTerms
def rightRaw : List Term := Proof.Events947.exact242647RawTerms
def group : MergeGroup := .operator 242833 242647
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242833) (leftOrdinal := 2)
    (rightResult := 242647) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54977⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54977⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨54977⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242837

namespace LeftMerge242838
def owner : Owner := ⟨.program ⟨257⟩, ⟨55479⟩⟩
def mergeEvent : Nat := 242838
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩] } }
def leftRaw : List Term := Proof.Events948.exact242833RawTerms
def rightRaw : List Term := Proof.Events947.exact242647RawTerms
def group : MergeGroup := .operator 242833 242647
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242833) (leftOrdinal := 1)
    (rightResult := 242647) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242838

namespace LeftMerge242846
def owner : Owner := ⟨.program ⟨257⟩, ⟨55872⟩⟩
def mergeEvent : Nat := 242846
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55870⟩⟩] } }
def leftRaw : List Term := Proof.Events948.exact242840RawTerms
def rightRaw : List Term := Proof.Events947.exact242563RawTerms
def group : MergeGroup := .operator 242840 242563
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242840) (leftOrdinal := 0)
    (rightResult := 242563) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55870⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55870⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242846

namespace LeftMerge242847
def owner : Owner := ⟨.program ⟨257⟩, ⟨55872⟩⟩
def mergeEvent : Nat := 242847
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55870⟩⟩] } }
def leftRaw : List Term := Proof.Events948.exact242840RawTerms
def rightRaw : List Term := Proof.Events947.exact242563RawTerms
def group : MergeGroup := .operator 242840 242563
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242840) (leftOrdinal := 1)
    (rightResult := 242563) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55870⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55870⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242847

namespace LeftMerge242849
def owner : Owner := ⟨.program ⟨257⟩, ⟨55872⟩⟩
def mergeEvent : Nat := 242849
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55123⟩⟩] } }
def rhsRaw : List Term := Proof.Events947.exact242560RawTerms
def group : MergeGroup := .relation 242848
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 242848) (rhsResult := 242560)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55870⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55870⟩⟩) ⟨55123⟩ 242560) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55123⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53852⟩⟩], [⟨.program ⟨257⟩, ⟨55123⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242849

namespace LeftMerge242863
def owner : Owner := ⟨.program ⟨257⟩, ⟨54699⟩⟩
def mergeEvent : Nat := 242863
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54696⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236870RawTerms
def rightRaw : List Term := Proof.Events948.exact242857RawTerms
def group : MergeGroup := .operator 236870 242857
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236870) (leftOrdinal := 0)
    (rightResult := 242857) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54696⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54696⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242863

namespace LeftMerge242984
def owner : Owner := ⟨.program ⟨257⟩, ⟨55340⟩⟩
def mergeEvent : Nat := 242984
def frameStart : Nat := 242918
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events949.exact242980RawTerms
def rightRaw : List Term := Proof.Events949.exact242978RawTerms
def group : MergeGroup := .operator 242980 242978
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242980) (leftOrdinal := 0)
    (rightResult := 242978) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53852⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242984

namespace LeftMerge242996
def owner : Owner := ⟨.program ⟨257⟩, ⟨55871⟩⟩
def mergeEvent : Nat := 242996
def frameStart : Nat := 242918
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55870⟩⟩] } }
def leftRaw : List Term := Proof.Events949.exact242992RawTerms
def rightRaw : List Term := Proof.Events949.exact242969RawTerms
def group : MergeGroup := .operator 242992 242969
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242992) (leftOrdinal := 0)
    (rightResult := 242969) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55870⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55870⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242996

namespace LeftMerge242997
def owner : Owner := ⟨.program ⟨257⟩, ⟨55871⟩⟩
def mergeEvent : Nat := 242997
def frameStart : Nat := 242918
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55870⟩⟩] } }
def leftRaw : List Term := Proof.Events949.exact242992RawTerms
def rightRaw : List Term := Proof.Events949.exact242969RawTerms
def group : MergeGroup := .operator 242992 242969
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242992) (leftOrdinal := 1)
    (rightResult := 242969) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55870⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55870⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242997

namespace LeftMerge242999
def owner : Owner := ⟨.program ⟨257⟩, ⟨55871⟩⟩
def mergeEvent : Nat := 242999
def frameStart : Nat := 242918
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55123⟩⟩] } }
def rhsRaw : List Term := Proof.Events949.exact242966RawTerms
def group : MergeGroup := .relation 242998
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 242998) (rhsResult := 242966)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨53852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55870⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55870⟩⟩) ⟨55123⟩ 242966) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55123⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨53852⟩⟩], [⟨.program ⟨257⟩, ⟨55123⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242999

namespace LeftMerge243007
def owner : Owner := ⟨.program ⟨257⟩, ⟨54105⟩⟩
def mergeEvent : Nat := 243007
def frameStart : Nat := 242918
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54103⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events949.exact242980RawTerms
def rightRaw : List Term := Proof.Events949.exact243003RawTerms
def group : MergeGroup := .operator 242980 243003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242980) (leftOrdinal := 0)
    (rightResult := 243003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54103⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨54103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243007

namespace LeftMerge243024
def owner : Owner := ⟨.program ⟨257⟩, ⟨54699⟩⟩
def mergeEvent : Nat := 243024
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }
def rhsRaw : List Term := Proof.Events949.exact243021RawTerms
def group : MergeGroup := .relation 243023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243023) (rhsResult := 243021)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 243022 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54696⟩⟩]⟩) (none) 243021) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243024

namespace LeftMerge243025
def owner : Owner := ⟨.program ⟨257⟩, ⟨54699⟩⟩
def mergeEvent : Nat := 243025
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55870⟩⟩] } }
def rhsRaw : List Term := Proof.Events949.exact243021RawTerms
def group : MergeGroup := .relation 243023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243023) (rhsResult := 243021)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 243022 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54696⟩⟩]⟩) (none) 243021) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55870⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55870⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243025

namespace LeftMerge243026
def owner : Owner := ⟨.program ⟨257⟩, ⟨54699⟩⟩
def mergeEvent : Nat := 243026
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55123⟩⟩] } }
def rhsRaw : List Term := Proof.Events949.exact243021RawTerms
def group : MergeGroup := .relation 243023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243023) (rhsResult := 243021)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 243022 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54696⟩⟩]⟩) (none) 243021) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55123⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53852⟩⟩], [⟨.program ⟨257⟩, ⟨55123⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243026

namespace LeftMerge243027
def owner : Owner := ⟨.program ⟨257⟩, ⟨54699⟩⟩
def mergeEvent : Nat := 243027
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨54103⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events949.exact243021RawTerms
def group : MergeGroup := .relation 243023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243023) (rhsResult := 243021)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 243022 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54696⟩⟩]⟩) (none) 243021) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54103⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨54103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243027

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
