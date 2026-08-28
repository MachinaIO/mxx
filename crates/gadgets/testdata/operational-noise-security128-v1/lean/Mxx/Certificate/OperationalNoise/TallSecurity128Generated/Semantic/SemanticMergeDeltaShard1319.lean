import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge215008
def owner : Owner := ⟨.program ⟨257⟩, ⟨21810⟩⟩
def mergeEvent : Nat := 215008
def frameStart : Nat := 214905
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events839.exact214961RawTerms
def rightRaw : List Term := Proof.Events839.exact215004RawTerms
def group : MergeGroup := .operator 214961 215004
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214961) (leftOrdinal := 0)
    (rightResult := 215004) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21808⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215008

namespace LeftMerge215025
def owner : Owner := ⟨.program ⟨257⟩, ⟨22372⟩⟩
def mergeEvent : Nat := 215025
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }
def rhsRaw : List Term := Proof.Events839.exact215022RawTerms
def group : MergeGroup := .relation 215024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 215024) (rhsResult := 215022)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 215023 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22369⟩⟩]⟩) (none) 215022) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215025

namespace LeftMerge215026
def owner : Owner := ⟨.program ⟨257⟩, ⟨22372⟩⟩
def mergeEvent : Nat := 215026
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23439⟩⟩] } }
def rhsRaw : List Term := Proof.Events839.exact215022RawTerms
def group : MergeGroup := .relation 215024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 215024) (rhsResult := 215022)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 215023 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22369⟩⟩]⟩) (none) 215022) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23439⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23439⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge215026

namespace LeftMerge215027
def owner : Owner := ⟨.program ⟨257⟩, ⟨22372⟩⟩
def mergeEvent : Nat := 215027
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21101⟩⟩, ⟨.program ⟨257⟩, ⟨21494⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22929⟩⟩] } }
def rhsRaw : List Term := Proof.Events839.exact215022RawTerms
def group : MergeGroup := .relation 215024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 215024) (rhsResult := 215022)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 215023 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22369⟩⟩]⟩) (none) 215022) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21101⟩⟩, ⟨.program ⟨257⟩, ⟨21494⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22929⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21101⟩⟩, ⟨.program ⟨257⟩, ⟨21494⟩⟩], [⟨.program ⟨257⟩, ⟨22929⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215027

namespace LeftMerge215028
def owner : Owner := ⟨.program ⟨257⟩, ⟨22372⟩⟩
def mergeEvent : Nat := 215028
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events839.exact215022RawTerms
def group : MergeGroup := .relation 215024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 215024) (rhsResult := 215022)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 215023 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22369⟩⟩]⟩) (none) 215022) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge215028

namespace LeftMerge215033
def owner : Owner := ⟨.program ⟨257⟩, ⟨23441⟩⟩
def mergeEvent : Nat := 215033
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21101⟩⟩, ⟨.program ⟨257⟩, ⟨21494⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22929⟩⟩] } }
def leftRaw : List Term := Proof.Events839.exact215029RawTerms
def rightRaw : List Term := Proof.Events839.exact214843RawTerms
def group : MergeGroup := .operator 215029 214843
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215029) (leftOrdinal := 2)
    (rightResult := 214843) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21101⟩⟩, ⟨.program ⟨257⟩, ⟨21494⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22929⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21101⟩⟩, ⟨.program ⟨257⟩, ⟨21494⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22929⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21101⟩⟩, ⟨.program ⟨257⟩, ⟨21494⟩⟩], [⟨.program ⟨257⟩, ⟨22929⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge215033

namespace LeftMerge215034
def owner : Owner := ⟨.program ⟨257⟩, ⟨23441⟩⟩
def mergeEvent : Nat := 215034
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23439⟩⟩] } }
def leftRaw : List Term := Proof.Events839.exact215029RawTerms
def rightRaw : List Term := Proof.Events839.exact214843RawTerms
def group : MergeGroup := .operator 215029 214843
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215029) (leftOrdinal := 1)
    (rightResult := 214843) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23439⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23439⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23439⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215034

namespace LeftMerge215042
def owner : Owner := ⟨.program ⟨257⟩, ⟨23874⟩⟩
def mergeEvent : Nat := 215042
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23872⟩⟩] } }
def leftRaw : List Term := Proof.Events839.exact215036RawTerms
def rightRaw : List Term := Proof.Events838.exact214759RawTerms
def group : MergeGroup := .operator 215036 214759
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215036) (leftOrdinal := 0)
    (rightResult := 214759) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23872⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23872⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215042

namespace LeftMerge215043
def owner : Owner := ⟨.program ⟨257⟩, ⟨23874⟩⟩
def mergeEvent : Nat := 215043
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23872⟩⟩] } }
def leftRaw : List Term := Proof.Events839.exact215036RawTerms
def rightRaw : List Term := Proof.Events838.exact214759RawTerms
def group : MergeGroup := .operator 215036 214759
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215036) (leftOrdinal := 1)
    (rightResult := 214759) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23872⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge215043

namespace LeftMerge215045
def owner : Owner := ⟨.program ⟨257⟩, ⟨23874⟩⟩
def mergeEvent : Nat := 215045
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23081⟩⟩] } }
def rhsRaw : List Term := Proof.Events838.exact214756RawTerms
def group : MergeGroup := .relation 215044
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 215044) (rhsResult := 214756)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23872⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23872⟩⟩) ⟨23081⟩ 214756) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23081⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21808⟩⟩], [⟨.program ⟨257⟩, ⟨23081⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge215045

namespace LeftMerge215059
def owner : Owner := ⟨.program ⟨257⟩, ⟨22679⟩⟩
def mergeEvent : Nat := 215059
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22676⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207620RawTerms
def rightRaw : List Term := Proof.Events840.exact215053RawTerms
def group : MergeGroup := .operator 207620 215053
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207620) (leftOrdinal := 0)
    (rightResult := 215053) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22676⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22676⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215059

namespace LeftMerge215180
def owner : Owner := ⟨.program ⟨257⟩, ⟨23288⟩⟩
def mergeEvent : Nat := 215180
def frameStart : Nat := 215114
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events840.exact215176RawTerms
def rightRaw : List Term := Proof.Events840.exact215174RawTerms
def group : MergeGroup := .operator 215176 215174
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215176) (leftOrdinal := 0)
    (rightResult := 215174) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21808⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215180

namespace LeftMerge215192
def owner : Owner := ⟨.program ⟨257⟩, ⟨23873⟩⟩
def mergeEvent : Nat := 215192
def frameStart : Nat := 215114
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23872⟩⟩] } }
def leftRaw : List Term := Proof.Events840.exact215188RawTerms
def rightRaw : List Term := Proof.Events840.exact215165RawTerms
def group : MergeGroup := .operator 215188 215165
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215188) (leftOrdinal := 0)
    (rightResult := 215165) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23872⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23872⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215192

namespace LeftMerge215193
def owner : Owner := ⟨.program ⟨257⟩, ⟨23873⟩⟩
def mergeEvent : Nat := 215193
def frameStart : Nat := 215114
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23872⟩⟩] } }
def leftRaw : List Term := Proof.Events840.exact215188RawTerms
def rightRaw : List Term := Proof.Events840.exact215165RawTerms
def group : MergeGroup := .operator 215188 215165
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215188) (leftOrdinal := 1)
    (rightResult := 215165) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23872⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge215193

namespace LeftMerge215195
def owner : Owner := ⟨.program ⟨257⟩, ⟨23873⟩⟩
def mergeEvent : Nat := 215195
def frameStart : Nat := 215114
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23081⟩⟩] } }
def rhsRaw : List Term := Proof.Events840.exact215162RawTerms
def group : MergeGroup := .relation 215194
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 215194) (rhsResult := 215162)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23872⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23872⟩⟩) ⟨23081⟩ 215162) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23081⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21808⟩⟩], [⟨.program ⟨257⟩, ⟨23081⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge215195

namespace LeftMerge215203
def owner : Owner := ⟨.program ⟨257⟩, ⟨22088⟩⟩
def mergeEvent : Nat := 215203
def frameStart : Nat := 215114
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events840.exact215176RawTerms
def rightRaw : List Term := Proof.Events840.exact215199RawTerms
def group : MergeGroup := .operator 215176 215199
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215176) (leftOrdinal := 0)
    (rightResult := 215199) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215203

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
