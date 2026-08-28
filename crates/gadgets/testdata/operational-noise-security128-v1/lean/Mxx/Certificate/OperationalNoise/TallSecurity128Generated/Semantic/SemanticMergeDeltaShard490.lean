import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge82893
def owner : Owner := ⟨.program ⟨257⟩, ⟨33528⟩⟩
def mergeEvent : Nat := 82893
def frameStart : Nat := 82798
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32985⟩⟩] } }
def rhsRaw : List Term := Proof.Events323.exact82840RawTerms
def group : MergeGroup := .relation 82892
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 82892) (rhsResult := 82840)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33525⟩⟩) ⟨32985⟩ 82840) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32985⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨32985⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge82893

namespace LeftMerge82901
def owner : Owner := ⟨.program ⟨257⟩, ⟨31878⟩⟩
def mergeEvent : Nat := 82901
def frameStart : Nat := 82798
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events323.exact82854RawTerms
def rightRaw : List Term := Proof.Events323.exact82897RawTerms
def group : MergeGroup := .operator 82854 82897
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 82854) (leftOrdinal := 0)
    (rightResult := 82897) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82901

namespace LeftMerge82918
def owner : Owner := ⟨.program ⟨257⟩, ⟨32452⟩⟩
def mergeEvent : Nat := 82918
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }
def rhsRaw : List Term := Proof.Events323.exact82915RawTerms
def group : MergeGroup := .relation 82917
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 82917) (rhsResult := 82915)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 82916 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32449⟩⟩]⟩) (none) 82915) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82918

namespace LeftMerge82919
def owner : Owner := ⟨.program ⟨257⟩, ⟨32452⟩⟩
def mergeEvent : Nat := 82919
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩] } }
def rhsRaw : List Term := Proof.Events323.exact82915RawTerms
def group : MergeGroup := .relation 82917
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 82917) (rhsResult := 82915)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 82916 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32449⟩⟩]⟩) (none) 82915) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge82919

namespace LeftMerge82920
def owner : Owner := ⟨.program ⟨257⟩, ⟨32452⟩⟩
def mergeEvent : Nat := 82920
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32985⟩⟩] } }
def rhsRaw : List Term := Proof.Events323.exact82915RawTerms
def group : MergeGroup := .relation 82917
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 82917) (rhsResult := 82915)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 82916 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32449⟩⟩]⟩) (none) 82915) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32985⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨32985⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82920

namespace LeftMerge82921
def owner : Owner := ⟨.program ⟨257⟩, ⟨32452⟩⟩
def mergeEvent : Nat := 82921
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events323.exact82915RawTerms
def group : MergeGroup := .relation 82917
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 82917) (rhsResult := 82915)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 82916 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32449⟩⟩]⟩) (none) 82915) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge82921

namespace LeftMerge82926
def owner : Owner := ⟨.program ⟨257⟩, ⟨33527⟩⟩
def mergeEvent : Nat := 82926
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32985⟩⟩] } }
def leftRaw : List Term := Proof.Events323.exact82922RawTerms
def rightRaw : List Term := Proof.Events323.exact82736RawTerms
def group : MergeGroup := .operator 82922 82736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 82922) (leftOrdinal := 2)
    (rightResult := 82736) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32985⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32985⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨32985⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge82926

namespace LeftMerge82927
def owner : Owner := ⟨.program ⟨257⟩, ⟨33527⟩⟩
def mergeEvent : Nat := 82927
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩] } }
def leftRaw : List Term := Proof.Events323.exact82922RawTerms
def rightRaw : List Term := Proof.Events323.exact82736RawTerms
def group : MergeGroup := .operator 82922 82736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 82922) (leftOrdinal := 1)
    (rightResult := 82736) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82927

namespace LeftMerge82935
def owner : Owner := ⟨.program ⟨257⟩, ⟨34080⟩⟩
def mergeEvent : Nat := 82935
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩] } }
def leftRaw : List Term := Proof.Events323.exact82929RawTerms
def rightRaw : List Term := Proof.Events322.exact82652RawTerms
def group : MergeGroup := .operator 82929 82652
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 82929) (leftOrdinal := 0)
    (rightResult := 82652) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨34078⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82935

namespace LeftMerge82936
def owner : Owner := ⟨.program ⟨257⟩, ⟨34080⟩⟩
def mergeEvent : Nat := 82936
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩] } }
def leftRaw : List Term := Proof.Events323.exact82929RawTerms
def rightRaw : List Term := Proof.Events322.exact82652RawTerms
def group : MergeGroup := .operator 82929 82652
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 82929) (leftOrdinal := 1)
    (rightResult := 82652) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨34078⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge82936

namespace LeftMerge82938
def owner : Owner := ⟨.program ⟨257⟩, ⟨34080⟩⟩
def mergeEvent : Nat := 82938
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33155⟩⟩] } }
def rhsRaw : List Term := Proof.Events322.exact82649RawTerms
def group : MergeGroup := .relation 82937
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 82937) (rhsResult := 82649)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨34078⟩⟩) ⟨33155⟩ 82649) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33155⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨33155⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge82938

namespace LeftMerge82952
def owner : Owner := ⟨.program ⟨257⟩, ⟨32819⟩⟩
def mergeEvent : Nat := 82952
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32816⟩⟩] } }
def leftRaw : List Term := Proof.Events296.exact75995RawTerms
def rightRaw : List Term := Proof.Events324.exact82946RawTerms
def group : MergeGroup := .operator 75995 82946
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75995) (leftOrdinal := 0)
    (rightResult := 82946) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32816⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32816⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82952

namespace LeftMerge83073
def owner : Owner := ⟨.program ⟨257⟩, ⟨33332⟩⟩
def mergeEvent : Nat := 83073
def frameStart : Nat := 83007
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events324.exact83069RawTerms
def rightRaw : List Term := Proof.Events324.exact83067RawTerms
def group : MergeGroup := .operator 83069 83067
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83069) (leftOrdinal := 0)
    (rightResult := 83067) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83073

namespace LeftMerge83085
def owner : Owner := ⟨.program ⟨257⟩, ⟨34079⟩⟩
def mergeEvent : Nat := 83085
def frameStart : Nat := 83007
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩] } }
def leftRaw : List Term := Proof.Events324.exact83081RawTerms
def rightRaw : List Term := Proof.Events324.exact83058RawTerms
def group : MergeGroup := .operator 83081 83058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83081) (leftOrdinal := 0)
    (rightResult := 83058) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨34078⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83085

namespace LeftMerge83086
def owner : Owner := ⟨.program ⟨257⟩, ⟨34079⟩⟩
def mergeEvent : Nat := 83086
def frameStart : Nat := 83007
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩] } }
def leftRaw : List Term := Proof.Events324.exact83081RawTerms
def rightRaw : List Term := Proof.Events324.exact83058RawTerms
def group : MergeGroup := .operator 83081 83058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83081) (leftOrdinal := 1)
    (rightResult := 83058) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨34078⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83086

namespace LeftMerge83088
def owner : Owner := ⟨.program ⟨257⟩, ⟨34079⟩⟩
def mergeEvent : Nat := 83088
def frameStart : Nat := 83007
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33155⟩⟩] } }
def rhsRaw : List Term := Proof.Events324.exact83055RawTerms
def group : MergeGroup := .relation 83087
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 83087) (rhsResult := 83055)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨34078⟩⟩) ⟨33155⟩ 83055) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33155⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨33155⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83088

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
