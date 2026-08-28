import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge188801
def owner : Owner := ⟨.program ⟨257⟩, ⟨47420⟩⟩
def mergeEvent : Nat := 188801
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47418⟩⟩] } }
def leftRaw : List Term := Proof.Events699.exact179038RawTerms
def rightRaw : List Term := Proof.Events737.exact188794RawTerms
def group : MergeGroup := .operator 179038 188794
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 179038) (leftOrdinal := 1)
    (rightResult := 188794) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47418⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47418⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188801

namespace LeftMerge188803
def owner : Owner := ⟨.program ⟨257⟩, ⟨47420⟩⟩
def mergeEvent : Nat := 188803
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46647⟩⟩] } }
def rhsRaw : List Term := Proof.Events737.exact188791RawTerms
def group : MergeGroup := .relation 188802
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188802) (rhsResult := 188791)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47418⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47418⟩⟩) ⟨46647⟩ 188791) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46647⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], [⟨.program ⟨257⟩, ⟨46647⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188803

namespace LeftMerge188817
def owner : Owner := ⟨.program ⟨257⟩, ⟨46275⟩⟩
def mergeEvent : Nat := 188817
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46272⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events737.exact188811RawTerms
def group : MergeGroup := .operator 178370 188811
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 188811) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46272⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46272⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188817

namespace LeftMerge188938
def owner : Owner := ⟨.program ⟨257⟩, ⟨46840⟩⟩
def mergeEvent : Nat := 188938
def frameStart : Nat := 188872
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events738.exact188934RawTerms
def rightRaw : List Term := Proof.Events738.exact188932RawTerms
def group : MergeGroup := .operator 188934 188932
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188934) (leftOrdinal := 0)
    (rightResult := 188932) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188938

namespace LeftMerge188950
def owner : Owner := ⟨.program ⟨257⟩, ⟨47419⟩⟩
def mergeEvent : Nat := 188950
def frameStart : Nat := 188872
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47418⟩⟩] } }
def leftRaw : List Term := Proof.Events738.exact188946RawTerms
def rightRaw : List Term := Proof.Events737.exact188923RawTerms
def group : MergeGroup := .operator 188946 188923
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188946) (leftOrdinal := 0)
    (rightResult := 188923) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47418⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47418⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188950

namespace LeftMerge188951
def owner : Owner := ⟨.program ⟨257⟩, ⟨47419⟩⟩
def mergeEvent : Nat := 188951
def frameStart : Nat := 188872
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47418⟩⟩] } }
def leftRaw : List Term := Proof.Events738.exact188946RawTerms
def rightRaw : List Term := Proof.Events737.exact188923RawTerms
def group : MergeGroup := .operator 188946 188923
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188946) (leftOrdinal := 1)
    (rightResult := 188923) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47418⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47418⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188951

namespace LeftMerge188953
def owner : Owner := ⟨.program ⟨257⟩, ⟨47419⟩⟩
def mergeEvent : Nat := 188953
def frameStart : Nat := 188872
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46647⟩⟩] } }
def rhsRaw : List Term := Proof.Events737.exact188920RawTerms
def group : MergeGroup := .relation 188952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188952) (rhsResult := 188920)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47418⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47418⟩⟩) ⟨46647⟩ 188920) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46647⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45492⟩⟩], [⟨.program ⟨257⟩, ⟨46647⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188953

namespace LeftMerge188961
def owner : Owner := ⟨.program ⟨257⟩, ⟨45720⟩⟩
def mergeEvent : Nat := 188961
def frameStart : Nat := 188872
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events738.exact188934RawTerms
def rightRaw : List Term := Proof.Events738.exact188957RawTerms
def group : MergeGroup := .operator 188934 188957
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188934) (leftOrdinal := 0)
    (rightResult := 188957) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45718⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188961

namespace LeftMerge188978
def owner : Owner := ⟨.program ⟨257⟩, ⟨46275⟩⟩
def mergeEvent : Nat := 188978
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7229⟩⟩] } }
def rhsRaw : List Term := Proof.Events738.exact188975RawTerms
def group : MergeGroup := .relation 188977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188977) (rhsResult := 188975)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46272⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 188976 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46272⟩⟩]⟩) (none) 188975) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7229⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188978

namespace LeftMerge188979
def owner : Owner := ⟨.program ⟨257⟩, ⟨46275⟩⟩
def mergeEvent : Nat := 188979
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47418⟩⟩] } }
def rhsRaw : List Term := Proof.Events738.exact188975RawTerms
def group : MergeGroup := .relation 188977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188977) (rhsResult := 188975)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46272⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 188976 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46272⟩⟩]⟩) (none) 188975) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47418⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47418⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188979

namespace LeftMerge188980
def owner : Owner := ⟨.program ⟨257⟩, ⟨46275⟩⟩
def mergeEvent : Nat := 188980
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46647⟩⟩] } }
def rhsRaw : List Term := Proof.Events738.exact188975RawTerms
def group : MergeGroup := .relation 188977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188977) (rhsResult := 188975)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46272⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 188976 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46272⟩⟩]⟩) (none) 188975) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46647⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], [⟨.program ⟨257⟩, ⟨46647⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188980

namespace LeftMerge188981
def owner : Owner := ⟨.program ⟨257⟩, ⟨46275⟩⟩
def mergeEvent : Nat := 188981
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events738.exact188975RawTerms
def group : MergeGroup := .relation 188977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188977) (rhsResult := 188975)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46272⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 188976 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46272⟩⟩]⟩) (none) 188975) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188981

namespace LeftMerge188986
def owner : Owner := ⟨.program ⟨257⟩, ⟨47421⟩⟩
def mergeEvent : Nat := 188986
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47418⟩⟩] } }
def leftRaw : List Term := Proof.Events738.exact188982RawTerms
def rightRaw : List Term := Proof.Events737.exact188804RawTerms
def group : MergeGroup := .operator 188982 188804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188982) (leftOrdinal := 0)
    (rightResult := 188804) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47418⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47418⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47418⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188986

namespace LeftMerge188987
def owner : Owner := ⟨.program ⟨257⟩, ⟨47421⟩⟩
def mergeEvent : Nat := 188987
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46647⟩⟩] } }
def leftRaw : List Term := Proof.Events738.exact188982RawTerms
def rightRaw : List Term := Proof.Events737.exact188804RawTerms
def group : MergeGroup := .operator 188982 188804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188982) (leftOrdinal := 2)
    (rightResult := 188804) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46647⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46647⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], [⟨.program ⟨257⟩, ⟨46647⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188987

namespace LeftMerge188995
def owner : Owner := ⟨.program ⟨257⟩, ⟨47422⟩⟩
def mergeEvent : Nat := 188995
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩] } }
def leftRaw : List Term := Proof.Events738.exact188989RawTerms
def rightRaw : List Term := Proof.Events060.exact15562RawTerms
def group : MergeGroup := .operator 188989 15562
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188989) (leftOrdinal := 0)
    (rightResult := 15562) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7229⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7151⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188995

namespace LeftMerge188996
def owner : Owner := ⟨.program ⟨257⟩, ⟨47422⟩⟩
def mergeEvent : Nat := 188996
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩] } }
def leftRaw : List Term := Proof.Events738.exact188989RawTerms
def rightRaw : List Term := Proof.Events060.exact15562RawTerms
def group : MergeGroup := .operator 188989 15562
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188989) (leftOrdinal := 1)
    (rightResult := 15562) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7151⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188996

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
