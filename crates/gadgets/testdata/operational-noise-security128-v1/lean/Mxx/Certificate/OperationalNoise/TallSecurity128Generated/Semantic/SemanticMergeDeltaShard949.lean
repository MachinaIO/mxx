import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge156016
def owner : Owner := ⟨.program ⟨257⟩, ⟨33429⟩⟩
def mergeEvent : Nat := 156016
def frameStart : Nat := 155923
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24254⟩⟩, ⟨.program ⟨257⟩, ⟨31404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33426⟩⟩] } }
def leftRaw : List Term := Proof.Events609.exact156011RawTerms
def rightRaw : List Term := Proof.Events609.exact155968RawTerms
def group : MergeGroup := .operator 156011 155968
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 156011) (leftOrdinal := 1)
    (rightResult := 155968) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24254⟩⟩, ⟨.program ⟨257⟩, ⟨31404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33426⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24254⟩⟩, ⟨.program ⟨257⟩, ⟨31404⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33426⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge156016

namespace LeftMerge156018
def owner : Owner := ⟨.program ⟨257⟩, ⟨33429⟩⟩
def mergeEvent : Nat := 156018
def frameStart : Nat := 155923
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24254⟩⟩, ⟨.program ⟨257⟩, ⟨31404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32931⟩⟩] } }
def rhsRaw : List Term := Proof.Events609.exact155965RawTerms
def group : MergeGroup := .relation 156017
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 156017) (rhsResult := 155965)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24254⟩⟩, ⟨.program ⟨257⟩, ⟨31404⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33426⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33426⟩⟩) ⟨32931⟩ 155965) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32931⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24254⟩⟩, ⟨.program ⟨257⟩, ⟨31404⟩⟩], [⟨.program ⟨257⟩, ⟨32931⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge156018

namespace LeftMerge156026
def owner : Owner := ⟨.program ⟨257⟩, ⟨31806⟩⟩
def mergeEvent : Nat := 156026
def frameStart : Nat := 155923
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events609.exact155979RawTerms
def rightRaw : List Term := Proof.Events609.exact156022RawTerms
def group : MergeGroup := .operator 155979 156022
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 155979) (leftOrdinal := 0)
    (rightResult := 156022) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge156026

namespace LeftMerge156043
def owner : Owner := ⟨.program ⟨257⟩, ⟨32362⟩⟩
def mergeEvent : Nat := 156043
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }
def rhsRaw : List Term := Proof.Events609.exact156040RawTerms
def group : MergeGroup := .relation 156042
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 156042) (rhsResult := 156040)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 156041 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32359⟩⟩]⟩) (none) 156040) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge156043

namespace LeftMerge156044
def owner : Owner := ⟨.program ⟨257⟩, ⟨32362⟩⟩
def mergeEvent : Nat := 156044
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33426⟩⟩] } }
def rhsRaw : List Term := Proof.Events609.exact156040RawTerms
def group : MergeGroup := .relation 156042
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 156042) (rhsResult := 156040)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 156041 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32359⟩⟩]⟩) (none) 156040) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33426⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33426⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge156044

namespace LeftMerge156045
def owner : Owner := ⟨.program ⟨257⟩, ⟨32362⟩⟩
def mergeEvent : Nat := 156045
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24254⟩⟩, ⟨.program ⟨257⟩, ⟨31404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32931⟩⟩] } }
def rhsRaw : List Term := Proof.Events609.exact156040RawTerms
def group : MergeGroup := .relation 156042
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 156042) (rhsResult := 156040)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 156041 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32359⟩⟩]⟩) (none) 156040) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24254⟩⟩, ⟨.program ⟨257⟩, ⟨31404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32931⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24254⟩⟩, ⟨.program ⟨257⟩, ⟨31404⟩⟩], [⟨.program ⟨257⟩, ⟨32931⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge156045

namespace LeftMerge156046
def owner : Owner := ⟨.program ⟨257⟩, ⟨32362⟩⟩
def mergeEvent : Nat := 156046
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events609.exact156040RawTerms
def group : MergeGroup := .relation 156042
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 156042) (rhsResult := 156040)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 156041 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32359⟩⟩]⟩) (none) 156040) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨31804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge156046

namespace LeftMerge156051
def owner : Owner := ⟨.program ⟨257⟩, ⟨33428⟩⟩
def mergeEvent : Nat := 156051
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24254⟩⟩, ⟨.program ⟨257⟩, ⟨31404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32931⟩⟩] } }
def leftRaw : List Term := Proof.Events609.exact156047RawTerms
def rightRaw : List Term := Proof.Events608.exact155861RawTerms
def group : MergeGroup := .operator 156047 155861
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 156047) (leftOrdinal := 2)
    (rightResult := 155861) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24254⟩⟩, ⟨.program ⟨257⟩, ⟨31404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32931⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24254⟩⟩, ⟨.program ⟨257⟩, ⟨31404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32931⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24254⟩⟩, ⟨.program ⟨257⟩, ⟨31404⟩⟩], [⟨.program ⟨257⟩, ⟨32931⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge156051

namespace LeftMerge156052
def owner : Owner := ⟨.program ⟨257⟩, ⟨33428⟩⟩
def mergeEvent : Nat := 156052
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33426⟩⟩] } }
def leftRaw : List Term := Proof.Events609.exact156047RawTerms
def rightRaw : List Term := Proof.Events608.exact155861RawTerms
def group : MergeGroup := .operator 156047 155861
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 156047) (leftOrdinal := 1)
    (rightResult := 155861) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33426⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33426⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33426⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge156052

namespace LeftMerge156060
def owner : Owner := ⟨.program ⟨257⟩, ⟨33801⟩⟩
def mergeEvent : Nat := 156060
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33799⟩⟩] } }
def leftRaw : List Term := Proof.Events609.exact156054RawTerms
def rightRaw : List Term := Proof.Events608.exact155777RawTerms
def group : MergeGroup := .operator 156054 155777
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 156054) (leftOrdinal := 0)
    (rightResult := 155777) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33799⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33799⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge156060

namespace LeftMerge156061
def owner : Owner := ⟨.program ⟨257⟩, ⟨33801⟩⟩
def mergeEvent : Nat := 156061
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33799⟩⟩] } }
def leftRaw : List Term := Proof.Events609.exact156054RawTerms
def rightRaw : List Term := Proof.Events608.exact155777RawTerms
def group : MergeGroup := .operator 156054 155777
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 156054) (leftOrdinal := 1)
    (rightResult := 155777) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33799⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨31804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33799⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge156061

namespace LeftMerge156063
def owner : Owner := ⟨.program ⟨257⟩, ⟨33801⟩⟩
def mergeEvent : Nat := 156063
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33074⟩⟩] } }
def rhsRaw : List Term := Proof.Events608.exact155774RawTerms
def group : MergeGroup := .relation 156062
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 156062) (rhsResult := 155774)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨31804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33799⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33799⟩⟩) ⟨33074⟩ 155774) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33074⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨31804⟩⟩], [⟨.program ⟨257⟩, ⟨33074⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge156063

namespace LeftMerge156077
def owner : Owner := ⟨.program ⟨257⟩, ⟨32639⟩⟩
def mergeEvent : Nat := 156077
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32636⟩⟩] } }
def leftRaw : List Term := Proof.Events582.exact149120RawTerms
def rightRaw : List Term := Proof.Events609.exact156071RawTerms
def group : MergeGroup := .operator 149120 156071
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149120) (leftOrdinal := 0)
    (rightResult := 156071) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32636⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32636⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge156077

namespace LeftMerge156198
def owner : Owner := ⟨.program ⟨257⟩, ⟨33296⟩⟩
def mergeEvent : Nat := 156198
def frameStart : Nat := 156132
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events610.exact156194RawTerms
def rightRaw : List Term := Proof.Events610.exact156192RawTerms
def group : MergeGroup := .operator 156194 156192
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 156194) (leftOrdinal := 0)
    (rightResult := 156192) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge156198

namespace LeftMerge156210
def owner : Owner := ⟨.program ⟨257⟩, ⟨33800⟩⟩
def mergeEvent : Nat := 156210
def frameStart : Nat := 156132
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33799⟩⟩] } }
def leftRaw : List Term := Proof.Events610.exact156206RawTerms
def rightRaw : List Term := Proof.Events610.exact156183RawTerms
def group : MergeGroup := .operator 156206 156183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 156206) (leftOrdinal := 0)
    (rightResult := 156183) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33799⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33799⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge156210

namespace LeftMerge156211
def owner : Owner := ⟨.program ⟨257⟩, ⟨33800⟩⟩
def mergeEvent : Nat := 156211
def frameStart : Nat := 156132
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33799⟩⟩] } }
def leftRaw : List Term := Proof.Events610.exact156206RawTerms
def rightRaw : List Term := Proof.Events610.exact156183RawTerms
def group : MergeGroup := .operator 156206 156183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 156206) (leftOrdinal := 1)
    (rightResult := 156183) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33799⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33799⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge156211

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
