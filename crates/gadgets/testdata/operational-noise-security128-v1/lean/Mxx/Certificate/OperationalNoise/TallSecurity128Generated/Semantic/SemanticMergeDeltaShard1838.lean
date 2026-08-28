import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge296937
def owner : Owner := ⟨.program ⟨257⟩, ⟨37772⟩⟩
def mergeEvent : Nat := 296937
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37769⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1159.exact296931RawTerms
def group : MergeGroup := .operator 295195 296931
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 296931) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨37769⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37769⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296937

namespace LeftMerge296992
def owner : Owner := ⟨.program ⟨257⟩, ⟨36875⟩⟩
def mergeEvent : Nat := 296992
def frameStart : Nat := 296974
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13731⟩⟩, ⟨.program ⟨257⟩, ⟨36874⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1160.exact296988RawTerms
def rightRaw : List Term := Proof.Events1160.exact296985RawTerms
def group : MergeGroup := .operator 296988 296985
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 296988) (leftOrdinal := 0)
    (rightResult := 296985) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13731⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨36874⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13731⟩⟩, ⟨.program ⟨257⟩, ⟨36874⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296992

namespace LeftMerge297022
def owner : Owner := ⟨.program ⟨257⟩, ⟨38668⟩⟩
def mergeEvent : Nat := 297022
def frameStart : Nat := 296974
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13731⟩⟩, ⟨.program ⟨257⟩, ⟨36874⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1160.exact297018RawTerms
def rightRaw : List Term := Proof.Events1160.exact297016RawTerms
def group : MergeGroup := .operator 297018 297016
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297018) (leftOrdinal := 0)
    (rightResult := 297016) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13731⟩⟩, ⟨.program ⟨257⟩, ⟨36874⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13731⟩⟩, ⟨.program ⟨257⟩, ⟨36874⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297022

namespace LeftMerge297045
def owner : Owner := ⟨.program ⟨257⟩, ⟨9555⟩⟩
def mergeEvent : Nat := 297045
def frameStart : Nat := 296974
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }
def leftRaw : List Term := Proof.Events1160.exact297041RawTerms
def rightRaw : List Term := Proof.Events1160.exact297038RawTerms
def group : MergeGroup := .operator 297041 297038
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297041) (leftOrdinal := 0)
    (rightResult := 297038) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9553⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297045

namespace LeftMerge297054
def owner : Owner := ⟨.program ⟨257⟩, ⟨38832⟩⟩
def mergeEvent : Nat := 297054
def frameStart : Nat := 296974
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38829⟩⟩] } }
def leftRaw : List Term := Proof.Events1160.exact297050RawTerms
def rightRaw : List Term := Proof.Events1160.exact297007RawTerms
def group : MergeGroup := .operator 297050 297007
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297050) (leftOrdinal := 0)
    (rightResult := 297007) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38829⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38829⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297054

namespace LeftMerge297055
def owner : Owner := ⟨.program ⟨257⟩, ⟨38832⟩⟩
def mergeEvent : Nat := 297055
def frameStart : Nat := 296974
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13731⟩⟩, ⟨.program ⟨257⟩, ⟨36874⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38829⟩⟩] } }
def leftRaw : List Term := Proof.Events1160.exact297050RawTerms
def rightRaw : List Term := Proof.Events1160.exact297007RawTerms
def group : MergeGroup := .operator 297050 297007
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297050) (leftOrdinal := 1)
    (rightResult := 297007) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13731⟩⟩, ⟨.program ⟨257⟩, ⟨36874⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38829⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13731⟩⟩, ⟨.program ⟨257⟩, ⟨36874⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38829⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge297055

namespace LeftMerge297057
def owner : Owner := ⟨.program ⟨257⟩, ⟨38832⟩⟩
def mergeEvent : Nat := 297057
def frameStart : Nat := 296974
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13731⟩⟩, ⟨.program ⟨257⟩, ⟨36874⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38369⟩⟩] } }
def rhsRaw : List Term := Proof.Events1160.exact297004RawTerms
def group : MergeGroup := .relation 297056
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 297056) (rhsResult := 297004)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13731⟩⟩, ⟨.program ⟨257⟩, ⟨36874⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38829⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38829⟩⟩) ⟨38369⟩ 297004) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38369⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13731⟩⟩, ⟨.program ⟨257⟩, ⟨36874⟩⟩], [⟨.program ⟨257⟩, ⟨38369⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge297057

namespace LeftMerge297065
def owner : Owner := ⟨.program ⟨257⟩, ⟨37350⟩⟩
def mergeEvent : Nat := 297065
def frameStart : Nat := 296974
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1160.exact297018RawTerms
def rightRaw : List Term := Proof.Events1160.exact297061RawTerms
def group : MergeGroup := .operator 297018 297061
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297018) (leftOrdinal := 0)
    (rightResult := 297061) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37348⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37348⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297065

namespace LeftMerge297082
def owner : Owner := ⟨.program ⟨257⟩, ⟨37772⟩⟩
def mergeEvent : Nat := 297082
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }
def rhsRaw : List Term := Proof.Events1160.exact297079RawTerms
def group : MergeGroup := .relation 297081
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 297081) (rhsResult := 297079)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37769⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 297080 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37769⟩⟩]⟩) (none) 297079) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297082

namespace LeftMerge297083
def owner : Owner := ⟨.program ⟨257⟩, ⟨37772⟩⟩
def mergeEvent : Nat := 297083
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38829⟩⟩] } }
def rhsRaw : List Term := Proof.Events1160.exact297079RawTerms
def group : MergeGroup := .relation 297081
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 297081) (rhsResult := 297079)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37769⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 297080 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37769⟩⟩]⟩) (none) 297079) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38829⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38829⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge297083

namespace LeftMerge297084
def owner : Owner := ⟨.program ⟨257⟩, ⟨37772⟩⟩
def mergeEvent : Nat := 297084
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13731⟩⟩, ⟨.program ⟨257⟩, ⟨36874⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38369⟩⟩] } }
def rhsRaw : List Term := Proof.Events1160.exact297079RawTerms
def group : MergeGroup := .relation 297081
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 297081) (rhsResult := 297079)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37769⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 297080 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37769⟩⟩]⟩) (none) 297079) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13731⟩⟩, ⟨.program ⟨257⟩, ⟨36874⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38369⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13731⟩⟩, ⟨.program ⟨257⟩, ⟨36874⟩⟩], [⟨.program ⟨257⟩, ⟨38369⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297084

namespace LeftMerge297085
def owner : Owner := ⟨.program ⟨257⟩, ⟨37772⟩⟩
def mergeEvent : Nat := 297085
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1160.exact297079RawTerms
def group : MergeGroup := .relation 297081
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 297081) (rhsResult := 297079)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37769⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 297080 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37769⟩⟩]⟩) (none) 297079) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37348⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge297085

namespace LeftMerge297090
def owner : Owner := ⟨.program ⟨257⟩, ⟨38831⟩⟩
def mergeEvent : Nat := 297090
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13731⟩⟩, ⟨.program ⟨257⟩, ⟨36874⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38369⟩⟩] } }
def leftRaw : List Term := Proof.Events1160.exact297086RawTerms
def rightRaw : List Term := Proof.Events1159.exact296924RawTerms
def group : MergeGroup := .operator 297086 296924
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297086) (leftOrdinal := 2)
    (rightResult := 296924) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13731⟩⟩, ⟨.program ⟨257⟩, ⟨36874⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38369⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13731⟩⟩, ⟨.program ⟨257⟩, ⟨36874⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38369⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13731⟩⟩, ⟨.program ⟨257⟩, ⟨36874⟩⟩], [⟨.program ⟨257⟩, ⟨38369⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge297090

namespace LeftMerge297091
def owner : Owner := ⟨.program ⟨257⟩, ⟨38831⟩⟩
def mergeEvent : Nat := 297091
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38829⟩⟩] } }
def leftRaw : List Term := Proof.Events1160.exact297086RawTerms
def rightRaw : List Term := Proof.Events1159.exact296924RawTerms
def group : MergeGroup := .operator 297086 296924
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297086) (leftOrdinal := 1)
    (rightResult := 296924) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38829⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38829⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38829⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297091

namespace LeftMerge297099
def owner : Owner := ⟨.program ⟨257⟩, ⟨39061⟩⟩
def mergeEvent : Nat := 297099
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39059⟩⟩] } }
def leftRaw : List Term := Proof.Events1160.exact297093RawTerms
def rightRaw : List Term := Proof.Events1159.exact296840RawTerms
def group : MergeGroup := .operator 297093 296840
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297093) (leftOrdinal := 0)
    (rightResult := 296840) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39059⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39059⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297099

namespace LeftMerge297100
def owner : Owner := ⟨.program ⟨257⟩, ⟨39061⟩⟩
def mergeEvent : Nat := 297100
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39059⟩⟩] } }
def leftRaw : List Term := Proof.Events1160.exact297093RawTerms
def rightRaw : List Term := Proof.Events1159.exact296840RawTerms
def group : MergeGroup := .operator 297093 296840
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297093) (leftOrdinal := 1)
    (rightResult := 296840) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39059⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37348⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39059⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge297100

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
