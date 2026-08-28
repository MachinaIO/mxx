import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge153061
def owner : Owner := ⟨.program ⟨257⟩, ⟨65365⟩⟩
def mergeEvent : Nat := 153061
def frameStart : Nat := 153031
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25694⟩⟩, ⟨.program ⟨257⟩, ⟨65364⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events597.exact153057RawTerms
def rightRaw : List Term := Proof.Events597.exact153054RawTerms
def group : MergeGroup := .operator 153057 153054
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 153057) (leftOrdinal := 0)
    (rightResult := 153054) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65364⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25694⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25694⟩⟩, ⟨.program ⟨257⟩, ⟨65364⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge153061

namespace LeftMerge153091
def owner : Owner := ⟨.program ⟨257⟩, ⟨68917⟩⟩
def mergeEvent : Nat := 153091
def frameStart : Nat := 153031
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25694⟩⟩, ⟨.program ⟨257⟩, ⟨65364⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events597.exact153087RawTerms
def rightRaw : List Term := Proof.Events597.exact153085RawTerms
def group : MergeGroup := .operator 153087 153085
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 153087) (leftOrdinal := 0)
    (rightResult := 153085) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25694⟩⟩, ⟨.program ⟨257⟩, ⟨65364⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25694⟩⟩, ⟨.program ⟨257⟩, ⟨65364⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge153091

namespace LeftMerge153114
def owner : Owner := ⟨.program ⟨257⟩, ⟨9543⟩⟩
def mergeEvent : Nat := 153114
def frameStart : Nat := 153031
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }
def leftRaw : List Term := Proof.Events598.exact153110RawTerms
def rightRaw : List Term := Proof.Events598.exact153107RawTerms
def group : MergeGroup := .operator 153110 153107
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 153110) (leftOrdinal := 0)
    (rightResult := 153107) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9541⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge153114

namespace LeftMerge153123
def owner : Owner := ⟨.program ⟨257⟩, ⟨69210⟩⟩
def mergeEvent : Nat := 153123
def frameStart : Nat := 153031
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69207⟩⟩] } }
def leftRaw : List Term := Proof.Events598.exact153119RawTerms
def rightRaw : List Term := Proof.Events597.exact153076RawTerms
def group : MergeGroup := .operator 153119 153076
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 153119) (leftOrdinal := 0)
    (rightResult := 153076) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69207⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69207⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge153123

namespace LeftMerge153124
def owner : Owner := ⟨.program ⟨257⟩, ⟨69210⟩⟩
def mergeEvent : Nat := 153124
def frameStart : Nat := 153031
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25694⟩⟩, ⟨.program ⟨257⟩, ⟨65364⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69207⟩⟩] } }
def leftRaw : List Term := Proof.Events598.exact153119RawTerms
def rightRaw : List Term := Proof.Events597.exact153076RawTerms
def group : MergeGroup := .operator 153119 153076
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 153119) (leftOrdinal := 1)
    (rightResult := 153076) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25694⟩⟩, ⟨.program ⟨257⟩, ⟨65364⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69207⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25694⟩⟩, ⟨.program ⟨257⟩, ⟨65364⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69207⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge153124

namespace LeftMerge153126
def owner : Owner := ⟨.program ⟨257⟩, ⟨69210⟩⟩
def mergeEvent : Nat := 153126
def frameStart : Nat := 153031
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25694⟩⟩, ⟨.program ⟨257⟩, ⟨65364⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68512⟩⟩] } }
def rhsRaw : List Term := Proof.Events597.exact153073RawTerms
def group : MergeGroup := .relation 153125
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 153125) (rhsResult := 153073)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25694⟩⟩, ⟨.program ⟨257⟩, ⟨65364⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69207⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69207⟩⟩) ⟨68512⟩ 153073) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68512⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25694⟩⟩, ⟨.program ⟨257⟩, ⟨65364⟩⟩], [⟨.program ⟨257⟩, ⟨68512⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge153126

namespace LeftMerge153134
def owner : Owner := ⟨.program ⟨257⟩, ⟨65766⟩⟩
def mergeEvent : Nat := 153134
def frameStart : Nat := 153031
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events597.exact153087RawTerms
def rightRaw : List Term := Proof.Events598.exact153130RawTerms
def group : MergeGroup := .operator 153087 153130
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 153087) (leftOrdinal := 0)
    (rightResult := 153130) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65764⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge153134

namespace LeftMerge153151
def owner : Owner := ⟨.program ⟨257⟩, ⟨67743⟩⟩
def mergeEvent : Nat := 153151
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }
def rhsRaw : List Term := Proof.Events598.exact153148RawTerms
def group : MergeGroup := .relation 153150
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 153150) (rhsResult := 153148)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67740⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 153149 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67740⟩⟩]⟩) (none) 153148) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge153151

namespace LeftMerge153152
def owner : Owner := ⟨.program ⟨257⟩, ⟨67743⟩⟩
def mergeEvent : Nat := 153152
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69207⟩⟩] } }
def rhsRaw : List Term := Proof.Events598.exact153148RawTerms
def group : MergeGroup := .relation 153150
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 153150) (rhsResult := 153148)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67740⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 153149 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67740⟩⟩]⟩) (none) 153148) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69207⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69207⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge153152

namespace LeftMerge153153
def owner : Owner := ⟨.program ⟨257⟩, ⟨67743⟩⟩
def mergeEvent : Nat := 153153
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25694⟩⟩, ⟨.program ⟨257⟩, ⟨65364⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68512⟩⟩] } }
def rhsRaw : List Term := Proof.Events598.exact153148RawTerms
def group : MergeGroup := .relation 153150
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 153150) (rhsResult := 153148)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67740⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 153149 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67740⟩⟩]⟩) (none) 153148) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25694⟩⟩, ⟨.program ⟨257⟩, ⟨65364⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68512⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25694⟩⟩, ⟨.program ⟨257⟩, ⟨65364⟩⟩], [⟨.program ⟨257⟩, ⟨68512⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge153153

namespace LeftMerge153154
def owner : Owner := ⟨.program ⟨257⟩, ⟨67743⟩⟩
def mergeEvent : Nat := 153154
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨65764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events598.exact153148RawTerms
def group : MergeGroup := .relation 153150
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 153150) (rhsResult := 153148)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67740⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 153149 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67740⟩⟩]⟩) (none) 153148) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨65764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge153154

namespace LeftMerge153159
def owner : Owner := ⟨.program ⟨257⟩, ⟨69209⟩⟩
def mergeEvent : Nat := 153159
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25694⟩⟩, ⟨.program ⟨257⟩, ⟨65364⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68512⟩⟩] } }
def leftRaw : List Term := Proof.Events598.exact153155RawTerms
def rightRaw : List Term := Proof.Events597.exact152969RawTerms
def group : MergeGroup := .operator 153155 152969
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 153155) (leftOrdinal := 2)
    (rightResult := 152969) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25694⟩⟩, ⟨.program ⟨257⟩, ⟨65364⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68512⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25694⟩⟩, ⟨.program ⟨257⟩, ⟨65364⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68512⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25694⟩⟩, ⟨.program ⟨257⟩, ⟨65364⟩⟩], [⟨.program ⟨257⟩, ⟨68512⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge153159

namespace LeftMerge153160
def owner : Owner := ⟨.program ⟨257⟩, ⟨69209⟩⟩
def mergeEvent : Nat := 153160
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69207⟩⟩] } }
def leftRaw : List Term := Proof.Events598.exact153155RawTerms
def rightRaw : List Term := Proof.Events597.exact152969RawTerms
def group : MergeGroup := .operator 153155 152969
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 153155) (leftOrdinal := 1)
    (rightResult := 152969) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69207⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69207⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69207⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge153160

namespace LeftMerge153168
def owner : Owner := ⟨.program ⟨257⟩, ⟨69942⟩⟩
def mergeEvent : Nat := 153168
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69940⟩⟩] } }
def leftRaw : List Term := Proof.Events598.exact153162RawTerms
def rightRaw : List Term := Proof.Events597.exact152885RawTerms
def group : MergeGroup := .operator 153162 152885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 153162) (leftOrdinal := 0)
    (rightResult := 152885) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69940⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69940⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge153168

namespace LeftMerge153169
def owner : Owner := ⟨.program ⟨257⟩, ⟨69942⟩⟩
def mergeEvent : Nat := 153169
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨65764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69940⟩⟩] } }
def leftRaw : List Term := Proof.Events598.exact153162RawTerms
def rightRaw : List Term := Proof.Events597.exact152885RawTerms
def group : MergeGroup := .operator 153162 152885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 153162) (leftOrdinal := 1)
    (rightResult := 152885) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨65764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69940⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨65764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69940⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge153169

namespace LeftMerge153171
def owner : Owner := ⟨.program ⟨257⟩, ⟨69942⟩⟩
def mergeEvent : Nat := 153171
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨65764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68655⟩⟩] } }
def rhsRaw : List Term := Proof.Events597.exact152882RawTerms
def group : MergeGroup := .relation 153170
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 153170) (rhsResult := 152882)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨65764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69940⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69940⟩⟩) ⟨68655⟩ 152882) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68655⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨65764⟩⟩], [⟨.program ⟨257⟩, ⟨68655⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge153171

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
