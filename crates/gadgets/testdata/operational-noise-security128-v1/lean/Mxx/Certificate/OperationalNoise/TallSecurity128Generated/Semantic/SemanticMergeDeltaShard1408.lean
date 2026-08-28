import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge229140
def owner : Owner := ⟨.program ⟨257⟩, ⟨33451⟩⟩
def mergeEvent : Nat := 229140
def frameStart : Nat := 229048
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33448⟩⟩] } }
def leftRaw : List Term := Proof.Events895.exact229136RawTerms
def rightRaw : List Term := Proof.Events894.exact229093RawTerms
def group : MergeGroup := .operator 229136 229093
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 229136) (leftOrdinal := 0)
    (rightResult := 229093) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33448⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33448⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge229140

namespace LeftMerge229141
def owner : Owner := ⟨.program ⟨257⟩, ⟨33451⟩⟩
def mergeEvent : Nat := 229141
def frameStart : Nat := 229048
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24278⟩⟩, ⟨.program ⟨257⟩, ⟨31458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33448⟩⟩] } }
def leftRaw : List Term := Proof.Events895.exact229136RawTerms
def rightRaw : List Term := Proof.Events894.exact229093RawTerms
def group : MergeGroup := .operator 229136 229093
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 229136) (leftOrdinal := 1)
    (rightResult := 229093) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24278⟩⟩, ⟨.program ⟨257⟩, ⟨31458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33448⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24278⟩⟩, ⟨.program ⟨257⟩, ⟨31458⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33448⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge229141

namespace LeftMerge229143
def owner : Owner := ⟨.program ⟨257⟩, ⟨33451⟩⟩
def mergeEvent : Nat := 229143
def frameStart : Nat := 229048
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24278⟩⟩, ⟨.program ⟨257⟩, ⟨31458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32943⟩⟩] } }
def rhsRaw : List Term := Proof.Events894.exact229090RawTerms
def group : MergeGroup := .relation 229142
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 229142) (rhsResult := 229090)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24278⟩⟩, ⟨.program ⟨257⟩, ⟨31458⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33448⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33448⟩⟩) ⟨32943⟩ 229090) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32943⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24278⟩⟩, ⟨.program ⟨257⟩, ⟨31458⟩⟩], [⟨.program ⟨257⟩, ⟨32943⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge229143

namespace LeftMerge229151
def owner : Owner := ⟨.program ⟨257⟩, ⟨31822⟩⟩
def mergeEvent : Nat := 229151
def frameStart : Nat := 229048
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events894.exact229104RawTerms
def rightRaw : List Term := Proof.Events895.exact229147RawTerms
def group : MergeGroup := .operator 229104 229147
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 229104) (leftOrdinal := 0)
    (rightResult := 229147) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31820⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge229151

namespace LeftMerge229168
def owner : Owner := ⟨.program ⟨257⟩, ⟨32382⟩⟩
def mergeEvent : Nat := 229168
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }
def rhsRaw : List Term := Proof.Events895.exact229165RawTerms
def group : MergeGroup := .relation 229167
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 229167) (rhsResult := 229165)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 229166 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32379⟩⟩]⟩) (none) 229165) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge229168

namespace LeftMerge229169
def owner : Owner := ⟨.program ⟨257⟩, ⟨32382⟩⟩
def mergeEvent : Nat := 229169
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33448⟩⟩] } }
def rhsRaw : List Term := Proof.Events895.exact229165RawTerms
def group : MergeGroup := .relation 229167
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 229167) (rhsResult := 229165)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 229166 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32379⟩⟩]⟩) (none) 229165) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33448⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33448⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge229169

namespace LeftMerge229170
def owner : Owner := ⟨.program ⟨257⟩, ⟨32382⟩⟩
def mergeEvent : Nat := 229170
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24278⟩⟩, ⟨.program ⟨257⟩, ⟨31458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32943⟩⟩] } }
def rhsRaw : List Term := Proof.Events895.exact229165RawTerms
def group : MergeGroup := .relation 229167
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 229167) (rhsResult := 229165)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 229166 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32379⟩⟩]⟩) (none) 229165) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24278⟩⟩, ⟨.program ⟨257⟩, ⟨31458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32943⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24278⟩⟩, ⟨.program ⟨257⟩, ⟨31458⟩⟩], [⟨.program ⟨257⟩, ⟨32943⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge229170

namespace LeftMerge229171
def owner : Owner := ⟨.program ⟨257⟩, ⟨32382⟩⟩
def mergeEvent : Nat := 229171
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨31820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events895.exact229165RawTerms
def group : MergeGroup := .relation 229167
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 229167) (rhsResult := 229165)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 229166 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32379⟩⟩]⟩) (none) 229165) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨31820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge229171

namespace LeftMerge229176
def owner : Owner := ⟨.program ⟨257⟩, ⟨33450⟩⟩
def mergeEvent : Nat := 229176
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24278⟩⟩, ⟨.program ⟨257⟩, ⟨31458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32943⟩⟩] } }
def leftRaw : List Term := Proof.Events895.exact229172RawTerms
def rightRaw : List Term := Proof.Events894.exact228986RawTerms
def group : MergeGroup := .operator 229172 228986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 229172) (leftOrdinal := 2)
    (rightResult := 228986) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24278⟩⟩, ⟨.program ⟨257⟩, ⟨31458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32943⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24278⟩⟩, ⟨.program ⟨257⟩, ⟨31458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32943⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24278⟩⟩, ⟨.program ⟨257⟩, ⟨31458⟩⟩], [⟨.program ⟨257⟩, ⟨32943⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge229176

namespace LeftMerge229177
def owner : Owner := ⟨.program ⟨257⟩, ⟨33450⟩⟩
def mergeEvent : Nat := 229177
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33448⟩⟩] } }
def leftRaw : List Term := Proof.Events895.exact229172RawTerms
def rightRaw : List Term := Proof.Events894.exact228986RawTerms
def group : MergeGroup := .operator 229172 228986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 229172) (leftOrdinal := 1)
    (rightResult := 228986) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33448⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33448⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33448⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge229177

namespace LeftMerge229185
def owner : Owner := ⟨.program ⟨257⟩, ⟨33863⟩⟩
def mergeEvent : Nat := 229185
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33861⟩⟩] } }
def leftRaw : List Term := Proof.Events895.exact229179RawTerms
def rightRaw : List Term := Proof.Events894.exact228902RawTerms
def group : MergeGroup := .operator 229179 228902
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 229179) (leftOrdinal := 0)
    (rightResult := 228902) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33861⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33861⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge229185

namespace LeftMerge229186
def owner : Owner := ⟨.program ⟨257⟩, ⟨33863⟩⟩
def mergeEvent : Nat := 229186
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨31820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33861⟩⟩] } }
def leftRaw : List Term := Proof.Events895.exact229179RawTerms
def rightRaw : List Term := Proof.Events894.exact228902RawTerms
def group : MergeGroup := .operator 229179 228902
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 229179) (leftOrdinal := 1)
    (rightResult := 228902) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨31820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33861⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨31820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33861⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge229186

namespace LeftMerge229188
def owner : Owner := ⟨.program ⟨257⟩, ⟨33863⟩⟩
def mergeEvent : Nat := 229188
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨31820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33092⟩⟩] } }
def rhsRaw : List Term := Proof.Events894.exact228899RawTerms
def group : MergeGroup := .relation 229187
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 229187) (rhsResult := 228899)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨31820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33861⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33861⟩⟩) ⟨33092⟩ 228899) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33092⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨31820⟩⟩], [⟨.program ⟨257⟩, ⟨33092⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge229188

namespace LeftMerge229202
def owner : Owner := ⟨.program ⟨257⟩, ⟨32679⟩⟩
def mergeEvent : Nat := 229202
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32676⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events895.exact229196RawTerms
def group : MergeGroup := .operator 222245 229196
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 229196) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32676⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32676⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge229202

namespace LeftMerge229323
def owner : Owner := ⟨.program ⟨257⟩, ⟨33304⟩⟩
def mergeEvent : Nat := 229323
def frameStart : Nat := 229257
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events895.exact229319RawTerms
def rightRaw : List Term := Proof.Events895.exact229317RawTerms
def group : MergeGroup := .operator 229319 229317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 229319) (leftOrdinal := 0)
    (rightResult := 229317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31820⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge229323

namespace LeftMerge229335
def owner : Owner := ⟨.program ⟨257⟩, ⟨33862⟩⟩
def mergeEvent : Nat := 229335
def frameStart : Nat := 229257
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33861⟩⟩] } }
def leftRaw : List Term := Proof.Events895.exact229331RawTerms
def rightRaw : List Term := Proof.Events895.exact229308RawTerms
def group : MergeGroup := .operator 229331 229308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 229331) (leftOrdinal := 0)
    (rightResult := 229308) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33861⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33861⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge229335

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
