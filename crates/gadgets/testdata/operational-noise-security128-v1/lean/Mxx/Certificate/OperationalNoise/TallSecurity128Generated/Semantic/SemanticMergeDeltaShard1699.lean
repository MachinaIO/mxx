import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge276169
def owner : Owner := ⟨.program ⟨257⟩, ⟨70980⟩⟩
def mergeEvent : Nat := 276169
def frameStart : Nat := 275461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40212⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def rhsRaw : List Term := Proof.Events1078.exact275974RawTerms
def group : MergeGroup := .relation 276168
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 276168) (rhsResult := 275974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276169

namespace LeftMerge276170
def owner : Owner := ⟨.program ⟨257⟩, ⟨70980⟩⟩
def mergeEvent : Nat := 276170
def frameStart : Nat := 275461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37536⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1078.exact276136RawTerms
def rightRaw : List Term := Proof.Events1078.exact275977RawTerms
def group : MergeGroup := .operator 276136 275977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 276136) (leftOrdinal := 25)
    (rightResult := 275977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37536⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276170

namespace LeftMerge276172
def owner : Owner := ⟨.program ⟨257⟩, ⟨70980⟩⟩
def mergeEvent : Nat := 276172
def frameStart : Nat := 275461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37536⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def rhsRaw : List Term := Proof.Events1078.exact275974RawTerms
def group : MergeGroup := .relation 276171
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 276171) (rhsResult := 275974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276172

namespace LeftMerge276173
def owner : Owner := ⟨.program ⟨257⟩, ⟨70980⟩⟩
def mergeEvent : Nat := 276173
def frameStart : Nat := 275461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1078.exact276136RawTerms
def rightRaw : List Term := Proof.Events1078.exact275977RawTerms
def group : MergeGroup := .operator 276136 275977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 276136) (leftOrdinal := 24)
    (rightResult := 275977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276173

namespace LeftMerge276175
def owner : Owner := ⟨.program ⟨257⟩, ⟨70980⟩⟩
def mergeEvent : Nat := 276175
def frameStart : Nat := 275461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def rhsRaw : List Term := Proof.Events1078.exact275974RawTerms
def group : MergeGroup := .relation 276174
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 276174) (rhsResult := 275974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276175

namespace LeftMerge276176
def owner : Owner := ⟨.program ⟨257⟩, ⟨70980⟩⟩
def mergeEvent : Nat := 276176
def frameStart : Nat := 275461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29192⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1078.exact276136RawTerms
def rightRaw : List Term := Proof.Events1078.exact275977RawTerms
def group : MergeGroup := .operator 276136 275977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 276136) (leftOrdinal := 22)
    (rightResult := 275977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29192⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276176

namespace LeftMerge276178
def owner : Owner := ⟨.program ⟨257⟩, ⟨70980⟩⟩
def mergeEvent : Nat := 276178
def frameStart : Nat := 275461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29192⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def rhsRaw : List Term := Proof.Events1078.exact275974RawTerms
def group : MergeGroup := .relation 276177
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 276177) (rhsResult := 275974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276178

namespace LeftMerge276179
def owner : Owner := ⟨.program ⟨257⟩, ⟨70980⟩⟩
def mergeEvent : Nat := 276179
def frameStart : Nat := 275461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1078.exact276136RawTerms
def rightRaw : List Term := Proof.Events1078.exact275977RawTerms
def group : MergeGroup := .operator 276136 275977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 276136) (leftOrdinal := 21)
    (rightResult := 275977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276179

namespace LeftMerge276181
def owner : Owner := ⟨.program ⟨257⟩, ⟨70980⟩⟩
def mergeEvent : Nat := 276181
def frameStart : Nat := 275461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def rhsRaw : List Term := Proof.Events1078.exact275974RawTerms
def group : MergeGroup := .relation 276180
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 276180) (rhsResult := 275974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276181

namespace LeftMerge276182
def owner : Owner := ⟨.program ⟨257⟩, ⟨70980⟩⟩
def mergeEvent : Nat := 276182
def frameStart : Nat := 275461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1078.exact276136RawTerms
def rightRaw : List Term := Proof.Events1078.exact275977RawTerms
def group : MergeGroup := .operator 276136 275977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 276136) (leftOrdinal := 35)
    (rightResult := 275977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276182

namespace LeftMerge276184
def owner : Owner := ⟨.program ⟨257⟩, ⟨70980⟩⟩
def mergeEvent : Nat := 276184
def frameStart : Nat := 275461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def rhsRaw : List Term := Proof.Events1078.exact275974RawTerms
def group : MergeGroup := .relation 276183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 276183) (rhsResult := 275974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276184

namespace LeftMerge276185
def owner : Owner := ⟨.program ⟨257⟩, ⟨70980⟩⟩
def mergeEvent : Nat := 276185
def frameStart : Nat := 275461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62924⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1078.exact276136RawTerms
def rightRaw : List Term := Proof.Events1078.exact275977RawTerms
def group : MergeGroup := .operator 276136 275977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 276136) (leftOrdinal := 34)
    (rightResult := 275977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62924⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276185

namespace LeftMerge276187
def owner : Owner := ⟨.program ⟨257⟩, ⟨70980⟩⟩
def mergeEvent : Nat := 276187
def frameStart : Nat := 275461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62924⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def rhsRaw : List Term := Proof.Events1078.exact275974RawTerms
def group : MergeGroup := .relation 276186
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 276186) (rhsResult := 275974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276187

namespace LeftMerge276188
def owner : Owner := ⟨.program ⟨257⟩, ⟨70980⟩⟩
def mergeEvent : Nat := 276188
def frameStart : Nat := 275461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1078.exact276136RawTerms
def rightRaw : List Term := Proof.Events1078.exact275977RawTerms
def group : MergeGroup := .operator 276136 275977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 276136) (leftOrdinal := 33)
    (rightResult := 275977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276188

namespace LeftMerge276190
def owner : Owner := ⟨.program ⟨257⟩, ⟨70980⟩⟩
def mergeEvent : Nat := 276190
def frameStart : Nat := 275461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def rhsRaw : List Term := Proof.Events1078.exact275974RawTerms
def group : MergeGroup := .relation 276189
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 276189) (rhsResult := 275974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276190

namespace LeftMerge276191
def owner : Owner := ⟨.program ⟨257⟩, ⟨70980⟩⟩
def mergeEvent : Nat := 276191
def frameStart : Nat := 275461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56964⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1078.exact276136RawTerms
def rightRaw : List Term := Proof.Events1078.exact275977RawTerms
def group : MergeGroup := .operator 276136 275977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 276136) (leftOrdinal := 32)
    (rightResult := 275977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56964⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276191

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
