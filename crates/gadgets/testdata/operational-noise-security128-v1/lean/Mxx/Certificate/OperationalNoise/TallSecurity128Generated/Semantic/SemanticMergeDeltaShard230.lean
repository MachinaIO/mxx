import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge42169
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42169
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }
def rhsRaw : List Term := Proof.Events163.exact41974RawTerms
def group : MergeGroup := .relation 42168
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42168) (rhsResult := 41974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 41974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40436⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42169

namespace LeftMerge42170
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42170
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 25)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42170

namespace LeftMerge42172
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42172
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }
def rhsRaw : List Term := Proof.Events163.exact41974RawTerms
def group : MergeGroup := .relation 42171
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42171) (rhsResult := 41974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 41974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37760⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42172

namespace LeftMerge42173
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42173
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨35080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 24)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨35080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨35080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42173

namespace LeftMerge42175
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42175
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨35080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }
def rhsRaw : List Term := Proof.Events163.exact41974RawTerms
def group : MergeGroup := .relation 42174
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42174) (rhsResult := 41974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨35080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 41974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨35080⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42175

namespace LeftMerge42176
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42176
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 22)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42176

namespace LeftMerge42178
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42178
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }
def rhsRaw : List Term := Proof.Events163.exact41974RawTerms
def group : MergeGroup := .relation 42177
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42177) (rhsResult := 41974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 41974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29416⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42178

namespace LeftMerge42179
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42179
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26736⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 21)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26736⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26736⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42179

namespace LeftMerge42181
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42181
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26736⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }
def rhsRaw : List Term := Proof.Events163.exact41974RawTerms
def group : MergeGroup := .relation 42180
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42180) (rhsResult := 41974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨26736⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 41974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨26736⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42181

namespace LeftMerge42182
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42182
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨67231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 35)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨67231⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42182

namespace LeftMerge42184
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42184
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨67231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }
def rhsRaw : List Term := Proof.Events163.exact41974RawTerms
def group : MergeGroup := .relation 42183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42183) (rhsResult := 41974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨67231⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 41974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨67231⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42184

namespace LeftMerge42185
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42185
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63252⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 34)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63252⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨63252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42185

namespace LeftMerge42187
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42187
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63252⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }
def rhsRaw : List Term := Proof.Events163.exact41974RawTerms
def group : MergeGroup := .relation 42186
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42186) (rhsResult := 41974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨63252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 41974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨63252⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42187

namespace LeftMerge42188
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42188
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60272⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 33)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60272⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨60272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42188

namespace LeftMerge42190
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42190
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60272⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }
def rhsRaw : List Term := Proof.Events163.exact41974RawTerms
def group : MergeGroup := .relation 42189
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42189) (rhsResult := 41974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨60272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 41974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨60272⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42190

namespace LeftMerge42191
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42191
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57292⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 32)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57292⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨57292⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42191

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
