import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge42149
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42149
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 8)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42149

namespace LeftMerge42150
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42150
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 7)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42150

namespace LeftMerge42151
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42151
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 6)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42151

namespace LeftMerge42152
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42152
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 5)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42152

namespace LeftMerge42153
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42153
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 4)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42153

namespace LeftMerge42154
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42154
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 3)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42154

namespace LeftMerge42155
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42155
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 2)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42155

namespace LeftMerge42156
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42156
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 1)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42156

namespace LeftMerge42157
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42157
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 0)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42157

namespace LeftMerge42158
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42158
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48480⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 29)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48480⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48480⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42158

namespace LeftMerge42160
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42160
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48480⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }
def rhsRaw : List Term := Proof.Events163.exact41974RawTerms
def group : MergeGroup := .relation 42159
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42159) (rhsResult := 41974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48480⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 41974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48480⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42160

namespace LeftMerge42161
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42161
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 28)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42161

namespace LeftMerge42163
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42163
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }
def rhsRaw : List Term := Proof.Events163.exact41974RawTerms
def group : MergeGroup := .relation 42162
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42162) (rhsResult := 41974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 41974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45800⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42163

namespace LeftMerge42164
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42164
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨43116⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 27)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨43116⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨43116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42164

namespace LeftMerge42166
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42166
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨43116⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }
def rhsRaw : List Term := Proof.Events163.exact41974RawTerms
def group : MergeGroup := .relation 42165
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42165) (rhsResult := 41974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨43116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 41974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨43116⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42166

namespace LeftMerge42167
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42167
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 26)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42167

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
