import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge170126
def owner : Owner := ⟨.program ⟨257⟩, ⟨52304⟩⟩
def mergeEvent : Nat := 170126
def frameStart : Nat := 170066
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24578⟩⟩, ⟨.program ⟨257⟩, ⟨50653⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events664.exact170122RawTerms
def rightRaw : List Term := Proof.Events664.exact170120RawTerms
def group : MergeGroup := .operator 170122 170120
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 170122) (leftOrdinal := 0)
    (rightResult := 170120) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24578⟩⟩, ⟨.program ⟨257⟩, ⟨50653⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24578⟩⟩, ⟨.program ⟨257⟩, ⟨50653⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge170126

namespace LeftMerge170149
def owner : Owner := ⟨.program ⟨257⟩, ⟨9582⟩⟩
def mergeEvent : Nat := 170149
def frameStart : Nat := 170066
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }
def leftRaw : List Term := Proof.Events664.exact170145RawTerms
def rightRaw : List Term := Proof.Events664.exact170142RawTerms
def group : MergeGroup := .operator 170145 170142
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 170145) (leftOrdinal := 0)
    (rightResult := 170142) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9580⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge170149

namespace LeftMerge170158
def owner : Owner := ⟨.program ⟨257⟩, ⟨52566⟩⟩
def mergeEvent : Nat := 170158
def frameStart : Nat := 170066
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52563⟩⟩] } }
def leftRaw : List Term := Proof.Events664.exact170154RawTerms
def rightRaw : List Term := Proof.Events664.exact170111RawTerms
def group : MergeGroup := .operator 170154 170111
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 170154) (leftOrdinal := 0)
    (rightResult := 170111) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52563⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52563⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge170158

namespace LeftMerge170159
def owner : Owner := ⟨.program ⟨257⟩, ⟨52566⟩⟩
def mergeEvent : Nat := 170159
def frameStart : Nat := 170066
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24578⟩⟩, ⟨.program ⟨257⟩, ⟨50653⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52563⟩⟩] } }
def leftRaw : List Term := Proof.Events664.exact170154RawTerms
def rightRaw : List Term := Proof.Events664.exact170111RawTerms
def group : MergeGroup := .operator 170154 170111
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 170154) (leftOrdinal := 1)
    (rightResult := 170111) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24578⟩⟩, ⟨.program ⟨257⟩, ⟨50653⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52563⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24578⟩⟩, ⟨.program ⟨257⟩, ⟨50653⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52563⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge170159

namespace LeftMerge170161
def owner : Owner := ⟨.program ⟨257⟩, ⟨52566⟩⟩
def mergeEvent : Nat := 170161
def frameStart : Nat := 170066
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24578⟩⟩, ⟨.program ⟨257⟩, ⟨50653⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52033⟩⟩] } }
def rhsRaw : List Term := Proof.Events664.exact170108RawTerms
def group : MergeGroup := .relation 170160
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 170160) (rhsResult := 170108)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24578⟩⟩, ⟨.program ⟨257⟩, ⟨50653⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52563⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52563⟩⟩) ⟨52033⟩ 170108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52033⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24578⟩⟩, ⟨.program ⟨257⟩, ⟨50653⟩⟩], [⟨.program ⟨257⟩, ⟨52033⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge170161

namespace LeftMerge170169
def owner : Owner := ⟨.program ⟨257⟩, ⟨50922⟩⟩
def mergeEvent : Nat := 170169
def frameStart : Nat := 170066
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50920⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events664.exact170122RawTerms
def rightRaw : List Term := Proof.Events664.exact170165RawTerms
def group : MergeGroup := .operator 170122 170165
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 170122) (leftOrdinal := 0)
    (rightResult := 170165) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50920⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge170169

namespace LeftMerge170186
def owner : Owner := ⟨.program ⟨257⟩, ⟨51492⟩⟩
def mergeEvent : Nat := 170186
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }
def rhsRaw : List Term := Proof.Events664.exact170183RawTerms
def group : MergeGroup := .relation 170185
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 170185) (rhsResult := 170183)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 170184 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51489⟩⟩]⟩) (none) 170183) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge170186

namespace LeftMerge170187
def owner : Owner := ⟨.program ⟨257⟩, ⟨51492⟩⟩
def mergeEvent : Nat := 170187
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52563⟩⟩] } }
def rhsRaw : List Term := Proof.Events664.exact170183RawTerms
def group : MergeGroup := .relation 170185
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 170185) (rhsResult := 170183)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 170184 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51489⟩⟩]⟩) (none) 170183) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52563⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52563⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge170187

namespace LeftMerge170188
def owner : Owner := ⟨.program ⟨257⟩, ⟨51492⟩⟩
def mergeEvent : Nat := 170188
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨24578⟩⟩, ⟨.program ⟨257⟩, ⟨50653⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52033⟩⟩] } }
def rhsRaw : List Term := Proof.Events664.exact170183RawTerms
def group : MergeGroup := .relation 170185
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 170185) (rhsResult := 170183)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 170184 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51489⟩⟩]⟩) (none) 170183) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24578⟩⟩, ⟨.program ⟨257⟩, ⟨50653⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52033⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨24578⟩⟩, ⟨.program ⟨257⟩, ⟨50653⟩⟩], [⟨.program ⟨257⟩, ⟨52033⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge170188

namespace LeftMerge170189
def owner : Owner := ⟨.program ⟨257⟩, ⟨51492⟩⟩
def mergeEvent : Nat := 170189
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨50920⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events664.exact170183RawTerms
def group : MergeGroup := .relation 170185
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 170185) (rhsResult := 170183)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 170184 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51489⟩⟩]⟩) (none) 170183) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50920⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨50920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge170189

namespace LeftMerge170194
def owner : Owner := ⟨.program ⟨257⟩, ⟨52565⟩⟩
def mergeEvent : Nat := 170194
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨24578⟩⟩, ⟨.program ⟨257⟩, ⟨50653⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52033⟩⟩] } }
def leftRaw : List Term := Proof.Events664.exact170190RawTerms
def rightRaw : List Term := Proof.Events664.exact170004RawTerms
def group : MergeGroup := .operator 170190 170004
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 170190) (leftOrdinal := 2)
    (rightResult := 170004) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨24578⟩⟩, ⟨.program ⟨257⟩, ⟨50653⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52033⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨24578⟩⟩, ⟨.program ⟨257⟩, ⟨50653⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52033⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨24578⟩⟩, ⟨.program ⟨257⟩, ⟨50653⟩⟩], [⟨.program ⟨257⟩, ⟨52033⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge170194

namespace LeftMerge170195
def owner : Owner := ⟨.program ⟨257⟩, ⟨52565⟩⟩
def mergeEvent : Nat := 170195
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52563⟩⟩] } }
def leftRaw : List Term := Proof.Events664.exact170190RawTerms
def rightRaw : List Term := Proof.Events664.exact170004RawTerms
def group : MergeGroup := .operator 170190 170004
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 170190) (leftOrdinal := 1)
    (rightResult := 170004) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52563⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52563⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52563⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge170195

namespace LeftMerge170203
def owner : Owner := ⟨.program ⟨257⟩, ⟨53078⟩⟩
def mergeEvent : Nat := 170203
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53076⟩⟩] } }
def leftRaw : List Term := Proof.Events664.exact170197RawTerms
def rightRaw : List Term := Proof.Events663.exact169920RawTerms
def group : MergeGroup := .operator 170197 169920
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 170197) (leftOrdinal := 0)
    (rightResult := 169920) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53076⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53076⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge170203

namespace LeftMerge170204
def owner : Owner := ⟨.program ⟨257⟩, ⟨53078⟩⟩
def mergeEvent : Nat := 170204
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨50920⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53076⟩⟩] } }
def leftRaw : List Term := Proof.Events664.exact170197RawTerms
def rightRaw : List Term := Proof.Events663.exact169920RawTerms
def group : MergeGroup := .operator 170197 169920
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 170197) (leftOrdinal := 1)
    (rightResult := 169920) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨50920⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53076⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨50920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53076⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge170204

namespace LeftMerge170206
def owner : Owner := ⟨.program ⟨257⟩, ⟨53078⟩⟩
def mergeEvent : Nat := 170206
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨50920⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52197⟩⟩] } }
def rhsRaw : List Term := Proof.Events663.exact169917RawTerms
def group : MergeGroup := .relation 170205
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 170205) (rhsResult := 169917)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨50920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53076⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨53076⟩⟩) ⟨52197⟩ 169917) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52197⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨50920⟩⟩], [⟨.program ⟨257⟩, ⟨52197⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge170206

namespace LeftMerge170220
def owner : Owner := ⟨.program ⟨257⟩, ⟨51839⟩⟩
def mergeEvent : Nat := 170220
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51836⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events664.exact170214RawTerms
def group : MergeGroup := .operator 163745 170214
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 170214) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51836⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge170220

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
