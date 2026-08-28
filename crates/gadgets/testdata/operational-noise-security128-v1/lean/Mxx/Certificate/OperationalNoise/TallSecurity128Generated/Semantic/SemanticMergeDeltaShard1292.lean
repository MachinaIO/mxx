import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge210006
def owner : Owner := ⟨.program ⟨257⟩, ⟨13585⟩⟩
def mergeEvent : Nat := 210006
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }
def leftRaw : List Term := Proof.Events820.exact209997RawTerms
def rightRaw : List Term := Proof.Events076.exact19615RawTerms
def group : MergeGroup := .operator 209997 19615
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209997) (leftOrdinal := 0)
    (rightResult := 19615) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9550⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210006

namespace LeftMerge210011
def owner : Owner := ⟨.program ⟨257⟩, ⟨34441⟩⟩
def mergeEvent : Nat := 210011
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13581⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }
def leftRaw : List Term := Proof.Events820.exact210007RawTerms
def rightRaw : List Term := Proof.Events820.exact209977RawTerms
def group : MergeGroup := .operator 210007 209977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210007) (leftOrdinal := 1)
    (rightResult := 209977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13581⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13581⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13581⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210011

namespace LeftMerge210019
def owner : Owner := ⟨.program ⟨257⟩, ⟨36260⟩⟩
def mergeEvent : Nat := 210019
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36259⟩⟩] } }
def leftRaw : List Term := Proof.Events820.exact210013RawTerms
def rightRaw : List Term := Proof.Events820.exact209949RawTerms
def group : MergeGroup := .operator 210013 209949
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210013) (leftOrdinal := 1)
    (rightResult := 209949) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36259⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36259⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge210019

namespace LeftMerge210021
def owner : Owner := ⟨.program ⟨257⟩, ⟨36260⟩⟩
def mergeEvent : Nat := 210021
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35749⟩⟩] } }
def rhsRaw : List Term := Proof.Events820.exact209946RawTerms
def group : MergeGroup := .relation 210020
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 210020) (rhsResult := 209946)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36259⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36259⟩⟩) ⟨35749⟩ 209946) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35749⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], [⟨.program ⟨257⟩, ⟨35749⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge210021

namespace LeftMerge210022
def owner : Owner := ⟨.program ⟨257⟩, ⟨36260⟩⟩
def mergeEvent : Nat := 210022
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36259⟩⟩] } }
def leftRaw : List Term := Proof.Events820.exact210013RawTerms
def rightRaw : List Term := Proof.Events820.exact209949RawTerms
def group : MergeGroup := .operator 210013 209949
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210013) (leftOrdinal := 0)
    (rightResult := 209949) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36259⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210022

namespace LeftMerge210036
def owner : Owner := ⟨.program ⟨257⟩, ⟨35192⟩⟩
def mergeEvent : Nat := 210036
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35189⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207620RawTerms
def rightRaw : List Term := Proof.Events820.exact210030RawTerms
def group : MergeGroup := .operator 207620 210030
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207620) (leftOrdinal := 0)
    (rightResult := 210030) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35189⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210036

namespace LeftMerge210115
def owner : Owner := ⟨.program ⟨257⟩, ⟨34435⟩⟩
def mergeEvent : Nat := 210115
def frameStart : Nat := 210085
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events820.exact210111RawTerms
def rightRaw : List Term := Proof.Events820.exact210108RawTerms
def group : MergeGroup := .operator 210111 210108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210111) (leftOrdinal := 0)
    (rightResult := 210108) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13581⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34434⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210115

namespace LeftMerge210145
def owner : Owner := ⟨.program ⟨257⟩, ⟨36028⟩⟩
def mergeEvent : Nat := 210145
def frameStart : Nat := 210085
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events820.exact210141RawTerms
def rightRaw : List Term := Proof.Events820.exact210139RawTerms
def group : MergeGroup := .operator 210141 210139
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210141) (leftOrdinal := 0)
    (rightResult := 210139) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210145

namespace LeftMerge210168
def owner : Owner := ⟨.program ⟨257⟩, ⟨9552⟩⟩
def mergeEvent : Nat := 210168
def frameStart : Nat := 210085
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }
def leftRaw : List Term := Proof.Events820.exact210164RawTerms
def rightRaw : List Term := Proof.Events820.exact210161RawTerms
def group : MergeGroup := .operator 210164 210161
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210164) (leftOrdinal := 0)
    (rightResult := 210161) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9550⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210168

namespace LeftMerge210177
def owner : Owner := ⟨.program ⟨257⟩, ⟨36262⟩⟩
def mergeEvent : Nat := 210177
def frameStart : Nat := 210085
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36259⟩⟩] } }
def leftRaw : List Term := Proof.Events820.exact210173RawTerms
def rightRaw : List Term := Proof.Events820.exact210130RawTerms
def group : MergeGroup := .operator 210173 210130
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210173) (leftOrdinal := 0)
    (rightResult := 210130) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36259⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210177

namespace LeftMerge210178
def owner : Owner := ⟨.program ⟨257⟩, ⟨36262⟩⟩
def mergeEvent : Nat := 210178
def frameStart : Nat := 210085
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36259⟩⟩] } }
def leftRaw : List Term := Proof.Events820.exact210173RawTerms
def rightRaw : List Term := Proof.Events820.exact210130RawTerms
def group : MergeGroup := .operator 210173 210130
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210173) (leftOrdinal := 1)
    (rightResult := 210130) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36259⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36259⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge210178

namespace LeftMerge210180
def owner : Owner := ⟨.program ⟨257⟩, ⟨36262⟩⟩
def mergeEvent : Nat := 210180
def frameStart : Nat := 210085
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35749⟩⟩] } }
def rhsRaw : List Term := Proof.Events820.exact210127RawTerms
def group : MergeGroup := .relation 210179
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 210179) (rhsResult := 210127)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36259⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36259⟩⟩) ⟨35749⟩ 210127) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35749⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], [⟨.program ⟨257⟩, ⟨35749⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge210180

namespace LeftMerge210188
def owner : Owner := ⟨.program ⟨257⟩, ⟨34750⟩⟩
def mergeEvent : Nat := 210188
def frameStart : Nat := 210085
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events820.exact210141RawTerms
def rightRaw : List Term := Proof.Events821.exact210184RawTerms
def group : MergeGroup := .operator 210141 210184
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210141) (leftOrdinal := 0)
    (rightResult := 210184) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34748⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210188

namespace LeftMerge210205
def owner : Owner := ⟨.program ⟨257⟩, ⟨35192⟩⟩
def mergeEvent : Nat := 210205
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }
def rhsRaw : List Term := Proof.Events821.exact210202RawTerms
def group : MergeGroup := .relation 210204
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 210204) (rhsResult := 210202)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35189⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 210203 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35189⟩⟩]⟩) (none) 210202) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210205

namespace LeftMerge210206
def owner : Owner := ⟨.program ⟨257⟩, ⟨35192⟩⟩
def mergeEvent : Nat := 210206
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36259⟩⟩] } }
def rhsRaw : List Term := Proof.Events821.exact210202RawTerms
def group : MergeGroup := .relation 210204
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 210204) (rhsResult := 210202)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35189⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 210203 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35189⟩⟩]⟩) (none) 210202) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36259⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36259⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge210206

namespace LeftMerge210207
def owner : Owner := ⟨.program ⟨257⟩, ⟨35192⟩⟩
def mergeEvent : Nat := 210207
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35749⟩⟩] } }
def rhsRaw : List Term := Proof.Events821.exact210202RawTerms
def group : MergeGroup := .relation 210204
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 210204) (rhsResult := 210202)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35189⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 210203 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35189⟩⟩]⟩) (none) 210202) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35749⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], [⟨.program ⟨257⟩, ⟨35749⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210207

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
