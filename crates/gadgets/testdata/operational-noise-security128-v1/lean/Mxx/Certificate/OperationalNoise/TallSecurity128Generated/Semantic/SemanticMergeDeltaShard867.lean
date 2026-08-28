import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge143170
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143170
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 27)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143170

namespace LeftMerge143172
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143172
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events524.exact134375RawTerms
def group : MergeGroup := .relation 143171
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 143171) (rhsResult := 134375)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143172

namespace LeftMerge143173
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143173
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 14)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge143173

namespace LeftMerge143174
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143174
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40228⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 26)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40228⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143174

namespace LeftMerge143176
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143176
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40228⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events524.exact134375RawTerms
def group : MergeGroup := .relation 143175
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 143175) (rhsResult := 134375)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143176

namespace LeftMerge143177
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143177
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 13)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge143177

namespace LeftMerge143178
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143178
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 25)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143178

namespace LeftMerge143180
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143180
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events524.exact134375RawTerms
def group : MergeGroup := .relation 143179
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 143179) (rhsResult := 134375)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143180

namespace LeftMerge143181
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143181
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 12)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge143181

namespace LeftMerge143182
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143182
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 24)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143182

namespace LeftMerge143184
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143184
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events524.exact134375RawTerms
def group : MergeGroup := .relation 143183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 143183) (rhsResult := 134375)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143184

namespace LeftMerge143185
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143185
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 11)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge143185

namespace LeftMerge143186
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143186
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 22)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143186

namespace LeftMerge143188
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143188
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events524.exact134375RawTerms
def group : MergeGroup := .relation 143187
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 143187) (rhsResult := 134375)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143188

namespace LeftMerge143189
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143189
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 10)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge143189

namespace LeftMerge143190
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143190
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 21)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143190

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
