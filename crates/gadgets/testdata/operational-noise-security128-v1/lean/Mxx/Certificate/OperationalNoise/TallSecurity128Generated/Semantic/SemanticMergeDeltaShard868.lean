import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge143192
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143192
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events524.exact134375RawTerms
def group : MergeGroup := .relation 143191
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 143191) (rhsResult := 134375)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143192

namespace LeftMerge143193
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143193
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 9)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge143193

namespace LeftMerge143194
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143194
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 35)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143194

namespace LeftMerge143196
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143196
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events524.exact134375RawTerms
def group : MergeGroup := .relation 143195
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 143195) (rhsResult := 134375)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143196

namespace LeftMerge143197
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143197
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 8)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge143197

namespace LeftMerge143198
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143198
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 34)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143198

namespace LeftMerge143200
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143200
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events524.exact134375RawTerms
def group : MergeGroup := .relation 143199
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 143199) (rhsResult := 134375)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143200

namespace LeftMerge143201
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143201
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 7)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge143201

namespace LeftMerge143202
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143202
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 33)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143202

namespace LeftMerge143204
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143204
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events524.exact134375RawTerms
def group : MergeGroup := .relation 143203
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 143203) (rhsResult := 134375)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143204

namespace LeftMerge143205
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143205
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 6)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge143205

namespace LeftMerge143206
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143206
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 32)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143206

namespace LeftMerge143208
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143208
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events524.exact134375RawTerms
def group : MergeGroup := .relation 143207
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 143207) (rhsResult := 134375)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143208

namespace LeftMerge143209
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143209
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 5)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge143209

namespace LeftMerge143210
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143210
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events559.exact143155RawTerms
def rightRaw : List Term := Proof.Events524.exact134378RawTerms
def group : MergeGroup := .operator 143155 134378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 143155) (leftOrdinal := 31)
    (rightResult := 134378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143210

namespace LeftMerge143212
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def mergeEvent : Nat := 143212
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def rhsRaw : List Term := Proof.Events524.exact134375RawTerms
def group : MergeGroup := .relation 143211
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 143211) (rhsResult := 134375)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge143212

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
