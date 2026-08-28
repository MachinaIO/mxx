import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge35196
def owner : Owner := ⟨.program ⟨257⟩, ⟨30700⟩⟩
def mergeEvent : Nat := 35196
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30698⟩⟩] } }
def leftRaw : List Term := Proof.Events137.exact35191RawTerms
def rightRaw : List Term := Proof.Events136.exact35005RawTerms
def group : MergeGroup := .operator 35191 35005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35191) (leftOrdinal := 1)
    (rightResult := 35005) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30698⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30698⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30698⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35196

namespace LeftMerge35204
def owner : Owner := ⟨.program ⟨257⟩, ⟨31196⟩⟩
def mergeEvent : Nat := 35204
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31194⟩⟩] } }
def leftRaw : List Term := Proof.Events137.exact35198RawTerms
def rightRaw : List Term := Proof.Events136.exact34921RawTerms
def group : MergeGroup := .operator 35198 34921
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35198) (leftOrdinal := 0)
    (rightResult := 34921) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31194⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31194⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35204

namespace LeftMerge35205
def owner : Owner := ⟨.program ⟨257⟩, ⟨31196⟩⟩
def mergeEvent : Nat := 35205
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29160⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31194⟩⟩] } }
def leftRaw : List Term := Proof.Events137.exact35198RawTerms
def rightRaw : List Term := Proof.Events136.exact34921RawTerms
def group : MergeGroup := .operator 35198 34921
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35198) (leftOrdinal := 1)
    (rightResult := 34921) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29160⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31194⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31194⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35205

namespace LeftMerge35207
def owner : Owner := ⟨.program ⟨257⟩, ⟨31196⟩⟩
def mergeEvent : Nat := 35207
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29160⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30322⟩⟩] } }
def rhsRaw : List Term := Proof.Events136.exact34918RawTerms
def group : MergeGroup := .relation 35206
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 35206) (rhsResult := 34918)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31194⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨31194⟩⟩) ⟨30322⟩ 34918) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30322⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29160⟩⟩], [⟨.program ⟨257⟩, ⟨30322⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35207

namespace LeftMerge35221
def owner : Owner := ⟨.program ⟨257⟩, ⟨30019⟩⟩
def mergeEvent : Nat := 35221
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨30016⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32120RawTerms
def rightRaw : List Term := Proof.Events137.exact35215RawTerms
def group : MergeGroup := .operator 32120 35215
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32120) (leftOrdinal := 0)
    (rightResult := 35215) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30016⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨30016⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35221

namespace LeftMerge35342
def owner : Owner := ⟨.program ⟨257⟩, ⟨30484⟩⟩
def mergeEvent : Nat := 35342
def frameStart : Nat := 35276
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29160⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events138.exact35338RawTerms
def rightRaw : List Term := Proof.Events138.exact35336RawTerms
def group : MergeGroup := .operator 35338 35336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35338) (leftOrdinal := 0)
    (rightResult := 35336) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29160⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35342

namespace LeftMerge35354
def owner : Owner := ⟨.program ⟨257⟩, ⟨31195⟩⟩
def mergeEvent : Nat := 35354
def frameStart : Nat := 35276
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31194⟩⟩] } }
def leftRaw : List Term := Proof.Events138.exact35350RawTerms
def rightRaw : List Term := Proof.Events137.exact35327RawTerms
def group : MergeGroup := .operator 35350 35327
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35350) (leftOrdinal := 0)
    (rightResult := 35327) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31194⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31194⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35354

namespace LeftMerge35355
def owner : Owner := ⟨.program ⟨257⟩, ⟨31195⟩⟩
def mergeEvent : Nat := 35355
def frameStart : Nat := 35276
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29160⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31194⟩⟩] } }
def leftRaw : List Term := Proof.Events138.exact35350RawTerms
def rightRaw : List Term := Proof.Events137.exact35327RawTerms
def group : MergeGroup := .operator 35350 35327
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35350) (leftOrdinal := 1)
    (rightResult := 35327) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29160⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31194⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31194⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35355

namespace LeftMerge35357
def owner : Owner := ⟨.program ⟨257⟩, ⟨31195⟩⟩
def mergeEvent : Nat := 35357
def frameStart : Nat := 35276
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29160⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30322⟩⟩] } }
def rhsRaw : List Term := Proof.Events137.exact35324RawTerms
def group : MergeGroup := .relation 35356
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 35356) (rhsResult := 35324)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31194⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨31194⟩⟩) ⟨30322⟩ 35324) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30322⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29160⟩⟩], [⟨.program ⟨257⟩, ⟨30322⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35357

namespace LeftMerge35365
def owner : Owner := ⟨.program ⟨257⟩, ⟨29417⟩⟩
def mergeEvent : Nat := 35365
def frameStart : Nat := 35276
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events138.exact35338RawTerms
def rightRaw : List Term := Proof.Events138.exact35361RawTerms
def group : MergeGroup := .operator 35338 35361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35338) (leftOrdinal := 0)
    (rightResult := 35361) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29416⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35365

namespace LeftMerge35382
def owner : Owner := ⟨.program ⟨257⟩, ⟨30019⟩⟩
def mergeEvent : Nat := 35382
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }
def rhsRaw : List Term := Proof.Events138.exact35379RawTerms
def group : MergeGroup := .relation 35381
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 35381) (rhsResult := 35379)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨30016⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 35380 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨30016⟩⟩]⟩) (none) 35379) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35382

namespace LeftMerge35383
def owner : Owner := ⟨.program ⟨257⟩, ⟨30019⟩⟩
def mergeEvent : Nat := 35383
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31194⟩⟩] } }
def rhsRaw : List Term := Proof.Events138.exact35379RawTerms
def group : MergeGroup := .relation 35381
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 35381) (rhsResult := 35379)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨30016⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 35380 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨30016⟩⟩]⟩) (none) 35379) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31194⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31194⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35383

namespace LeftMerge35384
def owner : Owner := ⟨.program ⟨257⟩, ⟨30019⟩⟩
def mergeEvent : Nat := 35384
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29160⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30322⟩⟩] } }
def rhsRaw : List Term := Proof.Events138.exact35379RawTerms
def group : MergeGroup := .relation 35381
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 35381) (rhsResult := 35379)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨30016⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 35380 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨30016⟩⟩]⟩) (none) 35379) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29160⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30322⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29160⟩⟩], [⟨.program ⟨257⟩, ⟨30322⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35384

namespace LeftMerge35385
def owner : Owner := ⟨.program ⟨257⟩, ⟨30019⟩⟩
def mergeEvent : Nat := 35385
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events138.exact35379RawTerms
def group : MergeGroup := .relation 35381
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 35381) (rhsResult := 35379)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨30016⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 35380 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨30016⟩⟩]⟩) (none) 35379) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35385

namespace LeftMerge35390
def owner : Owner := ⟨.program ⟨257⟩, ⟨31197⟩⟩
def mergeEvent : Nat := 35390
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31194⟩⟩] } }
def leftRaw : List Term := Proof.Events138.exact35386RawTerms
def rightRaw : List Term := Proof.Events137.exact35208RawTerms
def group : MergeGroup := .operator 35386 35208
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35386) (leftOrdinal := 0)
    (rightResult := 35208) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31194⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31194⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35390

namespace LeftMerge35391
def owner : Owner := ⟨.program ⟨257⟩, ⟨31197⟩⟩
def mergeEvent : Nat := 35391
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29160⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30322⟩⟩] } }
def leftRaw : List Term := Proof.Events138.exact35386RawTerms
def rightRaw : List Term := Proof.Events137.exact35208RawTerms
def group : MergeGroup := .operator 35386 35208
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35386) (leftOrdinal := 2)
    (rightResult := 35208) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29160⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30322⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29160⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30322⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29160⟩⟩], [⟨.program ⟨257⟩, ⟨30322⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35391

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
