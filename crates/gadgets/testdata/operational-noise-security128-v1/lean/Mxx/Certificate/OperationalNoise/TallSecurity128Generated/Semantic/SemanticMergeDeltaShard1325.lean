import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge216156
def owner : Owner := ⟨.program ⟨257⟩, ⟨17762⟩⟩
def mergeEvent : Nat := 216156
def frameStart : Nat := 216078
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩] } }
def leftRaw : List Term := Proof.Events844.exact216152RawTerms
def rightRaw : List Term := Proof.Events844.exact216129RawTerms
def group : MergeGroup := .operator 216152 216129
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 216152) (leftOrdinal := 0)
    (rightResult := 216129) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17761⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge216156

namespace LeftMerge216157
def owner : Owner := ⟨.program ⟨257⟩, ⟨17762⟩⟩
def mergeEvent : Nat := 216157
def frameStart : Nat := 216078
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩] } }
def leftRaw : List Term := Proof.Events844.exact216152RawTerms
def rightRaw : List Term := Proof.Events844.exact216129RawTerms
def group : MergeGroup := .operator 216152 216129
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 216152) (leftOrdinal := 1)
    (rightResult := 216129) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17761⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge216157

namespace LeftMerge216159
def owner : Owner := ⟨.program ⟨257⟩, ⟨17762⟩⟩
def mergeEvent : Nat := 216159
def frameStart : Nat := 216078
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17001⟩⟩] } }
def rhsRaw : List Term := Proof.Events844.exact216126RawTerms
def group : MergeGroup := .relation 216158
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 216158) (rhsResult := 216126)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17761⟩⟩) ⟨17001⟩ 216126) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17001⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15788⟩⟩], [⟨.program ⟨257⟩, ⟨17001⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge216159

namespace LeftMerge216167
def owner : Owner := ⟨.program ⟨257⟩, ⟨16036⟩⟩
def mergeEvent : Nat := 216167
def frameStart : Nat := 216078
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events844.exact216140RawTerms
def rightRaw : List Term := Proof.Events844.exact216163RawTerms
def group : MergeGroup := .operator 216140 216163
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 216140) (leftOrdinal := 0)
    (rightResult := 216163) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge216167

namespace LeftMerge216184
def owner : Owner := ⟨.program ⟨257⟩, ⟨16599⟩⟩
def mergeEvent : Nat := 216184
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }
def rhsRaw : List Term := Proof.Events844.exact216181RawTerms
def group : MergeGroup := .relation 216183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 216183) (rhsResult := 216181)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16596⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 216182 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16596⟩⟩]⟩) (none) 216181) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge216184

namespace LeftMerge216185
def owner : Owner := ⟨.program ⟨257⟩, ⟨16599⟩⟩
def mergeEvent : Nat := 216185
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩] } }
def rhsRaw : List Term := Proof.Events844.exact216181RawTerms
def group : MergeGroup := .relation 216183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 216183) (rhsResult := 216181)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16596⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 216182 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16596⟩⟩]⟩) (none) 216181) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge216185

namespace LeftMerge216186
def owner : Owner := ⟨.program ⟨257⟩, ⟨16599⟩⟩
def mergeEvent : Nat := 216186
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17001⟩⟩] } }
def rhsRaw : List Term := Proof.Events844.exact216181RawTerms
def group : MergeGroup := .relation 216183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 216183) (rhsResult := 216181)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16596⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 216182 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16596⟩⟩]⟩) (none) 216181) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17001⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15788⟩⟩], [⟨.program ⟨257⟩, ⟨17001⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge216186

namespace LeftMerge216187
def owner : Owner := ⟨.program ⟨257⟩, ⟨16599⟩⟩
def mergeEvent : Nat := 216187
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events844.exact216181RawTerms
def group : MergeGroup := .relation 216183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 216183) (rhsResult := 216181)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16596⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 216182 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16596⟩⟩]⟩) (none) 216181) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge216187

namespace LeftMerge216192
def owner : Owner := ⟨.program ⟨257⟩, ⟨17764⟩⟩
def mergeEvent : Nat := 216192
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩] } }
def leftRaw : List Term := Proof.Events844.exact216188RawTerms
def rightRaw : List Term := Proof.Events843.exact216010RawTerms
def group : MergeGroup := .operator 216188 216010
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 216188) (leftOrdinal := 0)
    (rightResult := 216010) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge216192

namespace LeftMerge216193
def owner : Owner := ⟨.program ⟨257⟩, ⟨17764⟩⟩
def mergeEvent : Nat := 216193
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17001⟩⟩] } }
def leftRaw : List Term := Proof.Events844.exact216188RawTerms
def rightRaw : List Term := Proof.Events843.exact216010RawTerms
def group : MergeGroup := .operator 216188 216010
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 216188) (leftOrdinal := 2)
    (rightResult := 216010) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17001⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17001⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15788⟩⟩], [⟨.program ⟨257⟩, ⟨17001⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge216193

namespace LeftMerge216286
def owner : Owner := ⟨.program ⟨257⟩, ⟨71238⟩⟩
def mergeEvent : Nat := 216286
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events844.exact216280RawTerms
def rightRaw : List Term := Proof.Events810.exact207503RawTerms
def group : MergeGroup := .operator 216280 207503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 216280) (leftOrdinal := 17)
    (rightResult := 207503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge216286

namespace LeftMerge216287
def owner : Owner := ⟨.program ⟨257⟩, ⟨71238⟩⟩
def mergeEvent : Nat := 216287
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events844.exact216280RawTerms
def rightRaw : List Term := Proof.Events810.exact207503RawTerms
def group : MergeGroup := .operator 216280 207503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 216280) (leftOrdinal := 29)
    (rightResult := 207503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge216287

namespace LeftMerge216289
def owner : Owner := ⟨.program ⟨257⟩, ⟨71238⟩⟩
def mergeEvent : Nat := 216289
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }
def rhsRaw : List Term := Proof.Events810.exact207500RawTerms
def group : MergeGroup := .relation 216288
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 216288) (rhsResult := 207500)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 207500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge216289

namespace LeftMerge216290
def owner : Owner := ⟨.program ⟨257⟩, ⟨71238⟩⟩
def mergeEvent : Nat := 216290
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events844.exact216280RawTerms
def rightRaw : List Term := Proof.Events810.exact207503RawTerms
def group : MergeGroup := .operator 216280 207503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 216280) (leftOrdinal := 16)
    (rightResult := 207503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge216290

namespace LeftMerge216291
def owner : Owner := ⟨.program ⟨257⟩, ⟨71238⟩⟩
def mergeEvent : Nat := 216291
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events844.exact216280RawTerms
def rightRaw : List Term := Proof.Events810.exact207503RawTerms
def group : MergeGroup := .operator 216280 207503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 216280) (leftOrdinal := 28)
    (rightResult := 207503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge216291

namespace LeftMerge216293
def owner : Owner := ⟨.program ⟨257⟩, ⟨71238⟩⟩
def mergeEvent : Nat := 216293
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }
def rhsRaw : List Term := Proof.Events810.exact207500RawTerms
def group : MergeGroup := .relation 216292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 216292) (rhsResult := 207500)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 207500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge216293

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
