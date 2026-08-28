import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge172116
def owner : Owner := ⟨.program ⟨257⟩, ⟨16332⟩⟩
def mergeEvent : Nat := 172116
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨12441⟩⟩, ⟨.program ⟨257⟩, ⟨15570⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16873⟩⟩] } }
def rhsRaw : List Term := Proof.Events672.exact172111RawTerms
def group : MergeGroup := .relation 172113
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 172113) (rhsResult := 172111)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16329⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 172112 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16329⟩⟩]⟩) (none) 172111) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12441⟩⟩, ⟨.program ⟨257⟩, ⟨15570⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16873⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨12441⟩⟩, ⟨.program ⟨257⟩, ⟨15570⟩⟩], [⟨.program ⟨257⟩, ⟨16873⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge172116

namespace LeftMerge172117
def owner : Owner := ⟨.program ⟨257⟩, ⟨16332⟩⟩
def mergeEvent : Nat := 172117
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events672.exact172111RawTerms
def group : MergeGroup := .relation 172113
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 172113) (rhsResult := 172111)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16329⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 172112 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16329⟩⟩]⟩) (none) 172111) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge172117

namespace LeftMerge172122
def owner : Owner := ⟨.program ⟨257⟩, ⟨17405⟩⟩
def mergeEvent : Nat := 172122
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨12441⟩⟩, ⟨.program ⟨257⟩, ⟨15570⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16873⟩⟩] } }
def leftRaw : List Term := Proof.Events672.exact172118RawTerms
def rightRaw : List Term := Proof.Events671.exact171932RawTerms
def group : MergeGroup := .operator 172118 171932
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 172118) (leftOrdinal := 2)
    (rightResult := 171932) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨12441⟩⟩, ⟨.program ⟨257⟩, ⟨15570⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16873⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨12441⟩⟩, ⟨.program ⟨257⟩, ⟨15570⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16873⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨12441⟩⟩, ⟨.program ⟨257⟩, ⟨15570⟩⟩], [⟨.program ⟨257⟩, ⟨16873⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge172122

namespace LeftMerge172123
def owner : Owner := ⟨.program ⟨257⟩, ⟨17405⟩⟩
def mergeEvent : Nat := 172123
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17403⟩⟩] } }
def leftRaw : List Term := Proof.Events672.exact172118RawTerms
def rightRaw : List Term := Proof.Events671.exact171932RawTerms
def group : MergeGroup := .operator 172118 171932
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 172118) (leftOrdinal := 1)
    (rightResult := 171932) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17403⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17403⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17403⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge172123

namespace LeftMerge172131
def owner : Owner := ⟨.program ⟨257⟩, ⟨17875⟩⟩
def mergeEvent : Nat := 172131
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩] } }
def leftRaw : List Term := Proof.Events672.exact172125RawTerms
def rightRaw : List Term := Proof.Events671.exact171848RawTerms
def group : MergeGroup := .operator 172125 171848
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 172125) (leftOrdinal := 0)
    (rightResult := 171848) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17873⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge172131

namespace LeftMerge172132
def owner : Owner := ⟨.program ⟨257⟩, ⟨17875⟩⟩
def mergeEvent : Nat := 172132
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩] } }
def leftRaw : List Term := Proof.Events672.exact172125RawTerms
def rightRaw : List Term := Proof.Events671.exact171848RawTerms
def group : MergeGroup := .operator 172125 171848
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 172125) (leftOrdinal := 1)
    (rightResult := 171848) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17873⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge172132

namespace LeftMerge172134
def owner : Owner := ⟨.program ⟨257⟩, ⟨17875⟩⟩
def mergeEvent : Nat := 172134
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17037⟩⟩] } }
def rhsRaw : List Term := Proof.Events671.exact171845RawTerms
def group : MergeGroup := .relation 172133
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 172133) (rhsResult := 171845)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17873⟩⟩) ⟨17037⟩ 171845) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17037⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨17037⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge172134

namespace LeftMerge172148
def owner : Owner := ⟨.program ⟨257⟩, ⟨16679⟩⟩
def mergeEvent : Nat := 172148
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16676⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events672.exact172142RawTerms
def group : MergeGroup := .operator 163745 172142
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 172142) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16676⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16676⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge172148

namespace LeftMerge172269
def owner : Owner := ⟨.program ⟨257⟩, ⟨17224⟩⟩
def mergeEvent : Nat := 172269
def frameStart : Nat := 172203
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events672.exact172265RawTerms
def rightRaw : List Term := Proof.Events672.exact172263RawTerms
def group : MergeGroup := .operator 172265 172263
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 172265) (leftOrdinal := 0)
    (rightResult := 172263) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15820⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge172269

namespace LeftMerge172281
def owner : Owner := ⟨.program ⟨257⟩, ⟨17874⟩⟩
def mergeEvent : Nat := 172281
def frameStart : Nat := 172203
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩] } }
def leftRaw : List Term := Proof.Events672.exact172277RawTerms
def rightRaw : List Term := Proof.Events672.exact172254RawTerms
def group : MergeGroup := .operator 172277 172254
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 172277) (leftOrdinal := 0)
    (rightResult := 172254) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17873⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge172281

namespace LeftMerge172282
def owner : Owner := ⟨.program ⟨257⟩, ⟨17874⟩⟩
def mergeEvent : Nat := 172282
def frameStart : Nat := 172203
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩] } }
def leftRaw : List Term := Proof.Events672.exact172277RawTerms
def rightRaw : List Term := Proof.Events672.exact172254RawTerms
def group : MergeGroup := .operator 172277 172254
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 172277) (leftOrdinal := 1)
    (rightResult := 172254) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17873⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge172282

namespace LeftMerge172284
def owner : Owner := ⟨.program ⟨257⟩, ⟨17874⟩⟩
def mergeEvent : Nat := 172284
def frameStart : Nat := 172203
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17037⟩⟩] } }
def rhsRaw : List Term := Proof.Events672.exact172251RawTerms
def group : MergeGroup := .relation 172283
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 172283) (rhsResult := 172251)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17873⟩⟩) ⟨17037⟩ 172251) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17037⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨17037⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge172284

namespace LeftMerge172292
def owner : Owner := ⟨.program ⟨257⟩, ⟨16100⟩⟩
def mergeEvent : Nat := 172292
def frameStart : Nat := 172203
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16099⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events672.exact172265RawTerms
def rightRaw : List Term := Proof.Events673.exact172288RawTerms
def group : MergeGroup := .operator 172265 172288
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 172265) (leftOrdinal := 0)
    (rightResult := 172288) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16099⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge172292

namespace LeftMerge172309
def owner : Owner := ⟨.program ⟨257⟩, ⟨16679⟩⟩
def mergeEvent : Nat := 172309
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }
def rhsRaw : List Term := Proof.Events673.exact172306RawTerms
def group : MergeGroup := .relation 172308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 172308) (rhsResult := 172306)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16676⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 172307 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16676⟩⟩]⟩) (none) 172306) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge172309

namespace LeftMerge172310
def owner : Owner := ⟨.program ⟨257⟩, ⟨16679⟩⟩
def mergeEvent : Nat := 172310
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩] } }
def rhsRaw : List Term := Proof.Events673.exact172306RawTerms
def group : MergeGroup := .relation 172308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 172308) (rhsResult := 172306)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16676⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 172307 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16676⟩⟩]⟩) (none) 172306) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge172310

namespace LeftMerge172311
def owner : Owner := ⟨.program ⟨257⟩, ⟨16679⟩⟩
def mergeEvent : Nat := 172311
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17037⟩⟩] } }
def rhsRaw : List Term := Proof.Events673.exact172306RawTerms
def group : MergeGroup := .relation 172308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 172308) (rhsResult := 172306)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16676⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 172307 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16676⟩⟩]⟩) (none) 172306) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17037⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨17037⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge172311

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
