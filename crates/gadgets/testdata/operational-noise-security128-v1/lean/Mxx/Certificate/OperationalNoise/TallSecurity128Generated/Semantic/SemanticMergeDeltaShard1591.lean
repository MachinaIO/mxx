import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge258218
def owner : Owner := ⟨.program ⟨257⟩, ⟨31357⟩⟩
def mergeEvent : Nat := 258218
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }
def rhsRaw : List Term := Proof.Events094.exact24094RawTerms
def group : MergeGroup := .relation 258217
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 258217) (rhsResult := 24094)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9577⟩⟩) ⟨7307⟩ 24094) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge258218

namespace LeftMerge258219
def owner : Owner := ⟨.program ⟨257⟩, ⟨31357⟩⟩
def mergeEvent : Nat := 258219
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }
def leftRaw : List Term := Proof.Events1008.exact258210RawTerms
def rightRaw : List Term := Proof.Events094.exact24124RawTerms
def group : MergeGroup := .operator 258210 24124
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258210) (leftOrdinal := 0)
    (rightResult := 24124) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9577⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258219

namespace LeftMerge258224
def owner : Owner := ⟨.program ⟨257⟩, ⟨31358⟩⟩
def mergeEvent : Nat := 258224
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }
def leftRaw : List Term := Proof.Events1008.exact258220RawTerms
def rightRaw : List Term := Proof.Events1008.exact258190RawTerms
def group : MergeGroup := .operator 258220 258190
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258220) (leftOrdinal := 1)
    (rightResult := 258190) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258224

namespace LeftMerge258232
def owner : Owner := ⟨.program ⟨257⟩, ⟨33405⟩⟩
def mergeEvent : Nat := 258232
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33404⟩⟩] } }
def leftRaw : List Term := Proof.Events1008.exact258226RawTerms
def rightRaw : List Term := Proof.Events1008.exact258162RawTerms
def group : MergeGroup := .operator 258226 258162
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258226) (leftOrdinal := 1)
    (rightResult := 258162) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33404⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33404⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge258232

namespace LeftMerge258234
def owner : Owner := ⟨.program ⟨257⟩, ⟨33405⟩⟩
def mergeEvent : Nat := 258234
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32919⟩⟩] } }
def rhsRaw : List Term := Proof.Events1008.exact258159RawTerms
def group : MergeGroup := .relation 258233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 258233) (rhsResult := 258159)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33404⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33404⟩⟩) ⟨32919⟩ 258159) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32919⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], [⟨.program ⟨257⟩, ⟨32919⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge258234

namespace LeftMerge258235
def owner : Owner := ⟨.program ⟨257⟩, ⟨33405⟩⟩
def mergeEvent : Nat := 258235
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33404⟩⟩] } }
def leftRaw : List Term := Proof.Events1008.exact258226RawTerms
def rightRaw : List Term := Proof.Events1008.exact258162RawTerms
def group : MergeGroup := .operator 258226 258162
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258226) (leftOrdinal := 0)
    (rightResult := 258162) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33404⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33404⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258235

namespace LeftMerge258249
def owner : Owner := ⟨.program ⟨257⟩, ⟨32342⟩⟩
def mergeEvent : Nat := 258249
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32339⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events1008.exact258243RawTerms
def group : MergeGroup := .operator 251495 258243
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 258243) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32339⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32339⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258249

namespace LeftMerge258328
def owner : Owner := ⟨.program ⟨257⟩, ⟨31351⟩⟩
def mergeEvent : Nat := 258328
def frameStart : Nat := 258298
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1009.exact258324RawTerms
def rightRaw : List Term := Proof.Events1009.exact258321RawTerms
def group : MergeGroup := .operator 258324 258321
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258324) (leftOrdinal := 0)
    (rightResult := 258321) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31350⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24230⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258328

namespace LeftMerge258358
def owner : Owner := ⟨.program ⟨257⟩, ⟨33208⟩⟩
def mergeEvent : Nat := 258358
def frameStart : Nat := 258298
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1009.exact258354RawTerms
def rightRaw : List Term := Proof.Events1009.exact258352RawTerms
def group : MergeGroup := .operator 258354 258352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258354) (leftOrdinal := 0)
    (rightResult := 258352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258358

namespace LeftMerge258381
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def mergeEvent : Nat := 258381
def frameStart : Nat := 258298
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }
def leftRaw : List Term := Proof.Events1009.exact258377RawTerms
def rightRaw : List Term := Proof.Events1009.exact258374RawTerms
def group : MergeGroup := .operator 258377 258374
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258377) (leftOrdinal := 0)
    (rightResult := 258374) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9577⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258381

namespace LeftMerge258390
def owner : Owner := ⟨.program ⟨257⟩, ⟨33407⟩⟩
def mergeEvent : Nat := 258390
def frameStart : Nat := 258298
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33404⟩⟩] } }
def leftRaw : List Term := Proof.Events1009.exact258386RawTerms
def rightRaw : List Term := Proof.Events1009.exact258343RawTerms
def group : MergeGroup := .operator 258386 258343
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258386) (leftOrdinal := 0)
    (rightResult := 258343) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33404⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33404⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258390

namespace LeftMerge258391
def owner : Owner := ⟨.program ⟨257⟩, ⟨33407⟩⟩
def mergeEvent : Nat := 258391
def frameStart : Nat := 258298
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33404⟩⟩] } }
def leftRaw : List Term := Proof.Events1009.exact258386RawTerms
def rightRaw : List Term := Proof.Events1009.exact258343RawTerms
def group : MergeGroup := .operator 258386 258343
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258386) (leftOrdinal := 1)
    (rightResult := 258343) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33404⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33404⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge258391

namespace LeftMerge258393
def owner : Owner := ⟨.program ⟨257⟩, ⟨33407⟩⟩
def mergeEvent : Nat := 258393
def frameStart : Nat := 258298
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32919⟩⟩] } }
def rhsRaw : List Term := Proof.Events1009.exact258340RawTerms
def group : MergeGroup := .relation 258392
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 258392) (rhsResult := 258340)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33404⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33404⟩⟩) ⟨32919⟩ 258340) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32919⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], [⟨.program ⟨257⟩, ⟨32919⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge258393

namespace LeftMerge258401
def owner : Owner := ⟨.program ⟨257⟩, ⟨31790⟩⟩
def mergeEvent : Nat := 258401
def frameStart : Nat := 258298
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1009.exact258354RawTerms
def rightRaw : List Term := Proof.Events1009.exact258397RawTerms
def group : MergeGroup := .operator 258354 258397
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258354) (leftOrdinal := 0)
    (rightResult := 258397) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31788⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258401

namespace LeftMerge258418
def owner : Owner := ⟨.program ⟨257⟩, ⟨32342⟩⟩
def mergeEvent : Nat := 258418
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }
def rhsRaw : List Term := Proof.Events1009.exact258415RawTerms
def group : MergeGroup := .relation 258417
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 258417) (rhsResult := 258415)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32339⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 258416 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32339⟩⟩]⟩) (none) 258415) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258418

namespace LeftMerge258419
def owner : Owner := ⟨.program ⟨257⟩, ⟨32342⟩⟩
def mergeEvent : Nat := 258419
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33404⟩⟩] } }
def rhsRaw : List Term := Proof.Events1009.exact258415RawTerms
def group : MergeGroup := .relation 258417
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 258417) (rhsResult := 258415)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32339⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 258416 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32339⟩⟩]⟩) (none) 258415) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33404⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33404⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge258419

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
