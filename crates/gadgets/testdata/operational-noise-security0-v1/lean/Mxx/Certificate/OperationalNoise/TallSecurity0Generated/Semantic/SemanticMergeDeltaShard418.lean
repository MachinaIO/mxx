import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge68260
def owner : Owner := ⟨.program ⟨214⟩, ⟨11956⟩⟩
def mergeEvent : Nat := 68260
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6784⟩⟩] } }
def leftRaw : List Term := Proof.Events266.exact68256RawTerms
def rightRaw : List Term := Proof.Events266.exact68226RawTerms
def group : MergeGroup := .operator 68256 68226
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 68256) (leftOrdinal := 1)
    (rightResult := 68226) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6784⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6784⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68260

namespace LeftMerge68268
def owner : Owner := ⟨.program ⟨214⟩, ⟨25215⟩⟩
def mergeEvent : Nat := 68268
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩] } }
def leftRaw : List Term := Proof.Events266.exact68262RawTerms
def rightRaw : List Term := Proof.Events266.exact68198RawTerms
def group : MergeGroup := .operator 68262 68198
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 68262) (leftOrdinal := 1)
    (rightResult := 68198) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25214⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge68268

namespace LeftMerge68270
def owner : Owner := ⟨.program ⟨214⟩, ⟨25215⟩⟩
def mergeEvent : Nat := 68270
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23120⟩⟩] } }
def rhsRaw : List Term := Proof.Events266.exact68195RawTerms
def group : MergeGroup := .relation 68269
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 68269) (rhsResult := 68195)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25214⟩⟩) ⟨23120⟩ 68195) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23120⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨23120⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge68270

namespace LeftMerge68271
def owner : Owner := ⟨.program ⟨214⟩, ⟨25215⟩⟩
def mergeEvent : Nat := 68271
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩] } }
def leftRaw : List Term := Proof.Events266.exact68262RawTerms
def rightRaw : List Term := Proof.Events266.exact68198RawTerms
def group : MergeGroup := .operator 68262 68198
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 68262) (leftOrdinal := 0)
    (rightResult := 68198) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25214⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68271

namespace LeftMerge68285
def owner : Owner := ⟨.program ⟨214⟩, ⟨19815⟩⟩
def mergeEvent : Nat := 68285
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19812⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65387RawTerms
def rightRaw : List Term := Proof.Events266.exact68279RawTerms
def group : MergeGroup := .operator 65387 68279
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65387) (leftOrdinal := 0)
    (rightResult := 68279) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19812⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68285

namespace LeftMerge68364
def owner : Owner := ⟨.program ⟨214⟩, ⟨11950⟩⟩
def mergeEvent : Nat := 68364
def frameStart : Nat := 68334
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events267.exact68360RawTerms
def rightRaw : List Term := Proof.Events267.exact68357RawTerms
def group : MergeGroup := .operator 68360 68357
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 68360) (leftOrdinal := 0)
    (rightResult := 68357) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9710⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11949⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68364

namespace LeftMerge68394
def owner : Owner := ⟨.program ⟨214⟩, ⟨12051⟩⟩
def mergeEvent : Nat := 68394
def frameStart : Nat := 68334
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events267.exact68390RawTerms
def rightRaw : List Term := Proof.Events267.exact68388RawTerms
def group : MergeGroup := .operator 68390 68388
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 68390) (leftOrdinal := 0)
    (rightResult := 68388) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68394

namespace LeftMerge68417
def owner : Owner := ⟨.program ⟨214⟩, ⟨7866⟩⟩
def mergeEvent : Nat := 68417
def frameStart : Nat := 68334
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩] } }
def leftRaw : List Term := Proof.Events267.exact68413RawTerms
def rightRaw : List Term := Proof.Events267.exact68410RawTerms
def group : MergeGroup := .operator 68413 68410
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 68413) (leftOrdinal := 0)
    (rightResult := 68410) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7864⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68417

namespace LeftMerge68426
def owner : Owner := ⟨.program ⟨214⟩, ⟨25217⟩⟩
def mergeEvent : Nat := 68426
def frameStart : Nat := 68334
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩] } }
def leftRaw : List Term := Proof.Events267.exact68422RawTerms
def rightRaw : List Term := Proof.Events267.exact68379RawTerms
def group : MergeGroup := .operator 68422 68379
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 68422) (leftOrdinal := 0)
    (rightResult := 68379) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25214⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68426

namespace LeftMerge68427
def owner : Owner := ⟨.program ⟨214⟩, ⟨25217⟩⟩
def mergeEvent : Nat := 68427
def frameStart : Nat := 68334
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩] } }
def leftRaw : List Term := Proof.Events267.exact68422RawTerms
def rightRaw : List Term := Proof.Events267.exact68379RawTerms
def group : MergeGroup := .operator 68422 68379
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 68422) (leftOrdinal := 1)
    (rightResult := 68379) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25214⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge68427

namespace LeftMerge68429
def owner : Owner := ⟨.program ⟨214⟩, ⟨25217⟩⟩
def mergeEvent : Nat := 68429
def frameStart : Nat := 68334
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23120⟩⟩] } }
def rhsRaw : List Term := Proof.Events267.exact68376RawTerms
def group : MergeGroup := .relation 68428
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 68428) (rhsResult := 68376)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25214⟩⟩) ⟨23120⟩ 68376) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23120⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨23120⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge68429

namespace LeftMerge68437
def owner : Owner := ⟨.program ⟨214⟩, ⟨16379⟩⟩
def mergeEvent : Nat := 68437
def frameStart : Nat := 68334
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16377⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events267.exact68390RawTerms
def rightRaw : List Term := Proof.Events267.exact68433RawTerms
def group : MergeGroup := .operator 68390 68433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 68390) (leftOrdinal := 0)
    (rightResult := 68433) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16377⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68437

namespace LeftMerge68454
def owner : Owner := ⟨.program ⟨214⟩, ⟨19815⟩⟩
def mergeEvent : Nat := 68454
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩] } }
def rhsRaw : List Term := Proof.Events267.exact68451RawTerms
def group : MergeGroup := .relation 68453
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 68453) (rhsResult := 68451)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 68452 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩) (none) 68451) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68454

namespace LeftMerge68455
def owner : Owner := ⟨.program ⟨214⟩, ⟨19815⟩⟩
def mergeEvent : Nat := 68455
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩] } }
def rhsRaw : List Term := Proof.Events267.exact68451RawTerms
def group : MergeGroup := .relation 68453
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 68453) (rhsResult := 68451)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 68452 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩) (none) 68451) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge68455

namespace LeftMerge68456
def owner : Owner := ⟨.program ⟨214⟩, ⟨19815⟩⟩
def mergeEvent : Nat := 68456
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23120⟩⟩] } }
def rhsRaw : List Term := Proof.Events267.exact68451RawTerms
def group : MergeGroup := .relation 68453
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 68453) (rhsResult := 68451)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 68452 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩) (none) 68451) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23120⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨23120⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68456

namespace LeftMerge68457
def owner : Owner := ⟨.program ⟨214⟩, ⟨19815⟩⟩
def mergeEvent : Nat := 68457
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16377⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events267.exact68451RawTerms
def group : MergeGroup := .relation 68453
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 68453) (rhsResult := 68451)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 68452 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩) (none) 68451) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16377⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge68457

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
