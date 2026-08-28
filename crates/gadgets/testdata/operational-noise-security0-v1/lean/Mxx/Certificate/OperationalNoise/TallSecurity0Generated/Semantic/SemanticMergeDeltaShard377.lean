import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge62402
def owner : Owner := ⟨.program ⟨214⟩, ⟨28741⟩⟩
def mergeEvent : Nat := 62402
def frameStart : Nat := 62324
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28740⟩⟩] } }
def leftRaw : List Term := Proof.Events243.exact62398RawTerms
def rightRaw : List Term := Proof.Events243.exact62375RawTerms
def group : MergeGroup := .operator 62398 62375
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62398) (leftOrdinal := 0)
    (rightResult := 62375) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28740⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28740⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62402

namespace LeftMerge62403
def owner : Owner := ⟨.program ⟨214⟩, ⟨28741⟩⟩
def mergeEvent : Nat := 62403
def frameStart : Nat := 62324
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28740⟩⟩] } }
def leftRaw : List Term := Proof.Events243.exact62398RawTerms
def rightRaw : List Term := Proof.Events243.exact62375RawTerms
def group : MergeGroup := .operator 62398 62375
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62398) (leftOrdinal := 1)
    (rightResult := 62375) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28740⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16385⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28740⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62403

namespace LeftMerge62405
def owner : Owner := ⟨.program ⟨214⟩, ⟨28741⟩⟩
def mergeEvent : Nat := 62405
def frameStart : Nat := 62324
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24416⟩⟩] } }
def rhsRaw : List Term := Proof.Events243.exact62372RawTerms
def group : MergeGroup := .relation 62404
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62404) (rhsResult := 62372)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16385⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28740⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28740⟩⟩) ⟨24416⟩ 62372) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24416⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16385⟩⟩], [⟨.program ⟨214⟩, ⟨24416⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62405

namespace LeftMerge62413
def owner : Owner := ⟨.program ⟨214⟩, ⟨18857⟩⟩
def mergeEvent : Nat := 62413
def frameStart : Nat := 62324
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18848⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events243.exact62386RawTerms
def rightRaw : List Term := Proof.Events243.exact62409RawTerms
def group : MergeGroup := .operator 62386 62409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62386) (leftOrdinal := 0)
    (rightResult := 62409) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18848⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18848⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62413

namespace LeftMerge62430
def owner : Owner := ⟨.program ⟨214⟩, ⟨21911⟩⟩
def mergeEvent : Nat := 62430
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6730⟩⟩] } }
def rhsRaw : List Term := Proof.Events243.exact62427RawTerms
def group : MergeGroup := .relation 62429
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62429) (rhsResult := 62427)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21908⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62428 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21908⟩⟩]⟩) (none) 62427) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6730⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6730⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62430

namespace LeftMerge62431
def owner : Owner := ⟨.program ⟨214⟩, ⟨21911⟩⟩
def mergeEvent : Nat := 62431
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28740⟩⟩] } }
def rhsRaw : List Term := Proof.Events243.exact62427RawTerms
def group : MergeGroup := .relation 62429
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62429) (rhsResult := 62427)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21908⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62428 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21908⟩⟩]⟩) (none) 62427) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28740⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28740⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62431

namespace LeftMerge62432
def owner : Owner := ⟨.program ⟨214⟩, ⟨21911⟩⟩
def mergeEvent : Nat := 62432
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24416⟩⟩] } }
def rhsRaw : List Term := Proof.Events243.exact62427RawTerms
def group : MergeGroup := .relation 62429
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62429) (rhsResult := 62427)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21908⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62428 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21908⟩⟩]⟩) (none) 62427) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24416⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16385⟩⟩], [⟨.program ⟨214⟩, ⟨24416⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62432

namespace LeftMerge62433
def owner : Owner := ⟨.program ⟨214⟩, ⟨21911⟩⟩
def mergeEvent : Nat := 62433
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events243.exact62427RawTerms
def group : MergeGroup := .relation 62429
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62429) (rhsResult := 62427)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21908⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62428 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21908⟩⟩]⟩) (none) 62427) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18848⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62433

namespace LeftMerge62438
def owner : Owner := ⟨.program ⟨214⟩, ⟨28743⟩⟩
def mergeEvent : Nat := 62438
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28740⟩⟩] } }
def leftRaw : List Term := Proof.Events243.exact62434RawTerms
def rightRaw : List Term := Proof.Events243.exact62256RawTerms
def group : MergeGroup := .operator 62434 62256
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62434) (leftOrdinal := 0)
    (rightResult := 62256) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28740⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28740⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28740⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62438

namespace LeftMerge62439
def owner : Owner := ⟨.program ⟨214⟩, ⟨28743⟩⟩
def mergeEvent : Nat := 62439
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24416⟩⟩] } }
def leftRaw : List Term := Proof.Events243.exact62434RawTerms
def rightRaw : List Term := Proof.Events243.exact62256RawTerms
def group : MergeGroup := .operator 62434 62256
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62434) (leftOrdinal := 2)
    (rightResult := 62256) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24416⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24416⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16385⟩⟩], [⟨.program ⟨214⟩, ⟨24416⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62439

namespace LeftMerge62447
def owner : Owner := ⟨.program ⟨214⟩, ⟨28744⟩⟩
def mergeEvent : Nat := 62447
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩] } }
def leftRaw : List Term := Proof.Events243.exact62441RawTerms
def rightRaw : List Term := Proof.Events022.exact5639RawTerms
def group : MergeGroup := .operator 62441 5639
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62441) (leftOrdinal := 0)
    (rightResult := 5639) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6730⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6673⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62447

namespace LeftMerge62448
def owner : Owner := ⟨.program ⟨214⟩, ⟨28744⟩⟩
def mergeEvent : Nat := 62448
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩] } }
def leftRaw : List Term := Proof.Events243.exact62441RawTerms
def rightRaw : List Term := Proof.Events022.exact5639RawTerms
def group : MergeGroup := .operator 62441 5639
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62441) (leftOrdinal := 1)
    (rightResult := 5639) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6673⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62448

namespace LeftMerge62450
def owner : Owner := ⟨.program ⟨214⟩, ⟨28744⟩⟩
def mergeEvent : Nat := 62450
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events022.exact5632RawTerms
def group : MergeGroup := .relation 62449
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62449) (rhsResult := 5632)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6673⟩⟩) ⟨6608⟩ 5632) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62450

namespace LeftMerge62464
def owner : Owner := ⟨.program ⟨214⟩, ⟨28525⟩⟩
def mergeEvent : Nat := 62464
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28523⟩⟩] } }
def leftRaw : List Term := Proof.Events212.exact54322RawTerms
def rightRaw : List Term := Proof.Events243.exact62458RawTerms
def group : MergeGroup := .operator 54322 62458
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54322) (leftOrdinal := 0)
    (rightResult := 62458) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28523⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28523⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62464

namespace LeftMerge62465
def owner : Owner := ⟨.program ⟨214⟩, ⟨28525⟩⟩
def mergeEvent : Nat := 62465
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16266⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28523⟩⟩] } }
def leftRaw : List Term := Proof.Events212.exact54322RawTerms
def rightRaw : List Term := Proof.Events243.exact62458RawTerms
def group : MergeGroup := .operator 54322 62458
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54322) (leftOrdinal := 1)
    (rightResult := 62458) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16266⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28523⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16266⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28523⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62465

namespace LeftMerge62467
def owner : Owner := ⟨.program ⟨214⟩, ⟨28525⟩⟩
def mergeEvent : Nat := 62467
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16266⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24353⟩⟩] } }
def rhsRaw : List Term := Proof.Events243.exact62455RawTerms
def group : MergeGroup := .relation 62466
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62466) (rhsResult := 62455)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16266⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28523⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28523⟩⟩) ⟨24353⟩ 62455) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24353⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16266⟩⟩], [⟨.program ⟨214⟩, ⟨24353⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62467

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
