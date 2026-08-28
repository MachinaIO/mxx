import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge28398
def owner : Owner := ⟨.program ⟨214⟩, ⟨7842⟩⟩
def mergeEvent : Nat := 28398
def frameStart : Nat := 28315
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }
def leftRaw : List Term := Proof.Events110.exact28394RawTerms
def rightRaw : List Term := Proof.Events110.exact28391RawTerms
def group : MergeGroup := .operator 28394 28391
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28394) (leftOrdinal := 0)
    (rightResult := 28391) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7840⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28398

namespace LeftMerge28407
def owner : Owner := ⟨.program ⟨214⟩, ⟨25314⟩⟩
def mergeEvent : Nat := 28407
def frameStart : Nat := 28315
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩] } }
def leftRaw : List Term := Proof.Events110.exact28403RawTerms
def rightRaw : List Term := Proof.Events110.exact28360RawTerms
def group : MergeGroup := .operator 28403 28360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28403) (leftOrdinal := 0)
    (rightResult := 28360) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25311⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28407

namespace LeftMerge28408
def owner : Owner := ⟨.program ⟨214⟩, ⟨25314⟩⟩
def mergeEvent : Nat := 28408
def frameStart : Nat := 28315
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩] } }
def leftRaw : List Term := Proof.Events110.exact28403RawTerms
def rightRaw : List Term := Proof.Events110.exact28360RawTerms
def group : MergeGroup := .operator 28403 28360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28403) (leftOrdinal := 1)
    (rightResult := 28360) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25311⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28408

namespace LeftMerge28410
def owner : Owner := ⟨.program ⟨214⟩, ⟨25314⟩⟩
def mergeEvent : Nat := 28410
def frameStart : Nat := 28315
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23170⟩⟩] } }
def rhsRaw : List Term := Proof.Events110.exact28357RawTerms
def group : MergeGroup := .relation 28409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28409) (rhsResult := 28357)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25311⟩⟩) ⟨23170⟩ 28357) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23170⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨23170⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28410

namespace LeftMerge28418
def owner : Owner := ⟨.program ⟨214⟩, ⟨15436⟩⟩
def mergeEvent : Nat := 28418
def frameStart : Nat := 28315
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15434⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events110.exact28371RawTerms
def rightRaw : List Term := Proof.Events110.exact28414RawTerms
def group : MergeGroup := .operator 28371 28414
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28371) (leftOrdinal := 0)
    (rightResult := 28414) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15434⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28418

namespace LeftMerge28435
def owner : Owner := ⟨.program ⟨214⟩, ⟨19255⟩⟩
def mergeEvent : Nat := 28435
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩] } }
def rhsRaw : List Term := Proof.Events111.exact28432RawTerms
def group : MergeGroup := .relation 28434
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28434) (rhsResult := 28432)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19252⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 28433 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19252⟩⟩]⟩) (none) 28432) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28435

namespace LeftMerge28436
def owner : Owner := ⟨.program ⟨214⟩, ⟨19255⟩⟩
def mergeEvent : Nat := 28436
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩] } }
def rhsRaw : List Term := Proof.Events111.exact28432RawTerms
def group : MergeGroup := .relation 28434
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28434) (rhsResult := 28432)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19252⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 28433 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19252⟩⟩]⟩) (none) 28432) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28436

namespace LeftMerge28437
def owner : Owner := ⟨.program ⟨214⟩, ⟨19255⟩⟩
def mergeEvent : Nat := 28437
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23170⟩⟩] } }
def rhsRaw : List Term := Proof.Events111.exact28432RawTerms
def group : MergeGroup := .relation 28434
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28434) (rhsResult := 28432)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19252⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 28433 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19252⟩⟩]⟩) (none) 28432) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23170⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨23170⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28437

namespace LeftMerge28438
def owner : Owner := ⟨.program ⟨214⟩, ⟨19255⟩⟩
def mergeEvent : Nat := 28438
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15434⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events111.exact28432RawTerms
def group : MergeGroup := .relation 28434
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28434) (rhsResult := 28432)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19252⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 28433 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19252⟩⟩]⟩) (none) 28432) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15434⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28438

namespace LeftMerge28443
def owner : Owner := ⟨.program ⟨214⟩, ⟨25313⟩⟩
def mergeEvent : Nat := 28443
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23170⟩⟩] } }
def leftRaw : List Term := Proof.Events111.exact28439RawTerms
def rightRaw : List Term := Proof.Events110.exact28253RawTerms
def group : MergeGroup := .operator 28439 28253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28439) (leftOrdinal := 2)
    (rightResult := 28253) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23170⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23170⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨23170⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28443

namespace LeftMerge28444
def owner : Owner := ⟨.program ⟨214⟩, ⟨25313⟩⟩
def mergeEvent : Nat := 28444
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩] } }
def leftRaw : List Term := Proof.Events111.exact28439RawTerms
def rightRaw : List Term := Proof.Events110.exact28253RawTerms
def group : MergeGroup := .operator 28439 28253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28439) (leftOrdinal := 1)
    (rightResult := 28253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28444

namespace LeftMerge28452
def owner : Owner := ⟨.program ⟨214⟩, ⟨27039⟩⟩
def mergeEvent : Nat := 28452
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27037⟩⟩] } }
def leftRaw : List Term := Proof.Events111.exact28446RawTerms
def rightRaw : List Term := Proof.Events110.exact28169RawTerms
def group : MergeGroup := .operator 28446 28169
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28446) (leftOrdinal := 0)
    (rightResult := 28169) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27037⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28452

namespace LeftMerge28453
def owner : Owner := ⟨.program ⟨214⟩, ⟨27039⟩⟩
def mergeEvent : Nat := 28453
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15434⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27037⟩⟩] } }
def leftRaw : List Term := Proof.Events111.exact28446RawTerms
def rightRaw : List Term := Proof.Events110.exact28169RawTerms
def group : MergeGroup := .operator 28446 28169
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28446) (leftOrdinal := 1)
    (rightResult := 28169) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15434⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27037⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28453

namespace LeftMerge28455
def owner : Owner := ⟨.program ⟨214⟩, ⟨27039⟩⟩
def mergeEvent : Nat := 28455
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15434⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23919⟩⟩] } }
def rhsRaw : List Term := Proof.Events110.exact28166RawTerms
def group : MergeGroup := .relation 28454
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28454) (rhsResult := 28166)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27037⟩⟩) ⟨23919⟩ 28166) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23919⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨23919⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28455

namespace LeftMerge28469
def owner : Owner := ⟨.program ⟨214⟩, ⟨20839⟩⟩
def mergeEvent : Nat := 28469
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20836⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21512RawTerms
def rightRaw : List Term := Proof.Events111.exact28463RawTerms
def group : MergeGroup := .operator 21512 28463
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21512) (leftOrdinal := 0)
    (rightResult := 28463) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20836⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28469

namespace LeftMerge28590
def owner : Owner := ⟨.program ⟨214⟩, ⟨15476⟩⟩
def mergeEvent : Nat := 28590
def frameStart : Nat := 28524
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15434⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events111.exact28586RawTerms
def rightRaw : List Term := Proof.Events111.exact28584RawTerms
def group : MergeGroup := .operator 28586 28584
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28586) (leftOrdinal := 0)
    (rightResult := 28584) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15434⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28590

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
