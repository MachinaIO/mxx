import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge67312
def owner : Owner := ⟨.program ⟨257⟩, ⟨53926⟩⟩
def mergeEvent : Nat := 67312
def frameStart : Nat := 67209
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53924⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events262.exact67265RawTerms
def rightRaw : List Term := Proof.Events262.exact67308RawTerms
def group : MergeGroup := .operator 67265 67308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67265) (leftOrdinal := 0)
    (rightResult := 67308) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53924⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67312

namespace LeftMerge67329
def owner : Owner := ⟨.program ⟨257⟩, ⟨54502⟩⟩
def mergeEvent : Nat := 67329
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }
def rhsRaw : List Term := Proof.Events262.exact67326RawTerms
def group : MergeGroup := .relation 67328
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67328) (rhsResult := 67326)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54499⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 67327 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54499⟩⟩]⟩) (none) 67326) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67329

namespace LeftMerge67330
def owner : Owner := ⟨.program ⟨257⟩, ⟨54502⟩⟩
def mergeEvent : Nat := 67330
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55576⟩⟩] } }
def rhsRaw : List Term := Proof.Events262.exact67326RawTerms
def group : MergeGroup := .relation 67328
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67328) (rhsResult := 67326)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54499⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 67327 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54499⟩⟩]⟩) (none) 67326) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55576⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55576⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67330

namespace LeftMerge67331
def owner : Owner := ⟨.program ⟨257⟩, ⟨54502⟩⟩
def mergeEvent : Nat := 67331
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24854⟩⟩, ⟨.program ⟨257⟩, ⟨53714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55031⟩⟩] } }
def rhsRaw : List Term := Proof.Events262.exact67326RawTerms
def group : MergeGroup := .relation 67328
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67328) (rhsResult := 67326)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54499⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 67327 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54499⟩⟩]⟩) (none) 67326) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24854⟩⟩, ⟨.program ⟨257⟩, ⟨53714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55031⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24854⟩⟩, ⟨.program ⟨257⟩, ⟨53714⟩⟩], [⟨.program ⟨257⟩, ⟨55031⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67331

namespace LeftMerge67332
def owner : Owner := ⟨.program ⟨257⟩, ⟨54502⟩⟩
def mergeEvent : Nat := 67332
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨53924⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events262.exact67326RawTerms
def group : MergeGroup := .relation 67328
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67328) (rhsResult := 67326)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54499⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 67327 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54499⟩⟩]⟩) (none) 67326) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53924⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨53924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67332

namespace LeftMerge67337
def owner : Owner := ⟨.program ⟨257⟩, ⟨55578⟩⟩
def mergeEvent : Nat := 67337
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24854⟩⟩, ⟨.program ⟨257⟩, ⟨53714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55031⟩⟩] } }
def leftRaw : List Term := Proof.Events263.exact67333RawTerms
def rightRaw : List Term := Proof.Events262.exact67147RawTerms
def group : MergeGroup := .operator 67333 67147
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67333) (leftOrdinal := 2)
    (rightResult := 67147) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24854⟩⟩, ⟨.program ⟨257⟩, ⟨53714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55031⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24854⟩⟩, ⟨.program ⟨257⟩, ⟨53714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55031⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24854⟩⟩, ⟨.program ⟨257⟩, ⟨53714⟩⟩], [⟨.program ⟨257⟩, ⟨55031⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67337

namespace LeftMerge67338
def owner : Owner := ⟨.program ⟨257⟩, ⟨55578⟩⟩
def mergeEvent : Nat := 67338
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55576⟩⟩] } }
def leftRaw : List Term := Proof.Events263.exact67333RawTerms
def rightRaw : List Term := Proof.Events262.exact67147RawTerms
def group : MergeGroup := .operator 67333 67147
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67333) (leftOrdinal := 1)
    (rightResult := 67147) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55576⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55576⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55576⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67338

namespace LeftMerge67346
def owner : Owner := ⟨.program ⟨257⟩, ⟨56151⟩⟩
def mergeEvent : Nat := 67346
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56149⟩⟩] } }
def leftRaw : List Term := Proof.Events263.exact67340RawTerms
def rightRaw : List Term := Proof.Events261.exact67063RawTerms
def group : MergeGroup := .operator 67340 67063
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67340) (leftOrdinal := 0)
    (rightResult := 67063) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56149⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56149⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67346

namespace LeftMerge67347
def owner : Owner := ⟨.program ⟨257⟩, ⟨56151⟩⟩
def mergeEvent : Nat := 67347
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨53924⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56149⟩⟩] } }
def leftRaw : List Term := Proof.Events263.exact67340RawTerms
def rightRaw : List Term := Proof.Events261.exact67063RawTerms
def group : MergeGroup := .operator 67340 67063
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67340) (leftOrdinal := 1)
    (rightResult := 67063) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨53924⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56149⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨53924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56149⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67347

namespace LeftMerge67349
def owner : Owner := ⟨.program ⟨257⟩, ⟨56151⟩⟩
def mergeEvent : Nat := 67349
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨53924⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55204⟩⟩] } }
def rhsRaw : List Term := Proof.Events261.exact67060RawTerms
def group : MergeGroup := .relation 67348
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67348) (rhsResult := 67060)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨53924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56149⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨56149⟩⟩) ⟨55204⟩ 67060) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55204⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨53924⟩⟩], [⟨.program ⟨257⟩, ⟨55204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67349

namespace LeftMerge67363
def owner : Owner := ⟨.program ⟨257⟩, ⟨54879⟩⟩
def mergeEvent : Nat := 67363
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54876⟩⟩] } }
def leftRaw : List Term := Proof.Events239.exact61370RawTerms
def rightRaw : List Term := Proof.Events263.exact67357RawTerms
def group : MergeGroup := .operator 61370 67357
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61370) (leftOrdinal := 0)
    (rightResult := 67357) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54876⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54876⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67363

namespace LeftMerge67484
def owner : Owner := ⟨.program ⟨257⟩, ⟨55376⟩⟩
def mergeEvent : Nat := 67484
def frameStart : Nat := 67418
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53924⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events263.exact67480RawTerms
def rightRaw : List Term := Proof.Events263.exact67478RawTerms
def group : MergeGroup := .operator 67480 67478
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67480) (leftOrdinal := 0)
    (rightResult := 67478) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53924⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67484

namespace LeftMerge67496
def owner : Owner := ⟨.program ⟨257⟩, ⟨56150⟩⟩
def mergeEvent : Nat := 67496
def frameStart : Nat := 67418
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56149⟩⟩] } }
def leftRaw : List Term := Proof.Events263.exact67492RawTerms
def rightRaw : List Term := Proof.Events263.exact67469RawTerms
def group : MergeGroup := .operator 67492 67469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67492) (leftOrdinal := 0)
    (rightResult := 67469) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56149⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56149⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67496

namespace LeftMerge67497
def owner : Owner := ⟨.program ⟨257⟩, ⟨56150⟩⟩
def mergeEvent : Nat := 67497
def frameStart : Nat := 67418
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53924⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56149⟩⟩] } }
def leftRaw : List Term := Proof.Events263.exact67492RawTerms
def rightRaw : List Term := Proof.Events263.exact67469RawTerms
def group : MergeGroup := .operator 67492 67469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67492) (leftOrdinal := 1)
    (rightResult := 67469) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53924⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56149⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56149⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67497

namespace LeftMerge67499
def owner : Owner := ⟨.program ⟨257⟩, ⟨56150⟩⟩
def mergeEvent : Nat := 67499
def frameStart : Nat := 67418
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53924⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55204⟩⟩] } }
def rhsRaw : List Term := Proof.Events263.exact67466RawTerms
def group : MergeGroup := .relation 67498
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67498) (rhsResult := 67466)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨53924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56149⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨56149⟩⟩) ⟨55204⟩ 67466) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55204⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨53924⟩⟩], [⟨.program ⟨257⟩, ⟨55204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67499

namespace LeftMerge67507
def owner : Owner := ⟨.program ⟨257⟩, ⟨54276⟩⟩
def mergeEvent : Nat := 67507
def frameStart : Nat := 67418
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events263.exact67480RawTerms
def rightRaw : List Term := Proof.Events263.exact67503RawTerms
def group : MergeGroup := .operator 67480 67503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67480) (leftOrdinal := 0)
    (rightResult := 67503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54274⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67507

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
