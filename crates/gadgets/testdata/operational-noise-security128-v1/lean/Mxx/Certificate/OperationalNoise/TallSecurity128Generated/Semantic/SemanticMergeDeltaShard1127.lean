import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge184146
def owner : Owner := ⟨.program ⟨257⟩, ⟨55533⟩⟩
def mergeEvent : Nat := 184146
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55532⟩⟩] } }
def leftRaw : List Term := Proof.Events719.exact184137RawTerms
def rightRaw : List Term := Proof.Events719.exact184073RawTerms
def group : MergeGroup := .operator 184137 184073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184137) (leftOrdinal := 0)
    (rightResult := 184073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55532⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184146

namespace LeftMerge184160
def owner : Owner := ⟨.program ⟨257⟩, ⟨54462⟩⟩
def mergeEvent : Nat := 184160
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54459⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events719.exact184154RawTerms
def group : MergeGroup := .operator 178370 184154
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 184154) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54459⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54459⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184160

namespace LeftMerge184239
def owner : Owner := ⟨.program ⟨257⟩, ⟨53607⟩⟩
def mergeEvent : Nat := 184239
def frameStart : Nat := 184209
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events719.exact184235RawTerms
def rightRaw : List Term := Proof.Events719.exact184232RawTerms
def group : MergeGroup := .operator 184235 184232
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184235) (leftOrdinal := 0)
    (rightResult := 184232) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53606⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184239

namespace LeftMerge184269
def owner : Owner := ⟨.program ⟨257⟩, ⟨55280⟩⟩
def mergeEvent : Nat := 184269
def frameStart : Nat := 184209
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events719.exact184265RawTerms
def rightRaw : List Term := Proof.Events719.exact184263RawTerms
def group : MergeGroup := .operator 184265 184263
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184265) (leftOrdinal := 0)
    (rightResult := 184263) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184269

namespace LeftMerge184292
def owner : Owner := ⟨.program ⟨257⟩, ⟨9531⟩⟩
def mergeEvent : Nat := 184292
def frameStart : Nat := 184209
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }
def leftRaw : List Term := Proof.Events719.exact184288RawTerms
def rightRaw : List Term := Proof.Events719.exact184285RawTerms
def group : MergeGroup := .operator 184288 184285
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184288) (leftOrdinal := 0)
    (rightResult := 184285) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9529⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184292

namespace LeftMerge184301
def owner : Owner := ⟨.program ⟨257⟩, ⟨55535⟩⟩
def mergeEvent : Nat := 184301
def frameStart : Nat := 184209
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55532⟩⟩] } }
def leftRaw : List Term := Proof.Events719.exact184297RawTerms
def rightRaw : List Term := Proof.Events719.exact184254RawTerms
def group : MergeGroup := .operator 184297 184254
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184297) (leftOrdinal := 0)
    (rightResult := 184254) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55532⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184301

namespace LeftMerge184302
def owner : Owner := ⟨.program ⟨257⟩, ⟨55535⟩⟩
def mergeEvent : Nat := 184302
def frameStart : Nat := 184209
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55532⟩⟩] } }
def leftRaw : List Term := Proof.Events719.exact184297RawTerms
def rightRaw : List Term := Proof.Events719.exact184254RawTerms
def group : MergeGroup := .operator 184297 184254
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184297) (leftOrdinal := 1)
    (rightResult := 184254) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55532⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55532⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184302

namespace LeftMerge184304
def owner : Owner := ⟨.program ⟨257⟩, ⟨55535⟩⟩
def mergeEvent : Nat := 184304
def frameStart : Nat := 184209
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55007⟩⟩] } }
def rhsRaw : List Term := Proof.Events719.exact184251RawTerms
def group : MergeGroup := .relation 184303
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 184303) (rhsResult := 184251)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55532⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55532⟩⟩) ⟨55007⟩ 184251) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55007⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], [⟨.program ⟨257⟩, ⟨55007⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184304

namespace LeftMerge184312
def owner : Owner := ⟨.program ⟨257⟩, ⟨53894⟩⟩
def mergeEvent : Nat := 184312
def frameStart : Nat := 184209
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events719.exact184265RawTerms
def rightRaw : List Term := Proof.Events719.exact184308RawTerms
def group : MergeGroup := .operator 184265 184308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184265) (leftOrdinal := 0)
    (rightResult := 184308) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184312

namespace LeftMerge184329
def owner : Owner := ⟨.program ⟨257⟩, ⟨54462⟩⟩
def mergeEvent : Nat := 184329
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }
def rhsRaw : List Term := Proof.Events720.exact184326RawTerms
def group : MergeGroup := .relation 184328
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 184328) (rhsResult := 184326)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54459⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 184327 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54459⟩⟩]⟩) (none) 184326) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184329

namespace LeftMerge184330
def owner : Owner := ⟨.program ⟨257⟩, ⟨54462⟩⟩
def mergeEvent : Nat := 184330
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55532⟩⟩] } }
def rhsRaw : List Term := Proof.Events720.exact184326RawTerms
def group : MergeGroup := .relation 184328
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 184328) (rhsResult := 184326)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54459⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 184327 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54459⟩⟩]⟩) (none) 184326) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55532⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55532⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184330

namespace LeftMerge184331
def owner : Owner := ⟨.program ⟨257⟩, ⟨54462⟩⟩
def mergeEvent : Nat := 184331
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55007⟩⟩] } }
def rhsRaw : List Term := Proof.Events720.exact184326RawTerms
def group : MergeGroup := .relation 184328
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 184328) (rhsResult := 184326)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54459⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 184327 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54459⟩⟩]⟩) (none) 184326) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55007⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], [⟨.program ⟨257⟩, ⟨55007⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184331

namespace LeftMerge184332
def owner : Owner := ⟨.program ⟨257⟩, ⟨54462⟩⟩
def mergeEvent : Nat := 184332
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events720.exact184326RawTerms
def group : MergeGroup := .relation 184328
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 184328) (rhsResult := 184326)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54459⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 184327 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54459⟩⟩]⟩) (none) 184326) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184332

namespace LeftMerge184337
def owner : Owner := ⟨.program ⟨257⟩, ⟨55534⟩⟩
def mergeEvent : Nat := 184337
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55007⟩⟩] } }
def leftRaw : List Term := Proof.Events720.exact184333RawTerms
def rightRaw : List Term := Proof.Events719.exact184147RawTerms
def group : MergeGroup := .operator 184333 184147
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184333) (leftOrdinal := 2)
    (rightResult := 184147) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55007⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55007⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], [⟨.program ⟨257⟩, ⟨55007⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184337

namespace LeftMerge184338
def owner : Owner := ⟨.program ⟨257⟩, ⟨55534⟩⟩
def mergeEvent : Nat := 184338
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55532⟩⟩] } }
def leftRaw : List Term := Proof.Events720.exact184333RawTerms
def rightRaw : List Term := Proof.Events719.exact184147RawTerms
def group : MergeGroup := .operator 184333 184147
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184333) (leftOrdinal := 1)
    (rightResult := 184147) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55532⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184338

namespace LeftMerge184346
def owner : Owner := ⟨.program ⟨257⟩, ⟨56027⟩⟩
def mergeEvent : Nat := 184346
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56025⟩⟩] } }
def leftRaw : List Term := Proof.Events720.exact184340RawTerms
def rightRaw : List Term := Proof.Events718.exact184063RawTerms
def group : MergeGroup := .operator 184340 184063
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184340) (leftOrdinal := 0)
    (rightResult := 184063) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56025⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56025⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184346

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
