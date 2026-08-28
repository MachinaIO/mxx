import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge62310
def owner : Owner := ⟨.program ⟨257⟩, ⟨14590⟩⟩
def mergeEvent : Nat := 62310
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }
def leftRaw : List Term := Proof.Events243.exact62301RawTerms
def rightRaw : List Term := Proof.Events070.exact18112RawTerms
def group : MergeGroup := .operator 62301 18112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62301) (leftOrdinal := 0)
    (rightResult := 18112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9559⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62310

namespace LeftMerge62315
def owner : Owner := ⟨.program ⟨257⟩, ⟨42649⟩⟩
def mergeEvent : Nat := 62315
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }
def leftRaw : List Term := Proof.Events243.exact62311RawTerms
def rightRaw : List Term := Proof.Events243.exact62281RawTerms
def group : MergeGroup := .operator 62311 62281
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62311) (leftOrdinal := 1)
    (rightResult := 62281) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62315

namespace LeftMerge62323
def owner : Owner := ⟨.program ⟨257⟩, ⟨44377⟩⟩
def mergeEvent : Nat := 62323
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44376⟩⟩] } }
def leftRaw : List Term := Proof.Events243.exact62317RawTerms
def rightRaw : List Term := Proof.Events243.exact62253RawTerms
def group : MergeGroup := .operator 62317 62253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62317) (leftOrdinal := 1)
    (rightResult := 62253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44376⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44376⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62323

namespace LeftMerge62325
def owner : Owner := ⟨.program ⟨257⟩, ⟨44377⟩⟩
def mergeEvent : Nat := 62325
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43831⟩⟩] } }
def rhsRaw : List Term := Proof.Events243.exact62250RawTerms
def group : MergeGroup := .relation 62324
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62324) (rhsResult := 62250)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44376⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44376⟩⟩) ⟨43831⟩ 62250) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43831⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], [⟨.program ⟨257⟩, ⟨43831⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62325

namespace LeftMerge62326
def owner : Owner := ⟨.program ⟨257⟩, ⟨44377⟩⟩
def mergeEvent : Nat := 62326
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44376⟩⟩] } }
def leftRaw : List Term := Proof.Events243.exact62317RawTerms
def rightRaw : List Term := Proof.Events243.exact62253RawTerms
def group : MergeGroup := .operator 62317 62253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62317) (leftOrdinal := 0)
    (rightResult := 62253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44376⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44376⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62326

namespace LeftMerge62340
def owner : Owner := ⟨.program ⟨257⟩, ⟨43302⟩⟩
def mergeEvent : Nat := 62340
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43299⟩⟩] } }
def leftRaw : List Term := Proof.Events239.exact61370RawTerms
def rightRaw : List Term := Proof.Events243.exact62334RawTerms
def group : MergeGroup := .operator 61370 62334
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61370) (leftOrdinal := 0)
    (rightResult := 62334) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43299⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43299⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62340

namespace LeftMerge62419
def owner : Owner := ⟨.program ⟨257⟩, ⟨42643⟩⟩
def mergeEvent : Nat := 62419
def frameStart : Nat := 62389
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events243.exact62415RawTerms
def rightRaw : List Term := Proof.Events243.exact62412RawTerms
def group : MergeGroup := .operator 62415 62412
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62415) (leftOrdinal := 0)
    (rightResult := 62412) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14586⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42642⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62419

namespace LeftMerge62449
def owner : Owner := ⟨.program ⟨257⟩, ⟨44096⟩⟩
def mergeEvent : Nat := 62449
def frameStart : Nat := 62389
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events243.exact62445RawTerms
def rightRaw : List Term := Proof.Events243.exact62443RawTerms
def group : MergeGroup := .operator 62445 62443
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62445) (leftOrdinal := 0)
    (rightResult := 62443) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62449

namespace LeftMerge62472
def owner : Owner := ⟨.program ⟨257⟩, ⟨9561⟩⟩
def mergeEvent : Nat := 62472
def frameStart : Nat := 62389
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }
def leftRaw : List Term := Proof.Events244.exact62468RawTerms
def rightRaw : List Term := Proof.Events244.exact62465RawTerms
def group : MergeGroup := .operator 62468 62465
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62468) (leftOrdinal := 0)
    (rightResult := 62465) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9559⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62472

namespace LeftMerge62481
def owner : Owner := ⟨.program ⟨257⟩, ⟨44379⟩⟩
def mergeEvent : Nat := 62481
def frameStart : Nat := 62389
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44376⟩⟩] } }
def leftRaw : List Term := Proof.Events244.exact62477RawTerms
def rightRaw : List Term := Proof.Events243.exact62434RawTerms
def group : MergeGroup := .operator 62477 62434
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62477) (leftOrdinal := 0)
    (rightResult := 62434) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44376⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44376⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62481

namespace LeftMerge62482
def owner : Owner := ⟨.program ⟨257⟩, ⟨44379⟩⟩
def mergeEvent : Nat := 62482
def frameStart : Nat := 62389
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44376⟩⟩] } }
def leftRaw : List Term := Proof.Events244.exact62477RawTerms
def rightRaw : List Term := Proof.Events243.exact62434RawTerms
def group : MergeGroup := .operator 62477 62434
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62477) (leftOrdinal := 1)
    (rightResult := 62434) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44376⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44376⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62482

namespace LeftMerge62484
def owner : Owner := ⟨.program ⟨257⟩, ⟨44379⟩⟩
def mergeEvent : Nat := 62484
def frameStart : Nat := 62389
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43831⟩⟩] } }
def rhsRaw : List Term := Proof.Events243.exact62431RawTerms
def group : MergeGroup := .relation 62483
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62483) (rhsResult := 62431)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44376⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44376⟩⟩) ⟨43831⟩ 62431) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43831⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], [⟨.program ⟨257⟩, ⟨43831⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62484

namespace LeftMerge62492
def owner : Owner := ⟨.program ⟨257⟩, ⟨42846⟩⟩
def mergeEvent : Nat := 62492
def frameStart : Nat := 62389
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events243.exact62445RawTerms
def rightRaw : List Term := Proof.Events244.exact62488RawTerms
def group : MergeGroup := .operator 62445 62488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62445) (leftOrdinal := 0)
    (rightResult := 62488) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42844⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62492

namespace LeftMerge62509
def owner : Owner := ⟨.program ⟨257⟩, ⟨43302⟩⟩
def mergeEvent : Nat := 62509
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }
def rhsRaw : List Term := Proof.Events244.exact62506RawTerms
def group : MergeGroup := .relation 62508
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62508) (rhsResult := 62506)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43299⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62507 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43299⟩⟩]⟩) (none) 62506) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62509

namespace LeftMerge62510
def owner : Owner := ⟨.program ⟨257⟩, ⟨43302⟩⟩
def mergeEvent : Nat := 62510
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44376⟩⟩] } }
def rhsRaw : List Term := Proof.Events244.exact62506RawTerms
def group : MergeGroup := .relation 62508
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62508) (rhsResult := 62506)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43299⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62507 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43299⟩⟩]⟩) (none) 62506) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44376⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44376⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62510

namespace LeftMerge62511
def owner : Owner := ⟨.program ⟨257⟩, ⟨43302⟩⟩
def mergeEvent : Nat := 62511
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43831⟩⟩] } }
def rhsRaw : List Term := Proof.Events244.exact62506RawTerms
def group : MergeGroup := .relation 62508
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62508) (rhsResult := 62506)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43299⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62507 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43299⟩⟩]⟩) (none) 62506) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43831⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], [⟨.program ⟨257⟩, ⟨43831⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62511

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
