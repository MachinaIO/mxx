import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge46339
def owner : Owner := ⟨.program ⟨214⟩, ⟨30168⟩⟩
def mergeEvent : Nat := 46339
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18503⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46332RawTerms
def rightRaw : List Term := Proof.Events021.exact5499RawTerms
def group : MergeGroup := .operator 46332 5499
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46332) (leftOrdinal := 1)
    (rightResult := 5499) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18503⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6651⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18503⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46339

namespace LeftMerge46341
def owner : Owner := ⟨.program ⟨214⟩, ⟨30168⟩⟩
def mergeEvent : Nat := 46341
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18503⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events021.exact5492RawTerms
def group : MergeGroup := .relation 46340
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46340) (rhsResult := 5492)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18503⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6651⟩⟩) ⟨6597⟩ 5492) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18503⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46341

namespace LeftMerge46355
def owner : Owner := ⟨.program ⟨214⟩, ⟨30156⟩⟩
def mergeEvent : Nat := 46355
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30154⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36323RawTerms
def rightRaw : List Term := Proof.Events181.exact46349RawTerms
def group : MergeGroup := .operator 36323 46349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36323) (leftOrdinal := 0)
    (rightResult := 46349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30154⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30154⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46355

namespace LeftMerge46356
def owner : Owner := ⟨.program ⟨214⟩, ⟨30156⟩⟩
def mergeEvent : Nat := 46356
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30154⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36323RawTerms
def rightRaw : List Term := Proof.Events181.exact46349RawTerms
def group : MergeGroup := .operator 36323 46349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36323) (leftOrdinal := 1)
    (rightResult := 46349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30154⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30154⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46356

namespace LeftMerge46358
def owner : Owner := ⟨.program ⟨214⟩, ⟨30156⟩⟩
def mergeEvent : Nat := 46358
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24797⟩⟩] } }
def rhsRaw : List Term := Proof.Events181.exact46346RawTerms
def group : MergeGroup := .relation 46357
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46357) (rhsResult := 46346)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30154⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨30154⟩⟩) ⟨24797⟩ 46346) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24797⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], [⟨.program ⟨214⟩, ⟨24797⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46358

namespace LeftMerge46372
def owner : Owner := ⟨.program ⟨214⟩, ⟨22779⟩⟩
def mergeEvent : Nat := 46372
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22776⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events181.exact46366RawTerms
def group : MergeGroup := .operator 36137 46366
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 46366) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22776⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22776⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46372

namespace LeftMerge46493
def owner : Owner := ⟨.program ⟨214⟩, ⟨17061⟩⟩
def mergeEvent : Nat := 46493
def frameStart : Nat := 46427
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46489RawTerms
def rightRaw : List Term := Proof.Events181.exact46487RawTerms
def group : MergeGroup := .operator 46489 46487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46489) (leftOrdinal := 0)
    (rightResult := 46487) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17019⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46493

namespace LeftMerge46505
def owner : Owner := ⟨.program ⟨214⟩, ⟨30155⟩⟩
def mergeEvent : Nat := 46505
def frameStart : Nat := 46427
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30154⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46501RawTerms
def rightRaw : List Term := Proof.Events181.exact46478RawTerms
def group : MergeGroup := .operator 46501 46478
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46501) (leftOrdinal := 0)
    (rightResult := 46478) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30154⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30154⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46505

namespace LeftMerge46506
def owner : Owner := ⟨.program ⟨214⟩, ⟨30155⟩⟩
def mergeEvent : Nat := 46506
def frameStart : Nat := 46427
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30154⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46501RawTerms
def rightRaw : List Term := Proof.Events181.exact46478RawTerms
def group : MergeGroup := .operator 46501 46478
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46501) (leftOrdinal := 1)
    (rightResult := 46478) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30154⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17019⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30154⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46506

namespace LeftMerge46508
def owner : Owner := ⟨.program ⟨214⟩, ⟨30155⟩⟩
def mergeEvent : Nat := 46508
def frameStart : Nat := 46427
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24797⟩⟩] } }
def rhsRaw : List Term := Proof.Events181.exact46475RawTerms
def group : MergeGroup := .relation 46507
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46507) (rhsResult := 46475)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨17019⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30154⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨30154⟩⟩) ⟨24797⟩ 46475) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24797⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨17019⟩⟩], [⟨.program ⟨214⟩, ⟨24797⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46508

namespace LeftMerge46516
def owner : Owner := ⟨.program ⟨214⟩, ⟨18134⟩⟩
def mergeEvent : Nat := 46516
def frameStart : Nat := 46427
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18132⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46489RawTerms
def rightRaw : List Term := Proof.Events181.exact46512RawTerms
def group : MergeGroup := .operator 46489 46512
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46489) (leftOrdinal := 0)
    (rightResult := 46512) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18132⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18132⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46516

namespace LeftMerge46533
def owner : Owner := ⟨.program ⟨214⟩, ⟨22779⟩⟩
def mergeEvent : Nat := 46533
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6742⟩⟩] } }
def rhsRaw : List Term := Proof.Events181.exact46530RawTerms
def group : MergeGroup := .relation 46532
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46532) (rhsResult := 46530)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22776⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 46531 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22776⟩⟩]⟩) (none) 46530) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6742⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6742⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46533

namespace LeftMerge46534
def owner : Owner := ⟨.program ⟨214⟩, ⟨22779⟩⟩
def mergeEvent : Nat := 46534
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30154⟩⟩] } }
def rhsRaw : List Term := Proof.Events181.exact46530RawTerms
def group : MergeGroup := .relation 46532
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46532) (rhsResult := 46530)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22776⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 46531 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22776⟩⟩]⟩) (none) 46530) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30154⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30154⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46534

namespace LeftMerge46535
def owner : Owner := ⟨.program ⟨214⟩, ⟨22779⟩⟩
def mergeEvent : Nat := 46535
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24797⟩⟩] } }
def rhsRaw : List Term := Proof.Events181.exact46530RawTerms
def group : MergeGroup := .relation 46532
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46532) (rhsResult := 46530)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22776⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 46531 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22776⟩⟩]⟩) (none) 46530) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24797⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], [⟨.program ⟨214⟩, ⟨24797⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46535

namespace LeftMerge46536
def owner : Owner := ⟨.program ⟨214⟩, ⟨22779⟩⟩
def mergeEvent : Nat := 46536
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18132⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events181.exact46530RawTerms
def group : MergeGroup := .relation 46532
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46532) (rhsResult := 46530)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22776⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 46531 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22776⟩⟩]⟩) (none) 46530) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18132⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18132⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46536

namespace LeftMerge46541
def owner : Owner := ⟨.program ⟨214⟩, ⟨30157⟩⟩
def mergeEvent : Nat := 46541
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30154⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46537RawTerms
def rightRaw : List Term := Proof.Events181.exact46359RawTerms
def group : MergeGroup := .operator 46537 46359
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46537) (leftOrdinal := 0)
    (rightResult := 46359) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30154⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30154⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30154⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46541

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
