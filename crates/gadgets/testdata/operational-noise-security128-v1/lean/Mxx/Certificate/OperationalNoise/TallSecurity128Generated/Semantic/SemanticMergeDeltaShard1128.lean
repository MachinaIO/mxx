import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge184347
def owner : Owner := ⟨.program ⟨257⟩, ⟨56027⟩⟩
def mergeEvent : Nat := 184347
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56025⟩⟩] } }
def leftRaw : List Term := Proof.Events720.exact184340RawTerms
def rightRaw : List Term := Proof.Events718.exact184063RawTerms
def group : MergeGroup := .operator 184340 184063
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184340) (leftOrdinal := 1)
    (rightResult := 184063) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56025⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56025⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184347

namespace LeftMerge184349
def owner : Owner := ⟨.program ⟨257⟩, ⟨56027⟩⟩
def mergeEvent : Nat := 184349
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55168⟩⟩] } }
def rhsRaw : List Term := Proof.Events718.exact184060RawTerms
def group : MergeGroup := .relation 184348
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 184348) (rhsResult := 184060)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56025⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨56025⟩⟩) ⟨55168⟩ 184060) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55168⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53892⟩⟩], [⟨.program ⟨257⟩, ⟨55168⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184349

namespace LeftMerge184363
def owner : Owner := ⟨.program ⟨257⟩, ⟨54799⟩⟩
def mergeEvent : Nat := 184363
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54796⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events720.exact184357RawTerms
def group : MergeGroup := .operator 178370 184357
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 184357) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54796⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54796⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184363

namespace LeftMerge184484
def owner : Owner := ⟨.program ⟨257⟩, ⟨55360⟩⟩
def mergeEvent : Nat := 184484
def frameStart : Nat := 184418
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events720.exact184480RawTerms
def rightRaw : List Term := Proof.Events720.exact184478RawTerms
def group : MergeGroup := .operator 184480 184478
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184480) (leftOrdinal := 0)
    (rightResult := 184478) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184484

namespace LeftMerge184496
def owner : Owner := ⟨.program ⟨257⟩, ⟨56026⟩⟩
def mergeEvent : Nat := 184496
def frameStart : Nat := 184418
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56025⟩⟩] } }
def leftRaw : List Term := Proof.Events720.exact184492RawTerms
def rightRaw : List Term := Proof.Events720.exact184469RawTerms
def group : MergeGroup := .operator 184492 184469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184492) (leftOrdinal := 0)
    (rightResult := 184469) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56025⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56025⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184496

namespace LeftMerge184497
def owner : Owner := ⟨.program ⟨257⟩, ⟨56026⟩⟩
def mergeEvent : Nat := 184497
def frameStart : Nat := 184418
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56025⟩⟩] } }
def leftRaw : List Term := Proof.Events720.exact184492RawTerms
def rightRaw : List Term := Proof.Events720.exact184469RawTerms
def group : MergeGroup := .operator 184492 184469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184492) (leftOrdinal := 1)
    (rightResult := 184469) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56025⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56025⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184497

namespace LeftMerge184499
def owner : Owner := ⟨.program ⟨257⟩, ⟨56026⟩⟩
def mergeEvent : Nat := 184499
def frameStart : Nat := 184418
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55168⟩⟩] } }
def rhsRaw : List Term := Proof.Events720.exact184466RawTerms
def group : MergeGroup := .relation 184498
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 184498) (rhsResult := 184466)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨53892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56025⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨56025⟩⟩) ⟨55168⟩ 184466) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55168⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨53892⟩⟩], [⟨.program ⟨257⟩, ⟨55168⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184499

namespace LeftMerge184507
def owner : Owner := ⟨.program ⟨257⟩, ⟨54200⟩⟩
def mergeEvent : Nat := 184507
def frameStart : Nat := 184418
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54198⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events720.exact184480RawTerms
def rightRaw : List Term := Proof.Events720.exact184503RawTerms
def group : MergeGroup := .operator 184480 184503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184480) (leftOrdinal := 0)
    (rightResult := 184503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54198⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184507

namespace LeftMerge184524
def owner : Owner := ⟨.program ⟨257⟩, ⟨54799⟩⟩
def mergeEvent : Nat := 184524
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }
def rhsRaw : List Term := Proof.Events720.exact184521RawTerms
def group : MergeGroup := .relation 184523
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 184523) (rhsResult := 184521)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54796⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 184522 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54796⟩⟩]⟩) (none) 184521) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184524

namespace LeftMerge184525
def owner : Owner := ⟨.program ⟨257⟩, ⟨54799⟩⟩
def mergeEvent : Nat := 184525
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56025⟩⟩] } }
def rhsRaw : List Term := Proof.Events720.exact184521RawTerms
def group : MergeGroup := .relation 184523
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 184523) (rhsResult := 184521)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54796⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 184522 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54796⟩⟩]⟩) (none) 184521) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56025⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56025⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184525

namespace LeftMerge184526
def owner : Owner := ⟨.program ⟨257⟩, ⟨54799⟩⟩
def mergeEvent : Nat := 184526
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55168⟩⟩] } }
def rhsRaw : List Term := Proof.Events720.exact184521RawTerms
def group : MergeGroup := .relation 184523
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 184523) (rhsResult := 184521)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54796⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 184522 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54796⟩⟩]⟩) (none) 184521) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55168⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53892⟩⟩], [⟨.program ⟨257⟩, ⟨55168⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184526

namespace LeftMerge184527
def owner : Owner := ⟨.program ⟨257⟩, ⟨54799⟩⟩
def mergeEvent : Nat := 184527
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events720.exact184521RawTerms
def group : MergeGroup := .relation 184523
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 184523) (rhsResult := 184521)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54796⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 184522 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54796⟩⟩]⟩) (none) 184521) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54198⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184527

namespace LeftMerge184532
def owner : Owner := ⟨.program ⟨257⟩, ⟨56028⟩⟩
def mergeEvent : Nat := 184532
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56025⟩⟩] } }
def leftRaw : List Term := Proof.Events720.exact184528RawTerms
def rightRaw : List Term := Proof.Events720.exact184350RawTerms
def group : MergeGroup := .operator 184528 184350
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184528) (leftOrdinal := 0)
    (rightResult := 184350) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56025⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56025⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56025⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184532

namespace LeftMerge184533
def owner : Owner := ⟨.program ⟨257⟩, ⟨56028⟩⟩
def mergeEvent : Nat := 184533
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55168⟩⟩] } }
def leftRaw : List Term := Proof.Events720.exact184528RawTerms
def rightRaw : List Term := Proof.Events720.exact184350RawTerms
def group : MergeGroup := .operator 184528 184350
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184528) (leftOrdinal := 2)
    (rightResult := 184350) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55168⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55168⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53892⟩⟩], [⟨.program ⟨257⟩, ⟨55168⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184533

namespace LeftMerge184559
def owner : Owner := ⟨.program ⟨257⟩, ⟨24567⟩⟩
def mergeEvent : Nat := 184559
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events033.exact8621RawTerms
def rightRaw : List Term := Proof.Events696.exact178278RawTerms
def group : MergeGroup := .operator 8621 178278
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8621) (leftOrdinal := 0)
    (rightResult := 178278) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24566⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24566⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184559

namespace LeftMerge184564
def owner : Owner := ⟨.program ⟨257⟩, ⟨8956⟩⟩
def mergeEvent : Nat := 184564
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } }
def leftRaw : List Term := Proof.Events695.exact178148RawTerms
def rightRaw : List Term := Proof.Events092.exact23593RawTerms
def group : MergeGroup := .operator 178148 23593
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178148) (leftOrdinal := 0)
    (rightResult := 23593) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184564

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
