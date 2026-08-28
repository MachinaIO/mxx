import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge107368
def owner : Owner := ⟨.program ⟨257⟩, ⟨39336⟩⟩
def mergeEvent : Nat := 107368
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38590⟩⟩] } }
def rhsRaw : List Term := Proof.Events418.exact107079RawTerms
def group : MergeGroup := .relation 107367
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 107367) (rhsResult := 107079)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39334⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39334⟩⟩) ⟨38590⟩ 107079) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38590⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37436⟩⟩], [⟨.program ⟨257⟩, ⟨38590⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge107368

namespace LeftMerge107382
def owner : Owner := ⟨.program ⟨257⟩, ⟨38199⟩⟩
def mergeEvent : Nat := 107382
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38196⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105245RawTerms
def rightRaw : List Term := Proof.Events419.exact107376RawTerms
def group : MergeGroup := .operator 105245 107376
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105245) (leftOrdinal := 0)
    (rightResult := 107376) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38196⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge107382

namespace LeftMerge107503
def owner : Owner := ⟨.program ⟨257⟩, ⟨38792⟩⟩
def mergeEvent : Nat := 107503
def frameStart : Nat := 107437
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events419.exact107499RawTerms
def rightRaw : List Term := Proof.Events419.exact107497RawTerms
def group : MergeGroup := .operator 107499 107497
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 107499) (leftOrdinal := 0)
    (rightResult := 107497) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37436⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge107503

namespace LeftMerge107515
def owner : Owner := ⟨.program ⟨257⟩, ⟨39335⟩⟩
def mergeEvent : Nat := 107515
def frameStart : Nat := 107437
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39334⟩⟩] } }
def leftRaw : List Term := Proof.Events419.exact107511RawTerms
def rightRaw : List Term := Proof.Events419.exact107488RawTerms
def group : MergeGroup := .operator 107511 107488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 107511) (leftOrdinal := 0)
    (rightResult := 107488) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39334⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39334⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge107515

namespace LeftMerge107516
def owner : Owner := ⟨.program ⟨257⟩, ⟨39335⟩⟩
def mergeEvent : Nat := 107516
def frameStart : Nat := 107437
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39334⟩⟩] } }
def leftRaw : List Term := Proof.Events419.exact107511RawTerms
def rightRaw : List Term := Proof.Events419.exact107488RawTerms
def group : MergeGroup := .operator 107511 107488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 107511) (leftOrdinal := 1)
    (rightResult := 107488) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39334⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39334⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge107516

namespace LeftMerge107518
def owner : Owner := ⟨.program ⟨257⟩, ⟨39335⟩⟩
def mergeEvent : Nat := 107518
def frameStart : Nat := 107437
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38590⟩⟩] } }
def rhsRaw : List Term := Proof.Events419.exact107485RawTerms
def group : MergeGroup := .relation 107517
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 107517) (rhsResult := 107485)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39334⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39334⟩⟩) ⟨38590⟩ 107485) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38590⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37436⟩⟩], [⟨.program ⟨257⟩, ⟨38590⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge107518

namespace LeftMerge107526
def owner : Owner := ⟨.program ⟨257⟩, ⟨37657⟩⟩
def mergeEvent : Nat := 107526
def frameStart : Nat := 107437
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37656⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events419.exact107499RawTerms
def rightRaw : List Term := Proof.Events420.exact107522RawTerms
def group : MergeGroup := .operator 107499 107522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 107499) (leftOrdinal := 0)
    (rightResult := 107522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37656⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37656⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge107526

namespace LeftMerge107543
def owner : Owner := ⟨.program ⟨257⟩, ⟨38199⟩⟩
def mergeEvent : Nat := 107543
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }
def rhsRaw : List Term := Proof.Events420.exact107540RawTerms
def group : MergeGroup := .relation 107542
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 107542) (rhsResult := 107540)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38196⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 107541 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38196⟩⟩]⟩) (none) 107540) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge107543

namespace LeftMerge107544
def owner : Owner := ⟨.program ⟨257⟩, ⟨38199⟩⟩
def mergeEvent : Nat := 107544
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39334⟩⟩] } }
def rhsRaw : List Term := Proof.Events420.exact107540RawTerms
def group : MergeGroup := .relation 107542
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 107542) (rhsResult := 107540)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38196⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 107541 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38196⟩⟩]⟩) (none) 107540) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39334⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39334⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge107544

namespace LeftMerge107545
def owner : Owner := ⟨.program ⟨257⟩, ⟨38199⟩⟩
def mergeEvent : Nat := 107545
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38590⟩⟩] } }
def rhsRaw : List Term := Proof.Events420.exact107540RawTerms
def group : MergeGroup := .relation 107542
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 107542) (rhsResult := 107540)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38196⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 107541 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38196⟩⟩]⟩) (none) 107540) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38590⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37436⟩⟩], [⟨.program ⟨257⟩, ⟨38590⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge107545

namespace LeftMerge107546
def owner : Owner := ⟨.program ⟨257⟩, ⟨38199⟩⟩
def mergeEvent : Nat := 107546
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37656⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events420.exact107540RawTerms
def group : MergeGroup := .relation 107542
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 107542) (rhsResult := 107540)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38196⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 107541 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38196⟩⟩]⟩) (none) 107540) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37656⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37656⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge107546

namespace LeftMerge107551
def owner : Owner := ⟨.program ⟨257⟩, ⟨39337⟩⟩
def mergeEvent : Nat := 107551
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39334⟩⟩] } }
def leftRaw : List Term := Proof.Events420.exact107547RawTerms
def rightRaw : List Term := Proof.Events419.exact107369RawTerms
def group : MergeGroup := .operator 107547 107369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 107547) (leftOrdinal := 0)
    (rightResult := 107369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39334⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39334⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39334⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge107551

namespace LeftMerge107552
def owner : Owner := ⟨.program ⟨257⟩, ⟨39337⟩⟩
def mergeEvent : Nat := 107552
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38590⟩⟩] } }
def leftRaw : List Term := Proof.Events420.exact107547RawTerms
def rightRaw : List Term := Proof.Events419.exact107369RawTerms
def group : MergeGroup := .operator 107547 107369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 107547) (leftOrdinal := 2)
    (rightResult := 107369) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38590⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38590⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37436⟩⟩], [⟨.program ⟨257⟩, ⟨38590⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge107552

namespace LeftMerge107578
def owner : Owner := ⟨.program ⟨257⟩, ⟨34461⟩⟩
def mergeEvent : Nat := 107578
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨34458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events018.exact4697RawTerms
def rightRaw : List Term := Proof.Events410.exact105153RawTerms
def group : MergeGroup := .operator 4697 105153
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4697) (leftOrdinal := 0)
    (rightResult := 105153) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34458⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨34458⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge107578

namespace LeftMerge107583
def owner : Owner := ⟨.program ⟨257⟩, ⟨8700⟩⟩
def mergeEvent : Nat := 107583
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }
def leftRaw : List Term := Proof.Events410.exact105023RawTerms
def rightRaw : List Term := Proof.Events076.exact19585RawTerms
def group : MergeGroup := .operator 105023 19585
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105023) (leftOrdinal := 0)
    (rightResult := 19585) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge107583

namespace LeftMerge107600
def owner : Owner := ⟨.program ⟨257⟩, ⟨34464⟩⟩
def mergeEvent : Nat := 107600
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13596⟩⟩, ⟨.program ⟨257⟩, ⟨34458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events420.exact107594RawTerms
def rightRaw : List Term := Proof.Events018.exact4700RawTerms
def group : MergeGroup := .operator 107594 4700
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 107594) (leftOrdinal := 1)
    (rightResult := 4700) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨34458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13596⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13596⟩⟩, ⟨.program ⟨257⟩, ⟨34458⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge107600

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
