import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge220383
def owner : Owner := ⟨.program ⟨257⟩, ⟨55927⟩⟩
def mergeEvent : Nat := 220383
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55925⟩⟩] } }
def leftRaw : List Term := Proof.Events834.exact213590RawTerms
def rightRaw : List Term := Proof.Events860.exact220376RawTerms
def group : MergeGroup := .operator 213590 220376
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213590) (leftOrdinal := 1)
    (rightResult := 220376) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55925⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55925⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge220383

namespace LeftMerge220385
def owner : Owner := ⟨.program ⟨257⟩, ⟨55927⟩⟩
def mergeEvent : Nat := 220385
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55140⟩⟩] } }
def rhsRaw : List Term := Proof.Events860.exact220373RawTerms
def group : MergeGroup := .relation 220384
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 220384) (rhsResult := 220373)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55925⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55925⟩⟩) ⟨55140⟩ 220373) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55140⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53868⟩⟩], [⟨.program ⟨257⟩, ⟨55140⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge220385

namespace LeftMerge220399
def owner : Owner := ⟨.program ⟨257⟩, ⟨54735⟩⟩
def mergeEvent : Nat := 220399
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54732⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207620RawTerms
def rightRaw : List Term := Proof.Events860.exact220393RawTerms
def group : MergeGroup := .operator 207620 220393
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207620) (leftOrdinal := 0)
    (rightResult := 220393) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54732⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54732⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge220399

namespace LeftMerge220520
def owner : Owner := ⟨.program ⟨257⟩, ⟨55348⟩⟩
def mergeEvent : Nat := 220520
def frameStart : Nat := 220454
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events861.exact220516RawTerms
def rightRaw : List Term := Proof.Events861.exact220514RawTerms
def group : MergeGroup := .operator 220516 220514
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 220516) (leftOrdinal := 0)
    (rightResult := 220514) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge220520

namespace LeftMerge220532
def owner : Owner := ⟨.program ⟨257⟩, ⟨55926⟩⟩
def mergeEvent : Nat := 220532
def frameStart : Nat := 220454
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55925⟩⟩] } }
def leftRaw : List Term := Proof.Events861.exact220528RawTerms
def rightRaw : List Term := Proof.Events861.exact220505RawTerms
def group : MergeGroup := .operator 220528 220505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 220528) (leftOrdinal := 0)
    (rightResult := 220505) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55925⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55925⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge220532

namespace LeftMerge220533
def owner : Owner := ⟨.program ⟨257⟩, ⟨55926⟩⟩
def mergeEvent : Nat := 220533
def frameStart : Nat := 220454
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55925⟩⟩] } }
def leftRaw : List Term := Proof.Events861.exact220528RawTerms
def rightRaw : List Term := Proof.Events861.exact220505RawTerms
def group : MergeGroup := .operator 220528 220505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 220528) (leftOrdinal := 1)
    (rightResult := 220505) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55925⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55925⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge220533

namespace LeftMerge220535
def owner : Owner := ⟨.program ⟨257⟩, ⟨55926⟩⟩
def mergeEvent : Nat := 220535
def frameStart : Nat := 220454
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55140⟩⟩] } }
def rhsRaw : List Term := Proof.Events861.exact220502RawTerms
def group : MergeGroup := .relation 220534
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 220534) (rhsResult := 220502)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨53868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55925⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55925⟩⟩) ⟨55140⟩ 220502) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55140⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨53868⟩⟩], [⟨.program ⟨257⟩, ⟨55140⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge220535

namespace LeftMerge220543
def owner : Owner := ⟨.program ⟨257⟩, ⟨54148⟩⟩
def mergeEvent : Nat := 220543
def frameStart : Nat := 220454
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54145⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events861.exact220516RawTerms
def rightRaw : List Term := Proof.Events861.exact220539RawTerms
def group : MergeGroup := .operator 220516 220539
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 220516) (leftOrdinal := 0)
    (rightResult := 220539) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54145⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨54145⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge220543

namespace LeftMerge220560
def owner : Owner := ⟨.program ⟨257⟩, ⟨54735⟩⟩
def mergeEvent : Nat := 220560
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7207⟩⟩] } }
def rhsRaw : List Term := Proof.Events861.exact220557RawTerms
def group : MergeGroup := .relation 220559
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 220559) (rhsResult := 220557)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54732⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 220558 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54732⟩⟩]⟩) (none) 220557) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7207⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge220560

namespace LeftMerge220561
def owner : Owner := ⟨.program ⟨257⟩, ⟨54735⟩⟩
def mergeEvent : Nat := 220561
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55925⟩⟩] } }
def rhsRaw : List Term := Proof.Events861.exact220557RawTerms
def group : MergeGroup := .relation 220559
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 220559) (rhsResult := 220557)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54732⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 220558 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54732⟩⟩]⟩) (none) 220557) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55925⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55925⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge220561

namespace LeftMerge220562
def owner : Owner := ⟨.program ⟨257⟩, ⟨54735⟩⟩
def mergeEvent : Nat := 220562
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55140⟩⟩] } }
def rhsRaw : List Term := Proof.Events861.exact220557RawTerms
def group : MergeGroup := .relation 220559
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 220559) (rhsResult := 220557)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54732⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 220558 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54732⟩⟩]⟩) (none) 220557) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55140⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53868⟩⟩], [⟨.program ⟨257⟩, ⟨55140⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge220562

namespace LeftMerge220563
def owner : Owner := ⟨.program ⟨257⟩, ⟨54735⟩⟩
def mergeEvent : Nat := 220563
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54145⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events861.exact220557RawTerms
def group : MergeGroup := .relation 220559
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 220559) (rhsResult := 220557)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54732⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 220558 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54732⟩⟩]⟩) (none) 220557) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54145⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54145⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge220563

namespace LeftMerge220568
def owner : Owner := ⟨.program ⟨257⟩, ⟨55928⟩⟩
def mergeEvent : Nat := 220568
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55925⟩⟩] } }
def leftRaw : List Term := Proof.Events861.exact220564RawTerms
def rightRaw : List Term := Proof.Events860.exact220386RawTerms
def group : MergeGroup := .operator 220564 220386
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 220564) (leftOrdinal := 0)
    (rightResult := 220386) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55925⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55925⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55925⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge220568

namespace LeftMerge220569
def owner : Owner := ⟨.program ⟨257⟩, ⟨55928⟩⟩
def mergeEvent : Nat := 220569
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55140⟩⟩] } }
def leftRaw : List Term := Proof.Events861.exact220564RawTerms
def rightRaw : List Term := Proof.Events860.exact220386RawTerms
def group : MergeGroup := .operator 220564 220386
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 220564) (leftOrdinal := 2)
    (rightResult := 220386) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55140⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55140⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53868⟩⟩], [⟨.program ⟨257⟩, ⟨55140⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge220569

namespace LeftMerge220577
def owner : Owner := ⟨.program ⟨257⟩, ⟨55929⟩⟩
def mergeEvent : Nat := 220577
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩] } }
def leftRaw : List Term := Proof.Events861.exact220571RawTerms
def rightRaw : List Term := Proof.Events061.exact15782RawTerms
def group : MergeGroup := .operator 220571 15782
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 220571) (leftOrdinal := 0)
    (rightResult := 15782) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7207⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7125⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge220577

namespace LeftMerge220578
def owner : Owner := ⟨.program ⟨257⟩, ⟨55929⟩⟩
def mergeEvent : Nat := 220578
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54145⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩] } }
def leftRaw : List Term := Proof.Events861.exact220571RawTerms
def rightRaw : List Term := Proof.Events061.exact15782RawTerms
def group : MergeGroup := .operator 220571 15782
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 220571) (leftOrdinal := 1)
    (rightResult := 15782) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54145⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7125⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54145⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge220578

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
