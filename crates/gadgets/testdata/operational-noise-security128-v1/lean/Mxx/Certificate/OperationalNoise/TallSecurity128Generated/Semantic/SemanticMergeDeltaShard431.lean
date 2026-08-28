import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge73674
def owner : Owner := ⟨.program ⟨257⟩, ⟨63815⟩⟩
def mergeEvent : Nat := 73674
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7213⟩⟩] } }
def rhsRaw : List Term := Proof.Events287.exact73671RawTerms
def group : MergeGroup := .relation 73673
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73673) (rhsResult := 73671)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63812⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 73672 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63812⟩⟩]⟩) (none) 73671) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7213⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73674

namespace LeftMerge73675
def owner : Owner := ⟨.program ⟨257⟩, ⟨63815⟩⟩
def mergeEvent : Nat := 73675
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65082⟩⟩] } }
def rhsRaw : List Term := Proof.Events287.exact73671RawTerms
def group : MergeGroup := .relation 73673
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73673) (rhsResult := 73671)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63812⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 73672 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63812⟩⟩]⟩) (none) 73671) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65082⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65082⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73675

namespace LeftMerge73676
def owner : Owner := ⟨.program ⟨257⟩, ⟨63815⟩⟩
def mergeEvent : Nat := 73676
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨62864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64143⟩⟩] } }
def rhsRaw : List Term := Proof.Events287.exact73671RawTerms
def group : MergeGroup := .relation 73673
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73673) (rhsResult := 73671)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63812⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 73672 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63812⟩⟩]⟩) (none) 73671) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64143⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨62864⟩⟩], [⟨.program ⟨257⟩, ⟨64143⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73676

namespace LeftMerge73677
def owner : Owner := ⟨.program ⟨257⟩, ⟨63815⟩⟩
def mergeEvent : Nat := 73677
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events287.exact73671RawTerms
def group : MergeGroup := .relation 73673
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73673) (rhsResult := 73671)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63812⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 73672 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63812⟩⟩]⟩) (none) 73671) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63218⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73677

namespace LeftMerge73682
def owner : Owner := ⟨.program ⟨257⟩, ⟨65085⟩⟩
def mergeEvent : Nat := 73682
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65082⟩⟩] } }
def leftRaw : List Term := Proof.Events287.exact73678RawTerms
def rightRaw : List Term := Proof.Events287.exact73500RawTerms
def group : MergeGroup := .operator 73678 73500
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73678) (leftOrdinal := 0)
    (rightResult := 73500) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65082⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73682

namespace LeftMerge73683
def owner : Owner := ⟨.program ⟨257⟩, ⟨65085⟩⟩
def mergeEvent : Nat := 73683
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨62864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64143⟩⟩] } }
def leftRaw : List Term := Proof.Events287.exact73678RawTerms
def rightRaw : List Term := Proof.Events287.exact73500RawTerms
def group : MergeGroup := .operator 73678 73500
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73678) (leftOrdinal := 2)
    (rightResult := 73500) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨62864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64143⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨62864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64143⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨62864⟩⟩], [⟨.program ⟨257⟩, ⟨64143⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73683

namespace LeftMerge73691
def owner : Owner := ⟨.program ⟨257⟩, ⟨65086⟩⟩
def mergeEvent : Nat := 73691
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩] } }
def leftRaw : List Term := Proof.Events287.exact73685RawTerms
def rightRaw : List Term := Proof.Events061.exact15722RawTerms
def group : MergeGroup := .operator 73685 15722
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73685) (leftOrdinal := 0)
    (rightResult := 15722) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7213⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7099⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73691

namespace LeftMerge73692
def owner : Owner := ⟨.program ⟨257⟩, ⟨65086⟩⟩
def mergeEvent : Nat := 73692
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩] } }
def leftRaw : List Term := Proof.Events287.exact73685RawTerms
def rightRaw : List Term := Proof.Events061.exact15722RawTerms
def group : MergeGroup := .operator 73685 15722
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73685) (leftOrdinal := 1)
    (rightResult := 15722) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7099⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73692

namespace LeftMerge73694
def owner : Owner := ⟨.program ⟨257⟩, ⟨65086⟩⟩
def mergeEvent : Nat := 73694
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15715RawTerms
def group : MergeGroup := .relation 73693
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73693) (rhsResult := 15715)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7099⟩⟩) ⟨7015⟩ 15715) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73694

namespace LeftMerge73708
def owner : Owner := ⟨.program ⟨257⟩, ⟨62104⟩⟩
def mergeEvent : Nat := 73708
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62102⟩⟩] } }
def leftRaw : List Term := Proof.Events259.exact66376RawTerms
def rightRaw : List Term := Proof.Events287.exact73702RawTerms
def group : MergeGroup := .operator 66376 73702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66376) (leftOrdinal := 0)
    (rightResult := 73702) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨62102⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62102⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73708

namespace LeftMerge73709
def owner : Owner := ⟨.program ⟨257⟩, ⟨62104⟩⟩
def mergeEvent : Nat := 73709
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62102⟩⟩] } }
def leftRaw : List Term := Proof.Events259.exact66376RawTerms
def rightRaw : List Term := Proof.Events287.exact73702RawTerms
def group : MergeGroup := .operator 66376 73702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66376) (leftOrdinal := 1)
    (rightResult := 73702) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨62102⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62102⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73709

namespace LeftMerge73711
def owner : Owner := ⟨.program ⟨257⟩, ⟨62104⟩⟩
def mergeEvent : Nat := 73711
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61163⟩⟩] } }
def rhsRaw : List Term := Proof.Events287.exact73699RawTerms
def group : MergeGroup := .relation 73710
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73710) (rhsResult := 73699)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62102⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨62102⟩⟩) ⟨61163⟩ 73699) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61163⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59884⟩⟩], [⟨.program ⟨257⟩, ⟨61163⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73711

namespace LeftMerge73725
def owner : Owner := ⟨.program ⟨257⟩, ⟨60835⟩⟩
def mergeEvent : Nat := 73725
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60832⟩⟩] } }
def leftRaw : List Term := Proof.Events239.exact61370RawTerms
def rightRaw : List Term := Proof.Events287.exact73719RawTerms
def group : MergeGroup := .operator 61370 73719
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61370) (leftOrdinal := 0)
    (rightResult := 73719) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60832⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60832⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73725

namespace LeftMerge73846
def owner : Owner := ⟨.program ⟨257⟩, ⟨61336⟩⟩
def mergeEvent : Nat := 73846
def frameStart : Nat := 73780
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events288.exact73842RawTerms
def rightRaw : List Term := Proof.Events288.exact73840RawTerms
def group : MergeGroup := .operator 73842 73840
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73842) (leftOrdinal := 0)
    (rightResult := 73840) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73846

namespace LeftMerge73858
def owner : Owner := ⟨.program ⟨257⟩, ⟨62103⟩⟩
def mergeEvent : Nat := 73858
def frameStart : Nat := 73780
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62102⟩⟩] } }
def leftRaw : List Term := Proof.Events288.exact73854RawTerms
def rightRaw : List Term := Proof.Events288.exact73831RawTerms
def group : MergeGroup := .operator 73854 73831
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73854) (leftOrdinal := 0)
    (rightResult := 73831) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨62102⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62102⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73858

namespace LeftMerge73859
def owner : Owner := ⟨.program ⟨257⟩, ⟨62103⟩⟩
def mergeEvent : Nat := 73859
def frameStart : Nat := 73780
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62102⟩⟩] } }
def leftRaw : List Term := Proof.Events288.exact73854RawTerms
def rightRaw : List Term := Proof.Events288.exact73831RawTerms
def group : MergeGroup := .operator 73854 73831
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73854) (leftOrdinal := 1)
    (rightResult := 73831) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨62102⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62102⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73859

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
