import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge13581
def owner : Owner := ⟨.program ⟨214⟩, ⟨19259⟩⟩
def mergeEvent : Nat := 13581
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19256⟩⟩] } }
def leftRaw : List Term := Proof.Events025.exact6561RawTerms
def rightRaw : List Term := Proof.Events053.exact13575RawTerms
def group : MergeGroup := .operator 6561 13575
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6561) (leftOrdinal := 0)
    (rightResult := 13575) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19256⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19256⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13581

namespace LeftMerge13660
def owner : Owner := ⟨.program ⟨214⟩, ⟨12200⟩⟩
def mergeEvent : Nat := 13660
def frameStart : Nat := 13630
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11149⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events053.exact13656RawTerms
def rightRaw : List Term := Proof.Events053.exact13653RawTerms
def group : MergeGroup := .operator 13656 13653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13656) (leftOrdinal := 0)
    (rightResult := 13653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12199⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11149⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11149⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13660

namespace LeftMerge13690
def owner : Owner := ⟨.program ⟨214⟩, ⟨12288⟩⟩
def mergeEvent : Nat := 13690
def frameStart : Nat := 13630
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11149⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events053.exact13686RawTerms
def rightRaw : List Term := Proof.Events053.exact13684RawTerms
def group : MergeGroup := .operator 13686 13684
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13686) (leftOrdinal := 0)
    (rightResult := 13684) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11149⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11149⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13690

namespace LeftMerge13713
def owner : Owner := ⟨.program ⟨214⟩, ⟨7842⟩⟩
def mergeEvent : Nat := 13713
def frameStart : Nat := 13630
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }
def leftRaw : List Term := Proof.Events053.exact13709RawTerms
def rightRaw : List Term := Proof.Events053.exact13706RawTerms
def group : MergeGroup := .operator 13709 13706
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13709) (leftOrdinal := 0)
    (rightResult := 13706) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7840⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13713

namespace LeftMerge13722
def owner : Owner := ⟨.program ⟨214⟩, ⟨25319⟩⟩
def mergeEvent : Nat := 13722
def frameStart : Nat := 13630
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11149⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25316⟩⟩] } }
def leftRaw : List Term := Proof.Events053.exact13718RawTerms
def rightRaw : List Term := Proof.Events053.exact13675RawTerms
def group : MergeGroup := .operator 13718 13675
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13718) (leftOrdinal := 1)
    (rightResult := 13675) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11149⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25316⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11149⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25316⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge13722

namespace LeftMerge13724
def owner : Owner := ⟨.program ⟨214⟩, ⟨25319⟩⟩
def mergeEvent : Nat := 13724
def frameStart : Nat := 13630
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11149⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23172⟩⟩] } }
def rhsRaw : List Term := Proof.Events053.exact13672RawTerms
def group : MergeGroup := .relation 13723
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 13723) (rhsResult := 13672)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11149⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25316⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25316⟩⟩) ⟨23172⟩ 13672) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23172⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11149⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], [⟨.program ⟨214⟩, ⟨23172⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge13724

namespace LeftMerge13725
def owner : Owner := ⟨.program ⟨214⟩, ⟨25319⟩⟩
def mergeEvent : Nat := 13725
def frameStart : Nat := 13630
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25316⟩⟩] } }
def leftRaw : List Term := Proof.Events053.exact13718RawTerms
def rightRaw : List Term := Proof.Events053.exact13675RawTerms
def group : MergeGroup := .operator 13718 13675
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13718) (leftOrdinal := 0)
    (rightResult := 13675) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25316⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25316⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13725

namespace LeftMerge13733
def owner : Owner := ⟨.program ⟨214⟩, ⟨15440⟩⟩
def mergeEvent : Nat := 13733
def frameStart : Nat := 13630
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15438⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events053.exact13686RawTerms
def rightRaw : List Term := Proof.Events053.exact13729RawTerms
def group : MergeGroup := .operator 13686 13729
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13686) (leftOrdinal := 0)
    (rightResult := 13729) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15438⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15438⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13733

namespace LeftMerge13750
def owner : Owner := ⟨.program ⟨214⟩, ⟨19259⟩⟩
def mergeEvent : Nat := 13750
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11149⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23172⟩⟩] } }
def rhsRaw : List Term := Proof.Events053.exact13747RawTerms
def group : MergeGroup := .relation 13749
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 13749) (rhsResult := 13747)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19256⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 13748 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19256⟩⟩]⟩) (none) 13747) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11149⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23172⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11149⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], [⟨.program ⟨214⟩, ⟨23172⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13750

namespace LeftMerge13751
def owner : Owner := ⟨.program ⟨214⟩, ⟨19259⟩⟩
def mergeEvent : Nat := 13751
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25316⟩⟩] } }
def rhsRaw : List Term := Proof.Events053.exact13747RawTerms
def group : MergeGroup := .relation 13749
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 13749) (rhsResult := 13747)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19256⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 13748 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19256⟩⟩]⟩) (none) 13747) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25316⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25316⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge13751

namespace LeftMerge13752
def owner : Owner := ⟨.program ⟨214⟩, ⟨19259⟩⟩
def mergeEvent : Nat := 13752
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15438⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events053.exact13747RawTerms
def group : MergeGroup := .relation 13749
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 13749) (rhsResult := 13747)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19256⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 13748 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19256⟩⟩]⟩) (none) 13747) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15438⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15438⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge13752

namespace LeftMerge13753
def owner : Owner := ⟨.program ⟨214⟩, ⟨19259⟩⟩
def mergeEvent : Nat := 13753
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩] } }
def rhsRaw : List Term := Proof.Events053.exact13747RawTerms
def group : MergeGroup := .relation 13749
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 13749) (rhsResult := 13747)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19256⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 13748 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19256⟩⟩]⟩) (none) 13747) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13753

namespace LeftMerge13758
def owner : Owner := ⟨.program ⟨214⟩, ⟨25318⟩⟩
def mergeEvent : Nat := 13758
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11149⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23172⟩⟩] } }
def leftRaw : List Term := Proof.Events053.exact13754RawTerms
def rightRaw : List Term := Proof.Events053.exact13568RawTerms
def group : MergeGroup := .operator 13754 13568
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13754) (leftOrdinal := 2)
    (rightResult := 13568) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11149⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23172⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11149⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23172⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11149⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], [⟨.program ⟨214⟩, ⟨23172⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge13758

namespace LeftMerge13759
def owner : Owner := ⟨.program ⟨214⟩, ⟨25318⟩⟩
def mergeEvent : Nat := 13759
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25316⟩⟩] } }
def leftRaw : List Term := Proof.Events053.exact13754RawTerms
def rightRaw : List Term := Proof.Events053.exact13568RawTerms
def group : MergeGroup := .operator 13754 13568
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13754) (leftOrdinal := 1)
    (rightResult := 13568) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25316⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25316⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25316⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13759

namespace LeftMerge13767
def owner : Owner := ⟨.program ⟨214⟩, ⟨27052⟩⟩
def mergeEvent : Nat := 13767
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15438⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27050⟩⟩] } }
def leftRaw : List Term := Proof.Events053.exact13761RawTerms
def rightRaw : List Term := Proof.Events052.exact13465RawTerms
def group : MergeGroup := .operator 13761 13465
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13761) (leftOrdinal := 1)
    (rightResult := 13465) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15438⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27050⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15438⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27050⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge13767

namespace LeftMerge13769
def owner : Owner := ⟨.program ⟨214⟩, ⟨27052⟩⟩
def mergeEvent : Nat := 13769
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15438⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23922⟩⟩] } }
def rhsRaw : List Term := Proof.Events052.exact13462RawTerms
def group : MergeGroup := .relation 13768
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 13768) (rhsResult := 13462)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15438⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27050⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27050⟩⟩) ⟨23922⟩ 13462) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23922⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15438⟩⟩], [⟨.program ⟨214⟩, ⟨23922⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge13769

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
