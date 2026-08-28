import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge49535
def owner : Owner := ⟨.program ⟨214⟩, ⟨26802⟩⟩
def mergeEvent : Nat := 49535
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26800⟩⟩] } }
def leftRaw : List Term := Proof.Events170.exact43553RawTerms
def rightRaw : List Term := Proof.Events193.exact49529RawTerms
def group : MergeGroup := .operator 43553 49529
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43553) (leftOrdinal := 0)
    (rightResult := 49529) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26800⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26800⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49535

namespace LeftMerge49536
def owner : Owner := ⟨.program ⟨214⟩, ⟨26802⟩⟩
def mergeEvent : Nat := 49536
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15122⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26800⟩⟩] } }
def leftRaw : List Term := Proof.Events170.exact43553RawTerms
def rightRaw : List Term := Proof.Events193.exact49529RawTerms
def group : MergeGroup := .operator 43553 49529
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43553) (leftOrdinal := 1)
    (rightResult := 49529) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15122⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26800⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15122⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26800⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49536

namespace LeftMerge49538
def owner : Owner := ⟨.program ⟨214⟩, ⟨26802⟩⟩
def mergeEvent : Nat := 49538
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15122⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23852⟩⟩] } }
def rhsRaw : List Term := Proof.Events193.exact49526RawTerms
def group : MergeGroup := .relation 49537
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49537) (rhsResult := 49526)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15122⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26800⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26800⟩⟩) ⟨23852⟩ 49526) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23852⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15122⟩⟩], [⟨.program ⟨214⟩, ⟨23852⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49538

namespace LeftMerge49552
def owner : Owner := ⟨.program ⟨214⟩, ⟨20619⟩⟩
def mergeEvent : Nat := 49552
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20616⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events193.exact49546RawTerms
def group : MergeGroup := .operator 36137 49546
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 49546) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20616⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49552

namespace LeftMerge49673
def owner : Owner := ⟨.program ⟨214⟩, ⟨15164⟩⟩
def mergeEvent : Nat := 49673
def frameStart : Nat := 49607
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15122⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events194.exact49669RawTerms
def rightRaw : List Term := Proof.Events194.exact49667RawTerms
def group : MergeGroup := .operator 49669 49667
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49669) (leftOrdinal := 0)
    (rightResult := 49667) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15122⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15122⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49673

namespace LeftMerge49685
def owner : Owner := ⟨.program ⟨214⟩, ⟨26801⟩⟩
def mergeEvent : Nat := 49685
def frameStart : Nat := 49607
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26800⟩⟩] } }
def leftRaw : List Term := Proof.Events194.exact49681RawTerms
def rightRaw : List Term := Proof.Events193.exact49658RawTerms
def group : MergeGroup := .operator 49681 49658
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49681) (leftOrdinal := 0)
    (rightResult := 49658) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26800⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26800⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49685

namespace LeftMerge49686
def owner : Owner := ⟨.program ⟨214⟩, ⟨26801⟩⟩
def mergeEvent : Nat := 49686
def frameStart : Nat := 49607
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15122⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26800⟩⟩] } }
def leftRaw : List Term := Proof.Events194.exact49681RawTerms
def rightRaw : List Term := Proof.Events193.exact49658RawTerms
def group : MergeGroup := .operator 49681 49658
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49681) (leftOrdinal := 1)
    (rightResult := 49658) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15122⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26800⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15122⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26800⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49686

namespace LeftMerge49688
def owner : Owner := ⟨.program ⟨214⟩, ⟨26801⟩⟩
def mergeEvent : Nat := 49688
def frameStart : Nat := 49607
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15122⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23852⟩⟩] } }
def rhsRaw : List Term := Proof.Events193.exact49655RawTerms
def group : MergeGroup := .relation 49687
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49687) (rhsResult := 49655)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15122⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26800⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26800⟩⟩) ⟨23852⟩ 49655) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23852⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15122⟩⟩], [⟨.program ⟨214⟩, ⟨23852⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49688

namespace LeftMerge49696
def owner : Owner := ⟨.program ⟨214⟩, ⟨15221⟩⟩
def mergeEvent : Nat := 49696
def frameStart : Nat := 49607
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15218⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events194.exact49669RawTerms
def rightRaw : List Term := Proof.Events194.exact49692RawTerms
def group : MergeGroup := .operator 49669 49692
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49669) (leftOrdinal := 0)
    (rightResult := 49692) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15218⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15218⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49696

namespace LeftMerge49713
def owner : Owner := ⟨.program ⟨214⟩, ⟨20619⟩⟩
def mergeEvent : Nat := 49713
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩] } }
def rhsRaw : List Term := Proof.Events194.exact49710RawTerms
def group : MergeGroup := .relation 49712
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49712) (rhsResult := 49710)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20616⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 49711 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20616⟩⟩]⟩) (none) 49710) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49713

namespace LeftMerge49714
def owner : Owner := ⟨.program ⟨214⟩, ⟨20619⟩⟩
def mergeEvent : Nat := 49714
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26800⟩⟩] } }
def rhsRaw : List Term := Proof.Events194.exact49710RawTerms
def group : MergeGroup := .relation 49712
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49712) (rhsResult := 49710)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20616⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 49711 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20616⟩⟩]⟩) (none) 49710) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26800⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26800⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49714

namespace LeftMerge49715
def owner : Owner := ⟨.program ⟨214⟩, ⟨20619⟩⟩
def mergeEvent : Nat := 49715
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15122⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23852⟩⟩] } }
def rhsRaw : List Term := Proof.Events194.exact49710RawTerms
def group : MergeGroup := .relation 49712
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49712) (rhsResult := 49710)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20616⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 49711 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20616⟩⟩]⟩) (none) 49710) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15122⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23852⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15122⟩⟩], [⟨.program ⟨214⟩, ⟨23852⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49715

namespace LeftMerge49716
def owner : Owner := ⟨.program ⟨214⟩, ⟨20619⟩⟩
def mergeEvent : Nat := 49716
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events194.exact49710RawTerms
def group : MergeGroup := .relation 49712
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49712) (rhsResult := 49710)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20616⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 49711 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20616⟩⟩]⟩) (none) 49710) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15218⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49716

namespace LeftMerge49721
def owner : Owner := ⟨.program ⟨214⟩, ⟨26803⟩⟩
def mergeEvent : Nat := 49721
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26800⟩⟩] } }
def leftRaw : List Term := Proof.Events194.exact49717RawTerms
def rightRaw : List Term := Proof.Events193.exact49539RawTerms
def group : MergeGroup := .operator 49717 49539
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49717) (leftOrdinal := 0)
    (rightResult := 49539) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26800⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26800⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26800⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49721

namespace LeftMerge49722
def owner : Owner := ⟨.program ⟨214⟩, ⟨26803⟩⟩
def mergeEvent : Nat := 49722
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15122⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23852⟩⟩] } }
def leftRaw : List Term := Proof.Events194.exact49717RawTerms
def rightRaw : List Term := Proof.Events193.exact49539RawTerms
def group : MergeGroup := .operator 49717 49539
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49717) (leftOrdinal := 2)
    (rightResult := 49539) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15122⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23852⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15122⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23852⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15122⟩⟩], [⟨.program ⟨214⟩, ⟨23852⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49722

namespace LeftMerge49730
def owner : Owner := ⟨.program ⟨214⟩, ⟨26804⟩⟩
def mergeEvent : Nat := 49730
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }
def leftRaw : List Term := Proof.Events194.exact49724RawTerms
def rightRaw : List Term := Proof.Events022.exact5819RawTerms
def group : MergeGroup := .operator 49724 5819
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49724) (leftOrdinal := 0)
    (rightResult := 5819) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6663⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49730

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
