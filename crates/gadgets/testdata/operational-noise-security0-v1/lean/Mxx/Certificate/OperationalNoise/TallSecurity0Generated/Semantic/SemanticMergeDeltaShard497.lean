import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge80577
def owner : Owner := ⟨.program ⟨214⟩, ⟨13155⟩⟩
def mergeEvent : Nat := 80577
def frameStart : Nat := 80547
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events314.exact80573RawTerms
def rightRaw : List Term := Proof.Events314.exact80570RawTerms
def group : MergeGroup := .operator 80573 80570
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80573) (leftOrdinal := 0)
    (rightResult := 80570) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10240⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13154⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80577

namespace LeftMerge80607
def owner : Owner := ⟨.program ⟨214⟩, ⟨13252⟩⟩
def mergeEvent : Nat := 80607
def frameStart : Nat := 80547
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events314.exact80603RawTerms
def rightRaw : List Term := Proof.Events314.exact80601RawTerms
def group : MergeGroup := .operator 80603 80601
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80603) (leftOrdinal := 0)
    (rightResult := 80601) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80607

namespace LeftMerge80628
def owner : Owner := ⟨.program ⟨214⟩, ⟨7881⟩⟩
def mergeEvent : Nat := 80628
def frameStart : Nat := 80547
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }
def leftRaw : List Term := Proof.Events314.exact80624RawTerms
def rightRaw : List Term := Proof.Events314.exact80621RawTerms
def group : MergeGroup := .operator 80624 80621
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80624) (leftOrdinal := 0)
    (rightResult := 80621) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7879⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80628

namespace LeftMerge80637
def owner : Owner := ⟨.program ⟨214⟩, ⟨25684⟩⟩
def mergeEvent : Nat := 80637
def frameStart : Nat := 80547
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩] } }
def leftRaw : List Term := Proof.Events314.exact80633RawTerms
def rightRaw : List Term := Proof.Events314.exact80592RawTerms
def group : MergeGroup := .operator 80633 80592
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80633) (leftOrdinal := 0)
    (rightResult := 80592) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25681⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80637

namespace LeftMerge80638
def owner : Owner := ⟨.program ⟨214⟩, ⟨25684⟩⟩
def mergeEvent : Nat := 80638
def frameStart : Nat := 80547
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩] } }
def leftRaw : List Term := Proof.Events314.exact80633RawTerms
def rightRaw : List Term := Proof.Events314.exact80592RawTerms
def group : MergeGroup := .operator 80633 80592
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80633) (leftOrdinal := 1)
    (rightResult := 80592) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80638

namespace LeftMerge80640
def owner : Owner := ⟨.program ⟨214⟩, ⟨25684⟩⟩
def mergeEvent : Nat := 80640
def frameStart : Nat := 80547
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23374⟩⟩] } }
def rhsRaw : List Term := Proof.Events314.exact80589RawTerms
def group : MergeGroup := .relation 80639
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80639) (rhsResult := 80589)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25681⟩⟩) ⟨23374⟩ 80589) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23374⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨23374⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80640

namespace LeftMerge80648
def owner : Owner := ⟨.program ⟨214⟩, ⟨16873⟩⟩
def mergeEvent : Nat := 80648
def frameStart : Nat := 80547
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16871⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events314.exact80603RawTerms
def rightRaw : List Term := Proof.Events315.exact80644RawTerms
def group : MergeGroup := .operator 80603 80644
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80603) (leftOrdinal := 0)
    (rightResult := 80644) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16871⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80648

namespace LeftMerge80665
def owner : Owner := ⟨.program ⟨214⟩, ⟨20179⟩⟩
def mergeEvent : Nat := 80665
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }
def rhsRaw : List Term := Proof.Events315.exact80662RawTerms
def group : MergeGroup := .relation 80664
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80664) (rhsResult := 80662)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20176⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 80663 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20176⟩⟩]⟩) (none) 80662) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80665

namespace LeftMerge80666
def owner : Owner := ⟨.program ⟨214⟩, ⟨20179⟩⟩
def mergeEvent : Nat := 80666
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩] } }
def rhsRaw : List Term := Proof.Events315.exact80662RawTerms
def group : MergeGroup := .relation 80664
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80664) (rhsResult := 80662)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20176⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 80663 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20176⟩⟩]⟩) (none) 80662) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80666

namespace LeftMerge80667
def owner : Owner := ⟨.program ⟨214⟩, ⟨20179⟩⟩
def mergeEvent : Nat := 80667
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23374⟩⟩] } }
def rhsRaw : List Term := Proof.Events315.exact80662RawTerms
def group : MergeGroup := .relation 80664
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80664) (rhsResult := 80662)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20176⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 80663 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20176⟩⟩]⟩) (none) 80662) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23374⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨23374⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80667

namespace LeftMerge80668
def owner : Owner := ⟨.program ⟨214⟩, ⟨20179⟩⟩
def mergeEvent : Nat := 80668
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events315.exact80662RawTerms
def group : MergeGroup := .relation 80664
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80664) (rhsResult := 80662)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20176⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 80663 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20176⟩⟩]⟩) (none) 80662) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16871⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80668

namespace LeftMerge80673
def owner : Owner := ⟨.program ⟨214⟩, ⟨25683⟩⟩
def mergeEvent : Nat := 80673
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23374⟩⟩] } }
def leftRaw : List Term := Proof.Events315.exact80669RawTerms
def rightRaw : List Term := Proof.Events314.exact80485RawTerms
def group : MergeGroup := .operator 80669 80485
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80669) (leftOrdinal := 2)
    (rightResult := 80485) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23374⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23374⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨23374⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80673

namespace LeftMerge80674
def owner : Owner := ⟨.program ⟨214⟩, ⟨25683⟩⟩
def mergeEvent : Nat := 80674
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩] } }
def leftRaw : List Term := Proof.Events315.exact80669RawTerms
def rightRaw : List Term := Proof.Events314.exact80485RawTerms
def group : MergeGroup := .operator 80669 80485
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80669) (leftOrdinal := 1)
    (rightResult := 80485) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80674

namespace LeftMerge80682
def owner : Owner := ⟨.program ⟨214⟩, ⟨29821⟩⟩
def mergeEvent : Nat := 80682
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩] } }
def leftRaw : List Term := Proof.Events315.exact80676RawTerms
def rightRaw : List Term := Proof.Events314.exact80401RawTerms
def group : MergeGroup := .operator 80676 80401
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80676) (leftOrdinal := 0)
    (rightResult := 80401) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29819⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80682

namespace LeftMerge80683
def owner : Owner := ⟨.program ⟨214⟩, ⟨29821⟩⟩
def mergeEvent : Nat := 80683
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩] } }
def leftRaw : List Term := Proof.Events315.exact80676RawTerms
def rightRaw : List Term := Proof.Events314.exact80401RawTerms
def group : MergeGroup := .operator 80676 80401
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80676) (leftOrdinal := 1)
    (rightResult := 80401) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29819⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80683

namespace LeftMerge80685
def owner : Owner := ⟨.program ⟨214⟩, ⟨29821⟩⟩
def mergeEvent : Nat := 80685
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24729⟩⟩] } }
def rhsRaw : List Term := Proof.Events314.exact80398RawTerms
def group : MergeGroup := .relation 80684
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80684) (rhsResult := 80398)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29819⟩⟩) ⟨24729⟩ 80398) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24729⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨24729⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80685

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
