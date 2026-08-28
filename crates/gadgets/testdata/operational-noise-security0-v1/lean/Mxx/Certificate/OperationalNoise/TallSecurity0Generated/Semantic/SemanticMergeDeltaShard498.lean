import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge80699
def owner : Owner := ⟨.program ⟨214⟩, ⟨22699⟩⟩
def mergeEvent : Nat := 80699
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22696⟩⟩] } }
def leftRaw : List Term := Proof.Events312.exact80012RawTerms
def rightRaw : List Term := Proof.Events315.exact80693RawTerms
def group : MergeGroup := .operator 80012 80693
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80012) (leftOrdinal := 0)
    (rightResult := 80693) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22696⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80699

namespace LeftMerge80820
def owner : Owner := ⟨.program ⟨214⟩, ⟨16969⟩⟩
def mergeEvent : Nat := 80820
def frameStart : Nat := 80754
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16871⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events315.exact80816RawTerms
def rightRaw : List Term := Proof.Events315.exact80814RawTerms
def group : MergeGroup := .operator 80816 80814
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80816) (leftOrdinal := 0)
    (rightResult := 80814) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16871⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80820

namespace LeftMerge80832
def owner : Owner := ⟨.program ⟨214⟩, ⟨29820⟩⟩
def mergeEvent : Nat := 80832
def frameStart : Nat := 80754
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩] } }
def leftRaw : List Term := Proof.Events315.exact80828RawTerms
def rightRaw : List Term := Proof.Events315.exact80805RawTerms
def group : MergeGroup := .operator 80828 80805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80828) (leftOrdinal := 0)
    (rightResult := 80805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29819⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80832

namespace LeftMerge80833
def owner : Owner := ⟨.program ⟨214⟩, ⟨29820⟩⟩
def mergeEvent : Nat := 80833
def frameStart : Nat := 80754
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16871⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩] } }
def leftRaw : List Term := Proof.Events315.exact80828RawTerms
def rightRaw : List Term := Proof.Events315.exact80805RawTerms
def group : MergeGroup := .operator 80828 80805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80828) (leftOrdinal := 1)
    (rightResult := 80805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16871⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29819⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80833

namespace LeftMerge80835
def owner : Owner := ⟨.program ⟨214⟩, ⟨29820⟩⟩
def mergeEvent : Nat := 80835
def frameStart : Nat := 80754
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16871⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24729⟩⟩] } }
def rhsRaw : List Term := Proof.Events315.exact80802RawTerms
def group : MergeGroup := .relation 80834
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80834) (rhsResult := 80802)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29819⟩⟩) ⟨24729⟩ 80802) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24729⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨24729⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80835

namespace LeftMerge80843
def owner : Owner := ⟨.program ⟨214⟩, ⟨17086⟩⟩
def mergeEvent : Nat := 80843
def frameStart : Nat := 80754
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17085⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events315.exact80816RawTerms
def rightRaw : List Term := Proof.Events315.exact80839RawTerms
def group : MergeGroup := .operator 80816 80839
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80816) (leftOrdinal := 0)
    (rightResult := 80839) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17085⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80843

namespace LeftMerge80860
def owner : Owner := ⟨.program ⟨214⟩, ⟨22699⟩⟩
def mergeEvent : Nat := 80860
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6741⟩⟩] } }
def rhsRaw : List Term := Proof.Events315.exact80857RawTerms
def group : MergeGroup := .relation 80859
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80859) (rhsResult := 80857)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 80858 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩) (none) 80857) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6741⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80860

namespace LeftMerge80861
def owner : Owner := ⟨.program ⟨214⟩, ⟨22699⟩⟩
def mergeEvent : Nat := 80861
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩] } }
def rhsRaw : List Term := Proof.Events315.exact80857RawTerms
def group : MergeGroup := .relation 80859
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80859) (rhsResult := 80857)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 80858 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩) (none) 80857) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80861

namespace LeftMerge80862
def owner : Owner := ⟨.program ⟨214⟩, ⟨22699⟩⟩
def mergeEvent : Nat := 80862
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24729⟩⟩] } }
def rhsRaw : List Term := Proof.Events315.exact80857RawTerms
def group : MergeGroup := .relation 80859
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80859) (rhsResult := 80857)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 80858 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩) (none) 80857) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16871⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24729⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨24729⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80862

namespace LeftMerge80863
def owner : Owner := ⟨.program ⟨214⟩, ⟨22699⟩⟩
def mergeEvent : Nat := 80863
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17085⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events315.exact80857RawTerms
def group : MergeGroup := .relation 80859
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80859) (rhsResult := 80857)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 80858 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩) (none) 80857) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17085⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80863

namespace LeftMerge80868
def owner : Owner := ⟨.program ⟨214⟩, ⟨29822⟩⟩
def mergeEvent : Nat := 80868
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩] } }
def leftRaw : List Term := Proof.Events315.exact80864RawTerms
def rightRaw : List Term := Proof.Events315.exact80686RawTerms
def group : MergeGroup := .operator 80864 80686
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80864) (leftOrdinal := 0)
    (rightResult := 80686) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80868

namespace LeftMerge80869
def owner : Owner := ⟨.program ⟨214⟩, ⟨29822⟩⟩
def mergeEvent : Nat := 80869
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24729⟩⟩] } }
def leftRaw : List Term := Proof.Events315.exact80864RawTerms
def rightRaw : List Term := Proof.Events315.exact80686RawTerms
def group : MergeGroup := .operator 80864 80686
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80864) (leftOrdinal := 2)
    (rightResult := 80686) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24729⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24729⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨24729⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80869

namespace LeftMerge80895
def owner : Owner := ⟨.program ⟨214⟩, ⟨12961⟩⟩
def mergeEvent : Nat := 80895
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events015.exact3874RawTerms
def rightRaw : List Term := Proof.Events312.exact79920RawTerms
def group : MergeGroup := .operator 3874 79920
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3874) (leftOrdinal := 0)
    (rightResult := 79920) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12958⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80895

namespace LeftMerge80900
def owner : Owner := ⟨.program ⟨214⟩, ⟨7244⟩⟩
def mergeEvent : Nat := 80900
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79790RawTerms
def rightRaw : List Term := Proof.Events029.exact7474RawTerms
def group : MergeGroup := .operator 79790 7474
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79790) (leftOrdinal := 0)
    (rightResult := 7474) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80900

namespace LeftMerge80917
def owner : Owner := ⟨.program ⟨214⟩, ⟨12964⟩⟩
def mergeEvent : Nat := 80917
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events316.exact80911RawTerms
def rightRaw : List Term := Proof.Events015.exact3877RawTerms
def group : MergeGroup := .operator 80911 3877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80911) (leftOrdinal := 1)
    (rightResult := 3877) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10135⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80917

namespace LeftMerge80918
def owner : Owner := ⟨.program ⟨214⟩, ⟨12964⟩⟩
def mergeEvent : Nat := 80918
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }
def leftRaw : List Term := Proof.Events316.exact80911RawTerms
def rightRaw : List Term := Proof.Events015.exact3877RawTerms
def group : MergeGroup := .operator 80911 3877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80911) (leftOrdinal := 0)
    (rightResult := 3877) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10135⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80918

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
