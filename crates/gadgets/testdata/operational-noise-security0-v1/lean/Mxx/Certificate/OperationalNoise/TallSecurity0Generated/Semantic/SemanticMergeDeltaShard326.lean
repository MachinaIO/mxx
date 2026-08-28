import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge53627
def owner : Owner := ⟨.program ⟨214⟩, ⟨9724⟩⟩
def mergeEvent : Nat := 53627
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩] } }
def leftRaw : List Term := Proof.Events209.exact53621RawTerms
def rightRaw : List Term := Proof.Events037.exact9508RawTerms
def group : MergeGroup := .operator 53621 9508
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53621) (leftOrdinal := 1)
    (rightResult := 9508) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7864⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53627

namespace LeftMerge53629
def owner : Owner := ⟨.program ⟨214⟩, ⟨9724⟩⟩
def mergeEvent : Nat := 53629
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6784⟩⟩] } }
def rhsRaw : List Term := Proof.Events037.exact9478RawTerms
def group : MergeGroup := .relation 53628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 53628) (rhsResult := 9478)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7864⟩⟩) ⟨6784⟩ 9478) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6784⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53629

namespace LeftMerge53630
def owner : Owner := ⟨.program ⟨214⟩, ⟨9724⟩⟩
def mergeEvent : Nat := 53630
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩] } }
def leftRaw : List Term := Proof.Events209.exact53621RawTerms
def rightRaw : List Term := Proof.Events037.exact9508RawTerms
def group : MergeGroup := .operator 53621 9508
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53621) (leftOrdinal := 0)
    (rightResult := 9508) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7864⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53630

namespace LeftMerge53635
def owner : Owner := ⟨.program ⟨214⟩, ⟨11972⟩⟩
def mergeEvent : Nat := 53635
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6784⟩⟩] } }
def leftRaw : List Term := Proof.Events209.exact53631RawTerms
def rightRaw : List Term := Proof.Events209.exact53601RawTerms
def group : MergeGroup := .operator 53631 53601
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53631) (leftOrdinal := 1)
    (rightResult := 53601) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6784⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6784⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53635

namespace LeftMerge53643
def owner : Owner := ⟨.program ⟨214⟩, ⟨25225⟩⟩
def mergeEvent : Nat := 53643
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25224⟩⟩] } }
def leftRaw : List Term := Proof.Events209.exact53637RawTerms
def rightRaw : List Term := Proof.Events209.exact53573RawTerms
def group : MergeGroup := .operator 53637 53573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53637) (leftOrdinal := 1)
    (rightResult := 53573) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25224⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25224⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53643

namespace LeftMerge53645
def owner : Owner := ⟨.program ⟨214⟩, ⟨25225⟩⟩
def mergeEvent : Nat := 53645
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23124⟩⟩] } }
def rhsRaw : List Term := Proof.Events209.exact53570RawTerms
def group : MergeGroup := .relation 53644
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 53644) (rhsResult := 53570)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25224⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25224⟩⟩) ⟨23124⟩ 53570) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23124⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], [⟨.program ⟨214⟩, ⟨23124⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53645

namespace LeftMerge53646
def owner : Owner := ⟨.program ⟨214⟩, ⟨25225⟩⟩
def mergeEvent : Nat := 53646
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25224⟩⟩] } }
def leftRaw : List Term := Proof.Events209.exact53637RawTerms
def rightRaw : List Term := Proof.Events209.exact53573RawTerms
def group : MergeGroup := .operator 53637 53573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53637) (leftOrdinal := 0)
    (rightResult := 53573) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25224⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25224⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53646

namespace LeftMerge53660
def owner : Owner := ⟨.program ⟨214⟩, ⟨19823⟩⟩
def mergeEvent : Nat := 53660
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19820⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events209.exact53654RawTerms
def group : MergeGroup := .operator 50762 53654
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 53654) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19820⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19820⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53660

namespace LeftMerge53739
def owner : Owner := ⟨.program ⟨214⟩, ⟨11966⟩⟩
def mergeEvent : Nat := 53739
def frameStart : Nat := 53709
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events209.exact53735RawTerms
def rightRaw : List Term := Proof.Events209.exact53732RawTerms
def group : MergeGroup := .operator 53735 53732
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53735) (leftOrdinal := 0)
    (rightResult := 53732) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9720⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11965⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53739

namespace LeftMerge53769
def owner : Owner := ⟨.program ⟨214⟩, ⟨12059⟩⟩
def mergeEvent : Nat := 53769
def frameStart : Nat := 53709
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events210.exact53765RawTerms
def rightRaw : List Term := Proof.Events210.exact53763RawTerms
def group : MergeGroup := .operator 53765 53763
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53765) (leftOrdinal := 0)
    (rightResult := 53763) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53769

namespace LeftMerge53792
def owner : Owner := ⟨.program ⟨214⟩, ⟨7866⟩⟩
def mergeEvent : Nat := 53792
def frameStart : Nat := 53709
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩] } }
def leftRaw : List Term := Proof.Events210.exact53788RawTerms
def rightRaw : List Term := Proof.Events210.exact53785RawTerms
def group : MergeGroup := .operator 53788 53785
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53788) (leftOrdinal := 0)
    (rightResult := 53785) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7864⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53792

namespace LeftMerge53801
def owner : Owner := ⟨.program ⟨214⟩, ⟨25227⟩⟩
def mergeEvent : Nat := 53801
def frameStart : Nat := 53709
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25224⟩⟩] } }
def leftRaw : List Term := Proof.Events210.exact53797RawTerms
def rightRaw : List Term := Proof.Events209.exact53754RawTerms
def group : MergeGroup := .operator 53797 53754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53797) (leftOrdinal := 0)
    (rightResult := 53754) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25224⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25224⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53801

namespace LeftMerge53802
def owner : Owner := ⟨.program ⟨214⟩, ⟨25227⟩⟩
def mergeEvent : Nat := 53802
def frameStart : Nat := 53709
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25224⟩⟩] } }
def leftRaw : List Term := Proof.Events210.exact53797RawTerms
def rightRaw : List Term := Proof.Events209.exact53754RawTerms
def group : MergeGroup := .operator 53797 53754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53797) (leftOrdinal := 1)
    (rightResult := 53754) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25224⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25224⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53802

namespace LeftMerge53804
def owner : Owner := ⟨.program ⟨214⟩, ⟨25227⟩⟩
def mergeEvent : Nat := 53804
def frameStart : Nat := 53709
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23124⟩⟩] } }
def rhsRaw : List Term := Proof.Events209.exact53751RawTerms
def group : MergeGroup := .relation 53803
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 53803) (rhsResult := 53751)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25224⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25224⟩⟩) ⟨23124⟩ 53751) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23124⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], [⟨.program ⟨214⟩, ⟨23124⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53804

namespace LeftMerge53812
def owner : Owner := ⟨.program ⟨214⟩, ⟨16387⟩⟩
def mergeEvent : Nat := 53812
def frameStart : Nat := 53709
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events210.exact53765RawTerms
def rightRaw : List Term := Proof.Events210.exact53808RawTerms
def group : MergeGroup := .operator 53765 53808
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53765) (leftOrdinal := 0)
    (rightResult := 53808) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16385⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53812

namespace LeftMerge53829
def owner : Owner := ⟨.program ⟨214⟩, ⟨19823⟩⟩
def mergeEvent : Nat := 53829
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩] } }
def rhsRaw : List Term := Proof.Events210.exact53826RawTerms
def group : MergeGroup := .relation 53828
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 53828) (rhsResult := 53826)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19820⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 53827 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19820⟩⟩]⟩) (none) 53826) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53829

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
