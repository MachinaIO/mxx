import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge72649
def owner : Owner := ⟨.program ⟨257⟩, ⟨36800⟩⟩
def mergeEvent : Nat := 72649
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨34804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36798⟩⟩] } }
def leftRaw : List Term := Proof.Events249.exact63966RawTerms
def rightRaw : List Term := Proof.Events283.exact72642RawTerms
def group : MergeGroup := .operator 63966 72642
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63966) (leftOrdinal := 1)
    (rightResult := 72642) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨34804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36798⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨34804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36798⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge72649

namespace LeftMerge72651
def owner : Owner := ⟨.program ⟨257⟩, ⟨36800⟩⟩
def mergeEvent : Nat := 72651
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨34804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35963⟩⟩] } }
def rhsRaw : List Term := Proof.Events283.exact72639RawTerms
def group : MergeGroup := .relation 72650
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 72650) (rhsResult := 72639)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨34804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36798⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36798⟩⟩) ⟨35963⟩ 72639) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35963⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨34804⟩⟩], [⟨.program ⟨257⟩, ⟨35963⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge72651

namespace LeftMerge72665
def owner : Owner := ⟨.program ⟨257⟩, ⟨35635⟩⟩
def mergeEvent : Nat := 72665
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35632⟩⟩] } }
def leftRaw : List Term := Proof.Events239.exact61370RawTerms
def rightRaw : List Term := Proof.Events283.exact72659RawTerms
def group : MergeGroup := .operator 61370 72659
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61370) (leftOrdinal := 0)
    (rightResult := 72659) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35632⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35632⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge72665

namespace LeftMerge72786
def owner : Owner := ⟨.program ⟨257⟩, ⟨36136⟩⟩
def mergeEvent : Nat := 72786
def frameStart : Nat := 72720
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events284.exact72782RawTerms
def rightRaw : List Term := Proof.Events284.exact72780RawTerms
def group : MergeGroup := .operator 72782 72780
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 72782) (leftOrdinal := 0)
    (rightResult := 72780) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34804⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge72786

namespace LeftMerge72798
def owner : Owner := ⟨.program ⟨257⟩, ⟨36799⟩⟩
def mergeEvent : Nat := 72798
def frameStart : Nat := 72720
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36798⟩⟩] } }
def leftRaw : List Term := Proof.Events284.exact72794RawTerms
def rightRaw : List Term := Proof.Events284.exact72771RawTerms
def group : MergeGroup := .operator 72794 72771
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 72794) (leftOrdinal := 0)
    (rightResult := 72771) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36798⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36798⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge72798

namespace LeftMerge72799
def owner : Owner := ⟨.program ⟨257⟩, ⟨36799⟩⟩
def mergeEvent : Nat := 72799
def frameStart : Nat := 72720
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36798⟩⟩] } }
def leftRaw : List Term := Proof.Events284.exact72794RawTerms
def rightRaw : List Term := Proof.Events284.exact72771RawTerms
def group : MergeGroup := .operator 72794 72771
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 72794) (leftOrdinal := 1)
    (rightResult := 72771) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36798⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36798⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge72799

namespace LeftMerge72801
def owner : Owner := ⟨.program ⟨257⟩, ⟨36799⟩⟩
def mergeEvent : Nat := 72801
def frameStart : Nat := 72720
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35963⟩⟩] } }
def rhsRaw : List Term := Proof.Events284.exact72768RawTerms
def group : MergeGroup := .relation 72800
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 72800) (rhsResult := 72768)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨34804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36798⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36798⟩⟩) ⟨35963⟩ 72768) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35963⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨34804⟩⟩], [⟨.program ⟨257⟩, ⟨35963⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge72801

namespace LeftMerge72809
def owner : Owner := ⟨.program ⟨257⟩, ⟨35052⟩⟩
def mergeEvent : Nat := 72809
def frameStart : Nat := 72720
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨35050⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events284.exact72782RawTerms
def rightRaw : List Term := Proof.Events284.exact72805RawTerms
def group : MergeGroup := .operator 72782 72805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 72782) (leftOrdinal := 0)
    (rightResult := 72805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨35050⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge72809

namespace LeftMerge72826
def owner : Owner := ⟨.program ⟨257⟩, ⟨35635⟩⟩
def mergeEvent : Nat := 72826
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7221⟩⟩] } }
def rhsRaw : List Term := Proof.Events284.exact72823RawTerms
def group : MergeGroup := .relation 72825
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 72825) (rhsResult := 72823)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35632⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 72824 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35632⟩⟩]⟩) (none) 72823) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7221⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge72826

namespace LeftMerge72827
def owner : Owner := ⟨.program ⟨257⟩, ⟨35635⟩⟩
def mergeEvent : Nat := 72827
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36798⟩⟩] } }
def rhsRaw : List Term := Proof.Events284.exact72823RawTerms
def group : MergeGroup := .relation 72825
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 72825) (rhsResult := 72823)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35632⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 72824 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35632⟩⟩]⟩) (none) 72823) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36798⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36798⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge72827

namespace LeftMerge72828
def owner : Owner := ⟨.program ⟨257⟩, ⟨35635⟩⟩
def mergeEvent : Nat := 72828
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨34804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35963⟩⟩] } }
def rhsRaw : List Term := Proof.Events284.exact72823RawTerms
def group : MergeGroup := .relation 72825
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 72825) (rhsResult := 72823)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35632⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 72824 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35632⟩⟩]⟩) (none) 72823) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35963⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨34804⟩⟩], [⟨.program ⟨257⟩, ⟨35963⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge72828

namespace LeftMerge72829
def owner : Owner := ⟨.program ⟨257⟩, ⟨35635⟩⟩
def mergeEvent : Nat := 72829
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events284.exact72823RawTerms
def group : MergeGroup := .relation 72825
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 72825) (rhsResult := 72823)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35632⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 72824 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35632⟩⟩]⟩) (none) 72823) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨35050⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge72829

namespace LeftMerge72834
def owner : Owner := ⟨.program ⟨257⟩, ⟨36801⟩⟩
def mergeEvent : Nat := 72834
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36798⟩⟩] } }
def leftRaw : List Term := Proof.Events284.exact72830RawTerms
def rightRaw : List Term := Proof.Events283.exact72652RawTerms
def group : MergeGroup := .operator 72830 72652
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 72830) (leftOrdinal := 0)
    (rightResult := 72652) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36798⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36798⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36798⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge72834

namespace LeftMerge72835
def owner : Owner := ⟨.program ⟨257⟩, ⟨36801⟩⟩
def mergeEvent : Nat := 72835
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨34804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35963⟩⟩] } }
def leftRaw : List Term := Proof.Events284.exact72830RawTerms
def rightRaw : List Term := Proof.Events283.exact72652RawTerms
def group : MergeGroup := .operator 72830 72652
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 72830) (leftOrdinal := 2)
    (rightResult := 72652) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨34804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35963⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨34804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35963⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨34804⟩⟩], [⟨.program ⟨257⟩, ⟨35963⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge72835

namespace LeftMerge72843
def owner : Owner := ⟨.program ⟨257⟩, ⟨36802⟩⟩
def mergeEvent : Nat := 72843
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩] } }
def leftRaw : List Term := Proof.Events284.exact72837RawTerms
def rightRaw : List Term := Proof.Events061.exact15642RawTerms
def group : MergeGroup := .operator 72837 15642
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 72837) (leftOrdinal := 0)
    (rightResult := 15642) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7221⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7163⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge72843

namespace LeftMerge72844
def owner : Owner := ⟨.program ⟨257⟩, ⟨36802⟩⟩
def mergeEvent : Nat := 72844
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩] } }
def leftRaw : List Term := Proof.Events284.exact72837RawTerms
def rightRaw : List Term := Proof.Events061.exact15642RawTerms
def group : MergeGroup := .operator 72837 15642
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 72837) (leftOrdinal := 1)
    (rightResult := 15642) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7163⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge72844

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
