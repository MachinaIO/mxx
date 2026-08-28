import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge67625
def owner : Owner := ⟨.program ⟨257⟩, ⟨52597⟩⟩
def mergeEvent : Nat := 67625
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52596⟩⟩] } }
def leftRaw : List Term := Proof.Events264.exact67619RawTerms
def rightRaw : List Term := Proof.Events263.exact67555RawTerms
def group : MergeGroup := .operator 67619 67555
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67619) (leftOrdinal := 1)
    (rightResult := 67555) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52596⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52596⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67625

namespace LeftMerge67627
def owner : Owner := ⟨.program ⟨257⟩, ⟨52597⟩⟩
def mergeEvent : Nat := 67627
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52051⟩⟩] } }
def rhsRaw : List Term := Proof.Events263.exact67552RawTerms
def group : MergeGroup := .relation 67626
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67626) (rhsResult := 67552)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52596⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52596⟩⟩) ⟨52051⟩ 67552) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52051⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], [⟨.program ⟨257⟩, ⟨52051⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67627

namespace LeftMerge67628
def owner : Owner := ⟨.program ⟨257⟩, ⟨52597⟩⟩
def mergeEvent : Nat := 67628
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52596⟩⟩] } }
def leftRaw : List Term := Proof.Events264.exact67619RawTerms
def rightRaw : List Term := Proof.Events263.exact67555RawTerms
def group : MergeGroup := .operator 67619 67555
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67619) (leftOrdinal := 0)
    (rightResult := 67555) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52596⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52596⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67628

namespace LeftMerge67642
def owner : Owner := ⟨.program ⟨257⟩, ⟨51522⟩⟩
def mergeEvent : Nat := 67642
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51519⟩⟩] } }
def leftRaw : List Term := Proof.Events239.exact61370RawTerms
def rightRaw : List Term := Proof.Events264.exact67636RawTerms
def group : MergeGroup := .operator 61370 67636
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61370) (leftOrdinal := 0)
    (rightResult := 67636) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51519⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51519⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67642

namespace LeftMerge67721
def owner : Owner := ⟨.program ⟨257⟩, ⟨50735⟩⟩
def mergeEvent : Nat := 67721
def frameStart : Nat := 67691
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events264.exact67717RawTerms
def rightRaw : List Term := Proof.Events264.exact67714RawTerms
def group : MergeGroup := .operator 67717 67714
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67717) (leftOrdinal := 0)
    (rightResult := 67714) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50734⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24614⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67721

namespace LeftMerge67751
def owner : Owner := ⟨.program ⟨257⟩, ⟨52316⟩⟩
def mergeEvent : Nat := 67751
def frameStart : Nat := 67691
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events264.exact67747RawTerms
def rightRaw : List Term := Proof.Events264.exact67745RawTerms
def group : MergeGroup := .operator 67747 67745
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67747) (leftOrdinal := 0)
    (rightResult := 67745) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67751

namespace LeftMerge67774
def owner : Owner := ⟨.program ⟨257⟩, ⟨9582⟩⟩
def mergeEvent : Nat := 67774
def frameStart : Nat := 67691
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }
def leftRaw : List Term := Proof.Events264.exact67770RawTerms
def rightRaw : List Term := Proof.Events264.exact67767RawTerms
def group : MergeGroup := .operator 67770 67767
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67770) (leftOrdinal := 0)
    (rightResult := 67767) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9580⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67774

namespace LeftMerge67783
def owner : Owner := ⟨.program ⟨257⟩, ⟨52599⟩⟩
def mergeEvent : Nat := 67783
def frameStart : Nat := 67691
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52596⟩⟩] } }
def leftRaw : List Term := Proof.Events264.exact67779RawTerms
def rightRaw : List Term := Proof.Events264.exact67736RawTerms
def group : MergeGroup := .operator 67779 67736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67779) (leftOrdinal := 0)
    (rightResult := 67736) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52596⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52596⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67783

namespace LeftMerge67784
def owner : Owner := ⟨.program ⟨257⟩, ⟨52599⟩⟩
def mergeEvent : Nat := 67784
def frameStart : Nat := 67691
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52596⟩⟩] } }
def leftRaw : List Term := Proof.Events264.exact67779RawTerms
def rightRaw : List Term := Proof.Events264.exact67736RawTerms
def group : MergeGroup := .operator 67779 67736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67779) (leftOrdinal := 1)
    (rightResult := 67736) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52596⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52596⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67784

namespace LeftMerge67786
def owner : Owner := ⟨.program ⟨257⟩, ⟨52599⟩⟩
def mergeEvent : Nat := 67786
def frameStart : Nat := 67691
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52051⟩⟩] } }
def rhsRaw : List Term := Proof.Events264.exact67733RawTerms
def group : MergeGroup := .relation 67785
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67785) (rhsResult := 67733)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52596⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52596⟩⟩) ⟨52051⟩ 67733) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52051⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], [⟨.program ⟨257⟩, ⟨52051⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67786

namespace LeftMerge67794
def owner : Owner := ⟨.program ⟨257⟩, ⟨50946⟩⟩
def mergeEvent : Nat := 67794
def frameStart : Nat := 67691
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events264.exact67747RawTerms
def rightRaw : List Term := Proof.Events264.exact67790RawTerms
def group : MergeGroup := .operator 67747 67790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67747) (leftOrdinal := 0)
    (rightResult := 67790) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50944⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67794

namespace LeftMerge67811
def owner : Owner := ⟨.program ⟨257⟩, ⟨51522⟩⟩
def mergeEvent : Nat := 67811
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }
def rhsRaw : List Term := Proof.Events264.exact67808RawTerms
def group : MergeGroup := .relation 67810
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67810) (rhsResult := 67808)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51519⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 67809 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51519⟩⟩]⟩) (none) 67808) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67811

namespace LeftMerge67812
def owner : Owner := ⟨.program ⟨257⟩, ⟨51522⟩⟩
def mergeEvent : Nat := 67812
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52596⟩⟩] } }
def rhsRaw : List Term := Proof.Events264.exact67808RawTerms
def group : MergeGroup := .relation 67810
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67810) (rhsResult := 67808)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51519⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 67809 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51519⟩⟩]⟩) (none) 67808) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52596⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52596⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67812

namespace LeftMerge67813
def owner : Owner := ⟨.program ⟨257⟩, ⟨51522⟩⟩
def mergeEvent : Nat := 67813
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52051⟩⟩] } }
def rhsRaw : List Term := Proof.Events264.exact67808RawTerms
def group : MergeGroup := .relation 67810
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67810) (rhsResult := 67808)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51519⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 67809 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51519⟩⟩]⟩) (none) 67808) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52051⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], [⟨.program ⟨257⟩, ⟨52051⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67813

namespace LeftMerge67814
def owner : Owner := ⟨.program ⟨257⟩, ⟨51522⟩⟩
def mergeEvent : Nat := 67814
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨50944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events264.exact67808RawTerms
def group : MergeGroup := .relation 67810
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67810) (rhsResult := 67808)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51519⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 67809 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51519⟩⟩]⟩) (none) 67808) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67814

namespace LeftMerge67819
def owner : Owner := ⟨.program ⟨257⟩, ⟨52598⟩⟩
def mergeEvent : Nat := 67819
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52051⟩⟩] } }
def leftRaw : List Term := Proof.Events264.exact67815RawTerms
def rightRaw : List Term := Proof.Events264.exact67629RawTerms
def group : MergeGroup := .operator 67815 67629
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67815) (leftOrdinal := 2)
    (rightResult := 67629) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52051⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52051⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], [⟨.program ⟨257⟩, ⟨52051⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67819

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
