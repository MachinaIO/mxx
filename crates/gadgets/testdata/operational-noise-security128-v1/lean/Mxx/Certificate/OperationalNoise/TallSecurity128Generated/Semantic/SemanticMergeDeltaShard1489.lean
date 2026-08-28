import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge241682
def owner : Owner := ⟨.program ⟨257⟩, ⟨61438⟩⟩
def mergeEvent : Nat := 241682
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61437⟩⟩] } }
def leftRaw : List Term := Proof.Events944.exact241673RawTerms
def rightRaw : List Term := Proof.Events943.exact241609RawTerms
def group : MergeGroup := .operator 241673 241609
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241673) (leftOrdinal := 0)
    (rightResult := 241609) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241682

namespace LeftMerge241696
def owner : Owner := ⟨.program ⟨257⟩, ⟨60372⟩⟩
def mergeEvent : Nat := 241696
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60369⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236870RawTerms
def rightRaw : List Term := Proof.Events944.exact241690RawTerms
def group : MergeGroup := .operator 236870 241690
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236870) (leftOrdinal := 0)
    (rightResult := 241690) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60369⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60369⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241696

namespace LeftMerge241775
def owner : Owner := ⟨.program ⟨257⟩, ⟨59432⟩⟩
def mergeEvent : Nat := 241775
def frameStart : Nat := 241745
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25226⟩⟩, ⟨.program ⟨257⟩, ⟨59431⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events944.exact241771RawTerms
def rightRaw : List Term := Proof.Events944.exact241768RawTerms
def group : MergeGroup := .operator 241771 241768
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241771) (leftOrdinal := 0)
    (rightResult := 241768) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59431⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25226⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25226⟩⟩, ⟨.program ⟨257⟩, ⟨59431⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241775

namespace LeftMerge241805
def owner : Owner := ⟨.program ⟨257⟩, ⟨61220⟩⟩
def mergeEvent : Nat := 241805
def frameStart : Nat := 241745
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25226⟩⟩, ⟨.program ⟨257⟩, ⟨59431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events944.exact241801RawTerms
def rightRaw : List Term := Proof.Events944.exact241799RawTerms
def group : MergeGroup := .operator 241801 241799
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241801) (leftOrdinal := 0)
    (rightResult := 241799) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25226⟩⟩, ⟨.program ⟨257⟩, ⟨59431⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25226⟩⟩, ⟨.program ⟨257⟩, ⟨59431⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241805

namespace LeftMerge241828
def owner : Owner := ⟨.program ⟨257⟩, ⟨9537⟩⟩
def mergeEvent : Nat := 241828
def frameStart : Nat := 241745
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }
def leftRaw : List Term := Proof.Events944.exact241824RawTerms
def rightRaw : List Term := Proof.Events944.exact241821RawTerms
def group : MergeGroup := .operator 241824 241821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241824) (leftOrdinal := 0)
    (rightResult := 241821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9535⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241828

namespace LeftMerge241837
def owner : Owner := ⟨.program ⟨257⟩, ⟨61440⟩⟩
def mergeEvent : Nat := 241837
def frameStart : Nat := 241745
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61437⟩⟩] } }
def leftRaw : List Term := Proof.Events944.exact241833RawTerms
def rightRaw : List Term := Proof.Events944.exact241790RawTerms
def group : MergeGroup := .operator 241833 241790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241833) (leftOrdinal := 0)
    (rightResult := 241790) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61437⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241837

namespace LeftMerge241838
def owner : Owner := ⟨.program ⟨257⟩, ⟨61440⟩⟩
def mergeEvent : Nat := 241838
def frameStart : Nat := 241745
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25226⟩⟩, ⟨.program ⟨257⟩, ⟨59431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61437⟩⟩] } }
def leftRaw : List Term := Proof.Events944.exact241833RawTerms
def rightRaw : List Term := Proof.Events944.exact241790RawTerms
def group : MergeGroup := .operator 241833 241790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241833) (leftOrdinal := 1)
    (rightResult := 241790) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25226⟩⟩, ⟨.program ⟨257⟩, ⟨59431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25226⟩⟩, ⟨.program ⟨257⟩, ⟨59431⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241838

namespace LeftMerge241840
def owner : Owner := ⟨.program ⟨257⟩, ⟨61440⟩⟩
def mergeEvent : Nat := 241840
def frameStart : Nat := 241745
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25226⟩⟩, ⟨.program ⟨257⟩, ⟨59431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60937⟩⟩] } }
def rhsRaw : List Term := Proof.Events944.exact241787RawTerms
def group : MergeGroup := .relation 241839
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 241839) (rhsResult := 241787)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25226⟩⟩, ⟨.program ⟨257⟩, ⟨59431⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61437⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61437⟩⟩) ⟨60937⟩ 241787) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60937⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25226⟩⟩, ⟨.program ⟨257⟩, ⟨59431⟩⟩], [⟨.program ⟨257⟩, ⟨60937⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241840

namespace LeftMerge241848
def owner : Owner := ⟨.program ⟨257⟩, ⟨59814⟩⟩
def mergeEvent : Nat := 241848
def frameStart : Nat := 241745
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events944.exact241801RawTerms
def rightRaw : List Term := Proof.Events944.exact241844RawTerms
def group : MergeGroup := .operator 241801 241844
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241801) (leftOrdinal := 0)
    (rightResult := 241844) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59812⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241848

namespace LeftMerge241865
def owner : Owner := ⟨.program ⟨257⟩, ⟨60372⟩⟩
def mergeEvent : Nat := 241865
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }
def rhsRaw : List Term := Proof.Events944.exact241862RawTerms
def group : MergeGroup := .relation 241864
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 241864) (rhsResult := 241862)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 241863 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60369⟩⟩]⟩) (none) 241862) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241865

namespace LeftMerge241866
def owner : Owner := ⟨.program ⟨257⟩, ⟨60372⟩⟩
def mergeEvent : Nat := 241866
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61437⟩⟩] } }
def rhsRaw : List Term := Proof.Events944.exact241862RawTerms
def group : MergeGroup := .relation 241864
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 241864) (rhsResult := 241862)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 241863 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60369⟩⟩]⟩) (none) 241862) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61437⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241866

namespace LeftMerge241867
def owner : Owner := ⟨.program ⟨257⟩, ⟨60372⟩⟩
def mergeEvent : Nat := 241867
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25226⟩⟩, ⟨.program ⟨257⟩, ⟨59431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60937⟩⟩] } }
def rhsRaw : List Term := Proof.Events944.exact241862RawTerms
def group : MergeGroup := .relation 241864
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 241864) (rhsResult := 241862)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 241863 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60369⟩⟩]⟩) (none) 241862) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25226⟩⟩, ⟨.program ⟨257⟩, ⟨59431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60937⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25226⟩⟩, ⟨.program ⟨257⟩, ⟨59431⟩⟩], [⟨.program ⟨257⟩, ⟨60937⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241867

namespace LeftMerge241868
def owner : Owner := ⟨.program ⟨257⟩, ⟨60372⟩⟩
def mergeEvent : Nat := 241868
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨59812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events944.exact241862RawTerms
def group : MergeGroup := .relation 241864
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 241864) (rhsResult := 241862)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 241863 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60369⟩⟩]⟩) (none) 241862) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨59812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241868

namespace LeftMerge241873
def owner : Owner := ⟨.program ⟨257⟩, ⟨61439⟩⟩
def mergeEvent : Nat := 241873
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25226⟩⟩, ⟨.program ⟨257⟩, ⟨59431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60937⟩⟩] } }
def leftRaw : List Term := Proof.Events944.exact241869RawTerms
def rightRaw : List Term := Proof.Events944.exact241683RawTerms
def group : MergeGroup := .operator 241869 241683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241869) (leftOrdinal := 2)
    (rightResult := 241683) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25226⟩⟩, ⟨.program ⟨257⟩, ⟨59431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60937⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25226⟩⟩, ⟨.program ⟨257⟩, ⟨59431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60937⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25226⟩⟩, ⟨.program ⟨257⟩, ⟨59431⟩⟩], [⟨.program ⟨257⟩, ⟨60937⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241873

namespace LeftMerge241874
def owner : Owner := ⟨.program ⟨257⟩, ⟨61439⟩⟩
def mergeEvent : Nat := 241874
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61437⟩⟩] } }
def leftRaw : List Term := Proof.Events944.exact241869RawTerms
def rightRaw : List Term := Proof.Events944.exact241683RawTerms
def group : MergeGroup := .operator 241869 241683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241869) (leftOrdinal := 1)
    (rightResult := 241683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61437⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241874

namespace LeftMerge241882
def owner : Owner := ⟨.program ⟨257⟩, ⟨61832⟩⟩
def mergeEvent : Nat := 241882
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61830⟩⟩] } }
def leftRaw : List Term := Proof.Events944.exact241876RawTerms
def rightRaw : List Term := Proof.Events943.exact241599RawTerms
def group : MergeGroup := .operator 241876 241599
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241876) (leftOrdinal := 0)
    (rightResult := 241599) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61830⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61830⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241882

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
