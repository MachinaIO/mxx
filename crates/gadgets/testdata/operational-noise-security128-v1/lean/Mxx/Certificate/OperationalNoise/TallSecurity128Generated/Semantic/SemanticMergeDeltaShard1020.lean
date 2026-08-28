import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge166812
def owner : Owner := ⟨.program ⟨257⟩, ⟨29572⟩⟩
def mergeEvent : Nat := 166812
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }
def rhsRaw : List Term := Proof.Events651.exact166809RawTerms
def group : MergeGroup := .relation 166811
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 166811) (rhsResult := 166809)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29569⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 166810 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29569⟩⟩]⟩) (none) 166809) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge166812

namespace LeftMerge166813
def owner : Owner := ⟨.program ⟨257⟩, ⟨29572⟩⟩
def mergeEvent : Nat := 166813
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30643⟩⟩] } }
def rhsRaw : List Term := Proof.Events651.exact166809RawTerms
def group : MergeGroup := .relation 166811
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 166811) (rhsResult := 166809)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29569⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 166810 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29569⟩⟩]⟩) (none) 166809) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30643⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30643⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge166813

namespace LeftMerge166814
def owner : Owner := ⟨.program ⟨257⟩, ⟨29572⟩⟩
def mergeEvent : Nat := 166814
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13341⟩⟩, ⟨.program ⟨257⟩, ⟨28870⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30113⟩⟩] } }
def rhsRaw : List Term := Proof.Events651.exact166809RawTerms
def group : MergeGroup := .relation 166811
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 166811) (rhsResult := 166809)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29569⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 166810 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29569⟩⟩]⟩) (none) 166809) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13341⟩⟩, ⟨.program ⟨257⟩, ⟨28870⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30113⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13341⟩⟩, ⟨.program ⟨257⟩, ⟨28870⟩⟩], [⟨.program ⟨257⟩, ⟨30113⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge166814

namespace LeftMerge166815
def owner : Owner := ⟨.program ⟨257⟩, ⟨29572⟩⟩
def mergeEvent : Nat := 166815
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events651.exact166809RawTerms
def group : MergeGroup := .relation 166811
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 166811) (rhsResult := 166809)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29569⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 166810 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29569⟩⟩]⟩) (none) 166809) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge166815

namespace LeftMerge166820
def owner : Owner := ⟨.program ⟨257⟩, ⟨30645⟩⟩
def mergeEvent : Nat := 166820
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13341⟩⟩, ⟨.program ⟨257⟩, ⟨28870⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30113⟩⟩] } }
def leftRaw : List Term := Proof.Events651.exact166816RawTerms
def rightRaw : List Term := Proof.Events650.exact166630RawTerms
def group : MergeGroup := .operator 166816 166630
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 166816) (leftOrdinal := 2)
    (rightResult := 166630) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13341⟩⟩, ⟨.program ⟨257⟩, ⟨28870⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30113⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13341⟩⟩, ⟨.program ⟨257⟩, ⟨28870⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13341⟩⟩, ⟨.program ⟨257⟩, ⟨28870⟩⟩], [⟨.program ⟨257⟩, ⟨30113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge166820

namespace LeftMerge166821
def owner : Owner := ⟨.program ⟨257⟩, ⟨30645⟩⟩
def mergeEvent : Nat := 166821
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30643⟩⟩] } }
def leftRaw : List Term := Proof.Events651.exact166816RawTerms
def rightRaw : List Term := Proof.Events650.exact166630RawTerms
def group : MergeGroup := .operator 166816 166630
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 166816) (leftOrdinal := 1)
    (rightResult := 166630) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30643⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30643⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30643⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge166821

namespace LeftMerge166829
def owner : Owner := ⟨.program ⟨257⟩, ⟨31071⟩⟩
def mergeEvent : Nat := 166829
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31069⟩⟩] } }
def leftRaw : List Term := Proof.Events651.exact166823RawTerms
def rightRaw : List Term := Proof.Events650.exact166546RawTerms
def group : MergeGroup := .operator 166823 166546
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 166823) (leftOrdinal := 0)
    (rightResult := 166546) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31069⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31069⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge166829

namespace LeftMerge166830
def owner : Owner := ⟨.program ⟨257⟩, ⟨31071⟩⟩
def mergeEvent : Nat := 166830
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31069⟩⟩] } }
def leftRaw : List Term := Proof.Events651.exact166823RawTerms
def rightRaw : List Term := Proof.Events650.exact166546RawTerms
def group : MergeGroup := .operator 166823 166546
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 166823) (leftOrdinal := 1)
    (rightResult := 166546) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31069⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31069⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge166830

namespace LeftMerge166832
def owner : Owner := ⟨.program ⟨257⟩, ⟨31071⟩⟩
def mergeEvent : Nat := 166832
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30277⟩⟩] } }
def rhsRaw : List Term := Proof.Events650.exact166543RawTerms
def group : MergeGroup := .relation 166831
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 166831) (rhsResult := 166543)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31069⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨31069⟩⟩) ⟨30277⟩ 166543) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30277⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29120⟩⟩], [⟨.program ⟨257⟩, ⟨30277⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge166832

namespace LeftMerge166846
def owner : Owner := ⟨.program ⟨257⟩, ⟨29919⟩⟩
def mergeEvent : Nat := 166846
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29916⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events651.exact166840RawTerms
def group : MergeGroup := .operator 163745 166840
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 166840) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨29916⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29916⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge166846

namespace LeftMerge166967
def owner : Owner := ⟨.program ⟨257⟩, ⟨30464⟩⟩
def mergeEvent : Nat := 166967
def frameStart : Nat := 166901
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events652.exact166963RawTerms
def rightRaw : List Term := Proof.Events652.exact166961RawTerms
def group : MergeGroup := .operator 166963 166961
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 166963) (leftOrdinal := 0)
    (rightResult := 166961) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29120⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge166967

namespace LeftMerge166979
def owner : Owner := ⟨.program ⟨257⟩, ⟨31070⟩⟩
def mergeEvent : Nat := 166979
def frameStart : Nat := 166901
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31069⟩⟩] } }
def leftRaw : List Term := Proof.Events652.exact166975RawTerms
def rightRaw : List Term := Proof.Events652.exact166952RawTerms
def group : MergeGroup := .operator 166975 166952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 166975) (leftOrdinal := 0)
    (rightResult := 166952) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31069⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31069⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge166979

namespace LeftMerge166980
def owner : Owner := ⟨.program ⟨257⟩, ⟨31070⟩⟩
def mergeEvent : Nat := 166980
def frameStart : Nat := 166901
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31069⟩⟩] } }
def leftRaw : List Term := Proof.Events652.exact166975RawTerms
def rightRaw : List Term := Proof.Events652.exact166952RawTerms
def group : MergeGroup := .operator 166975 166952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 166975) (leftOrdinal := 1)
    (rightResult := 166952) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31069⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31069⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge166980

namespace LeftMerge166982
def owner : Owner := ⟨.program ⟨257⟩, ⟨31070⟩⟩
def mergeEvent : Nat := 166982
def frameStart : Nat := 166901
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30277⟩⟩] } }
def rhsRaw : List Term := Proof.Events652.exact166949RawTerms
def group : MergeGroup := .relation 166981
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 166981) (rhsResult := 166949)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31069⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨31069⟩⟩) ⟨30277⟩ 166949) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30277⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29120⟩⟩], [⟨.program ⟨257⟩, ⟨30277⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge166982

namespace LeftMerge166990
def owner : Owner := ⟨.program ⟨257⟩, ⟨29352⟩⟩
def mergeEvent : Nat := 166990
def frameStart : Nat := 166901
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29351⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events652.exact166963RawTerms
def rightRaw : List Term := Proof.Events652.exact166986RawTerms
def group : MergeGroup := .operator 166963 166986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 166963) (leftOrdinal := 0)
    (rightResult := 166986) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29351⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29351⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge166990

namespace LeftMerge167007
def owner : Owner := ⟨.program ⟨257⟩, ⟨29919⟩⟩
def mergeEvent : Nat := 167007
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }
def rhsRaw : List Term := Proof.Events652.exact167004RawTerms
def group : MergeGroup := .relation 167006
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 167006) (rhsResult := 167004)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29916⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 167005 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29916⟩⟩]⟩) (none) 167004) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge167007

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
