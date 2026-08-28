import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge111661
def owner : Owner := ⟨.program ⟨257⟩, ⟨52533⟩⟩
def mergeEvent : Nat := 111661
def frameStart : Nat := 111566
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24542⟩⟩, ⟨.program ⟨257⟩, ⟨50572⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52015⟩⟩] } }
def rhsRaw : List Term := Proof.Events435.exact111608RawTerms
def group : MergeGroup := .relation 111660
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 111660) (rhsResult := 111608)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24542⟩⟩, ⟨.program ⟨257⟩, ⟨50572⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52530⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52530⟩⟩) ⟨52015⟩ 111608) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52015⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24542⟩⟩, ⟨.program ⟨257⟩, ⟨50572⟩⟩], [⟨.program ⟨257⟩, ⟨52015⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge111661

namespace LeftMerge111669
def owner : Owner := ⟨.program ⟨257⟩, ⟨50898⟩⟩
def mergeEvent : Nat := 111669
def frameStart : Nat := 111566
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events436.exact111622RawTerms
def rightRaw : List Term := Proof.Events436.exact111665RawTerms
def group : MergeGroup := .operator 111622 111665
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111622) (leftOrdinal := 0)
    (rightResult := 111665) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50896⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111669

namespace LeftMerge111686
def owner : Owner := ⟨.program ⟨257⟩, ⟨51462⟩⟩
def mergeEvent : Nat := 111686
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }
def rhsRaw : List Term := Proof.Events436.exact111683RawTerms
def group : MergeGroup := .relation 111685
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 111685) (rhsResult := 111683)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51459⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 111684 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51459⟩⟩]⟩) (none) 111683) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111686

namespace LeftMerge111687
def owner : Owner := ⟨.program ⟨257⟩, ⟨51462⟩⟩
def mergeEvent : Nat := 111687
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52530⟩⟩] } }
def rhsRaw : List Term := Proof.Events436.exact111683RawTerms
def group : MergeGroup := .relation 111685
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 111685) (rhsResult := 111683)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51459⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 111684 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51459⟩⟩]⟩) (none) 111683) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52530⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52530⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge111687

namespace LeftMerge111688
def owner : Owner := ⟨.program ⟨257⟩, ⟨51462⟩⟩
def mergeEvent : Nat := 111688
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24542⟩⟩, ⟨.program ⟨257⟩, ⟨50572⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52015⟩⟩] } }
def rhsRaw : List Term := Proof.Events436.exact111683RawTerms
def group : MergeGroup := .relation 111685
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 111685) (rhsResult := 111683)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51459⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 111684 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51459⟩⟩]⟩) (none) 111683) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24542⟩⟩, ⟨.program ⟨257⟩, ⟨50572⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52015⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24542⟩⟩, ⟨.program ⟨257⟩, ⟨50572⟩⟩], [⟨.program ⟨257⟩, ⟨52015⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111688

namespace LeftMerge111689
def owner : Owner := ⟨.program ⟨257⟩, ⟨51462⟩⟩
def mergeEvent : Nat := 111689
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨50896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events436.exact111683RawTerms
def group : MergeGroup := .relation 111685
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 111685) (rhsResult := 111683)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51459⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 111684 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51459⟩⟩]⟩) (none) 111683) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨50896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge111689

namespace LeftMerge111694
def owner : Owner := ⟨.program ⟨257⟩, ⟨52532⟩⟩
def mergeEvent : Nat := 111694
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24542⟩⟩, ⟨.program ⟨257⟩, ⟨50572⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52015⟩⟩] } }
def leftRaw : List Term := Proof.Events436.exact111690RawTerms
def rightRaw : List Term := Proof.Events435.exact111504RawTerms
def group : MergeGroup := .operator 111690 111504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111690) (leftOrdinal := 2)
    (rightResult := 111504) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24542⟩⟩, ⟨.program ⟨257⟩, ⟨50572⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52015⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24542⟩⟩, ⟨.program ⟨257⟩, ⟨50572⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52015⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24542⟩⟩, ⟨.program ⟨257⟩, ⟨50572⟩⟩], [⟨.program ⟨257⟩, ⟨52015⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge111694

namespace LeftMerge111695
def owner : Owner := ⟨.program ⟨257⟩, ⟨52532⟩⟩
def mergeEvent : Nat := 111695
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52530⟩⟩] } }
def leftRaw : List Term := Proof.Events436.exact111690RawTerms
def rightRaw : List Term := Proof.Events435.exact111504RawTerms
def group : MergeGroup := .operator 111690 111504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111690) (leftOrdinal := 1)
    (rightResult := 111504) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52530⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52530⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52530⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111695

namespace LeftMerge111703
def owner : Owner := ⟨.program ⟨257⟩, ⟨52985⟩⟩
def mergeEvent : Nat := 111703
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52983⟩⟩] } }
def leftRaw : List Term := Proof.Events436.exact111697RawTerms
def rightRaw : List Term := Proof.Events435.exact111420RawTerms
def group : MergeGroup := .operator 111697 111420
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111697) (leftOrdinal := 0)
    (rightResult := 111420) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52983⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52983⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111703

namespace LeftMerge111704
def owner : Owner := ⟨.program ⟨257⟩, ⟨52985⟩⟩
def mergeEvent : Nat := 111704
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨50896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52983⟩⟩] } }
def leftRaw : List Term := Proof.Events436.exact111697RawTerms
def rightRaw : List Term := Proof.Events435.exact111420RawTerms
def group : MergeGroup := .operator 111697 111420
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111697) (leftOrdinal := 1)
    (rightResult := 111420) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨50896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52983⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨50896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52983⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge111704

namespace LeftMerge111706
def owner : Owner := ⟨.program ⟨257⟩, ⟨52985⟩⟩
def mergeEvent : Nat := 111706
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨50896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52170⟩⟩] } }
def rhsRaw : List Term := Proof.Events435.exact111417RawTerms
def group : MergeGroup := .relation 111705
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 111705) (rhsResult := 111417)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨50896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52983⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52983⟩⟩) ⟨52170⟩ 111417) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52170⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨50896⟩⟩], [⟨.program ⟨257⟩, ⟨52170⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge111706

namespace LeftMerge111720
def owner : Owner := ⟨.program ⟨257⟩, ⟨51779⟩⟩
def mergeEvent : Nat := 111720
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51776⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105245RawTerms
def rightRaw : List Term := Proof.Events436.exact111714RawTerms
def group : MergeGroup := .operator 105245 111714
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105245) (leftOrdinal := 0)
    (rightResult := 111714) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51776⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51776⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111720

namespace LeftMerge111841
def owner : Owner := ⟨.program ⟨257⟩, ⟨52372⟩⟩
def mergeEvent : Nat := 111841
def frameStart : Nat := 111775
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events436.exact111837RawTerms
def rightRaw : List Term := Proof.Events436.exact111835RawTerms
def group : MergeGroup := .operator 111837 111835
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111837) (leftOrdinal := 0)
    (rightResult := 111835) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50896⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111841

namespace LeftMerge111853
def owner : Owner := ⟨.program ⟨257⟩, ⟨52984⟩⟩
def mergeEvent : Nat := 111853
def frameStart : Nat := 111775
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52983⟩⟩] } }
def leftRaw : List Term := Proof.Events436.exact111849RawTerms
def rightRaw : List Term := Proof.Events436.exact111826RawTerms
def group : MergeGroup := .operator 111849 111826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111849) (leftOrdinal := 0)
    (rightResult := 111826) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52983⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52983⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111853

namespace LeftMerge111854
def owner : Owner := ⟨.program ⟨257⟩, ⟨52984⟩⟩
def mergeEvent : Nat := 111854
def frameStart : Nat := 111775
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52983⟩⟩] } }
def leftRaw : List Term := Proof.Events436.exact111849RawTerms
def rightRaw : List Term := Proof.Events436.exact111826RawTerms
def group : MergeGroup := .operator 111849 111826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111849) (leftOrdinal := 1)
    (rightResult := 111826) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52983⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52983⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge111854

namespace LeftMerge111856
def owner : Owner := ⟨.program ⟨257⟩, ⟨52984⟩⟩
def mergeEvent : Nat := 111856
def frameStart : Nat := 111775
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52170⟩⟩] } }
def rhsRaw : List Term := Proof.Events436.exact111823RawTerms
def group : MergeGroup := .relation 111855
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 111855) (rhsResult := 111823)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨50896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52983⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52983⟩⟩) ⟨52170⟩ 111823) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52170⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨50896⟩⟩], [⟨.program ⟨257⟩, ⟨52170⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge111856

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
