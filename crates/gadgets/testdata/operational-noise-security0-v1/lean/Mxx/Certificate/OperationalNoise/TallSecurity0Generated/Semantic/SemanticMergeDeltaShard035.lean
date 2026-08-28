import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge6535
def owner : Owner := ⟨.program ⟨214⟩, ⟨25779⟩⟩
def mergeEvent : Nat := 6535
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25778⟩⟩] } }
def leftRaw : List Term := Proof.Events025.exact6529RawTerms
def rightRaw : List Term := Proof.Events025.exact6439RawTerms
def group : MergeGroup := .operator 6529 6439
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6529) (leftOrdinal := 1)
    (rightResult := 6439) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25778⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25778⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge6535

namespace LeftMerge6537
def owner : Owner := ⟨.program ⟨214⟩, ⟨25779⟩⟩
def mergeEvent : Nat := 6537
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23424⟩⟩] } }
def rhsRaw : List Term := Proof.Events025.exact6436RawTerms
def group : MergeGroup := .relation 6536
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 6536) (rhsResult := 6436)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25778⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25778⟩⟩) ⟨23424⟩ 6436) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23424⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], [⟨.program ⟨214⟩, ⟨23424⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge6537

namespace LeftMerge6538
def owner : Owner := ⟨.program ⟨214⟩, ⟨25779⟩⟩
def mergeEvent : Nat := 6538
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25778⟩⟩] } }
def leftRaw : List Term := Proof.Events025.exact6529RawTerms
def rightRaw : List Term := Proof.Events025.exact6439RawTerms
def group : MergeGroup := .operator 6529 6439
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6529) (leftOrdinal := 0)
    (rightResult := 6439) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25778⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25778⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6538

namespace LeftMerge6554
def owner : Owner := ⟨.program ⟨214⟩, ⟨5564⟩⟩
def mergeEvent : Nat := 6554
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6314RawTerms
def rightRaw : List Term := Proof.Events025.exact6550RawTerms
def group : MergeGroup := .operator 6314 6550
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6314) (leftOrdinal := 0)
    (rightResult := 6550) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6554

namespace LeftMerge6567
def owner : Owner := ⟨.program ⟨214⟩, ⟨20267⟩⟩
def mergeEvent : Nat := 6567
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20264⟩⟩] } }
def leftRaw : List Term := Proof.Events025.exact6561RawTerms
def rightRaw : List Term := Proof.Events025.exact6546RawTerms
def group : MergeGroup := .operator 6561 6546
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6561) (leftOrdinal := 0)
    (rightResult := 6546) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20264⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20264⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6567

namespace LeftMerge6646
def owner : Owner := ⟨.program ⟨214⟩, ⟨13383⟩⟩
def mergeEvent : Nat := 6646
def frameStart : Nat := 6616
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events025.exact6642RawTerms
def rightRaw : List Term := Proof.Events025.exact6639RawTerms
def group : MergeGroup := .operator 6642 6639
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6642) (leftOrdinal := 0)
    (rightResult := 6639) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10365⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13382⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6646

namespace LeftMerge6676
def owner : Owner := ⟨.program ⟨214⟩, ⟨13464⟩⟩
def mergeEvent : Nat := 6676
def frameStart : Nat := 6616
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events026.exact6672RawTerms
def rightRaw : List Term := Proof.Events026.exact6670RawTerms
def group : MergeGroup := .operator 6672 6670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6672) (leftOrdinal := 0)
    (rightResult := 6670) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6676

namespace LeftMerge6699
def owner : Owner := ⟨.program ⟨214⟩, ⟨7884⟩⟩
def mergeEvent : Nat := 6699
def frameStart : Nat := 6616
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }
def leftRaw : List Term := Proof.Events026.exact6695RawTerms
def rightRaw : List Term := Proof.Events026.exact6692RawTerms
def group : MergeGroup := .operator 6695 6692
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6695) (leftOrdinal := 0)
    (rightResult := 6692) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7882⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6699

namespace LeftMerge6708
def owner : Owner := ⟨.program ⟨214⟩, ⟨25781⟩⟩
def mergeEvent : Nat := 6708
def frameStart : Nat := 6616
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25778⟩⟩] } }
def leftRaw : List Term := Proof.Events026.exact6704RawTerms
def rightRaw : List Term := Proof.Events026.exact6661RawTerms
def group : MergeGroup := .operator 6704 6661
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6704) (leftOrdinal := 1)
    (rightResult := 6661) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25778⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25778⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge6708

namespace LeftMerge6710
def owner : Owner := ⟨.program ⟨214⟩, ⟨25781⟩⟩
def mergeEvent : Nat := 6710
def frameStart : Nat := 6616
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23424⟩⟩] } }
def rhsRaw : List Term := Proof.Events026.exact6658RawTerms
def group : MergeGroup := .relation 6709
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 6709) (rhsResult := 6658)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25778⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25778⟩⟩) ⟨23424⟩ 6658) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23424⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], [⟨.program ⟨214⟩, ⟨23424⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge6710

namespace LeftMerge6711
def owner : Owner := ⟨.program ⟨214⟩, ⟨25781⟩⟩
def mergeEvent : Nat := 6711
def frameStart : Nat := 6616
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25778⟩⟩] } }
def leftRaw : List Term := Proof.Events026.exact6704RawTerms
def rightRaw : List Term := Proof.Events026.exact6661RawTerms
def group : MergeGroup := .operator 6704 6661
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6704) (leftOrdinal := 0)
    (rightResult := 6661) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25778⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25778⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6711

namespace LeftMerge6719
def owner : Owner := ⟨.program ⟨214⟩, ⟨17029⟩⟩
def mergeEvent : Nat := 6719
def frameStart : Nat := 6616
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events026.exact6672RawTerms
def rightRaw : List Term := Proof.Events026.exact6715RawTerms
def group : MergeGroup := .operator 6672 6715
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6672) (leftOrdinal := 0)
    (rightResult := 6715) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6719

namespace LeftMerge6736
def owner : Owner := ⟨.program ⟨214⟩, ⟨20267⟩⟩
def mergeEvent : Nat := 6736
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23424⟩⟩] } }
def rhsRaw : List Term := Proof.Events026.exact6733RawTerms
def group : MergeGroup := .relation 6735
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 6735) (rhsResult := 6733)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20264⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 6734 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20264⟩⟩]⟩) (none) 6733) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23424⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], [⟨.program ⟨214⟩, ⟨23424⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6736

namespace LeftMerge6737
def owner : Owner := ⟨.program ⟨214⟩, ⟨20267⟩⟩
def mergeEvent : Nat := 6737
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25778⟩⟩] } }
def rhsRaw : List Term := Proof.Events026.exact6733RawTerms
def group : MergeGroup := .relation 6735
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 6735) (rhsResult := 6733)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20264⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 6734 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20264⟩⟩]⟩) (none) 6733) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25778⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25778⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge6737

namespace LeftMerge6738
def owner : Owner := ⟨.program ⟨214⟩, ⟨20267⟩⟩
def mergeEvent : Nat := 6738
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events026.exact6733RawTerms
def group : MergeGroup := .relation 6735
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 6735) (rhsResult := 6733)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20264⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 6734 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20264⟩⟩]⟩) (none) 6733) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge6738

namespace LeftMerge6739
def owner : Owner := ⟨.program ⟨214⟩, ⟨20267⟩⟩
def mergeEvent : Nat := 6739
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }
def rhsRaw : List Term := Proof.Events026.exact6733RawTerms
def group : MergeGroup := .relation 6735
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 6735) (rhsResult := 6733)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20264⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 6734 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20264⟩⟩]⟩) (none) 6733) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6739

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
