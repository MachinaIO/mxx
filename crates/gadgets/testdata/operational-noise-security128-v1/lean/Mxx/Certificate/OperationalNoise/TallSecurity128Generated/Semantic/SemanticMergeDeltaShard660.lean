import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge109574
def owner : Owner := ⟨.program ⟨257⟩, ⟨64451⟩⟩
def mergeEvent : Nat := 109574
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63935⟩⟩] } }
def rhsRaw : List Term := Proof.Events427.exact109499RawTerms
def group : MergeGroup := .relation 109573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 109573) (rhsResult := 109499)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64450⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64450⟩⟩) ⟨63935⟩ 109499) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63935⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], [⟨.program ⟨257⟩, ⟨63935⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge109574

namespace LeftMerge109575
def owner : Owner := ⟨.program ⟨257⟩, ⟨64451⟩⟩
def mergeEvent : Nat := 109575
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64450⟩⟩] } }
def leftRaw : List Term := Proof.Events427.exact109566RawTerms
def rightRaw : List Term := Proof.Events427.exact109502RawTerms
def group : MergeGroup := .operator 109566 109502
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 109566) (leftOrdinal := 0)
    (rightResult := 109502) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64450⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64450⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge109575

namespace LeftMerge109589
def owner : Owner := ⟨.program ⟨257⟩, ⟨63382⟩⟩
def mergeEvent : Nat := 109589
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63379⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105245RawTerms
def rightRaw : List Term := Proof.Events428.exact109583RawTerms
def group : MergeGroup := .operator 105245 109583
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105245) (leftOrdinal := 0)
    (rightResult := 109583) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63379⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63379⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge109589

namespace LeftMerge109668
def owner : Owner := ⟨.program ⟨257⟩, ⟨62493⟩⟩
def mergeEvent : Nat := 109668
def frameStart : Nat := 109638
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events428.exact109664RawTerms
def rightRaw : List Term := Proof.Events428.exact109661RawTerms
def group : MergeGroup := .operator 109664 109661
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 109664) (leftOrdinal := 0)
    (rightResult := 109661) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62492⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25502⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge109668

namespace LeftMerge109698
def owner : Owner := ⟨.program ⟨257⟩, ⟨64212⟩⟩
def mergeEvent : Nat := 109698
def frameStart : Nat := 109638
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events428.exact109694RawTerms
def rightRaw : List Term := Proof.Events428.exact109692RawTerms
def group : MergeGroup := .operator 109694 109692
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 109694) (leftOrdinal := 0)
    (rightResult := 109692) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge109698

namespace LeftMerge109721
def owner : Owner := ⟨.program ⟨257⟩, ⟨9540⟩⟩
def mergeEvent : Nat := 109721
def frameStart : Nat := 109638
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }
def leftRaw : List Term := Proof.Events428.exact109717RawTerms
def rightRaw : List Term := Proof.Events428.exact109714RawTerms
def group : MergeGroup := .operator 109717 109714
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 109717) (leftOrdinal := 0)
    (rightResult := 109714) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9538⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge109721

namespace LeftMerge109730
def owner : Owner := ⟨.program ⟨257⟩, ⟨64453⟩⟩
def mergeEvent : Nat := 109730
def frameStart : Nat := 109638
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64450⟩⟩] } }
def leftRaw : List Term := Proof.Events428.exact109726RawTerms
def rightRaw : List Term := Proof.Events428.exact109683RawTerms
def group : MergeGroup := .operator 109726 109683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 109726) (leftOrdinal := 0)
    (rightResult := 109683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64450⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64450⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge109730

namespace LeftMerge109731
def owner : Owner := ⟨.program ⟨257⟩, ⟨64453⟩⟩
def mergeEvent : Nat := 109731
def frameStart : Nat := 109638
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64450⟩⟩] } }
def leftRaw : List Term := Proof.Events428.exact109726RawTerms
def rightRaw : List Term := Proof.Events428.exact109683RawTerms
def group : MergeGroup := .operator 109726 109683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 109726) (leftOrdinal := 1)
    (rightResult := 109683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64450⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64450⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge109731

namespace LeftMerge109733
def owner : Owner := ⟨.program ⟨257⟩, ⟨64453⟩⟩
def mergeEvent : Nat := 109733
def frameStart : Nat := 109638
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63935⟩⟩] } }
def rhsRaw : List Term := Proof.Events428.exact109680RawTerms
def group : MergeGroup := .relation 109732
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 109732) (rhsResult := 109680)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64450⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64450⟩⟩) ⟨63935⟩ 109680) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63935⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], [⟨.program ⟨257⟩, ⟨63935⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge109733

namespace LeftMerge109741
def owner : Owner := ⟨.program ⟨257⟩, ⟨62818⟩⟩
def mergeEvent : Nat := 109741
def frameStart : Nat := 109638
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62816⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events428.exact109694RawTerms
def rightRaw : List Term := Proof.Events428.exact109737RawTerms
def group : MergeGroup := .operator 109694 109737
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 109694) (leftOrdinal := 0)
    (rightResult := 109737) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62816⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62816⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge109741

namespace LeftMerge109758
def owner : Owner := ⟨.program ⟨257⟩, ⟨63382⟩⟩
def mergeEvent : Nat := 109758
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }
def rhsRaw : List Term := Proof.Events428.exact109755RawTerms
def group : MergeGroup := .relation 109757
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 109757) (rhsResult := 109755)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 109756 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63379⟩⟩]⟩) (none) 109755) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge109758

namespace LeftMerge109759
def owner : Owner := ⟨.program ⟨257⟩, ⟨63382⟩⟩
def mergeEvent : Nat := 109759
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64450⟩⟩] } }
def rhsRaw : List Term := Proof.Events428.exact109755RawTerms
def group : MergeGroup := .relation 109757
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 109757) (rhsResult := 109755)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 109756 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63379⟩⟩]⟩) (none) 109755) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64450⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64450⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge109759

namespace LeftMerge109760
def owner : Owner := ⟨.program ⟨257⟩, ⟨63382⟩⟩
def mergeEvent : Nat := 109760
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63935⟩⟩] } }
def rhsRaw : List Term := Proof.Events428.exact109755RawTerms
def group : MergeGroup := .relation 109757
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 109757) (rhsResult := 109755)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 109756 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63379⟩⟩]⟩) (none) 109755) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63935⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], [⟨.program ⟨257⟩, ⟨63935⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge109760

namespace LeftMerge109761
def owner : Owner := ⟨.program ⟨257⟩, ⟨63382⟩⟩
def mergeEvent : Nat := 109761
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨62816⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events428.exact109755RawTerms
def group : MergeGroup := .relation 109757
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 109757) (rhsResult := 109755)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 109756 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63379⟩⟩]⟩) (none) 109755) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62816⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨62816⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge109761

namespace LeftMerge109766
def owner : Owner := ⟨.program ⟨257⟩, ⟨64452⟩⟩
def mergeEvent : Nat := 109766
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63935⟩⟩] } }
def leftRaw : List Term := Proof.Events428.exact109762RawTerms
def rightRaw : List Term := Proof.Events428.exact109576RawTerms
def group : MergeGroup := .operator 109762 109576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 109762) (leftOrdinal := 2)
    (rightResult := 109576) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63935⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63935⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25502⟩⟩, ⟨.program ⟨257⟩, ⟨62492⟩⟩], [⟨.program ⟨257⟩, ⟨63935⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge109766

namespace LeftMerge109767
def owner : Owner := ⟨.program ⟨257⟩, ⟨64452⟩⟩
def mergeEvent : Nat := 109767
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64450⟩⟩] } }
def leftRaw : List Term := Proof.Events428.exact109762RawTerms
def rightRaw : List Term := Proof.Events428.exact109576RawTerms
def group : MergeGroup := .operator 109762 109576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 109762) (leftOrdinal := 1)
    (rightResult := 109576) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64450⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64450⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64450⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge109767

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
