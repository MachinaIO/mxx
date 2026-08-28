import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge145773
def owner : Owner := ⟨.program ⟨257⟩, ⟨36450⟩⟩
def mergeEvent : Nat := 145773
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩] } }
def leftRaw : List Term := Proof.Events535.exact137091RawTerms
def rightRaw : List Term := Proof.Events569.exact145767RawTerms
def group : MergeGroup := .operator 137091 145767
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 137091) (leftOrdinal := 0)
    (rightResult := 145767) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36448⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge145773

namespace LeftMerge145774
def owner : Owner := ⟨.program ⟨257⟩, ⟨36450⟩⟩
def mergeEvent : Nat := 145774
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩] } }
def leftRaw : List Term := Proof.Events535.exact137091RawTerms
def rightRaw : List Term := Proof.Events569.exact145767RawTerms
def group : MergeGroup := .operator 137091 145767
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 137091) (leftOrdinal := 1)
    (rightResult := 145767) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36448⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge145774

namespace LeftMerge145776
def owner : Owner := ⟨.program ⟨257⟩, ⟨36450⟩⟩
def mergeEvent : Nat := 145776
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35837⟩⟩] } }
def rhsRaw : List Term := Proof.Events569.exact145764RawTerms
def group : MergeGroup := .relation 145775
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 145775) (rhsResult := 145764)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36448⟩⟩) ⟨35837⟩ 145764) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35837⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨35837⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge145776

namespace LeftMerge145790
def owner : Owner := ⟨.program ⟨257⟩, ⟨35355⟩⟩
def mergeEvent : Nat := 145790
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35352⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134495RawTerms
def rightRaw : List Term := Proof.Events569.exact145784RawTerms
def group : MergeGroup := .operator 134495 145784
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134495) (leftOrdinal := 0)
    (rightResult := 145784) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35352⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35352⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge145790

namespace LeftMerge145911
def owner : Owner := ⟨.program ⟨257⟩, ⟨36080⟩⟩
def mergeEvent : Nat := 145911
def frameStart : Nat := 145845
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34692⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events569.exact145907RawTerms
def rightRaw : List Term := Proof.Events569.exact145905RawTerms
def group : MergeGroup := .operator 145907 145905
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 145907) (leftOrdinal := 0)
    (rightResult := 145905) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34692⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge145911

namespace LeftMerge145923
def owner : Owner := ⟨.program ⟨257⟩, ⟨36449⟩⟩
def mergeEvent : Nat := 145923
def frameStart : Nat := 145845
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩] } }
def leftRaw : List Term := Proof.Events569.exact145919RawTerms
def rightRaw : List Term := Proof.Events569.exact145896RawTerms
def group : MergeGroup := .operator 145919 145896
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 145919) (leftOrdinal := 0)
    (rightResult := 145896) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36448⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge145923

namespace LeftMerge145924
def owner : Owner := ⟨.program ⟨257⟩, ⟨36449⟩⟩
def mergeEvent : Nat := 145924
def frameStart : Nat := 145845
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34692⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩] } }
def leftRaw : List Term := Proof.Events569.exact145919RawTerms
def rightRaw : List Term := Proof.Events569.exact145896RawTerms
def group : MergeGroup := .operator 145919 145896
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 145919) (leftOrdinal := 1)
    (rightResult := 145896) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34692⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36448⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge145924

namespace LeftMerge145926
def owner : Owner := ⟨.program ⟨257⟩, ⟨36449⟩⟩
def mergeEvent : Nat := 145926
def frameStart : Nat := 145845
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34692⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35837⟩⟩] } }
def rhsRaw : List Term := Proof.Events569.exact145893RawTerms
def group : MergeGroup := .relation 145925
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 145925) (rhsResult := 145893)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36448⟩⟩) ⟨35837⟩ 145893) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35837⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨35837⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge145926

namespace LeftMerge145934
def owner : Owner := ⟨.program ⟨257⟩, ⟨34870⟩⟩
def mergeEvent : Nat := 145934
def frameStart : Nat := 145845
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events569.exact145907RawTerms
def rightRaw : List Term := Proof.Events570.exact145930RawTerms
def group : MergeGroup := .operator 145907 145930
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 145907) (leftOrdinal := 0)
    (rightResult := 145930) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34868⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge145934

namespace LeftMerge145951
def owner : Owner := ⟨.program ⟨257⟩, ⟨35355⟩⟩
def mergeEvent : Nat := 145951
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7221⟩⟩] } }
def rhsRaw : List Term := Proof.Events570.exact145948RawTerms
def group : MergeGroup := .relation 145950
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 145950) (rhsResult := 145948)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35352⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 145949 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35352⟩⟩]⟩) (none) 145948) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7221⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge145951

namespace LeftMerge145952
def owner : Owner := ⟨.program ⟨257⟩, ⟨35355⟩⟩
def mergeEvent : Nat := 145952
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩] } }
def rhsRaw : List Term := Proof.Events570.exact145948RawTerms
def group : MergeGroup := .relation 145950
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 145950) (rhsResult := 145948)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35352⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 145949 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35352⟩⟩]⟩) (none) 145948) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge145952

namespace LeftMerge145953
def owner : Owner := ⟨.program ⟨257⟩, ⟨35355⟩⟩
def mergeEvent : Nat := 145953
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35837⟩⟩] } }
def rhsRaw : List Term := Proof.Events570.exact145948RawTerms
def group : MergeGroup := .relation 145950
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 145950) (rhsResult := 145948)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35352⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 145949 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35352⟩⟩]⟩) (none) 145948) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34692⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35837⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨35837⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge145953

namespace LeftMerge145954
def owner : Owner := ⟨.program ⟨257⟩, ⟨35355⟩⟩
def mergeEvent : Nat := 145954
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events570.exact145948RawTerms
def group : MergeGroup := .relation 145950
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 145950) (rhsResult := 145948)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35352⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 145949 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35352⟩⟩]⟩) (none) 145948) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge145954

namespace LeftMerge145959
def owner : Owner := ⟨.program ⟨257⟩, ⟨36451⟩⟩
def mergeEvent : Nat := 145959
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩] } }
def leftRaw : List Term := Proof.Events570.exact145955RawTerms
def rightRaw : List Term := Proof.Events569.exact145777RawTerms
def group : MergeGroup := .operator 145955 145777
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 145955) (leftOrdinal := 0)
    (rightResult := 145777) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge145959

namespace LeftMerge145960
def owner : Owner := ⟨.program ⟨257⟩, ⟨36451⟩⟩
def mergeEvent : Nat := 145960
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35837⟩⟩] } }
def leftRaw : List Term := Proof.Events570.exact145955RawTerms
def rightRaw : List Term := Proof.Events569.exact145777RawTerms
def group : MergeGroup := .operator 145955 145777
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 145955) (leftOrdinal := 2)
    (rightResult := 145777) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35837⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35837⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨35837⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge145960

namespace LeftMerge145968
def owner : Owner := ⟨.program ⟨257⟩, ⟨36452⟩⟩
def mergeEvent : Nat := 145968
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩] } }
def leftRaw : List Term := Proof.Events570.exact145962RawTerms
def rightRaw : List Term := Proof.Events061.exact15642RawTerms
def group : MergeGroup := .operator 145962 15642
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 145962) (leftOrdinal := 0)
    (rightResult := 15642) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7221⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7163⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge145968

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
