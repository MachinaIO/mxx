import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge87893
def owner : Owner := ⟨.program ⟨257⟩, ⟨28437⟩⟩
def mergeEvent : Nat := 87893
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩] } }
def leftRaw : List Term := Proof.Events343.exact87886RawTerms
def rightRaw : List Term := Proof.Events061.exact15682RawTerms
def group : MergeGroup := .operator 87886 15682
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87886) (leftOrdinal := 1)
    (rightResult := 15682) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7169⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87893

namespace LeftMerge87895
def owner : Owner := ⟨.program ⟨257⟩, ⟨28437⟩⟩
def mergeEvent : Nat := 87895
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15675RawTerms
def group : MergeGroup := .relation 87894
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87894) (rhsResult := 15675)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7169⟩⟩) ⟨7050⟩ 15675) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87895

namespace LeftMerge87909
def owner : Owner := ⟨.program ⟨257⟩, ⟨70638⟩⟩
def mergeEvent : Nat := 87909
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩] } }
def leftRaw : List Term := Proof.Events312.exact80037RawTerms
def rightRaw : List Term := Proof.Events343.exact87903RawTerms
def group : MergeGroup := .operator 80037 87903
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80037) (leftOrdinal := 0)
    (rightResult := 87903) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70636⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87909

namespace LeftMerge87910
def owner : Owner := ⟨.program ⟨257⟩, ⟨70638⟩⟩
def mergeEvent : Nat := 87910
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩] } }
def leftRaw : List Term := Proof.Events312.exact80037RawTerms
def rightRaw : List Term := Proof.Events343.exact87903RawTerms
def group : MergeGroup := .operator 80037 87903
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80037) (leftOrdinal := 1)
    (rightResult := 87903) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70636⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87910

namespace LeftMerge87912
def owner : Owner := ⟨.program ⟨257⟩, ⟨70638⟩⟩
def mergeEvent : Nat := 87912
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68735⟩⟩] } }
def rhsRaw : List Term := Proof.Events343.exact87900RawTerms
def group : MergeGroup := .relation 87911
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87911) (rhsResult := 87900)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70636⟩⟩) ⟨68735⟩ 87900) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68735⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨68735⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87912

namespace LeftMerge87926
def owner : Owner := ⟨.program ⟨257⟩, ⟨68196⟩⟩
def mergeEvent : Nat := 87926
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68193⟩⟩] } }
def leftRaw : List Term := Proof.Events296.exact75995RawTerms
def rightRaw : List Term := Proof.Events343.exact87920RawTerms
def group : MergeGroup := .operator 75995 87920
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75995) (leftOrdinal := 0)
    (rightResult := 87920) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68193⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68193⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87926

namespace LeftMerge88047
def owner : Owner := ⟨.program ⟨257⟩, ⟨69033⟩⟩
def mergeEvent : Nat := 88047
def frameStart : Nat := 87981
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events343.exact88043RawTerms
def rightRaw : List Term := Proof.Events343.exact88041RawTerms
def group : MergeGroup := .operator 88043 88041
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 88043) (leftOrdinal := 0)
    (rightResult := 88041) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65836⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge88047

namespace LeftMerge88059
def owner : Owner := ⟨.program ⟨257⟩, ⟨70637⟩⟩
def mergeEvent : Nat := 88059
def frameStart : Nat := 87981
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩] } }
def leftRaw : List Term := Proof.Events343.exact88055RawTerms
def rightRaw : List Term := Proof.Events343.exact88032RawTerms
def group : MergeGroup := .operator 88055 88032
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 88055) (leftOrdinal := 0)
    (rightResult := 88032) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70636⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge88059

namespace LeftMerge88060
def owner : Owner := ⟨.program ⟨257⟩, ⟨70637⟩⟩
def mergeEvent : Nat := 88060
def frameStart : Nat := 87981
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩] } }
def leftRaw : List Term := Proof.Events343.exact88055RawTerms
def rightRaw : List Term := Proof.Events343.exact88032RawTerms
def group : MergeGroup := .operator 88055 88032
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 88055) (leftOrdinal := 1)
    (rightResult := 88032) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70636⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge88060

namespace LeftMerge88062
def owner : Owner := ⟨.program ⟨257⟩, ⟨70637⟩⟩
def mergeEvent : Nat := 88062
def frameStart : Nat := 87981
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68735⟩⟩] } }
def rhsRaw : List Term := Proof.Events343.exact88029RawTerms
def group : MergeGroup := .relation 88061
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 88061) (rhsResult := 88029)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70636⟩⟩) ⟨68735⟩ 88029) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68735⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨68735⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge88062

namespace LeftMerge88070
def owner : Owner := ⟨.program ⟨257⟩, ⟨67019⟩⟩
def mergeEvent : Nat := 88070
def frameStart : Nat := 87981
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨67008⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events343.exact88043RawTerms
def rightRaw : List Term := Proof.Events344.exact88066RawTerms
def group : MergeGroup := .operator 88043 88066
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 88043) (leftOrdinal := 0)
    (rightResult := 88066) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67008⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨67008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge88070

namespace LeftMerge88087
def owner : Owner := ⟨.program ⟨257⟩, ⟨68196⟩⟩
def mergeEvent : Nat := 88087
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7215⟩⟩] } }
def rhsRaw : List Term := Proof.Events344.exact88084RawTerms
def group : MergeGroup := .relation 88086
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 88086) (rhsResult := 88084)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68193⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 88085 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68193⟩⟩]⟩) (none) 88084) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7215⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge88087

namespace LeftMerge88088
def owner : Owner := ⟨.program ⟨257⟩, ⟨68196⟩⟩
def mergeEvent : Nat := 88088
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩] } }
def rhsRaw : List Term := Proof.Events344.exact88084RawTerms
def group : MergeGroup := .relation 88086
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 88086) (rhsResult := 88084)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68193⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 88085 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68193⟩⟩]⟩) (none) 88084) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge88088

namespace LeftMerge88089
def owner : Owner := ⟨.program ⟨257⟩, ⟨68196⟩⟩
def mergeEvent : Nat := 88089
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68735⟩⟩] } }
def rhsRaw : List Term := Proof.Events344.exact88084RawTerms
def group : MergeGroup := .relation 88086
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 88086) (rhsResult := 88084)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68193⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 88085 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68193⟩⟩]⟩) (none) 88084) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68735⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨68735⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge88089

namespace LeftMerge88090
def owner : Owner := ⟨.program ⟨257⟩, ⟨68196⟩⟩
def mergeEvent : Nat := 88090
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events344.exact88084RawTerms
def group : MergeGroup := .relation 88086
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 88086) (rhsResult := 88084)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68193⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 88085 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68193⟩⟩]⟩) (none) 88084) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67008⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge88090

namespace LeftMerge88095
def owner : Owner := ⟨.program ⟨257⟩, ⟨70639⟩⟩
def mergeEvent : Nat := 88095
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩] } }
def leftRaw : List Term := Proof.Events344.exact88091RawTerms
def rightRaw : List Term := Proof.Events343.exact87913RawTerms
def group : MergeGroup := .operator 88091 87913
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 88091) (leftOrdinal := 0)
    (rightResult := 87913) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge88095

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
