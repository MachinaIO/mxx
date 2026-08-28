import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge139803
def owner : Owner := ⟨.program ⟨257⟩, ⟨57342⟩⟩
def mergeEvent : Nat := 139803
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57339⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134495RawTerms
def rightRaw : List Term := Proof.Events546.exact139797RawTerms
def group : MergeGroup := .operator 134495 139797
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134495) (leftOrdinal := 0)
    (rightResult := 139797) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57339⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57339⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139803

namespace LeftMerge139882
def owner : Owner := ⟨.program ⟨257⟩, ⟨56317⟩⟩
def mergeEvent : Nat := 139882
def frameStart : Nat := 139852
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24926⟩⟩, ⟨.program ⟨257⟩, ⟨56316⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events546.exact139878RawTerms
def rightRaw : List Term := Proof.Events546.exact139875RawTerms
def group : MergeGroup := .operator 139878 139875
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139878) (leftOrdinal := 0)
    (rightResult := 139875) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56316⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24926⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24926⟩⟩, ⟨.program ⟨257⟩, ⟨56316⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139882

namespace LeftMerge139912
def owner : Owner := ⟨.program ⟨257⟩, ⟨58220⟩⟩
def mergeEvent : Nat := 139912
def frameStart : Nat := 139852
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24926⟩⟩, ⟨.program ⟨257⟩, ⟨56316⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events546.exact139908RawTerms
def rightRaw : List Term := Proof.Events546.exact139906RawTerms
def group : MergeGroup := .operator 139908 139906
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139908) (leftOrdinal := 0)
    (rightResult := 139906) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24926⟩⟩, ⟨.program ⟨257⟩, ⟨56316⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24926⟩⟩, ⟨.program ⟨257⟩, ⟨56316⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139912

namespace LeftMerge139935
def owner : Owner := ⟨.program ⟨257⟩, ⟨9534⟩⟩
def mergeEvent : Nat := 139935
def frameStart : Nat := 139852
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }
def leftRaw : List Term := Proof.Events546.exact139931RawTerms
def rightRaw : List Term := Proof.Events546.exact139928RawTerms
def group : MergeGroup := .operator 139931 139928
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139931) (leftOrdinal := 0)
    (rightResult := 139928) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9532⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139935

namespace LeftMerge139944
def owner : Owner := ⟨.program ⟨257⟩, ⟨58405⟩⟩
def mergeEvent : Nat := 139944
def frameStart : Nat := 139852
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58402⟩⟩] } }
def leftRaw : List Term := Proof.Events546.exact139940RawTerms
def rightRaw : List Term := Proof.Events546.exact139897RawTerms
def group : MergeGroup := .operator 139940 139897
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139940) (leftOrdinal := 0)
    (rightResult := 139897) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58402⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58402⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139944

namespace LeftMerge139945
def owner : Owner := ⟨.program ⟨257⟩, ⟨58405⟩⟩
def mergeEvent : Nat := 139945
def frameStart : Nat := 139852
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24926⟩⟩, ⟨.program ⟨257⟩, ⟨56316⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58402⟩⟩] } }
def leftRaw : List Term := Proof.Events546.exact139940RawTerms
def rightRaw : List Term := Proof.Events546.exact139897RawTerms
def group : MergeGroup := .operator 139940 139897
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139940) (leftOrdinal := 1)
    (rightResult := 139897) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24926⟩⟩, ⟨.program ⟨257⟩, ⟨56316⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58402⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24926⟩⟩, ⟨.program ⟨257⟩, ⟨56316⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58402⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge139945

namespace LeftMerge139947
def owner : Owner := ⟨.program ⟨257⟩, ⟨58405⟩⟩
def mergeEvent : Nat := 139947
def frameStart : Nat := 139852
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24926⟩⟩, ⟨.program ⟨257⟩, ⟨56316⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57927⟩⟩] } }
def rhsRaw : List Term := Proof.Events546.exact139894RawTerms
def group : MergeGroup := .relation 139946
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 139946) (rhsResult := 139894)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24926⟩⟩, ⟨.program ⟨257⟩, ⟨56316⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58402⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58402⟩⟩) ⟨57927⟩ 139894) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57927⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24926⟩⟩, ⟨.program ⟨257⟩, ⟨56316⟩⟩], [⟨.program ⟨257⟩, ⟨57927⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge139947

namespace LeftMerge139955
def owner : Owner := ⟨.program ⟨257⟩, ⟨56794⟩⟩
def mergeEvent : Nat := 139955
def frameStart : Nat := 139852
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56792⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events546.exact139908RawTerms
def rightRaw : List Term := Proof.Events546.exact139951RawTerms
def group : MergeGroup := .operator 139908 139951
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139908) (leftOrdinal := 0)
    (rightResult := 139951) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56792⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56792⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139955

namespace LeftMerge139972
def owner : Owner := ⟨.program ⟨257⟩, ⟨57342⟩⟩
def mergeEvent : Nat := 139972
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }
def rhsRaw : List Term := Proof.Events546.exact139969RawTerms
def group : MergeGroup := .relation 139971
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 139971) (rhsResult := 139969)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57339⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 139970 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57339⟩⟩]⟩) (none) 139969) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139972

namespace LeftMerge139973
def owner : Owner := ⟨.program ⟨257⟩, ⟨57342⟩⟩
def mergeEvent : Nat := 139973
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58402⟩⟩] } }
def rhsRaw : List Term := Proof.Events546.exact139969RawTerms
def group : MergeGroup := .relation 139971
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 139971) (rhsResult := 139969)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57339⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 139970 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57339⟩⟩]⟩) (none) 139969) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58402⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58402⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge139973

namespace LeftMerge139974
def owner : Owner := ⟨.program ⟨257⟩, ⟨57342⟩⟩
def mergeEvent : Nat := 139974
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24926⟩⟩, ⟨.program ⟨257⟩, ⟨56316⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57927⟩⟩] } }
def rhsRaw : List Term := Proof.Events546.exact139969RawTerms
def group : MergeGroup := .relation 139971
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 139971) (rhsResult := 139969)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57339⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 139970 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57339⟩⟩]⟩) (none) 139969) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24926⟩⟩, ⟨.program ⟨257⟩, ⟨56316⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57927⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24926⟩⟩, ⟨.program ⟨257⟩, ⟨56316⟩⟩], [⟨.program ⟨257⟩, ⟨57927⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139974

namespace LeftMerge139975
def owner : Owner := ⟨.program ⟨257⟩, ⟨57342⟩⟩
def mergeEvent : Nat := 139975
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56792⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events546.exact139969RawTerms
def group : MergeGroup := .relation 139971
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 139971) (rhsResult := 139969)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57339⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 139970 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57339⟩⟩]⟩) (none) 139969) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56792⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56792⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge139975

namespace LeftMerge139980
def owner : Owner := ⟨.program ⟨257⟩, ⟨58404⟩⟩
def mergeEvent : Nat := 139980
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24926⟩⟩, ⟨.program ⟨257⟩, ⟨56316⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57927⟩⟩] } }
def leftRaw : List Term := Proof.Events546.exact139976RawTerms
def rightRaw : List Term := Proof.Events546.exact139790RawTerms
def group : MergeGroup := .operator 139976 139790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139976) (leftOrdinal := 2)
    (rightResult := 139790) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24926⟩⟩, ⟨.program ⟨257⟩, ⟨56316⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57927⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24926⟩⟩, ⟨.program ⟨257⟩, ⟨56316⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57927⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24926⟩⟩, ⟨.program ⟨257⟩, ⟨56316⟩⟩], [⟨.program ⟨257⟩, ⟨57927⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge139980

namespace LeftMerge139981
def owner : Owner := ⟨.program ⟨257⟩, ⟨58404⟩⟩
def mergeEvent : Nat := 139981
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58402⟩⟩] } }
def leftRaw : List Term := Proof.Events546.exact139976RawTerms
def rightRaw : List Term := Proof.Events546.exact139790RawTerms
def group : MergeGroup := .operator 139976 139790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139976) (leftOrdinal := 1)
    (rightResult := 139790) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58402⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58402⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58402⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139981

namespace LeftMerge139989
def owner : Owner := ⟨.program ⟨257⟩, ⟨58697⟩⟩
def mergeEvent : Nat := 139989
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58695⟩⟩] } }
def leftRaw : List Term := Proof.Events546.exact139983RawTerms
def rightRaw : List Term := Proof.Events545.exact139706RawTerms
def group : MergeGroup := .operator 139983 139706
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139983) (leftOrdinal := 0)
    (rightResult := 139706) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58695⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58695⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139989

namespace LeftMerge139990
def owner : Owner := ⟨.program ⟨257⟩, ⟨58697⟩⟩
def mergeEvent : Nat := 139990
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56792⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58695⟩⟩] } }
def leftRaw : List Term := Proof.Events546.exact139983RawTerms
def rightRaw : List Term := Proof.Events545.exact139706RawTerms
def group : MergeGroup := .operator 139983 139706
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139983) (leftOrdinal := 1)
    (rightResult := 139706) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56792⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58695⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56792⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58695⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge139990

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
