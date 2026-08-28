import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge140944
def owner : Owner := ⟨.program ⟨257⟩, ⟨52444⟩⟩
def mergeEvent : Nat := 140944
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51967⟩⟩] } }
def leftRaw : List Term := Proof.Events550.exact140940RawTerms
def rightRaw : List Term := Proof.Events549.exact140754RawTerms
def group : MergeGroup := .operator 140940 140754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140940) (leftOrdinal := 2)
    (rightResult := 140754) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51967⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51967⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], [⟨.program ⟨257⟩, ⟨51967⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge140944

namespace LeftMerge140945
def owner : Owner := ⟨.program ⟨257⟩, ⟨52444⟩⟩
def mergeEvent : Nat := 140945
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52442⟩⟩] } }
def leftRaw : List Term := Proof.Events550.exact140940RawTerms
def rightRaw : List Term := Proof.Events549.exact140754RawTerms
def group : MergeGroup := .operator 140940 140754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140940) (leftOrdinal := 1)
    (rightResult := 140754) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52442⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52442⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52442⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140945

namespace LeftMerge140953
def owner : Owner := ⟨.program ⟨257⟩, ⟨52737⟩⟩
def mergeEvent : Nat := 140953
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52735⟩⟩] } }
def leftRaw : List Term := Proof.Events550.exact140947RawTerms
def rightRaw : List Term := Proof.Events549.exact140670RawTerms
def group : MergeGroup := .operator 140947 140670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140947) (leftOrdinal := 0)
    (rightResult := 140670) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52735⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52735⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140953

namespace LeftMerge140954
def owner : Owner := ⟨.program ⟨257⟩, ⟨52737⟩⟩
def mergeEvent : Nat := 140954
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨50832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52735⟩⟩] } }
def leftRaw : List Term := Proof.Events550.exact140947RawTerms
def rightRaw : List Term := Proof.Events549.exact140670RawTerms
def group : MergeGroup := .operator 140947 140670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140947) (leftOrdinal := 1)
    (rightResult := 140670) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨50832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52735⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨50832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52735⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge140954

namespace LeftMerge140956
def owner : Owner := ⟨.program ⟨257⟩, ⟨52737⟩⟩
def mergeEvent : Nat := 140956
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨50832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52098⟩⟩] } }
def rhsRaw : List Term := Proof.Events549.exact140667RawTerms
def group : MergeGroup := .relation 140955
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 140955) (rhsResult := 140667)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨50832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52735⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52735⟩⟩) ⟨52098⟩ 140667) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52098⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨50832⟩⟩], [⟨.program ⟨257⟩, ⟨52098⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge140956

namespace LeftMerge140970
def owner : Owner := ⟨.program ⟨257⟩, ⟨51619⟩⟩
def mergeEvent : Nat := 140970
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51616⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134495RawTerms
def rightRaw : List Term := Proof.Events550.exact140964RawTerms
def group : MergeGroup := .operator 134495 140964
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134495) (leftOrdinal := 0)
    (rightResult := 140964) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51616⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140970

namespace LeftMerge141091
def owner : Owner := ⟨.program ⟨257⟩, ⟨52340⟩⟩
def mergeEvent : Nat := 141091
def frameStart : Nat := 141025
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events551.exact141087RawTerms
def rightRaw : List Term := Proof.Events551.exact141085RawTerms
def group : MergeGroup := .operator 141087 141085
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141087) (leftOrdinal := 0)
    (rightResult := 141085) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50832⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141091

namespace LeftMerge141103
def owner : Owner := ⟨.program ⟨257⟩, ⟨52736⟩⟩
def mergeEvent : Nat := 141103
def frameStart : Nat := 141025
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52735⟩⟩] } }
def leftRaw : List Term := Proof.Events551.exact141099RawTerms
def rightRaw : List Term := Proof.Events551.exact141076RawTerms
def group : MergeGroup := .operator 141099 141076
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141099) (leftOrdinal := 0)
    (rightResult := 141076) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52735⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52735⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141103

namespace LeftMerge141104
def owner : Owner := ⟨.program ⟨257⟩, ⟨52736⟩⟩
def mergeEvent : Nat := 141104
def frameStart : Nat := 141025
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52735⟩⟩] } }
def leftRaw : List Term := Proof.Events551.exact141099RawTerms
def rightRaw : List Term := Proof.Events551.exact141076RawTerms
def group : MergeGroup := .operator 141099 141076
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141099) (leftOrdinal := 1)
    (rightResult := 141076) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52735⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52735⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141104

namespace LeftMerge141106
def owner : Owner := ⟨.program ⟨257⟩, ⟨52736⟩⟩
def mergeEvent : Nat := 141106
def frameStart : Nat := 141025
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52098⟩⟩] } }
def rhsRaw : List Term := Proof.Events551.exact141073RawTerms
def group : MergeGroup := .relation 141105
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 141105) (rhsResult := 141073)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨50832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52735⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52735⟩⟩) ⟨52098⟩ 141073) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52098⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨50832⟩⟩], [⟨.program ⟨257⟩, ⟨52098⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141106

namespace LeftMerge141114
def owner : Owner := ⟨.program ⟨257⟩, ⟨51030⟩⟩
def mergeEvent : Nat := 141114
def frameStart : Nat := 141025
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events551.exact141087RawTerms
def rightRaw : List Term := Proof.Events551.exact141110RawTerms
def group : MergeGroup := .operator 141087 141110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141087) (leftOrdinal := 0)
    (rightResult := 141110) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51028⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141114

namespace LeftMerge141131
def owner : Owner := ⟨.program ⟨257⟩, ⟨51619⟩⟩
def mergeEvent : Nat := 141131
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }
def rhsRaw : List Term := Proof.Events551.exact141128RawTerms
def group : MergeGroup := .relation 141130
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 141130) (rhsResult := 141128)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51616⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 141129 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51616⟩⟩]⟩) (none) 141128) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141131

namespace LeftMerge141132
def owner : Owner := ⟨.program ⟨257⟩, ⟨51619⟩⟩
def mergeEvent : Nat := 141132
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52735⟩⟩] } }
def rhsRaw : List Term := Proof.Events551.exact141128RawTerms
def group : MergeGroup := .relation 141130
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 141130) (rhsResult := 141128)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51616⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 141129 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51616⟩⟩]⟩) (none) 141128) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52735⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52735⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141132

namespace LeftMerge141133
def owner : Owner := ⟨.program ⟨257⟩, ⟨51619⟩⟩
def mergeEvent : Nat := 141133
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨50832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52098⟩⟩] } }
def rhsRaw : List Term := Proof.Events551.exact141128RawTerms
def group : MergeGroup := .relation 141130
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 141130) (rhsResult := 141128)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51616⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 141129 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51616⟩⟩]⟩) (none) 141128) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52098⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨50832⟩⟩], [⟨.program ⟨257⟩, ⟨52098⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141133

namespace LeftMerge141134
def owner : Owner := ⟨.program ⟨257⟩, ⟨51619⟩⟩
def mergeEvent : Nat := 141134
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨51028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events551.exact141128RawTerms
def group : MergeGroup := .relation 141130
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 141130) (rhsResult := 141128)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51616⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 141129 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51616⟩⟩]⟩) (none) 141128) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141134

namespace LeftMerge141139
def owner : Owner := ⟨.program ⟨257⟩, ⟨52738⟩⟩
def mergeEvent : Nat := 141139
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52735⟩⟩] } }
def leftRaw : List Term := Proof.Events551.exact141135RawTerms
def rightRaw : List Term := Proof.Events550.exact140957RawTerms
def group : MergeGroup := .operator 141135 140957
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141135) (leftOrdinal := 0)
    (rightResult := 140957) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52735⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52735⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52735⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141139

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
