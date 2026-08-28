import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge84037
def owner : Owner := ⟨.program ⟨257⟩, ⟨20092⟩⟩
def mergeEvent : Nat := 84037
def frameStart : Nat := 83971
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18636⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events328.exact84033RawTerms
def rightRaw : List Term := Proof.Events328.exact84031RawTerms
def group : MergeGroup := .operator 84033 84031
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84033) (leftOrdinal := 0)
    (rightResult := 84031) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18636⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84037

namespace LeftMerge84049
def owner : Owner := ⟨.program ⟨257⟩, ⟨20839⟩⟩
def mergeEvent : Nat := 84049
def frameStart : Nat := 83971
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩] } }
def leftRaw : List Term := Proof.Events328.exact84045RawTerms
def rightRaw : List Term := Proof.Events328.exact84022RawTerms
def group : MergeGroup := .operator 84045 84022
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84045) (leftOrdinal := 0)
    (rightResult := 84022) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20838⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84049

namespace LeftMerge84050
def owner : Owner := ⟨.program ⟨257⟩, ⟨20839⟩⟩
def mergeEvent : Nat := 84050
def frameStart : Nat := 83971
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18636⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩] } }
def leftRaw : List Term := Proof.Events328.exact84045RawTerms
def rightRaw : List Term := Proof.Events328.exact84022RawTerms
def group : MergeGroup := .operator 84045 84022
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84045) (leftOrdinal := 1)
    (rightResult := 84022) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18636⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20838⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84050

namespace LeftMerge84052
def owner : Owner := ⟨.program ⟨257⟩, ⟨20839⟩⟩
def mergeEvent : Nat := 84052
def frameStart : Nat := 83971
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18636⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19915⟩⟩] } }
def rhsRaw : List Term := Proof.Events328.exact84019RawTerms
def group : MergeGroup := .relation 84051
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84051) (rhsResult := 84019)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20838⟩⟩) ⟨19915⟩ 84019) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19915⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨19915⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84052

namespace LeftMerge84060
def owner : Owner := ⟨.program ⟨257⟩, ⟨18982⟩⟩
def mergeEvent : Nat := 84060
def frameStart : Nat := 83971
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18980⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events328.exact84033RawTerms
def rightRaw : List Term := Proof.Events328.exact84056RawTerms
def group : MergeGroup := .operator 84033 84056
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84033) (leftOrdinal := 0)
    (rightResult := 84056) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18980⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84060

namespace LeftMerge84077
def owner : Owner := ⟨.program ⟨257⟩, ⟨19579⟩⟩
def mergeEvent : Nat := 84077
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }
def rhsRaw : List Term := Proof.Events328.exact84074RawTerms
def group : MergeGroup := .relation 84076
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84076) (rhsResult := 84074)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19576⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 84075 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19576⟩⟩]⟩) (none) 84074) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84077

namespace LeftMerge84078
def owner : Owner := ⟨.program ⟨257⟩, ⟨19579⟩⟩
def mergeEvent : Nat := 84078
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩] } }
def rhsRaw : List Term := Proof.Events328.exact84074RawTerms
def group : MergeGroup := .relation 84076
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84076) (rhsResult := 84074)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19576⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 84075 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19576⟩⟩]⟩) (none) 84074) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84078

namespace LeftMerge84079
def owner : Owner := ⟨.program ⟨257⟩, ⟨19579⟩⟩
def mergeEvent : Nat := 84079
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18636⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19915⟩⟩] } }
def rhsRaw : List Term := Proof.Events328.exact84074RawTerms
def group : MergeGroup := .relation 84076
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84076) (rhsResult := 84074)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19576⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 84075 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19576⟩⟩]⟩) (none) 84074) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18636⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19915⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨19915⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84079

namespace LeftMerge84080
def owner : Owner := ⟨.program ⟨257⟩, ⟨19579⟩⟩
def mergeEvent : Nat := 84080
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events328.exact84074RawTerms
def group : MergeGroup := .relation 84076
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84076) (rhsResult := 84074)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19576⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 84075 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19576⟩⟩]⟩) (none) 84074) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18980⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84080

namespace LeftMerge84085
def owner : Owner := ⟨.program ⟨257⟩, ⟨20841⟩⟩
def mergeEvent : Nat := 84085
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩] } }
def leftRaw : List Term := Proof.Events328.exact84081RawTerms
def rightRaw : List Term := Proof.Events327.exact83903RawTerms
def group : MergeGroup := .operator 84081 83903
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84081) (leftOrdinal := 0)
    (rightResult := 83903) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84085

namespace LeftMerge84086
def owner : Owner := ⟨.program ⟨257⟩, ⟨20841⟩⟩
def mergeEvent : Nat := 84086
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18636⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19915⟩⟩] } }
def leftRaw : List Term := Proof.Events328.exact84081RawTerms
def rightRaw : List Term := Proof.Events327.exact83903RawTerms
def group : MergeGroup := .operator 84081 83903
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84081) (leftOrdinal := 2)
    (rightResult := 83903) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18636⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19915⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18636⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19915⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨19915⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84086

namespace LeftMerge84112
def owner : Owner := ⟨.program ⟨257⟩, ⟨15621⟩⟩
def mergeEvent : Nat := 84112
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events013.exact3477RawTerms
def rightRaw : List Term := Proof.Events296.exact75903RawTerms
def group : MergeGroup := .operator 3477 75903
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3477) (leftOrdinal := 0)
    (rightResult := 75903) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15618⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84112

namespace LeftMerge84117
def owner : Owner := ⟨.program ⟨257⟩, ⟨10362⟩⟩
def mergeEvent : Nat := 84117
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } }
def leftRaw : List Term := Proof.Events295.exact75773RawTerms
def rightRaw : List Term := Proof.Events099.exact25597RawTerms
def group : MergeGroup := .operator 75773 25597
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75773) (leftOrdinal := 0)
    (rightResult := 25597) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84117

namespace LeftMerge84134
def owner : Owner := ⟨.program ⟨257⟩, ⟨15624⟩⟩
def mergeEvent : Nat := 84134
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events328.exact84128RawTerms
def rightRaw : List Term := Proof.Events013.exact3480RawTerms
def group : MergeGroup := .operator 84128 3480
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84128) (leftOrdinal := 1)
    (rightResult := 3480) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12471⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84134

namespace LeftMerge84135
def owner : Owner := ⟨.program ⟨257⟩, ⟨15624⟩⟩
def mergeEvent : Nat := 84135
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } }
def leftRaw : List Term := Proof.Events328.exact84128RawTerms
def rightRaw : List Term := Proof.Events013.exact3480RawTerms
def group : MergeGroup := .operator 84128 3480
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84128) (leftOrdinal := 0)
    (rightResult := 3480) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12471⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84135

namespace LeftMerge84140
def owner : Owner := ⟨.program ⟨257⟩, ⟨12472⟩⟩
def mergeEvent : Nat := 84140
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events013.exact3480RawTerms
def rightRaw : List Term := Proof.Events296.exact75903RawTerms
def group : MergeGroup := .operator 3480 75903
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3480) (leftOrdinal := 0)
    (rightResult := 75903) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12471⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84140

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
