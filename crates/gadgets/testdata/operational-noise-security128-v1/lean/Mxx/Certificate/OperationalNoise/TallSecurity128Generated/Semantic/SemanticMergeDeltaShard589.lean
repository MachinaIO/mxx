import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge98790
def owner : Owner := ⟨.program ⟨257⟩, ⟨12460⟩⟩
def mergeEvent : Nat := 98790
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }
def leftRaw : List Term := Proof.Events385.exact98781RawTerms
def rightRaw : List Term := Proof.Events100.exact25627RawTerms
def group : MergeGroup := .operator 98781 25627
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98781) (leftOrdinal := 0)
    (rightResult := 25627) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9568⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98790

namespace LeftMerge98795
def owner : Owner := ⟨.program ⟨257⟩, ⟨15601⟩⟩
def mergeEvent : Nat := 98795
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12456⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } }
def leftRaw : List Term := Proof.Events385.exact98791RawTerms
def rightRaw : List Term := Proof.Events385.exact98761RawTerms
def group : MergeGroup := .operator 98791 98761
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98791) (leftOrdinal := 1)
    (rightResult := 98761) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12456⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12456⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12456⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98795

namespace LeftMerge98803
def owner : Owner := ⟨.program ⟨257⟩, ⟨17415⟩⟩
def mergeEvent : Nat := 98803
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17414⟩⟩] } }
def leftRaw : List Term := Proof.Events385.exact98797RawTerms
def rightRaw : List Term := Proof.Events385.exact98733RawTerms
def group : MergeGroup := .operator 98797 98733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98797) (leftOrdinal := 1)
    (rightResult := 98733) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17414⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17414⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98803

namespace LeftMerge98805
def owner : Owner := ⟨.program ⟨257⟩, ⟨17415⟩⟩
def mergeEvent : Nat := 98805
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16879⟩⟩] } }
def rhsRaw : List Term := Proof.Events385.exact98730RawTerms
def group : MergeGroup := .relation 98804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98804) (rhsResult := 98730)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17414⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17414⟩⟩) ⟨16879⟩ 98730) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16879⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], [⟨.program ⟨257⟩, ⟨16879⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98805

namespace LeftMerge98806
def owner : Owner := ⟨.program ⟨257⟩, ⟨17415⟩⟩
def mergeEvent : Nat := 98806
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17414⟩⟩] } }
def leftRaw : List Term := Proof.Events385.exact98797RawTerms
def rightRaw : List Term := Proof.Events385.exact98733RawTerms
def group : MergeGroup := .operator 98797 98733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98797) (leftOrdinal := 0)
    (rightResult := 98733) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17414⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17414⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98806

namespace LeftMerge98820
def owner : Owner := ⟨.program ⟨257⟩, ⟨16342⟩⟩
def mergeEvent : Nat := 98820
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16339⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90620RawTerms
def rightRaw : List Term := Proof.Events385.exact98814RawTerms
def group : MergeGroup := .operator 90620 98814
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90620) (leftOrdinal := 0)
    (rightResult := 98814) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16339⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16339⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98820

namespace LeftMerge98899
def owner : Owner := ⟨.program ⟨257⟩, ⟨15595⟩⟩
def mergeEvent : Nat := 98899
def frameStart : Nat := 98869
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events386.exact98895RawTerms
def rightRaw : List Term := Proof.Events386.exact98892RawTerms
def group : MergeGroup := .operator 98895 98892
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98895) (leftOrdinal := 0)
    (rightResult := 98892) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12456⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15594⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98899

namespace LeftMerge98929
def owner : Owner := ⟨.program ⟨257⟩, ⟨17148⟩⟩
def mergeEvent : Nat := 98929
def frameStart : Nat := 98869
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events386.exact98925RawTerms
def rightRaw : List Term := Proof.Events386.exact98923RawTerms
def group : MergeGroup := .operator 98925 98923
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98925) (leftOrdinal := 0)
    (rightResult := 98923) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98929

namespace LeftMerge98952
def owner : Owner := ⟨.program ⟨257⟩, ⟨9570⟩⟩
def mergeEvent : Nat := 98952
def frameStart : Nat := 98869
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }
def leftRaw : List Term := Proof.Events386.exact98948RawTerms
def rightRaw : List Term := Proof.Events386.exact98945RawTerms
def group : MergeGroup := .operator 98948 98945
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98948) (leftOrdinal := 0)
    (rightResult := 98945) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9568⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98952

namespace LeftMerge98961
def owner : Owner := ⟨.program ⟨257⟩, ⟨17417⟩⟩
def mergeEvent : Nat := 98961
def frameStart : Nat := 98869
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17414⟩⟩] } }
def leftRaw : List Term := Proof.Events386.exact98957RawTerms
def rightRaw : List Term := Proof.Events386.exact98914RawTerms
def group : MergeGroup := .operator 98957 98914
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98957) (leftOrdinal := 0)
    (rightResult := 98914) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17414⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17414⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98961

namespace LeftMerge98962
def owner : Owner := ⟨.program ⟨257⟩, ⟨17417⟩⟩
def mergeEvent : Nat := 98962
def frameStart : Nat := 98869
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17414⟩⟩] } }
def leftRaw : List Term := Proof.Events386.exact98957RawTerms
def rightRaw : List Term := Proof.Events386.exact98914RawTerms
def group : MergeGroup := .operator 98957 98914
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98957) (leftOrdinal := 1)
    (rightResult := 98914) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17414⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17414⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98962

namespace LeftMerge98964
def owner : Owner := ⟨.program ⟨257⟩, ⟨17417⟩⟩
def mergeEvent : Nat := 98964
def frameStart : Nat := 98869
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16879⟩⟩] } }
def rhsRaw : List Term := Proof.Events386.exact98911RawTerms
def group : MergeGroup := .relation 98963
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98963) (rhsResult := 98911)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17414⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17414⟩⟩) ⟨16879⟩ 98911) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16879⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], [⟨.program ⟨257⟩, ⟨16879⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98964

namespace LeftMerge98972
def owner : Owner := ⟨.program ⟨257⟩, ⟨15830⟩⟩
def mergeEvent : Nat := 98972
def frameStart : Nat := 98869
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events386.exact98925RawTerms
def rightRaw : List Term := Proof.Events386.exact98968RawTerms
def group : MergeGroup := .operator 98925 98968
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98925) (leftOrdinal := 0)
    (rightResult := 98968) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15828⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98972

namespace LeftMerge98989
def owner : Owner := ⟨.program ⟨257⟩, ⟨16342⟩⟩
def mergeEvent : Nat := 98989
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }
def rhsRaw : List Term := Proof.Events386.exact98986RawTerms
def group : MergeGroup := .relation 98988
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98988) (rhsResult := 98986)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16339⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98987 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16339⟩⟩]⟩) (none) 98986) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98989

namespace LeftMerge98990
def owner : Owner := ⟨.program ⟨257⟩, ⟨16342⟩⟩
def mergeEvent : Nat := 98990
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17414⟩⟩] } }
def rhsRaw : List Term := Proof.Events386.exact98986RawTerms
def group : MergeGroup := .relation 98988
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98988) (rhsResult := 98986)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16339⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98987 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16339⟩⟩]⟩) (none) 98986) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17414⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17414⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98990

namespace LeftMerge98991
def owner : Owner := ⟨.program ⟨257⟩, ⟨16342⟩⟩
def mergeEvent : Nat := 98991
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16879⟩⟩] } }
def rhsRaw : List Term := Proof.Events386.exact98986RawTerms
def group : MergeGroup := .relation 98988
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98988) (rhsResult := 98986)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16339⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98987 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16339⟩⟩]⟩) (none) 98986) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16879⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], [⟨.program ⟨257⟩, ⟨16879⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98991

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
