import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge23043
def owner : Owner := ⟨.program ⟨214⟩, ⟨12787⟩⟩
def mergeEvent : Nat := 23043
def frameStart : Nat := 23013
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10045⟩⟩, ⟨.program ⟨214⟩, ⟨12786⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events089.exact23039RawTerms
def rightRaw : List Term := Proof.Events089.exact23036RawTerms
def group : MergeGroup := .operator 23039 23036
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23039) (leftOrdinal := 0)
    (rightResult := 23036) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10045⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12786⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10045⟩⟩, ⟨.program ⟨214⟩, ⟨12786⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23043

namespace LeftMerge23073
def owner : Owner := ⟨.program ⟨214⟩, ⟨12872⟩⟩
def mergeEvent : Nat := 23073
def frameStart : Nat := 23013
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10045⟩⟩, ⟨.program ⟨214⟩, ⟨12786⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events090.exact23069RawTerms
def rightRaw : List Term := Proof.Events090.exact23067RawTerms
def group : MergeGroup := .operator 23069 23067
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23069) (leftOrdinal := 0)
    (rightResult := 23067) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10045⟩⟩, ⟨.program ⟨214⟩, ⟨12786⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10045⟩⟩, ⟨.program ⟨214⟩, ⟨12786⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23073

namespace LeftMerge23096
def owner : Owner := ⟨.program ⟨214⟩, ⟨7875⟩⟩
def mergeEvent : Nat := 23096
def frameStart : Nat := 23013
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }
def leftRaw : List Term := Proof.Events090.exact23092RawTerms
def rightRaw : List Term := Proof.Events090.exact23089RawTerms
def group : MergeGroup := .operator 23092 23089
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23092) (leftOrdinal := 0)
    (rightResult := 23089) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7873⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23096

namespace LeftMerge23105
def owner : Owner := ⟨.program ⟨214⟩, ⟨25545⟩⟩
def mergeEvent : Nat := 23105
def frameStart : Nat := 23013
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25542⟩⟩] } }
def leftRaw : List Term := Proof.Events090.exact23101RawTerms
def rightRaw : List Term := Proof.Events090.exact23058RawTerms
def group : MergeGroup := .operator 23101 23058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23101) (leftOrdinal := 0)
    (rightResult := 23058) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25542⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25542⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23105

namespace LeftMerge23106
def owner : Owner := ⟨.program ⟨214⟩, ⟨25545⟩⟩
def mergeEvent : Nat := 23106
def frameStart : Nat := 23013
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10045⟩⟩, ⟨.program ⟨214⟩, ⟨12786⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25542⟩⟩] } }
def leftRaw : List Term := Proof.Events090.exact23101RawTerms
def rightRaw : List Term := Proof.Events090.exact23058RawTerms
def group : MergeGroup := .operator 23101 23058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23101) (leftOrdinal := 1)
    (rightResult := 23058) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10045⟩⟩, ⟨.program ⟨214⟩, ⟨12786⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25542⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10045⟩⟩, ⟨.program ⟨214⟩, ⟨12786⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25542⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23106

namespace LeftMerge23108
def owner : Owner := ⟨.program ⟨214⟩, ⟨25545⟩⟩
def mergeEvent : Nat := 23108
def frameStart : Nat := 23013
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10045⟩⟩, ⟨.program ⟨214⟩, ⟨12786⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23296⟩⟩] } }
def rhsRaw : List Term := Proof.Events090.exact23055RawTerms
def group : MergeGroup := .relation 23107
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23107) (rhsResult := 23055)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10045⟩⟩, ⟨.program ⟨214⟩, ⟨12786⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25542⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25542⟩⟩) ⟨23296⟩ 23055) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23296⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10045⟩⟩, ⟨.program ⟨214⟩, ⟨12786⟩⟩], [⟨.program ⟨214⟩, ⟨23296⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23108

namespace LeftMerge23116
def owner : Owner := ⟨.program ⟨214⟩, ⟨16647⟩⟩
def mergeEvent : Nat := 23116
def frameStart : Nat := 23013
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16645⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events090.exact23069RawTerms
def rightRaw : List Term := Proof.Events090.exact23112RawTerms
def group : MergeGroup := .operator 23069 23112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23069) (leftOrdinal := 0)
    (rightResult := 23112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16645⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16645⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23116

namespace LeftMerge23133
def owner : Owner := ⟨.program ⟨214⟩, ⟨20047⟩⟩
def mergeEvent : Nat := 23133
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }
def rhsRaw : List Term := Proof.Events090.exact23130RawTerms
def group : MergeGroup := .relation 23132
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23132) (rhsResult := 23130)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20044⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 23131 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20044⟩⟩]⟩) (none) 23130) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23133

namespace LeftMerge23134
def owner : Owner := ⟨.program ⟨214⟩, ⟨20047⟩⟩
def mergeEvent : Nat := 23134
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25542⟩⟩] } }
def rhsRaw : List Term := Proof.Events090.exact23130RawTerms
def group : MergeGroup := .relation 23132
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23132) (rhsResult := 23130)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20044⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 23131 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20044⟩⟩]⟩) (none) 23130) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25542⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25542⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23134

namespace LeftMerge23135
def owner : Owner := ⟨.program ⟨214⟩, ⟨20047⟩⟩
def mergeEvent : Nat := 23135
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10045⟩⟩, ⟨.program ⟨214⟩, ⟨12786⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23296⟩⟩] } }
def rhsRaw : List Term := Proof.Events090.exact23130RawTerms
def group : MergeGroup := .relation 23132
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23132) (rhsResult := 23130)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20044⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 23131 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20044⟩⟩]⟩) (none) 23130) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10045⟩⟩, ⟨.program ⟨214⟩, ⟨12786⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23296⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10045⟩⟩, ⟨.program ⟨214⟩, ⟨12786⟩⟩], [⟨.program ⟨214⟩, ⟨23296⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23135

namespace LeftMerge23136
def owner : Owner := ⟨.program ⟨214⟩, ⟨20047⟩⟩
def mergeEvent : Nat := 23136
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16645⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events090.exact23130RawTerms
def group : MergeGroup := .relation 23132
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23132) (rhsResult := 23130)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20044⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 23131 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20044⟩⟩]⟩) (none) 23130) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16645⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16645⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23136

namespace LeftMerge23141
def owner : Owner := ⟨.program ⟨214⟩, ⟨25544⟩⟩
def mergeEvent : Nat := 23141
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10045⟩⟩, ⟨.program ⟨214⟩, ⟨12786⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23296⟩⟩] } }
def leftRaw : List Term := Proof.Events090.exact23137RawTerms
def rightRaw : List Term := Proof.Events089.exact22951RawTerms
def group : MergeGroup := .operator 23137 22951
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23137) (leftOrdinal := 2)
    (rightResult := 22951) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10045⟩⟩, ⟨.program ⟨214⟩, ⟨12786⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23296⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10045⟩⟩, ⟨.program ⟨214⟩, ⟨12786⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23296⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10045⟩⟩, ⟨.program ⟨214⟩, ⟨12786⟩⟩], [⟨.program ⟨214⟩, ⟨23296⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23141

namespace LeftMerge23142
def owner : Owner := ⟨.program ⟨214⟩, ⟨25544⟩⟩
def mergeEvent : Nat := 23142
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25542⟩⟩] } }
def leftRaw : List Term := Proof.Events090.exact23137RawTerms
def rightRaw : List Term := Proof.Events089.exact22951RawTerms
def group : MergeGroup := .operator 23137 22951
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23137) (leftOrdinal := 1)
    (rightResult := 22951) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25542⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25542⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25542⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23142

namespace LeftMerge23150
def owner : Owner := ⟨.program ⟨214⟩, ⟨29426⟩⟩
def mergeEvent : Nat := 23150
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29424⟩⟩] } }
def leftRaw : List Term := Proof.Events090.exact23144RawTerms
def rightRaw : List Term := Proof.Events089.exact22867RawTerms
def group : MergeGroup := .operator 23144 22867
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23144) (leftOrdinal := 0)
    (rightResult := 22867) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29424⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29424⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23150

namespace LeftMerge23151
def owner : Owner := ⟨.program ⟨214⟩, ⟨29426⟩⟩
def mergeEvent : Nat := 23151
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16645⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29424⟩⟩] } }
def leftRaw : List Term := Proof.Events090.exact23144RawTerms
def rightRaw : List Term := Proof.Events089.exact22867RawTerms
def group : MergeGroup := .operator 23144 22867
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23144) (leftOrdinal := 1)
    (rightResult := 22867) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16645⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29424⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16645⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29424⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23151

namespace LeftMerge23153
def owner : Owner := ⟨.program ⟨214⟩, ⟨29426⟩⟩
def mergeEvent : Nat := 23153
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16645⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24612⟩⟩] } }
def rhsRaw : List Term := Proof.Events089.exact22864RawTerms
def group : MergeGroup := .relation 23152
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23152) (rhsResult := 22864)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16645⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29424⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29424⟩⟩) ⟨24612⟩ 22864) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16645⟩⟩], [⟨.program ⟨214⟩, ⟨24612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23153

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
