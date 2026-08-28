import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge98916
def owner : Owner := ⟨.program ⟨214⟩, ⟨7854⟩⟩
def mergeEvent : Nat := 98916
def frameStart : Nat := 98845
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩] } }
def leftRaw : List Term := Proof.Events386.exact98912RawTerms
def rightRaw : List Term := Proof.Events386.exact98909RawTerms
def group : MergeGroup := .operator 98912 98909
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98912) (leftOrdinal := 0)
    (rightResult := 98909) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7852⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98916

namespace LeftMerge98925
def owner : Owner := ⟨.program ⟨214⟩, ⟨26056⟩⟩
def mergeEvent : Nat := 98925
def frameStart : Nat := 98845
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26053⟩⟩] } }
def leftRaw : List Term := Proof.Events386.exact98921RawTerms
def rightRaw : List Term := Proof.Events386.exact98878RawTerms
def group : MergeGroup := .operator 98921 98878
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98921) (leftOrdinal := 0)
    (rightResult := 98878) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26053⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26053⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98925

namespace LeftMerge98926
def owner : Owner := ⟨.program ⟨214⟩, ⟨26056⟩⟩
def mergeEvent : Nat := 98926
def frameStart : Nat := 98845
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11457⟩⟩, ⟨.program ⟨214⟩, ⟨14180⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26053⟩⟩] } }
def leftRaw : List Term := Proof.Events386.exact98921RawTerms
def rightRaw : List Term := Proof.Events386.exact98878RawTerms
def group : MergeGroup := .operator 98921 98878
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98921) (leftOrdinal := 1)
    (rightResult := 98878) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11457⟩⟩, ⟨.program ⟨214⟩, ⟨14180⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26053⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11457⟩⟩, ⟨.program ⟨214⟩, ⟨14180⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26053⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98926

namespace LeftMerge98928
def owner : Owner := ⟨.program ⟨214⟩, ⟨26056⟩⟩
def mergeEvent : Nat := 98928
def frameStart : Nat := 98845
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11457⟩⟩, ⟨.program ⟨214⟩, ⟨14180⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23578⟩⟩] } }
def rhsRaw : List Term := Proof.Events386.exact98875RawTerms
def group : MergeGroup := .relation 98927
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98927) (rhsResult := 98875)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11457⟩⟩, ⟨.program ⟨214⟩, ⟨14180⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26053⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26053⟩⟩) ⟨23578⟩ 98875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23578⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11457⟩⟩, ⟨.program ⟨214⟩, ⟨14180⟩⟩], [⟨.program ⟨214⟩, ⟨23578⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98928

namespace LeftMerge98936
def owner : Owner := ⟨.program ⟨214⟩, ⟨15932⟩⟩
def mergeEvent : Nat := 98936
def frameStart : Nat := 98845
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15930⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events386.exact98889RawTerms
def rightRaw : List Term := Proof.Events386.exact98932RawTerms
def group : MergeGroup := .operator 98889 98932
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98889) (leftOrdinal := 0)
    (rightResult := 98932) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15930⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15930⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98936

namespace LeftMerge98953
def owner : Owner := ⟨.program ⟨214⟩, ⟨19520⟩⟩
def mergeEvent : Nat := 98953
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩] } }
def rhsRaw : List Term := Proof.Events386.exact98950RawTerms
def group : MergeGroup := .relation 98952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98952) (rhsResult := 98950)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19517⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98951 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19517⟩⟩]⟩) (none) 98950) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98953

namespace LeftMerge98954
def owner : Owner := ⟨.program ⟨214⟩, ⟨19520⟩⟩
def mergeEvent : Nat := 98954
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26053⟩⟩] } }
def rhsRaw : List Term := Proof.Events386.exact98950RawTerms
def group : MergeGroup := .relation 98952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98952) (rhsResult := 98950)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19517⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98951 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19517⟩⟩]⟩) (none) 98950) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26053⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26053⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98954

namespace LeftMerge98955
def owner : Owner := ⟨.program ⟨214⟩, ⟨19520⟩⟩
def mergeEvent : Nat := 98955
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11457⟩⟩, ⟨.program ⟨214⟩, ⟨14180⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23578⟩⟩] } }
def rhsRaw : List Term := Proof.Events386.exact98950RawTerms
def group : MergeGroup := .relation 98952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98952) (rhsResult := 98950)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19517⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98951 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19517⟩⟩]⟩) (none) 98950) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11457⟩⟩, ⟨.program ⟨214⟩, ⟨14180⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23578⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11457⟩⟩, ⟨.program ⟨214⟩, ⟨14180⟩⟩], [⟨.program ⟨214⟩, ⟨23578⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98955

namespace LeftMerge98956
def owner : Owner := ⟨.program ⟨214⟩, ⟨19520⟩⟩
def mergeEvent : Nat := 98956
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15930⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events386.exact98950RawTerms
def group : MergeGroup := .relation 98952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98952) (rhsResult := 98950)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19517⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98951 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19517⟩⟩]⟩) (none) 98950) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15930⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15930⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98956

namespace LeftMerge98961
def owner : Owner := ⟨.program ⟨214⟩, ⟨26055⟩⟩
def mergeEvent : Nat := 98961
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11457⟩⟩, ⟨.program ⟨214⟩, ⟨14180⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23578⟩⟩] } }
def leftRaw : List Term := Proof.Events386.exact98957RawTerms
def rightRaw : List Term := Proof.Events385.exact98795RawTerms
def group : MergeGroup := .operator 98957 98795
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98957) (leftOrdinal := 2)
    (rightResult := 98795) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11457⟩⟩, ⟨.program ⟨214⟩, ⟨14180⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23578⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11457⟩⟩, ⟨.program ⟨214⟩, ⟨14180⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23578⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11457⟩⟩, ⟨.program ⟨214⟩, ⟨14180⟩⟩], [⟨.program ⟨214⟩, ⟨23578⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98961

namespace LeftMerge98962
def owner : Owner := ⟨.program ⟨214⟩, ⟨26055⟩⟩
def mergeEvent : Nat := 98962
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26053⟩⟩] } }
def leftRaw : List Term := Proof.Events386.exact98957RawTerms
def rightRaw : List Term := Proof.Events385.exact98795RawTerms
def group : MergeGroup := .operator 98957 98795
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98957) (leftOrdinal := 1)
    (rightResult := 98795) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26053⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26053⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26053⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98962

namespace LeftMerge98970
def owner : Owner := ⟨.program ⟨214⟩, ⟨27833⟩⟩
def mergeEvent : Nat := 98970
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27831⟩⟩] } }
def leftRaw : List Term := Proof.Events386.exact98964RawTerms
def rightRaw : List Term := Proof.Events385.exact98711RawTerms
def group : MergeGroup := .operator 98964 98711
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98964) (leftOrdinal := 0)
    (rightResult := 98711) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27831⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27831⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98970

namespace LeftMerge98971
def owner : Owner := ⟨.program ⟨214⟩, ⟨27833⟩⟩
def mergeEvent : Nat := 98971
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15930⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27831⟩⟩] } }
def leftRaw : List Term := Proof.Events386.exact98964RawTerms
def rightRaw : List Term := Proof.Events385.exact98711RawTerms
def group : MergeGroup := .operator 98964 98711
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98964) (leftOrdinal := 1)
    (rightResult := 98711) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15930⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27831⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15930⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27831⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98971

namespace LeftMerge98973
def owner : Owner := ⟨.program ⟨214⟩, ⟨27833⟩⟩
def mergeEvent : Nat := 98973
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15930⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24153⟩⟩] } }
def rhsRaw : List Term := Proof.Events385.exact98708RawTerms
def group : MergeGroup := .relation 98972
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98972) (rhsResult := 98708)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15930⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27831⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27831⟩⟩) ⟨24153⟩ 98708) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24153⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15930⟩⟩], [⟨.program ⟨214⟩, ⟨24153⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98973

namespace LeftMerge98987
def owner : Owner := ⟨.program ⟨214⟩, ⟨21392⟩⟩
def mergeEvent : Nat := 98987
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21389⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94462RawTerms
def rightRaw : List Term := Proof.Events386.exact98981RawTerms
def group : MergeGroup := .operator 94462 98981
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94462) (leftOrdinal := 0)
    (rightResult := 98981) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21389⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21389⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98987

namespace LeftMerge99084
def owner : Owner := ⟨.program ⟨214⟩, ⟨16009⟩⟩
def mergeEvent : Nat := 99084
def frameStart : Nat := 99030
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15930⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events387.exact99080RawTerms
def rightRaw : List Term := Proof.Events387.exact99078RawTerms
def group : MergeGroup := .operator 99080 99078
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99080) (leftOrdinal := 0)
    (rightResult := 99078) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15930⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15930⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99084

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
