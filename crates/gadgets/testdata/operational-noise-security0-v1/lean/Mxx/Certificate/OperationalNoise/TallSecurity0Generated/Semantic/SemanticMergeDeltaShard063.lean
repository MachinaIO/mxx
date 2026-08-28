import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge11942
def owner : Owner := ⟨.program ⟨214⟩, ⟨21419⟩⟩
def mergeEvent : Nat := 11942
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27918⟩⟩] } }
def rhsRaw : List Term := Proof.Events046.exact11938RawTerms
def group : MergeGroup := .relation 11940
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 11940) (rhsResult := 11938)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21416⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 11939 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21416⟩⟩]⟩) (none) 11938) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27918⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27918⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge11942

namespace LeftMerge11943
def owner : Owner := ⟨.program ⟨214⟩, ⟨21419⟩⟩
def mergeEvent : Nat := 11943
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events046.exact11938RawTerms
def group : MergeGroup := .relation 11940
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 11940) (rhsResult := 11938)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21416⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 11939 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21416⟩⟩]⟩) (none) 11938) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge11943

namespace LeftMerge11944
def owner : Owner := ⟨.program ⟨214⟩, ⟨21419⟩⟩
def mergeEvent : Nat := 11944
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }
def rhsRaw : List Term := Proof.Events046.exact11938RawTerms
def group : MergeGroup := .relation 11940
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 11940) (rhsResult := 11938)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21416⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 11939 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21416⟩⟩]⟩) (none) 11938) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11944

namespace LeftMerge11949
def owner : Owner := ⟨.program ⟨214⟩, ⟨27921⟩⟩
def mergeEvent : Nat := 11949
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15956⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24174⟩⟩] } }
def leftRaw : List Term := Proof.Events046.exact11945RawTerms
def rightRaw : List Term := Proof.Events045.exact11767RawTerms
def group : MergeGroup := .operator 11945 11767
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11945) (leftOrdinal := 2)
    (rightResult := 11767) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15956⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24174⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15956⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24174⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15956⟩⟩], [⟨.program ⟨214⟩, ⟨24174⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge11949

namespace LeftMerge11950
def owner : Owner := ⟨.program ⟨214⟩, ⟨27921⟩⟩
def mergeEvent : Nat := 11950
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27918⟩⟩] } }
def leftRaw : List Term := Proof.Events046.exact11945RawTerms
def rightRaw : List Term := Proof.Events045.exact11767RawTerms
def group : MergeGroup := .operator 11945 11767
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11945) (leftOrdinal := 0)
    (rightResult := 11767) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27918⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27918⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27918⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11950

namespace LeftMerge11979
def owner : Owner := ⟨.program ⟨214⟩, ⟨11402⟩⟩
def mergeEvent : Nat := 11979
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11401⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events001.exact304RawTerms
def rightRaw : List Term := Proof.Events025.exact6449RawTerms
def group : MergeGroup := .operator 304 6449
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304) (leftOrdinal := 0)
    (rightResult := 6449) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11401⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11401⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11979

namespace LeftMerge11987
def owner : Owner := ⟨.program ⟨214⟩, ⟨7386⟩⟩
def mergeEvent : Nat := 11987
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6314RawTerms
def rightRaw : List Term := Proof.Events046.exact11983RawTerms
def group : MergeGroup := .operator 6314 11983
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6314) (leftOrdinal := 0)
    (rightResult := 11983) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11987

namespace LeftMerge12004
def owner : Owner := ⟨.program ⟨214⟩, ⟨14029⟩⟩
def mergeEvent : Nat := 12004
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11401⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events046.exact11998RawTerms
def rightRaw : List Term := Proof.Events001.exact307RawTerms
def group : MergeGroup := .operator 11998 307
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11998) (leftOrdinal := 1)
    (rightResult := 307) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11401⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14026⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11401⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge12004

namespace LeftMerge12005
def owner : Owner := ⟨.program ⟨214⟩, ⟨14029⟩⟩
def mergeEvent : Nat := 12005
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }
def leftRaw : List Term := Proof.Events046.exact11998RawTerms
def rightRaw : List Term := Proof.Events001.exact307RawTerms
def group : MergeGroup := .operator 11998 307
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11998) (leftOrdinal := 0)
    (rightResult := 307) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14026⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12005

namespace LeftMerge12020
def owner : Owner := ⟨.program ⟨214⟩, ⟨14030⟩⟩
def mergeEvent : Nat := 12020
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events001.exact307RawTerms
def rightRaw : List Term := Proof.Events025.exact6449RawTerms
def group : MergeGroup := .operator 307 6449
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 307) (leftOrdinal := 0)
    (rightResult := 6449) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14026⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12020

namespace LeftMerge12028
def owner : Owner := ⟨.program ⟨214⟩, ⟨7366⟩⟩
def mergeEvent : Nat := 12028
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6314RawTerms
def rightRaw : List Term := Proof.Events046.exact12024RawTerms
def group : MergeGroup := .operator 6314 12024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6314) (leftOrdinal := 0)
    (rightResult := 12024) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12028

namespace LeftMerge12045
def owner : Owner := ⟨.program ⟨214⟩, ⟨14033⟩⟩
def mergeEvent : Nat := 12045
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩] } }
def leftRaw : List Term := Proof.Events047.exact12039RawTerms
def rightRaw : List Term := Proof.Events046.exact12013RawTerms
def group : MergeGroup := .operator 12039 12013
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12039) (leftOrdinal := 1)
    (rightResult := 12013) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7849⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge12045

namespace LeftMerge12047
def owner : Owner := ⟨.program ⟨214⟩, ⟨14033⟩⟩
def mergeEvent : Nat := 12047
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }
def rhsRaw : List Term := Proof.Events046.exact11983RawTerms
def group : MergeGroup := .relation 12046
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 12046) (rhsResult := 11983)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7849⟩⟩) ⟨6778⟩ 11983) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge12047

namespace LeftMerge12048
def owner : Owner := ⟨.program ⟨214⟩, ⟨14033⟩⟩
def mergeEvent : Nat := 12048
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩] } }
def leftRaw : List Term := Proof.Events047.exact12039RawTerms
def rightRaw : List Term := Proof.Events046.exact12013RawTerms
def group : MergeGroup := .operator 12039 12013
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12039) (leftOrdinal := 0)
    (rightResult := 12013) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7849⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12048

namespace LeftMerge12053
def owner : Owner := ⟨.program ⟨214⟩, ⟨14034⟩⟩
def mergeEvent : Nat := 12053
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }
def leftRaw : List Term := Proof.Events047.exact12049RawTerms
def rightRaw : List Term := Proof.Events046.exact12006RawTerms
def group : MergeGroup := .operator 12049 12006
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12049) (leftOrdinal := 1)
    (rightResult := 12006) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12053

namespace LeftMerge12061
def owner : Owner := ⟨.program ⟨214⟩, ⟨26010⟩⟩
def mergeEvent : Nat := 12061
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11401⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26009⟩⟩] } }
def leftRaw : List Term := Proof.Events047.exact12055RawTerms
def rightRaw : List Term := Proof.Events046.exact11972RawTerms
def group : MergeGroup := .operator 12055 11972
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12055) (leftOrdinal := 1)
    (rightResult := 11972) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11401⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26009⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11401⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26009⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge12061

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
