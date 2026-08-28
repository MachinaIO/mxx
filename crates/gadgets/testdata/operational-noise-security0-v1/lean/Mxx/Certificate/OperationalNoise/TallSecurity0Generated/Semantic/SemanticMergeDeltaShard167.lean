import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge28918
def owner : Owner := ⟨.program ⟨214⟩, ⟨19183⟩⟩
def mergeEvent : Nat := 28918
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25080⟩⟩] } }
def rhsRaw : List Term := Proof.Events112.exact28914RawTerms
def group : MergeGroup := .relation 28916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28916) (rhsResult := 28914)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19180⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 28915 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19180⟩⟩]⟩) (none) 28914) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25080⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25080⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28918

namespace LeftMerge28919
def owner : Owner := ⟨.program ⟨214⟩, ⟨19183⟩⟩
def mergeEvent : Nat := 28919
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10857⟩⟩, ⟨.program ⟨214⟩, ⟨11001⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23044⟩⟩] } }
def rhsRaw : List Term := Proof.Events112.exact28914RawTerms
def group : MergeGroup := .relation 28916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28916) (rhsResult := 28914)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19180⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 28915 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19180⟩⟩]⟩) (none) 28914) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10857⟩⟩, ⟨.program ⟨214⟩, ⟨11001⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23044⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10857⟩⟩, ⟨.program ⟨214⟩, ⟨11001⟩⟩], [⟨.program ⟨214⟩, ⟨23044⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28919

namespace LeftMerge28920
def owner : Owner := ⟨.program ⟨214⟩, ⟨19183⟩⟩
def mergeEvent : Nat := 28920
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15126⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events112.exact28914RawTerms
def group : MergeGroup := .relation 28916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28916) (rhsResult := 28914)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19180⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 28915 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19180⟩⟩]⟩) (none) 28914) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15126⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15126⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28920

namespace LeftMerge28925
def owner : Owner := ⟨.program ⟨214⟩, ⟨25082⟩⟩
def mergeEvent : Nat := 28925
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10857⟩⟩, ⟨.program ⟨214⟩, ⟨11001⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23044⟩⟩] } }
def leftRaw : List Term := Proof.Events112.exact28921RawTerms
def rightRaw : List Term := Proof.Events112.exact28735RawTerms
def group : MergeGroup := .operator 28921 28735
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28921) (leftOrdinal := 2)
    (rightResult := 28735) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10857⟩⟩, ⟨.program ⟨214⟩, ⟨11001⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23044⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10857⟩⟩, ⟨.program ⟨214⟩, ⟨11001⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23044⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10857⟩⟩, ⟨.program ⟨214⟩, ⟨11001⟩⟩], [⟨.program ⟨214⟩, ⟨23044⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28925

namespace LeftMerge28926
def owner : Owner := ⟨.program ⟨214⟩, ⟨25082⟩⟩
def mergeEvent : Nat := 28926
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25080⟩⟩] } }
def leftRaw : List Term := Proof.Events112.exact28921RawTerms
def rightRaw : List Term := Proof.Events112.exact28735RawTerms
def group : MergeGroup := .operator 28921 28735
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28921) (leftOrdinal := 1)
    (rightResult := 28735) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25080⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25080⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25080⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28926

namespace LeftMerge28934
def owner : Owner := ⟨.program ⟨214⟩, ⟨26822⟩⟩
def mergeEvent : Nat := 28934
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26820⟩⟩] } }
def leftRaw : List Term := Proof.Events113.exact28928RawTerms
def rightRaw : List Term := Proof.Events111.exact28651RawTerms
def group : MergeGroup := .operator 28928 28651
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28928) (leftOrdinal := 0)
    (rightResult := 28651) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26820⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26820⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28934

namespace LeftMerge28935
def owner : Owner := ⟨.program ⟨214⟩, ⟨26822⟩⟩
def mergeEvent : Nat := 28935
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15126⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26820⟩⟩] } }
def leftRaw : List Term := Proof.Events113.exact28928RawTerms
def rightRaw : List Term := Proof.Events111.exact28651RawTerms
def group : MergeGroup := .operator 28928 28651
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28928) (leftOrdinal := 1)
    (rightResult := 28651) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15126⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26820⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15126⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26820⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28935

namespace LeftMerge28937
def owner : Owner := ⟨.program ⟨214⟩, ⟨26822⟩⟩
def mergeEvent : Nat := 28937
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15126⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23856⟩⟩] } }
def rhsRaw : List Term := Proof.Events111.exact28648RawTerms
def group : MergeGroup := .relation 28936
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28936) (rhsResult := 28648)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15126⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26820⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26820⟩⟩) ⟨23856⟩ 28648) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23856⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15126⟩⟩], [⟨.program ⟨214⟩, ⟨23856⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28937

namespace LeftMerge28951
def owner : Owner := ⟨.program ⟨214⟩, ⟨20695⟩⟩
def mergeEvent : Nat := 28951
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20692⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21512RawTerms
def rightRaw : List Term := Proof.Events113.exact28945RawTerms
def group : MergeGroup := .operator 21512 28945
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21512) (leftOrdinal := 0)
    (rightResult := 28945) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20692⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20692⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28951

namespace LeftMerge29072
def owner : Owner := ⟨.program ⟨214⟩, ⟨15168⟩⟩
def mergeEvent : Nat := 29072
def frameStart : Nat := 29006
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15126⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events113.exact29068RawTerms
def rightRaw : List Term := Proof.Events113.exact29066RawTerms
def group : MergeGroup := .operator 29068 29066
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29068) (leftOrdinal := 0)
    (rightResult := 29066) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15126⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15126⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29072

namespace LeftMerge29084
def owner : Owner := ⟨.program ⟨214⟩, ⟨26821⟩⟩
def mergeEvent : Nat := 29084
def frameStart : Nat := 29006
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26820⟩⟩] } }
def leftRaw : List Term := Proof.Events113.exact29080RawTerms
def rightRaw : List Term := Proof.Events113.exact29057RawTerms
def group : MergeGroup := .operator 29080 29057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29080) (leftOrdinal := 0)
    (rightResult := 29057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26820⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26820⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29084

namespace LeftMerge29085
def owner : Owner := ⟨.program ⟨214⟩, ⟨26821⟩⟩
def mergeEvent : Nat := 29085
def frameStart : Nat := 29006
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15126⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26820⟩⟩] } }
def leftRaw : List Term := Proof.Events113.exact29080RawTerms
def rightRaw : List Term := Proof.Events113.exact29057RawTerms
def group : MergeGroup := .operator 29080 29057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29080) (leftOrdinal := 1)
    (rightResult := 29057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15126⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26820⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15126⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26820⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29085

namespace LeftMerge29087
def owner : Owner := ⟨.program ⟨214⟩, ⟨26821⟩⟩
def mergeEvent : Nat := 29087
def frameStart : Nat := 29006
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15126⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23856⟩⟩] } }
def rhsRaw : List Term := Proof.Events113.exact29054RawTerms
def group : MergeGroup := .relation 29086
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29086) (rhsResult := 29054)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15126⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26820⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26820⟩⟩) ⟨23856⟩ 29054) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23856⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15126⟩⟩], [⟨.program ⟨214⟩, ⟨23856⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29087

namespace LeftMerge29095
def owner : Owner := ⟨.program ⟨214⟩, ⟨15380⟩⟩
def mergeEvent : Nat := 29095
def frameStart : Nat := 29006
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15378⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events113.exact29068RawTerms
def rightRaw : List Term := Proof.Events113.exact29091RawTerms
def group : MergeGroup := .operator 29068 29091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29068) (leftOrdinal := 0)
    (rightResult := 29091) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15378⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29095

namespace LeftMerge29112
def owner : Owner := ⟨.program ⟨214⟩, ⟨20695⟩⟩
def mergeEvent : Nat := 29112
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }
def rhsRaw : List Term := Proof.Events113.exact29109RawTerms
def group : MergeGroup := .relation 29111
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29111) (rhsResult := 29109)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20692⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 29110 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20692⟩⟩]⟩) (none) 29109) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29112

namespace LeftMerge29113
def owner : Owner := ⟨.program ⟨214⟩, ⟨20695⟩⟩
def mergeEvent : Nat := 29113
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26820⟩⟩] } }
def rhsRaw : List Term := Proof.Events113.exact29109RawTerms
def group : MergeGroup := .relation 29111
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29111) (rhsResult := 29109)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20692⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 29110 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20692⟩⟩]⟩) (none) 29109) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26820⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26820⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29113

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
