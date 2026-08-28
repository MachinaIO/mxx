import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge57984
def owner : Owner := ⟨.program ⟨214⟩, ⟨25071⟩⟩
def mergeEvent : Nat := 57984
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25070⟩⟩] } }
def leftRaw : List Term := Proof.Events226.exact57975RawTerms
def rightRaw : List Term := Proof.Events226.exact57911RawTerms
def group : MergeGroup := .operator 57975 57911
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57975) (leftOrdinal := 0)
    (rightResult := 57911) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25070⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25070⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57984

namespace LeftMerge57998
def owner : Owner := ⟨.program ⟨214⟩, ⟨19175⟩⟩
def mergeEvent : Nat := 57998
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19172⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events226.exact57992RawTerms
def group : MergeGroup := .operator 50762 57992
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 57992) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19172⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19172⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57998

namespace LeftMerge58077
def owner : Owner := ⟨.program ⟨214⟩, ⟨10986⟩⟩
def mergeEvent : Nat := 58077
def frameStart : Nat := 58047
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events226.exact58073RawTerms
def rightRaw : List Term := Proof.Events226.exact58070RawTerms
def group : MergeGroup := .operator 58073 58070
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58073) (leftOrdinal := 0)
    (rightResult := 58070) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10847⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10985⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58077

namespace LeftMerge58107
def owner : Owner := ⟨.program ⟨214⟩, ⟨11079⟩⟩
def mergeEvent : Nat := 58107
def frameStart : Nat := 58047
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events226.exact58103RawTerms
def rightRaw : List Term := Proof.Events226.exact58101RawTerms
def group : MergeGroup := .operator 58103 58101
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58103) (leftOrdinal := 0)
    (rightResult := 58101) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58107

namespace LeftMerge58130
def owner : Owner := ⟨.program ⟨214⟩, ⟨7839⟩⟩
def mergeEvent : Nat := 58130
def frameStart : Nat := 58047
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩] } }
def leftRaw : List Term := Proof.Events227.exact58126RawTerms
def rightRaw : List Term := Proof.Events227.exact58123RawTerms
def group : MergeGroup := .operator 58126 58123
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58126) (leftOrdinal := 0)
    (rightResult := 58123) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7837⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58130

namespace LeftMerge58139
def owner : Owner := ⟨.program ⟨214⟩, ⟨25073⟩⟩
def mergeEvent : Nat := 58139
def frameStart : Nat := 58047
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25070⟩⟩] } }
def leftRaw : List Term := Proof.Events227.exact58135RawTerms
def rightRaw : List Term := Proof.Events226.exact58092RawTerms
def group : MergeGroup := .operator 58135 58092
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58135) (leftOrdinal := 0)
    (rightResult := 58092) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25070⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25070⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58139

namespace LeftMerge58140
def owner : Owner := ⟨.program ⟨214⟩, ⟨25073⟩⟩
def mergeEvent : Nat := 58140
def frameStart : Nat := 58047
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25070⟩⟩] } }
def leftRaw : List Term := Proof.Events227.exact58135RawTerms
def rightRaw : List Term := Proof.Events226.exact58092RawTerms
def group : MergeGroup := .operator 58135 58092
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58135) (leftOrdinal := 1)
    (rightResult := 58092) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25070⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25070⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58140

namespace LeftMerge58142
def owner : Owner := ⟨.program ⟨214⟩, ⟨25073⟩⟩
def mergeEvent : Nat := 58142
def frameStart : Nat := 58047
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23040⟩⟩] } }
def rhsRaw : List Term := Proof.Events226.exact58089RawTerms
def group : MergeGroup := .relation 58141
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58141) (rhsResult := 58089)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25070⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25070⟩⟩) ⟨23040⟩ 58089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23040⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], [⟨.program ⟨214⟩, ⟨23040⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58142

namespace LeftMerge58150
def owner : Owner := ⟨.program ⟨214⟩, ⟨15120⟩⟩
def mergeEvent : Nat := 58150
def frameStart : Nat := 58047
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15118⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events226.exact58103RawTerms
def rightRaw : List Term := Proof.Events227.exact58146RawTerms
def group : MergeGroup := .operator 58103 58146
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58103) (leftOrdinal := 0)
    (rightResult := 58146) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15118⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15118⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58150

namespace LeftMerge58167
def owner : Owner := ⟨.program ⟨214⟩, ⟨19175⟩⟩
def mergeEvent : Nat := 58167
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩] } }
def rhsRaw : List Term := Proof.Events227.exact58164RawTerms
def group : MergeGroup := .relation 58166
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58166) (rhsResult := 58164)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19172⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 58165 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19172⟩⟩]⟩) (none) 58164) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58167

namespace LeftMerge58168
def owner : Owner := ⟨.program ⟨214⟩, ⟨19175⟩⟩
def mergeEvent : Nat := 58168
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25070⟩⟩] } }
def rhsRaw : List Term := Proof.Events227.exact58164RawTerms
def group : MergeGroup := .relation 58166
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58166) (rhsResult := 58164)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19172⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 58165 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19172⟩⟩]⟩) (none) 58164) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25070⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25070⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58168

namespace LeftMerge58169
def owner : Owner := ⟨.program ⟨214⟩, ⟨19175⟩⟩
def mergeEvent : Nat := 58169
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23040⟩⟩] } }
def rhsRaw : List Term := Proof.Events227.exact58164RawTerms
def group : MergeGroup := .relation 58166
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58166) (rhsResult := 58164)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19172⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 58165 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19172⟩⟩]⟩) (none) 58164) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23040⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], [⟨.program ⟨214⟩, ⟨23040⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58169

namespace LeftMerge58170
def owner : Owner := ⟨.program ⟨214⟩, ⟨19175⟩⟩
def mergeEvent : Nat := 58170
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15118⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events227.exact58164RawTerms
def group : MergeGroup := .relation 58166
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58166) (rhsResult := 58164)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19172⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 58165 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19172⟩⟩]⟩) (none) 58164) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15118⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15118⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58170

namespace LeftMerge58175
def owner : Owner := ⟨.program ⟨214⟩, ⟨25072⟩⟩
def mergeEvent : Nat := 58175
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23040⟩⟩] } }
def leftRaw : List Term := Proof.Events227.exact58171RawTerms
def rightRaw : List Term := Proof.Events226.exact57985RawTerms
def group : MergeGroup := .operator 58171 57985
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58171) (leftOrdinal := 2)
    (rightResult := 57985) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23040⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23040⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], [⟨.program ⟨214⟩, ⟨23040⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58175

namespace LeftMerge58176
def owner : Owner := ⟨.program ⟨214⟩, ⟨25072⟩⟩
def mergeEvent : Nat := 58176
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25070⟩⟩] } }
def leftRaw : List Term := Proof.Events227.exact58171RawTerms
def rightRaw : List Term := Proof.Events226.exact57985RawTerms
def group : MergeGroup := .operator 58171 57985
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58171) (leftOrdinal := 1)
    (rightResult := 57985) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25070⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25070⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25070⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58176

namespace LeftMerge58184
def owner : Owner := ⟨.program ⟨214⟩, ⟨26796⟩⟩
def mergeEvent : Nat := 58184
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26794⟩⟩] } }
def leftRaw : List Term := Proof.Events227.exact58178RawTerms
def rightRaw : List Term := Proof.Events226.exact57901RawTerms
def group : MergeGroup := .operator 58178 57901
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58178) (leftOrdinal := 0)
    (rightResult := 57901) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26794⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26794⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58184

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
