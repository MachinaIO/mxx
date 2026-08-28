import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge36990
def owner : Owner := ⟨.program ⟨214⟩, ⟨22707⟩⟩
def mergeEvent : Nat := 36990
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29845⟩⟩] } }
def rhsRaw : List Term := Proof.Events144.exact36986RawTerms
def group : MergeGroup := .relation 36988
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36988) (rhsResult := 36986)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22704⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 36987 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22704⟩⟩]⟩) (none) 36986) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29845⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29845⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36990

namespace LeftMerge36991
def owner : Owner := ⟨.program ⟨214⟩, ⟨22707⟩⟩
def mergeEvent : Nat := 36991
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24735⟩⟩] } }
def rhsRaw : List Term := Proof.Events144.exact36986RawTerms
def group : MergeGroup := .relation 36988
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36988) (rhsResult := 36986)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22704⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 36987 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22704⟩⟩]⟩) (none) 36986) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24735⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], [⟨.program ⟨214⟩, ⟨24735⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36991

namespace LeftMerge36992
def owner : Owner := ⟨.program ⟨214⟩, ⟨22707⟩⟩
def mergeEvent : Nat := 36992
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17091⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events144.exact36986RawTerms
def group : MergeGroup := .relation 36988
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36988) (rhsResult := 36986)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22704⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 36987 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22704⟩⟩]⟩) (none) 36986) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17091⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17091⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36992

namespace LeftMerge36997
def owner : Owner := ⟨.program ⟨214⟩, ⟨29848⟩⟩
def mergeEvent : Nat := 36997
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29845⟩⟩] } }
def leftRaw : List Term := Proof.Events144.exact36993RawTerms
def rightRaw : List Term := Proof.Events143.exact36815RawTerms
def group : MergeGroup := .operator 36993 36815
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36993) (leftOrdinal := 0)
    (rightResult := 36815) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29845⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29845⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29845⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36997

namespace LeftMerge36998
def owner : Owner := ⟨.program ⟨214⟩, ⟨29848⟩⟩
def mergeEvent : Nat := 36998
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24735⟩⟩] } }
def leftRaw : List Term := Proof.Events144.exact36993RawTerms
def rightRaw : List Term := Proof.Events143.exact36815RawTerms
def group : MergeGroup := .operator 36993 36815
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36993) (leftOrdinal := 2)
    (rightResult := 36815) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24735⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24735⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], [⟨.program ⟨214⟩, ⟨24735⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36998

namespace LeftMerge37024
def owner : Owner := ⟨.program ⟨214⟩, ⟨12977⟩⟩
def mergeEvent : Nat := 37024
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨12974⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events006.exact1636RawTerms
def rightRaw : List Term := Proof.Events140.exact36045RawTerms
def group : MergeGroup := .operator 1636 36045
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1636) (leftOrdinal := 0)
    (rightResult := 36045) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12974⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨12974⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37024

namespace LeftMerge37029
def owner : Owner := ⟨.program ⟨214⟩, ⟨7320⟩⟩
def mergeEvent : Nat := 37029
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35915RawTerms
def rightRaw : List Term := Proof.Events029.exact7474RawTerms
def group : MergeGroup := .operator 35915 7474
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35915) (leftOrdinal := 0)
    (rightResult := 7474) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37029

namespace LeftMerge37046
def owner : Owner := ⟨.program ⟨214⟩, ⟨12980⟩⟩
def mergeEvent : Nat := 37046
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩, ⟨.program ⟨214⟩, ⟨12974⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events144.exact37040RawTerms
def rightRaw : List Term := Proof.Events006.exact1639RawTerms
def group : MergeGroup := .operator 37040 1639
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37040) (leftOrdinal := 1)
    (rightResult := 1639) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨12974⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10145⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩, ⟨.program ⟨214⟩, ⟨12974⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37046

namespace LeftMerge37047
def owner : Owner := ⟨.program ⟨214⟩, ⟨12980⟩⟩
def mergeEvent : Nat := 37047
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }
def leftRaw : List Term := Proof.Events144.exact37040RawTerms
def rightRaw : List Term := Proof.Events006.exact1639RawTerms
def group : MergeGroup := .operator 37040 1639
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37040) (leftOrdinal := 0)
    (rightResult := 1639) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10145⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37047

namespace LeftMerge37052
def owner : Owner := ⟨.program ⟨214⟩, ⟨10146⟩⟩
def mergeEvent : Nat := 37052
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events006.exact1639RawTerms
def rightRaw : List Term := Proof.Events140.exact36045RawTerms
def group : MergeGroup := .operator 1639 36045
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1639) (leftOrdinal := 0)
    (rightResult := 36045) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10145⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37052

namespace LeftMerge37057
def owner : Owner := ⟨.program ⟨214⟩, ⟨7300⟩⟩
def mergeEvent : Nat := 37057
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35915RawTerms
def rightRaw : List Term := Proof.Events029.exact7515RawTerms
def group : MergeGroup := .operator 35915 7515
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35915) (leftOrdinal := 0)
    (rightResult := 7515) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37057

namespace LeftMerge37074
def owner : Owner := ⟨.program ⟨214⟩, ⟨10149⟩⟩
def mergeEvent : Nat := 37074
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }
def leftRaw : List Term := Proof.Events144.exact37068RawTerms
def rightRaw : List Term := Proof.Events029.exact7504RawTerms
def group : MergeGroup := .operator 37068 7504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37068) (leftOrdinal := 1)
    (rightResult := 7504) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7876⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37074

namespace LeftMerge37076
def owner : Owner := ⟨.program ⟨214⟩, ⟨10149⟩⟩
def mergeEvent : Nat := 37076
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }
def rhsRaw : List Term := Proof.Events029.exact7474RawTerms
def group : MergeGroup := .relation 37075
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 37075) (rhsResult := 7474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7876⟩⟩) ⟨6788⟩ 7474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37076

namespace LeftMerge37077
def owner : Owner := ⟨.program ⟨214⟩, ⟨10149⟩⟩
def mergeEvent : Nat := 37077
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }
def leftRaw : List Term := Proof.Events144.exact37068RawTerms
def rightRaw : List Term := Proof.Events029.exact7504RawTerms
def group : MergeGroup := .operator 37068 7504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37068) (leftOrdinal := 0)
    (rightResult := 7504) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7876⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37077

namespace LeftMerge37082
def owner : Owner := ⟨.program ⟨214⟩, ⟨12981⟩⟩
def mergeEvent : Nat := 37082
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }
def leftRaw : List Term := Proof.Events144.exact37078RawTerms
def rightRaw : List Term := Proof.Events144.exact37048RawTerms
def group : MergeGroup := .operator 37078 37048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37078) (leftOrdinal := 1)
    (rightResult := 37048) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37082

namespace LeftMerge37090
def owner : Owner := ⟨.program ⟨214⟩, ⟨25615⟩⟩
def mergeEvent : Nat := 37090
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩, ⟨.program ⟨214⟩, ⟨12974⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25614⟩⟩] } }
def leftRaw : List Term := Proof.Events144.exact37084RawTerms
def rightRaw : List Term := Proof.Events144.exact37020RawTerms
def group : MergeGroup := .operator 37084 37020
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37084) (leftOrdinal := 1)
    (rightResult := 37020) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩, ⟨.program ⟨214⟩, ⟨12974⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25614⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩, ⟨.program ⟨214⟩, ⟨12974⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25614⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37090

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
