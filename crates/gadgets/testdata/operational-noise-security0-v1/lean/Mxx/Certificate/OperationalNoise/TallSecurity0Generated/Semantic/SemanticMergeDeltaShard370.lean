import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge60981
def owner : Owner := ⟨.program ⟨214⟩, ⟨30134⟩⟩
def mergeEvent : Nat := 60981
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30132⟩⟩] } }
def leftRaw : List Term := Proof.Events199.exact50948RawTerms
def rightRaw : List Term := Proof.Events238.exact60974RawTerms
def group : MergeGroup := .operator 50948 60974
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50948) (leftOrdinal := 1)
    (rightResult := 60974) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30132⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30132⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60981

namespace LeftMerge60983
def owner : Owner := ⟨.program ⟨214⟩, ⟨30134⟩⟩
def mergeEvent : Nat := 60983
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24794⟩⟩] } }
def rhsRaw : List Term := Proof.Events238.exact60971RawTerms
def group : MergeGroup := .relation 60982
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60982) (rhsResult := 60971)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30132⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨30132⟩⟩) ⟨24794⟩ 60971) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24794⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], [⟨.program ⟨214⟩, ⟨24794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60983

namespace LeftMerge60997
def owner : Owner := ⟨.program ⟨214⟩, ⟨22775⟩⟩
def mergeEvent : Nat := 60997
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22772⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events238.exact60991RawTerms
def group : MergeGroup := .operator 50762 60991
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 60991) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22772⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22772⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60997

namespace LeftMerge61118
def owner : Owner := ⟨.program ⟨214⟩, ⟨17057⟩⟩
def mergeEvent : Nat := 61118
def frameStart : Nat := 61052
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61114RawTerms
def rightRaw : List Term := Proof.Events238.exact61112RawTerms
def group : MergeGroup := .operator 61114 61112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61114) (leftOrdinal := 0)
    (rightResult := 61112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61118

namespace LeftMerge61130
def owner : Owner := ⟨.program ⟨214⟩, ⟨30133⟩⟩
def mergeEvent : Nat := 61130
def frameStart : Nat := 61052
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30132⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61126RawTerms
def rightRaw : List Term := Proof.Events238.exact61103RawTerms
def group : MergeGroup := .operator 61126 61103
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61126) (leftOrdinal := 0)
    (rightResult := 61103) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30132⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30132⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61130

namespace LeftMerge61131
def owner : Owner := ⟨.program ⟨214⟩, ⟨30133⟩⟩
def mergeEvent : Nat := 61131
def frameStart : Nat := 61052
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30132⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61126RawTerms
def rightRaw : List Term := Proof.Events238.exact61103RawTerms
def group : MergeGroup := .operator 61126 61103
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61126) (leftOrdinal := 1)
    (rightResult := 61103) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30132⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30132⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge61131

namespace LeftMerge61133
def owner : Owner := ⟨.program ⟨214⟩, ⟨30133⟩⟩
def mergeEvent : Nat := 61133
def frameStart : Nat := 61052
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24794⟩⟩] } }
def rhsRaw : List Term := Proof.Events238.exact61100RawTerms
def group : MergeGroup := .relation 61132
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 61132) (rhsResult := 61100)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨17015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30132⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨30132⟩⟩) ⟨24794⟩ 61100) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24794⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨17015⟩⟩], [⟨.program ⟨214⟩, ⟨24794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge61133

namespace LeftMerge61141
def owner : Owner := ⟨.program ⟨214⟩, ⟨18130⟩⟩
def mergeEvent : Nat := 61141
def frameStart : Nat := 61052
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18128⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61114RawTerms
def rightRaw : List Term := Proof.Events238.exact61137RawTerms
def group : MergeGroup := .operator 61114 61137
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61114) (leftOrdinal := 0)
    (rightResult := 61137) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18128⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18128⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61141

namespace LeftMerge61158
def owner : Owner := ⟨.program ⟨214⟩, ⟨22775⟩⟩
def mergeEvent : Nat := 61158
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6742⟩⟩] } }
def rhsRaw : List Term := Proof.Events238.exact61155RawTerms
def group : MergeGroup := .relation 61157
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 61157) (rhsResult := 61155)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22772⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 61156 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22772⟩⟩]⟩) (none) 61155) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6742⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6742⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61158

namespace LeftMerge61159
def owner : Owner := ⟨.program ⟨214⟩, ⟨22775⟩⟩
def mergeEvent : Nat := 61159
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30132⟩⟩] } }
def rhsRaw : List Term := Proof.Events238.exact61155RawTerms
def group : MergeGroup := .relation 61157
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 61157) (rhsResult := 61155)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22772⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 61156 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22772⟩⟩]⟩) (none) 61155) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30132⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30132⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge61159

namespace LeftMerge61160
def owner : Owner := ⟨.program ⟨214⟩, ⟨22775⟩⟩
def mergeEvent : Nat := 61160
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24794⟩⟩] } }
def rhsRaw : List Term := Proof.Events238.exact61155RawTerms
def group : MergeGroup := .relation 61157
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 61157) (rhsResult := 61155)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22772⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 61156 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22772⟩⟩]⟩) (none) 61155) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24794⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], [⟨.program ⟨214⟩, ⟨24794⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61160

namespace LeftMerge61161
def owner : Owner := ⟨.program ⟨214⟩, ⟨22775⟩⟩
def mergeEvent : Nat := 61161
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18128⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events238.exact61155RawTerms
def group : MergeGroup := .relation 61157
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 61157) (rhsResult := 61155)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22772⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 61156 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22772⟩⟩]⟩) (none) 61155) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18128⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18128⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge61161

namespace LeftMerge61166
def owner : Owner := ⟨.program ⟨214⟩, ⟨30135⟩⟩
def mergeEvent : Nat := 61166
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30132⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61162RawTerms
def rightRaw : List Term := Proof.Events238.exact60984RawTerms
def group : MergeGroup := .operator 61162 60984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61162) (leftOrdinal := 0)
    (rightResult := 60984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30132⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30132⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30132⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61166

namespace LeftMerge61167
def owner : Owner := ⟨.program ⟨214⟩, ⟨30135⟩⟩
def mergeEvent : Nat := 61167
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24794⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61162RawTerms
def rightRaw : List Term := Proof.Events238.exact60984RawTerms
def group : MergeGroup := .operator 61162 60984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61162) (leftOrdinal := 2)
    (rightResult := 60984) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24794⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24794⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], [⟨.program ⟨214⟩, ⟨24794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge61167

namespace LeftMerge61175
def owner : Owner := ⟨.program ⟨214⟩, ⟨30136⟩⟩
def mergeEvent : Nat := 61175
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61169RawTerms
def rightRaw : List Term := Proof.Events021.exact5519RawTerms
def group : MergeGroup := .operator 61169 5519
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61169) (leftOrdinal := 0)
    (rightResult := 5519) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6742⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6657⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61175

namespace LeftMerge61176
def owner : Owner := ⟨.program ⟨214⟩, ⟨30136⟩⟩
def mergeEvent : Nat := 61176
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18128⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61169RawTerms
def rightRaw : List Term := Proof.Events021.exact5519RawTerms
def group : MergeGroup := .operator 61169 5519
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61169) (leftOrdinal := 1)
    (rightResult := 5519) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18128⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6657⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18128⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge61176

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
