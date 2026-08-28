import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge7934
def owner : Owner := ⟨.program ⟨214⟩, ⟨22571⟩⟩
def mergeEvent : Nat := 7934
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29654⟩⟩] } }
def rhsRaw : List Term := Proof.Events030.exact7930RawTerms
def group : MergeGroup := .relation 7932
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 7932) (rhsResult := 7930)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22568⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 7931 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22568⟩⟩]⟩) (none) 7930) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29654⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29654⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7934

namespace LeftMerge7935
def owner : Owner := ⟨.program ⟨214⟩, ⟨22571⟩⟩
def mergeEvent : Nat := 7935
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16810⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events030.exact7930RawTerms
def group : MergeGroup := .relation 7932
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 7932) (rhsResult := 7930)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22568⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 7931 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22568⟩⟩]⟩) (none) 7930) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16810⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7935

namespace LeftMerge7936
def owner : Owner := ⟨.program ⟨214⟩, ⟨22571⟩⟩
def mergeEvent : Nat := 7936
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩] } }
def rhsRaw : List Term := Proof.Events030.exact7930RawTerms
def group : MergeGroup := .relation 7932
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 7932) (rhsResult := 7930)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22568⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 7931 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22568⟩⟩]⟩) (none) 7930) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7936

namespace LeftMerge7941
def owner : Owner := ⟨.program ⟨214⟩, ⟨29657⟩⟩
def mergeEvent : Nat := 7941
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16768⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24678⟩⟩] } }
def leftRaw : List Term := Proof.Events031.exact7937RawTerms
def rightRaw : List Term := Proof.Events030.exact7759RawTerms
def group : MergeGroup := .operator 7937 7759
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7937) (leftOrdinal := 2)
    (rightResult := 7759) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16768⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24678⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16768⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24678⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16768⟩⟩], [⟨.program ⟨214⟩, ⟨24678⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7941

namespace LeftMerge7942
def owner : Owner := ⟨.program ⟨214⟩, ⟨29657⟩⟩
def mergeEvent : Nat := 7942
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29654⟩⟩] } }
def leftRaw : List Term := Proof.Events031.exact7937RawTerms
def rightRaw : List Term := Proof.Events030.exact7759RawTerms
def group : MergeGroup := .operator 7937 7759
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7937) (leftOrdinal := 0)
    (rightResult := 7759) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29654⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29654⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29654⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7942

namespace LeftMerge7971
def owner : Owner := ⟨.program ⟨214⟩, ⟨12797⟩⟩
def mergeEvent : Nat := 7971
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact120RawTerms
def rightRaw : List Term := Proof.Events025.exact6449RawTerms
def group : MergeGroup := .operator 120 6449
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120) (leftOrdinal := 0)
    (rightResult := 6449) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7971

namespace LeftMerge7979
def owner : Owner := ⟨.program ⟨214⟩, ⟨7395⟩⟩
def mergeEvent : Nat := 7979
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6314RawTerms
def rightRaw : List Term := Proof.Events031.exact7975RawTerms
def group : MergeGroup := .operator 6314 7975
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6314) (leftOrdinal := 0)
    (rightResult := 7975) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7979

namespace LeftMerge7996
def owner : Owner := ⟨.program ⟨214⟩, ⟨12800⟩⟩
def mergeEvent : Nat := 7996
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events031.exact7990RawTerms
def rightRaw : List Term := Proof.Events000.exact123RawTerms
def group : MergeGroup := .operator 7990 123
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7990) (leftOrdinal := 1)
    (rightResult := 123) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7996

namespace LeftMerge7997
def owner : Owner := ⟨.program ⟨214⟩, ⟨12800⟩⟩
def mergeEvent : Nat := 7997
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }
def leftRaw : List Term := Proof.Events031.exact7990RawTerms
def rightRaw : List Term := Proof.Events000.exact123RawTerms
def group : MergeGroup := .operator 7990 123
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7990) (leftOrdinal := 0)
    (rightResult := 123) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7997

namespace LeftMerge8012
def owner : Owner := ⟨.program ⟨214⟩, ⟨10051⟩⟩
def mergeEvent : Nat := 8012
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact123RawTerms
def rightRaw : List Term := Proof.Events025.exact6449RawTerms
def group : MergeGroup := .operator 123 6449
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 123) (leftOrdinal := 0)
    (rightResult := 6449) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8012

namespace LeftMerge8020
def owner : Owner := ⟨.program ⟨214⟩, ⟨7375⟩⟩
def mergeEvent : Nat := 8020
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6314RawTerms
def rightRaw : List Term := Proof.Events031.exact8016RawTerms
def group : MergeGroup := .operator 6314 8016
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6314) (leftOrdinal := 0)
    (rightResult := 8016) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8020

namespace LeftMerge8037
def owner : Owner := ⟨.program ⟨214⟩, ⟨10054⟩⟩
def mergeEvent : Nat := 8037
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }
def leftRaw : List Term := Proof.Events031.exact8031RawTerms
def rightRaw : List Term := Proof.Events031.exact8005RawTerms
def group : MergeGroup := .operator 8031 8005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8031) (leftOrdinal := 1)
    (rightResult := 8005) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7873⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge8037

namespace LeftMerge8039
def owner : Owner := ⟨.program ⟨214⟩, ⟨10054⟩⟩
def mergeEvent : Nat := 8039
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }
def rhsRaw : List Term := Proof.Events031.exact7975RawTerms
def group : MergeGroup := .relation 8038
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 8038) (rhsResult := 7975)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7873⟩⟩) ⟨6787⟩ 7975) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge8039

namespace LeftMerge8040
def owner : Owner := ⟨.program ⟨214⟩, ⟨10054⟩⟩
def mergeEvent : Nat := 8040
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }
def leftRaw : List Term := Proof.Events031.exact8031RawTerms
def rightRaw : List Term := Proof.Events031.exact8005RawTerms
def group : MergeGroup := .operator 8031 8005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8031) (leftOrdinal := 0)
    (rightResult := 8005) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7873⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8040

namespace LeftMerge8045
def owner : Owner := ⟨.program ⟨214⟩, ⟨12801⟩⟩
def mergeEvent : Nat := 8045
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }
def leftRaw : List Term := Proof.Events031.exact8041RawTerms
def rightRaw : List Term := Proof.Events031.exact7998RawTerms
def group : MergeGroup := .operator 8041 7998
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8041) (leftOrdinal := 1)
    (rightResult := 7998) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8045

namespace LeftMerge8053
def owner : Owner := ⟨.program ⟨214⟩, ⟨25548⟩⟩
def mergeEvent : Nat := 8053
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩] } }
def leftRaw : List Term := Proof.Events031.exact8047RawTerms
def rightRaw : List Term := Proof.Events031.exact7964RawTerms
def group : MergeGroup := .operator 8047 7964
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8047) (leftOrdinal := 1)
    (rightResult := 7964) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25547⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge8053

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
