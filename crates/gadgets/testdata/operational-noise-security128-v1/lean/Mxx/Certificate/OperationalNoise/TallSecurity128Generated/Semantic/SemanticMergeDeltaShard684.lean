import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge113934
def owner : Owner := ⟨.program ⟨257⟩, ⟨71269⟩⟩
def mergeEvent : Nat := 113934
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨34976⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events410.exact105125RawTerms
def group : MergeGroup := .relation 113933
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 113933) (rhsResult := 105125)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨34976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71267⟩⟩) ⟨68836⟩ 105125) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨34976⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge113934

namespace LeftMerge113935
def owner : Owner := ⟨.program ⟨257⟩, ⟨71269⟩⟩
def mergeEvent : Nat := 113935
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events444.exact113905RawTerms
def rightRaw : List Term := Proof.Events410.exact105128RawTerms
def group : MergeGroup := .operator 113905 105128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 113905) (leftOrdinal := 11)
    (rightResult := 105128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge113935

namespace LeftMerge113936
def owner : Owner := ⟨.program ⟨257⟩, ⟨71269⟩⟩
def mergeEvent : Nat := 113936
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29312⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events444.exact113905RawTerms
def rightRaw : List Term := Proof.Events410.exact105128RawTerms
def group : MergeGroup := .operator 113905 105128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 113905) (leftOrdinal := 22)
    (rightResult := 105128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29312⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge113936

namespace LeftMerge113938
def owner : Owner := ⟨.program ⟨257⟩, ⟨71269⟩⟩
def mergeEvent : Nat := 113938
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29312⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events410.exact105125RawTerms
def group : MergeGroup := .relation 113937
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 113937) (rhsResult := 105125)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71267⟩⟩) ⟨68836⟩ 105125) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29312⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge113938

namespace LeftMerge113939
def owner : Owner := ⟨.program ⟨257⟩, ⟨71269⟩⟩
def mergeEvent : Nat := 113939
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events444.exact113905RawTerms
def rightRaw : List Term := Proof.Events410.exact105128RawTerms
def group : MergeGroup := .operator 113905 105128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 113905) (leftOrdinal := 10)
    (rightResult := 105128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge113939

namespace LeftMerge113940
def owner : Owner := ⟨.program ⟨257⟩, ⟨71269⟩⟩
def mergeEvent : Nat := 113940
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events444.exact113905RawTerms
def rightRaw : List Term := Proof.Events410.exact105128RawTerms
def group : MergeGroup := .operator 113905 105128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 113905) (leftOrdinal := 21)
    (rightResult := 105128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge113940

namespace LeftMerge113942
def owner : Owner := ⟨.program ⟨257⟩, ⟨71269⟩⟩
def mergeEvent : Nat := 113942
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events410.exact105125RawTerms
def group : MergeGroup := .relation 113941
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 113941) (rhsResult := 105125)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71267⟩⟩) ⟨68836⟩ 105125) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge113942

namespace LeftMerge113943
def owner : Owner := ⟨.program ⟨257⟩, ⟨71269⟩⟩
def mergeEvent : Nat := 113943
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events444.exact113905RawTerms
def rightRaw : List Term := Proof.Events410.exact105128RawTerms
def group : MergeGroup := .operator 113905 105128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 113905) (leftOrdinal := 9)
    (rightResult := 105128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge113943

namespace LeftMerge113944
def owner : Owner := ⟨.program ⟨257⟩, ⟨71269⟩⟩
def mergeEvent : Nat := 113944
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨66671⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events444.exact113905RawTerms
def rightRaw : List Term := Proof.Events410.exact105128RawTerms
def group : MergeGroup := .operator 113905 105128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 113905) (leftOrdinal := 35)
    (rightResult := 105128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨66671⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨66671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge113944

namespace LeftMerge113946
def owner : Owner := ⟨.program ⟨257⟩, ⟨71269⟩⟩
def mergeEvent : Nat := 113946
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨66671⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events410.exact105125RawTerms
def group : MergeGroup := .relation 113945
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 113945) (rhsResult := 105125)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨66671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71267⟩⟩) ⟨68836⟩ 105125) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨66671⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge113946

namespace LeftMerge113947
def owner : Owner := ⟨.program ⟨257⟩, ⟨71269⟩⟩
def mergeEvent : Nat := 113947
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events444.exact113905RawTerms
def rightRaw : List Term := Proof.Events410.exact105128RawTerms
def group : MergeGroup := .operator 113905 105128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 113905) (leftOrdinal := 8)
    (rightResult := 105128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge113947

namespace LeftMerge113948
def owner : Owner := ⟨.program ⟨257⟩, ⟨71269⟩⟩
def mergeEvent : Nat := 113948
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨63100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events444.exact113905RawTerms
def rightRaw : List Term := Proof.Events410.exact105128RawTerms
def group : MergeGroup := .operator 113905 105128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 113905) (leftOrdinal := 34)
    (rightResult := 105128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨63100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨63100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge113948

namespace LeftMerge113950
def owner : Owner := ⟨.program ⟨257⟩, ⟨71269⟩⟩
def mergeEvent : Nat := 113950
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨63100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events410.exact105125RawTerms
def group : MergeGroup := .relation 113949
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 113949) (rhsResult := 105125)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨63100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71267⟩⟩) ⟨68836⟩ 105125) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨63100⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge113950

namespace LeftMerge113951
def owner : Owner := ⟨.program ⟨257⟩, ⟨71269⟩⟩
def mergeEvent : Nat := 113951
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events444.exact113905RawTerms
def rightRaw : List Term := Proof.Events410.exact105128RawTerms
def group : MergeGroup := .operator 113905 105128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 113905) (leftOrdinal := 7)
    (rightResult := 105128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge113951

namespace LeftMerge113952
def owner : Owner := ⟨.program ⟨257⟩, ⟨71269⟩⟩
def mergeEvent : Nat := 113952
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨60120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events444.exact113905RawTerms
def rightRaw : List Term := Proof.Events410.exact105128RawTerms
def group : MergeGroup := .operator 113905 105128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 113905) (leftOrdinal := 33)
    (rightResult := 105128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨60120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨60120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge113952

namespace LeftMerge113954
def owner : Owner := ⟨.program ⟨257⟩, ⟨71269⟩⟩
def mergeEvent : Nat := 113954
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨60120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events410.exact105125RawTerms
def group : MergeGroup := .relation 113953
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 113953) (rhsResult := 105125)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨60120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71267⟩⟩) ⟨68836⟩ 105125) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨60120⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge113954

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
