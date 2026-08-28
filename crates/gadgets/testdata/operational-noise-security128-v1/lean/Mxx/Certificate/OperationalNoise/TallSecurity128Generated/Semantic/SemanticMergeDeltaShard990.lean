import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge162916
def owner : Owner := ⟨.program ⟨257⟩, ⟨20555⟩⟩
def mergeEvent : Nat := 162916
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20552⟩⟩] } }
def leftRaw : List Term := Proof.Events636.exact162912RawTerms
def rightRaw : List Term := Proof.Events635.exact162734RawTerms
def group : MergeGroup := .operator 162912 162734
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 162912) (leftOrdinal := 0)
    (rightResult := 162734) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20552⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20552⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20552⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge162916

namespace LeftMerge162917
def owner : Owner := ⟨.program ⟨257⟩, ⟨20555⟩⟩
def mergeEvent : Nat := 162917
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18564⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19833⟩⟩] } }
def leftRaw : List Term := Proof.Events636.exact162912RawTerms
def rightRaw : List Term := Proof.Events635.exact162734RawTerms
def group : MergeGroup := .operator 162912 162734
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 162912) (leftOrdinal := 2)
    (rightResult := 162734) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18564⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19833⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18564⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19833⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18564⟩⟩], [⟨.program ⟨257⟩, ⟨19833⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162917

namespace LeftMerge162925
def owner : Owner := ⟨.program ⟨257⟩, ⟨20556⟩⟩
def mergeEvent : Nat := 162925
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩] } }
def leftRaw : List Term := Proof.Events636.exact162919RawTerms
def rightRaw : List Term := Proof.Events061.exact15862RawTerms
def group : MergeGroup := .operator 162919 15862
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 162919) (leftOrdinal := 0)
    (rightResult := 15862) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7199⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7165⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge162925

namespace LeftMerge162926
def owner : Owner := ⟨.program ⟨257⟩, ⟨20556⟩⟩
def mergeEvent : Nat := 162926
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩] } }
def leftRaw : List Term := Proof.Events636.exact162919RawTerms
def rightRaw : List Term := Proof.Events061.exact15862RawTerms
def group : MergeGroup := .operator 162919 15862
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 162919) (leftOrdinal := 1)
    (rightResult := 15862) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7165⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162926

namespace LeftMerge162928
def owner : Owner := ⟨.program ⟨257⟩, ⟨20556⟩⟩
def mergeEvent : Nat := 162928
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15855RawTerms
def group : MergeGroup := .relation 162927
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 162927) (rhsResult := 15855)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7165⟩⟩) ⟨7048⟩ 15855) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162928

namespace LeftMerge162942
def owner : Owner := ⟨.program ⟨257⟩, ⟨17672⟩⟩
def mergeEvent : Nat := 162942
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩] } }
def leftRaw : List Term := Proof.Events615.exact157500RawTerms
def rightRaw : List Term := Proof.Events636.exact162936RawTerms
def group : MergeGroup := .operator 157500 162936
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 157500) (leftOrdinal := 0)
    (rightResult := 162936) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17670⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge162942

namespace LeftMerge162943
def owner : Owner := ⟨.program ⟨257⟩, ⟨17672⟩⟩
def mergeEvent : Nat := 162943
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩] } }
def leftRaw : List Term := Proof.Events615.exact157500RawTerms
def rightRaw : List Term := Proof.Events636.exact162936RawTerms
def group : MergeGroup := .operator 157500 162936
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 157500) (leftOrdinal := 1)
    (rightResult := 162936) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17670⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162943

namespace LeftMerge162945
def owner : Owner := ⟨.program ⟨257⟩, ⟨17672⟩⟩
def mergeEvent : Nat := 162945
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16973⟩⟩] } }
def rhsRaw : List Term := Proof.Events636.exact162933RawTerms
def group : MergeGroup := .relation 162944
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 162944) (rhsResult := 162933)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17670⟩⟩) ⟨16973⟩ 162933) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16973⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨16973⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162945

namespace LeftMerge162959
def owner : Owner := ⟨.program ⟨257⟩, ⟨16535⟩⟩
def mergeEvent : Nat := 162959
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16532⟩⟩] } }
def leftRaw : List Term := Proof.Events582.exact149120RawTerms
def rightRaw : List Term := Proof.Events636.exact162953RawTerms
def group : MergeGroup := .operator 149120 162953
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149120) (leftOrdinal := 0)
    (rightResult := 162953) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16532⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge162959

namespace LeftMerge163080
def owner : Owner := ⟨.program ⟨257⟩, ⟨17196⟩⟩
def mergeEvent : Nat := 163080
def frameStart : Nat := 163014
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events637.exact163076RawTerms
def rightRaw : List Term := Proof.Events637.exact163074RawTerms
def group : MergeGroup := .operator 163076 163074
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163076) (leftOrdinal := 0)
    (rightResult := 163074) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15764⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163080

namespace LeftMerge163092
def owner : Owner := ⟨.program ⟨257⟩, ⟨17671⟩⟩
def mergeEvent : Nat := 163092
def frameStart : Nat := 163014
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩] } }
def leftRaw : List Term := Proof.Events637.exact163088RawTerms
def rightRaw : List Term := Proof.Events636.exact163065RawTerms
def group : MergeGroup := .operator 163088 163065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163088) (leftOrdinal := 0)
    (rightResult := 163065) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17670⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163092

namespace LeftMerge163093
def owner : Owner := ⟨.program ⟨257⟩, ⟨17671⟩⟩
def mergeEvent : Nat := 163093
def frameStart : Nat := 163014
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩] } }
def leftRaw : List Term := Proof.Events637.exact163088RawTerms
def rightRaw : List Term := Proof.Events636.exact163065RawTerms
def group : MergeGroup := .operator 163088 163065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163088) (leftOrdinal := 1)
    (rightResult := 163065) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17670⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge163093

namespace LeftMerge163095
def owner : Owner := ⟨.program ⟨257⟩, ⟨17671⟩⟩
def mergeEvent : Nat := 163095
def frameStart : Nat := 163014
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16973⟩⟩] } }
def rhsRaw : List Term := Proof.Events636.exact163062RawTerms
def group : MergeGroup := .relation 163094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 163094) (rhsResult := 163062)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17670⟩⟩) ⟨16973⟩ 163062) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16973⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨16973⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge163095

namespace LeftMerge163103
def owner : Owner := ⟨.program ⟨257⟩, ⟨15985⟩⟩
def mergeEvent : Nat := 163103
def frameStart : Nat := 163014
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15982⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events637.exact163076RawTerms
def rightRaw : List Term := Proof.Events637.exact163099RawTerms
def group : MergeGroup := .operator 163076 163099
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163076) (leftOrdinal := 0)
    (rightResult := 163099) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15982⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163103

namespace LeftMerge163120
def owner : Owner := ⟨.program ⟨257⟩, ⟨16535⟩⟩
def mergeEvent : Nat := 163120
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7197⟩⟩] } }
def rhsRaw : List Term := Proof.Events637.exact163117RawTerms
def group : MergeGroup := .relation 163119
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 163119) (rhsResult := 163117)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16532⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 163118 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16532⟩⟩]⟩) (none) 163117) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7197⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163120

namespace LeftMerge163121
def owner : Owner := ⟨.program ⟨257⟩, ⟨16535⟩⟩
def mergeEvent : Nat := 163121
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩] } }
def rhsRaw : List Term := Proof.Events637.exact163117RawTerms
def group : MergeGroup := .relation 163119
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 163119) (rhsResult := 163117)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16532⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 163118 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16532⟩⟩]⟩) (none) 163117) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge163121

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
