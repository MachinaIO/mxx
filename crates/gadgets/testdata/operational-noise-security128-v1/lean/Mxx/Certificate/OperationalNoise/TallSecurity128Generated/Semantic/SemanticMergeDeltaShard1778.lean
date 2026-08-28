import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge288093
def owner : Owner := ⟨.program ⟨257⟩, ⟨23376⟩⟩
def mergeEvent : Nat := 288093
def frameStart : Nat := 288000
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21011⟩⟩, ⟨.program ⟨257⟩, ⟨21350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22893⟩⟩] } }
def rhsRaw : List Term := Proof.Events1125.exact288042RawTerms
def group : MergeGroup := .relation 288092
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 288092) (rhsResult := 288042)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21011⟩⟩, ⟨.program ⟨257⟩, ⟨21350⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23373⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23373⟩⟩) ⟨22893⟩ 288042) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22893⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21011⟩⟩, ⟨.program ⟨257⟩, ⟨21350⟩⟩], [⟨.program ⟨257⟩, ⟨22893⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288093

namespace LeftMerge288101
def owner : Owner := ⟨.program ⟨257⟩, ⟨21762⟩⟩
def mergeEvent : Nat := 288101
def frameStart : Nat := 288000
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1125.exact288056RawTerms
def rightRaw : List Term := Proof.Events1125.exact288097RawTerms
def group : MergeGroup := .operator 288056 288097
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288056) (leftOrdinal := 0)
    (rightResult := 288097) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21760⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288101

namespace LeftMerge288118
def owner : Owner := ⟨.program ⟨257⟩, ⟨22312⟩⟩
def mergeEvent : Nat := 288118
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }
def rhsRaw : List Term := Proof.Events1125.exact288115RawTerms
def group : MergeGroup := .relation 288117
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 288117) (rhsResult := 288115)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22309⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 288116 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22309⟩⟩]⟩) (none) 288115) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288118

namespace LeftMerge288119
def owner : Owner := ⟨.program ⟨257⟩, ⟨22312⟩⟩
def mergeEvent : Nat := 288119
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23373⟩⟩] } }
def rhsRaw : List Term := Proof.Events1125.exact288115RawTerms
def group : MergeGroup := .relation 288117
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 288117) (rhsResult := 288115)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22309⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 288116 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22309⟩⟩]⟩) (none) 288115) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23373⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23373⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288119

namespace LeftMerge288120
def owner : Owner := ⟨.program ⟨257⟩, ⟨22312⟩⟩
def mergeEvent : Nat := 288120
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21011⟩⟩, ⟨.program ⟨257⟩, ⟨21350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22893⟩⟩] } }
def rhsRaw : List Term := Proof.Events1125.exact288115RawTerms
def group : MergeGroup := .relation 288117
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 288117) (rhsResult := 288115)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22309⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 288116 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22309⟩⟩]⟩) (none) 288115) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21011⟩⟩, ⟨.program ⟨257⟩, ⟨21350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22893⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21011⟩⟩, ⟨.program ⟨257⟩, ⟨21350⟩⟩], [⟨.program ⟨257⟩, ⟨22893⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288120

namespace LeftMerge288121
def owner : Owner := ⟨.program ⟨257⟩, ⟨22312⟩⟩
def mergeEvent : Nat := 288121
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1125.exact288115RawTerms
def group : MergeGroup := .relation 288117
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 288117) (rhsResult := 288115)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22309⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 288116 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22309⟩⟩]⟩) (none) 288115) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288121

namespace LeftMerge288126
def owner : Owner := ⟨.program ⟨257⟩, ⟨23375⟩⟩
def mergeEvent : Nat := 288126
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21011⟩⟩, ⟨.program ⟨257⟩, ⟨21350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22893⟩⟩] } }
def leftRaw : List Term := Proof.Events1125.exact288122RawTerms
def rightRaw : List Term := Proof.Events1124.exact287938RawTerms
def group : MergeGroup := .operator 288122 287938
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288122) (leftOrdinal := 2)
    (rightResult := 287938) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21011⟩⟩, ⟨.program ⟨257⟩, ⟨21350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22893⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21011⟩⟩, ⟨.program ⟨257⟩, ⟨21350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22893⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21011⟩⟩, ⟨.program ⟨257⟩, ⟨21350⟩⟩], [⟨.program ⟨257⟩, ⟨22893⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288126

namespace LeftMerge288127
def owner : Owner := ⟨.program ⟨257⟩, ⟨23375⟩⟩
def mergeEvent : Nat := 288127
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23373⟩⟩] } }
def leftRaw : List Term := Proof.Events1125.exact288122RawTerms
def rightRaw : List Term := Proof.Events1124.exact287938RawTerms
def group : MergeGroup := .operator 288122 287938
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288122) (leftOrdinal := 1)
    (rightResult := 287938) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23373⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23373⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23373⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288127

namespace LeftMerge288135
def owner : Owner := ⟨.program ⟨257⟩, ⟨23688⟩⟩
def mergeEvent : Nat := 288135
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23686⟩⟩] } }
def leftRaw : List Term := Proof.Events1125.exact288129RawTerms
def rightRaw : List Term := Proof.Events1124.exact287854RawTerms
def group : MergeGroup := .operator 288129 287854
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288129) (leftOrdinal := 0)
    (rightResult := 287854) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23686⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23686⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288135

namespace LeftMerge288136
def owner : Owner := ⟨.program ⟨257⟩, ⟨23688⟩⟩
def mergeEvent : Nat := 288136
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23686⟩⟩] } }
def leftRaw : List Term := Proof.Events1125.exact288129RawTerms
def rightRaw : List Term := Proof.Events1124.exact287854RawTerms
def group : MergeGroup := .operator 288129 287854
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288129) (leftOrdinal := 1)
    (rightResult := 287854) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23686⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23686⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288136

namespace LeftMerge288138
def owner : Owner := ⟨.program ⟨257⟩, ⟨23688⟩⟩
def mergeEvent : Nat := 288138
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23027⟩⟩] } }
def rhsRaw : List Term := Proof.Events1124.exact287851RawTerms
def group : MergeGroup := .relation 288137
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 288137) (rhsResult := 287851)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23686⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23686⟩⟩) ⟨23027⟩ 287851) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23027⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21760⟩⟩], [⟨.program ⟨257⟩, ⟨23027⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288138

namespace LeftMerge288152
def owner : Owner := ⟨.program ⟨257⟩, ⟨22559⟩⟩
def mergeEvent : Nat := 288152
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22556⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280745RawTerms
def rightRaw : List Term := Proof.Events1125.exact288146RawTerms
def group : MergeGroup := .operator 280745 288146
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280745) (leftOrdinal := 0)
    (rightResult := 288146) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22556⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288152

namespace LeftMerge288273
def owner : Owner := ⟨.program ⟨257⟩, ⟨23264⟩⟩
def mergeEvent : Nat := 288273
def frameStart : Nat := 288207
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1126.exact288269RawTerms
def rightRaw : List Term := Proof.Events1126.exact288267RawTerms
def group : MergeGroup := .operator 288269 288267
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288269) (leftOrdinal := 0)
    (rightResult := 288267) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21760⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288273

namespace LeftMerge288285
def owner : Owner := ⟨.program ⟨257⟩, ⟨23687⟩⟩
def mergeEvent : Nat := 288285
def frameStart : Nat := 288207
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23686⟩⟩] } }
def leftRaw : List Term := Proof.Events1126.exact288281RawTerms
def rightRaw : List Term := Proof.Events1126.exact288258RawTerms
def group : MergeGroup := .operator 288281 288258
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288281) (leftOrdinal := 0)
    (rightResult := 288258) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23686⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23686⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288285

namespace LeftMerge288286
def owner : Owner := ⟨.program ⟨257⟩, ⟨23687⟩⟩
def mergeEvent : Nat := 288286
def frameStart : Nat := 288207
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23686⟩⟩] } }
def leftRaw : List Term := Proof.Events1126.exact288281RawTerms
def rightRaw : List Term := Proof.Events1126.exact288258RawTerms
def group : MergeGroup := .operator 288281 288258
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288281) (leftOrdinal := 1)
    (rightResult := 288258) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23686⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23686⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288286

namespace LeftMerge288288
def owner : Owner := ⟨.program ⟨257⟩, ⟨23687⟩⟩
def mergeEvent : Nat := 288288
def frameStart : Nat := 288207
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23027⟩⟩] } }
def rhsRaw : List Term := Proof.Events1125.exact288255RawTerms
def group : MergeGroup := .relation 288287
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 288287) (rhsResult := 288255)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23686⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23686⟩⟩) ⟨23027⟩ 288255) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23027⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21760⟩⟩], [⟨.program ⟨257⟩, ⟨23027⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288288

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
