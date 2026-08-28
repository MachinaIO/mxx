import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge177069
def owner : Owner := ⟨.program ⟨257⟩, ⟨33324⟩⟩
def mergeEvent : Nat := 177069
def frameStart : Nat := 177003
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events691.exact177065RawTerms
def rightRaw : List Term := Proof.Events691.exact177063RawTerms
def group : MergeGroup := .operator 177065 177063
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 177065) (leftOrdinal := 0)
    (rightResult := 177063) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31860⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge177069

namespace LeftMerge177081
def owner : Owner := ⟨.program ⟨257⟩, ⟨34010⟩⟩
def mergeEvent : Nat := 177081
def frameStart : Nat := 177003
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34009⟩⟩] } }
def leftRaw : List Term := Proof.Events691.exact177077RawTerms
def rightRaw : List Term := Proof.Events691.exact177054RawTerms
def group : MergeGroup := .operator 177077 177054
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 177077) (leftOrdinal := 0)
    (rightResult := 177054) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨34009⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34009⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge177081

namespace LeftMerge177082
def owner : Owner := ⟨.program ⟨257⟩, ⟨34010⟩⟩
def mergeEvent : Nat := 177082
def frameStart : Nat := 177003
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34009⟩⟩] } }
def leftRaw : List Term := Proof.Events691.exact177077RawTerms
def rightRaw : List Term := Proof.Events691.exact177054RawTerms
def group : MergeGroup := .operator 177077 177054
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 177077) (leftOrdinal := 1)
    (rightResult := 177054) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨34009⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34009⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge177082

namespace LeftMerge177084
def owner : Owner := ⟨.program ⟨257⟩, ⟨34010⟩⟩
def mergeEvent : Nat := 177084
def frameStart : Nat := 177003
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33136⟩⟩] } }
def rhsRaw : List Term := Proof.Events691.exact177051RawTerms
def group : MergeGroup := .relation 177083
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 177083) (rhsResult := 177051)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨31860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34009⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨34009⟩⟩) ⟨33136⟩ 177051) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33136⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨31860⟩⟩], [⟨.program ⟨257⟩, ⟨33136⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge177084

namespace LeftMerge177092
def owner : Owner := ⟨.program ⟨257⟩, ⟨32180⟩⟩
def mergeEvent : Nat := 177092
def frameStart : Nat := 177003
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32177⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events691.exact177065RawTerms
def rightRaw : List Term := Proof.Events691.exact177088RawTerms
def group : MergeGroup := .operator 177065 177088
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 177065) (leftOrdinal := 0)
    (rightResult := 177088) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32177⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge177092

namespace LeftMerge177109
def owner : Owner := ⟨.program ⟨257⟩, ⟨32775⟩⟩
def mergeEvent : Nat := 177109
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7203⟩⟩] } }
def rhsRaw : List Term := Proof.Events691.exact177106RawTerms
def group : MergeGroup := .relation 177108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 177108) (rhsResult := 177106)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32772⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 177107 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32772⟩⟩]⟩) (none) 177106) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7203⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge177109

namespace LeftMerge177110
def owner : Owner := ⟨.program ⟨257⟩, ⟨32775⟩⟩
def mergeEvent : Nat := 177110
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34009⟩⟩] } }
def rhsRaw : List Term := Proof.Events691.exact177106RawTerms
def group : MergeGroup := .relation 177108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 177108) (rhsResult := 177106)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32772⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 177107 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32772⟩⟩]⟩) (none) 177106) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34009⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34009⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge177110

namespace LeftMerge177111
def owner : Owner := ⟨.program ⟨257⟩, ⟨32775⟩⟩
def mergeEvent : Nat := 177111
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨31860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33136⟩⟩] } }
def rhsRaw : List Term := Proof.Events691.exact177106RawTerms
def group : MergeGroup := .relation 177108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 177108) (rhsResult := 177106)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32772⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 177107 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32772⟩⟩]⟩) (none) 177106) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33136⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨31860⟩⟩], [⟨.program ⟨257⟩, ⟨33136⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge177111

namespace LeftMerge177112
def owner : Owner := ⟨.program ⟨257⟩, ⟨32775⟩⟩
def mergeEvent : Nat := 177112
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events691.exact177106RawTerms
def group : MergeGroup := .relation 177108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 177108) (rhsResult := 177106)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32772⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 177107 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32772⟩⟩]⟩) (none) 177106) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32177⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge177112

namespace LeftMerge177117
def owner : Owner := ⟨.program ⟨257⟩, ⟨34012⟩⟩
def mergeEvent : Nat := 177117
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34009⟩⟩] } }
def leftRaw : List Term := Proof.Events691.exact177113RawTerms
def rightRaw : List Term := Proof.Events691.exact176935RawTerms
def group : MergeGroup := .operator 177113 176935
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 177113) (leftOrdinal := 0)
    (rightResult := 176935) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34009⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34009⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34009⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge177117

namespace LeftMerge177118
def owner : Owner := ⟨.program ⟨257⟩, ⟨34012⟩⟩
def mergeEvent : Nat := 177118
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨31860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33136⟩⟩] } }
def leftRaw : List Term := Proof.Events691.exact177113RawTerms
def rightRaw : List Term := Proof.Events691.exact176935RawTerms
def group : MergeGroup := .operator 177113 176935
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 177113) (leftOrdinal := 2)
    (rightResult := 176935) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨31860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33136⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨31860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33136⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨31860⟩⟩], [⟨.program ⟨257⟩, ⟨33136⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge177118

namespace LeftMerge177126
def owner : Owner := ⟨.program ⟨257⟩, ⟨34013⟩⟩
def mergeEvent : Nat := 177126
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩] } }
def leftRaw : List Term := Proof.Events691.exact177120RawTerms
def rightRaw : List Term := Proof.Events061.exact15822RawTerms
def group : MergeGroup := .operator 177120 15822
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 177120) (leftOrdinal := 0)
    (rightResult := 15822) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7203⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7145⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge177126

namespace LeftMerge177127
def owner : Owner := ⟨.program ⟨257⟩, ⟨34013⟩⟩
def mergeEvent : Nat := 177127
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩] } }
def leftRaw : List Term := Proof.Events691.exact177120RawTerms
def rightRaw : List Term := Proof.Events061.exact15822RawTerms
def group : MergeGroup := .operator 177120 15822
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 177120) (leftOrdinal := 1)
    (rightResult := 15822) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7145⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge177127

namespace LeftMerge177129
def owner : Owner := ⟨.program ⟨257⟩, ⟨34013⟩⟩
def mergeEvent : Nat := 177129
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15815RawTerms
def group : MergeGroup := .relation 177128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 177128) (rhsResult := 15815)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7145⟩⟩) ⟨7038⟩ 15815) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge177129

namespace LeftMerge177143
def owner : Owner := ⟨.program ⟨257⟩, ⟨23991⟩⟩
def mergeEvent : Nat := 177143
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23989⟩⟩] } }
def leftRaw : List Term := Proof.Events668.exact171161RawTerms
def rightRaw : List Term := Proof.Events691.exact177137RawTerms
def group : MergeGroup := .operator 171161 177137
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171161) (leftOrdinal := 0)
    (rightResult := 177137) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23989⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23989⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge177143

namespace LeftMerge177144
def owner : Owner := ⟨.program ⟨257⟩, ⟨23991⟩⟩
def mergeEvent : Nat := 177144
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23989⟩⟩] } }
def leftRaw : List Term := Proof.Events668.exact171161RawTerms
def rightRaw : List Term := Proof.Events691.exact177137RawTerms
def group : MergeGroup := .operator 171161 177137
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171161) (leftOrdinal := 1)
    (rightResult := 177137) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23989⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23989⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge177144

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
