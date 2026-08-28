import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge280392
def owner : Owner := ⟨.program ⟨257⟩, ⟨70987⟩⟩
def mergeEvent : Nat := 280392
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9525⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280332RawTerms
def rightRaw : List Term := Proof.Events065.exact16664RawTerms
def group : MergeGroup := .operator 280332 16664
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280332) (leftOrdinal := 18)
    (rightResult := 16664) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9525⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9525⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge280392

namespace LeftMerge280394
def owner : Owner := ⟨.program ⟨257⟩, ⟨70987⟩⟩
def mergeEvent : Nat := 280394
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }
def rhsRaw : List Term := Proof.Events065.exact16657RawTerms
def group : MergeGroup := .relation 280393
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 280393) (rhsResult := 16657)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9525⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9525⟩⟩) ⟨7269⟩ 16657) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨7269⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge280394

namespace LeftMerge280395
def owner : Owner := ⟨.program ⟨257⟩, ⟨70987⟩⟩
def mergeEvent : Nat := 280395
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7268⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9525⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280332RawTerms
def rightRaw : List Term := Proof.Events065.exact16664RawTerms
def group : MergeGroup := .operator 280332 16664
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280332) (leftOrdinal := 0)
    (rightResult := 16664) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7268⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9525⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7268⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9525⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280395

namespace LeftMerge280400
def owner : Owner := ⟨.program ⟨257⟩, ⟨70988⟩⟩
def mergeEvent : Nat := 280400
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67300⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280396RawTerms
def rightRaw : List Term := Proof.Events1039.exact265993RawTerms
def group : MergeGroup := .operator 280396 265993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280396) (leftOrdinal := 6)
    (rightResult := 265993) (rightOrdinal := 24) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67300⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67300⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67300⟩⟩], [⟨.program ⟨257⟩, ⟨7269⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge280400

namespace LeftMerge280401
def owner : Owner := ⟨.program ⟨257⟩, ⟨70988⟩⟩
def mergeEvent : Nat := 280401
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48252⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280396RawTerms
def rightRaw : List Term := Proof.Events1039.exact265993RawTerms
def group : MergeGroup := .operator 280396 265993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280396) (leftOrdinal := 8)
    (rightResult := 265993) (rightOrdinal := 26) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48252⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48252⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48252⟩⟩], [⟨.program ⟨257⟩, ⟨7269⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280401

namespace LeftMerge280402
def owner : Owner := ⟨.program ⟨257⟩, ⟨70988⟩⟩
def mergeEvent : Nat := 280402
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45572⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280396RawTerms
def rightRaw : List Term := Proof.Events1039.exact265993RawTerms
def group : MergeGroup := .operator 280396 265993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280396) (leftOrdinal := 9)
    (rightResult := 265993) (rightOrdinal := 27) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45572⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45572⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45572⟩⟩], [⟨.program ⟨257⟩, ⟨7269⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280402

namespace LeftMerge280403
def owner : Owner := ⟨.program ⟨257⟩, ⟨70988⟩⟩
def mergeEvent : Nat := 280403
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42895⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280396RawTerms
def rightRaw : List Term := Proof.Events1039.exact265993RawTerms
def group : MergeGroup := .operator 280396 265993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280396) (leftOrdinal := 10)
    (rightResult := 265993) (rightOrdinal := 28) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42895⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42895⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42895⟩⟩], [⟨.program ⟨257⟩, ⟨7269⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280403

namespace LeftMerge280404
def owner : Owner := ⟨.program ⟨257⟩, ⟨70988⟩⟩
def mergeEvent : Nat := 280404
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280396RawTerms
def rightRaw : List Term := Proof.Events1039.exact265993RawTerms
def group : MergeGroup := .operator 280396 265993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280396) (leftOrdinal := 12)
    (rightResult := 265993) (rightOrdinal := 30) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40215⟩⟩], [⟨.program ⟨257⟩, ⟨7269⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280404

namespace LeftMerge280405
def owner : Owner := ⟨.program ⟨257⟩, ⟨70988⟩⟩
def mergeEvent : Nat := 280405
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280396RawTerms
def rightRaw : List Term := Proof.Events1039.exact265993RawTerms
def group : MergeGroup := .operator 280396 265993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280396) (leftOrdinal := 13)
    (rightResult := 265993) (rightOrdinal := 31) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37532⟩⟩], [⟨.program ⟨257⟩, ⟨7269⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280405

namespace LeftMerge280406
def owner : Owner := ⟨.program ⟨257⟩, ⟨70988⟩⟩
def mergeEvent : Nat := 280406
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280396RawTerms
def rightRaw : List Term := Proof.Events1039.exact265993RawTerms
def group : MergeGroup := .operator 280396 265993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280396) (leftOrdinal := 14)
    (rightResult := 265993) (rightOrdinal := 32) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34852⟩⟩], [⟨.program ⟨257⟩, ⟨7269⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280406

namespace LeftMerge280407
def owner : Owner := ⟨.program ⟨257⟩, ⟨70988⟩⟩
def mergeEvent : Nat := 280407
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280396RawTerms
def rightRaw : List Term := Proof.Events1039.exact265993RawTerms
def group : MergeGroup := .operator 280396 265993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280396) (leftOrdinal := 16)
    (rightResult := 265993) (rightOrdinal := 34) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29195⟩⟩], [⟨.program ⟨257⟩, ⟨7269⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280407

namespace LeftMerge280408
def owner : Owner := ⟨.program ⟨257⟩, ⟨70988⟩⟩
def mergeEvent : Nat := 280408
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26515⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280396RawTerms
def rightRaw : List Term := Proof.Events1039.exact265993RawTerms
def group : MergeGroup := .operator 280396 265993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280396) (leftOrdinal := 17)
    (rightResult := 265993) (rightOrdinal := 35) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26515⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26515⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26515⟩⟩], [⟨.program ⟨257⟩, ⟨7269⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280408

namespace LeftMerge280409
def owner : Owner := ⟨.program ⟨257⟩, ⟨70988⟩⟩
def mergeEvent : Nat := 280409
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66006⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280396RawTerms
def rightRaw : List Term := Proof.Events1039.exact265993RawTerms
def group : MergeGroup := .operator 280396 265993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280396) (leftOrdinal := 19)
    (rightResult := 265993) (rightOrdinal := 37) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66006⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66006⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66006⟩⟩], [⟨.program ⟨257⟩, ⟨7269⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280409

namespace LeftMerge280410
def owner : Owner := ⟨.program ⟨257⟩, ⟨70988⟩⟩
def mergeEvent : Nat := 280410
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280396RawTerms
def rightRaw : List Term := Proof.Events1039.exact265993RawTerms
def group : MergeGroup := .operator 280396 265993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280396) (leftOrdinal := 1)
    (rightResult := 265993) (rightOrdinal := 19) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨7269⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280410

namespace LeftMerge280411
def owner : Owner := ⟨.program ⟨257⟩, ⟨70988⟩⟩
def mergeEvent : Nat := 280411
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280396RawTerms
def rightRaw : List Term := Proof.Events1039.exact265993RawTerms
def group : MergeGroup := .operator 280396 265993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280396) (leftOrdinal := 2)
    (rightResult := 265993) (rightOrdinal := 20) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], [⟨.program ⟨257⟩, ⟨7269⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280411

namespace LeftMerge280412
def owner : Owner := ⟨.program ⟨257⟩, ⟨70988⟩⟩
def mergeEvent : Nat := 280412
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56968⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280396RawTerms
def rightRaw : List Term := Proof.Events1039.exact265993RawTerms
def group : MergeGroup := .operator 280396 265993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280396) (leftOrdinal := 3)
    (rightResult := 265993) (rightOrdinal := 21) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56968⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56968⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7269⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56968⟩⟩], [⟨.program ⟨257⟩, ⟨7269⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280412

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
