import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge251357
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251357
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26557⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 1)
    (rightResult := 12772) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26557⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26557⟩⟩], [⟨.program ⟨257⟩, ⟨7267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251357

namespace LeftMerge251358
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251358
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 1)
    (rightResult := 12772) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], [⟨.program ⟨257⟩, ⟨7267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251358

namespace LeftMerge251359
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251359
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 1)
    (rightResult := 12772) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], [⟨.program ⟨257⟩, ⟨7267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251359

namespace LeftMerge251360
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251360
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 1)
    (rightResult := 12772) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], [⟨.program ⟨257⟩, ⟨7267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251360

namespace LeftMerge251361
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251361
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 1)
    (rightResult := 12772) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], [⟨.program ⟨257⟩, ⟨7267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251361

namespace LeftMerge251362
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251362
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 1)
    (rightResult := 12772) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], [⟨.program ⟨257⟩, ⟨7267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251362

namespace LeftMerge251363
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251363
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 1)
    (rightResult := 12772) (rightOrdinal := 4) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], [⟨.program ⟨257⟩, ⟨7267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251363

namespace LeftMerge251364
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251364
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 1)
    (rightResult := 12772) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], [⟨.program ⟨257⟩, ⟨7267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251364

namespace LeftMerge251365
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251365
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 1)
    (rightResult := 12772) (rightOrdinal := 10) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], [⟨.program ⟨257⟩, ⟨7267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251365

namespace LeftMerge251366
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251366
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 1)
    (rightResult := 12772) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], [⟨.program ⟨257⟩, ⟨7267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251366

namespace LeftMerge251367
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251367
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 1)
    (rightResult := 12772) (rightOrdinal := 17) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], [⟨.program ⟨257⟩, ⟨7267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251367

namespace LeftMerge251402
def owner : Owner := ⟨.program ⟨257⟩, ⟨6925⟩⟩
def mergeEvent : Nat := 251402
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251273RawTerms
def rightRaw : List Term := Proof.Events000.exact2RawTerms
def group : MergeGroup := .operator 251273 2
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251273) (leftOrdinal := 0)
    (rightResult := 2) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251402

namespace LeftMerge251407
def owner : Owner := ⟨.program ⟨257⟩, ⟨47717⟩⟩
def mergeEvent : Nat := 251407
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events047.exact12062RawTerms
def rightRaw : List Term := Proof.Events982.exact251403RawTerms
def group : MergeGroup := .operator 12062 251403
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12062) (leftOrdinal := 0)
    (rightResult := 251403) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨47714⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251407

namespace LeftMerge251412
def owner : Owner := ⟨.program ⟨257⟩, ⟨8021⟩⟩
def mergeEvent : Nat := 251412
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251273RawTerms
def rightRaw : List Term := Proof.Events066.exact17065RawTerms
def group : MergeGroup := .operator 251273 17065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251273) (leftOrdinal := 0)
    (rightResult := 17065) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251412

namespace LeftMerge251429
def owner : Owner := ⟨.program ⟨257⟩, ⟨47720⟩⟩
def mergeEvent : Nat := 251429
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251423RawTerms
def rightRaw : List Term := Proof.Events047.exact12065RawTerms
def group : MergeGroup := .operator 251423 12065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251423) (leftOrdinal := 1)
    (rightResult := 12065) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15006⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251429

namespace LeftMerge251430
def owner : Owner := ⟨.program ⟨257⟩, ⟨47720⟩⟩
def mergeEvent : Nat := 251430
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251423RawTerms
def rightRaw : List Term := Proof.Events047.exact12065RawTerms
def group : MergeGroup := .operator 251423 12065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251423) (leftOrdinal := 0)
    (rightResult := 12065) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15006⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251430

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
