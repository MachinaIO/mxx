import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge251341
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251341
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 0)
    (rightResult := 12772) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251341

namespace LeftMerge251342
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251342
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 0)
    (rightResult := 12772) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251342

namespace LeftMerge251343
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251343
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 0)
    (rightResult := 12772) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251343

namespace LeftMerge251344
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251344
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 0)
    (rightResult := 12772) (rightOrdinal := 4) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251344

namespace LeftMerge251345
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251345
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 0)
    (rightResult := 12772) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251345

namespace LeftMerge251346
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251346
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 0)
    (rightResult := 12772) (rightOrdinal := 10) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251346

namespace LeftMerge251347
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251347
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 0)
    (rightResult := 12772) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251347

namespace LeftMerge251348
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251348
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 0)
    (rightResult := 12772) (rightOrdinal := 17) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251348

namespace LeftMerge251349
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251349
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 1)
    (rightResult := 12772) (rightOrdinal := 5) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67363⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67363⟩⟩], [⟨.program ⟨257⟩, ⟨7267⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251349

namespace LeftMerge251350
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251350
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48294⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 1)
    (rightResult := 12772) (rightOrdinal := 7) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48294⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48294⟩⟩], [⟨.program ⟨257⟩, ⟨7267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251350

namespace LeftMerge251351
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251351
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45614⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 1)
    (rightResult := 12772) (rightOrdinal := 8) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45614⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45614⟩⟩], [⟨.program ⟨257⟩, ⟨7267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251351

namespace LeftMerge251352
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251352
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42937⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 1)
    (rightResult := 12772) (rightOrdinal := 9) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42937⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42937⟩⟩], [⟨.program ⟨257⟩, ⟨7267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251352

namespace LeftMerge251353
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251353
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40257⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 1)
    (rightResult := 12772) (rightOrdinal := 11) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40257⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40257⟩⟩], [⟨.program ⟨257⟩, ⟨7267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251353

namespace LeftMerge251354
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251354
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37574⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 1)
    (rightResult := 12772) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37574⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37574⟩⟩], [⟨.program ⟨257⟩, ⟨7267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251354

namespace LeftMerge251355
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251355
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34894⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 1)
    (rightResult := 12772) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34894⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34894⟩⟩], [⟨.program ⟨257⟩, ⟨7267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251355

namespace LeftMerge251356
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def mergeEvent : Nat := 251356
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29237⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251288RawTerms
def rightRaw : List Term := Proof.Events049.exact12772RawTerms
def group : MergeGroup := .operator 251288 12772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251288) (leftOrdinal := 1)
    (rightResult := 12772) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7267⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29237⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29237⟩⟩], [⟨.program ⟨257⟩, ⟨7267⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251356

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
