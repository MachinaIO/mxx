import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge5303
def owner : Owner := ⟨.program ⟨257⟩, ⟨67480⟩⟩
def mergeEvent : Nat := 5303
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29315⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events020.exact5292RawTerms
def rightRaw : List Term := Proof.Events017.exact4569RawTerms
def group : MergeGroup := .operator 5292 4569
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5292) (leftOrdinal := 15)
    (rightResult := 4569) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29315⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29315⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5303

namespace LeftMerge5304
def owner : Owner := ⟨.program ⟨257⟩, ⟨67480⟩⟩
def mergeEvent : Nat := 5304
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26635⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events020.exact5292RawTerms
def rightRaw : List Term := Proof.Events017.exact4569RawTerms
def group : MergeGroup := .operator 5292 4569
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5292) (leftOrdinal := 16)
    (rightResult := 4569) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26635⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26635⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5304

namespace LeftMerge5305
def owner : Owner := ⟨.program ⟨257⟩, ⟨67480⟩⟩
def mergeEvent : Nat := 5305
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66658⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events020.exact5292RawTerms
def rightRaw : List Term := Proof.Events017.exact4569RawTerms
def group : MergeGroup := .operator 5292 4569
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5292) (leftOrdinal := 18)
    (rightResult := 4569) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66658⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66658⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5305

namespace LeftMerge5306
def owner : Owner := ⟨.program ⟨257⟩, ⟨67480⟩⟩
def mergeEvent : Nat := 5306
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨63104⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events020.exact5292RawTerms
def rightRaw : List Term := Proof.Events017.exact4569RawTerms
def group : MergeGroup := .operator 5292 4569
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5292) (leftOrdinal := 0)
    (rightResult := 4569) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63104⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨63104⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5306

namespace LeftMerge5307
def owner : Owner := ⟨.program ⟨257⟩, ⟨67480⟩⟩
def mergeEvent : Nat := 5307
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨60124⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events020.exact5292RawTerms
def rightRaw : List Term := Proof.Events017.exact4569RawTerms
def group : MergeGroup := .operator 5292 4569
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5292) (leftOrdinal := 1)
    (rightResult := 4569) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60124⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨60124⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5307

namespace LeftMerge5308
def owner : Owner := ⟨.program ⟨257⟩, ⟨67480⟩⟩
def mergeEvent : Nat := 5308
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨57144⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events020.exact5292RawTerms
def rightRaw : List Term := Proof.Events017.exact4569RawTerms
def group : MergeGroup := .operator 5292 4569
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5292) (leftOrdinal := 2)
    (rightResult := 4569) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57144⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨57144⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5308

namespace LeftMerge5309
def owner : Owner := ⟨.program ⟨257⟩, ⟨67480⟩⟩
def mergeEvent : Nat := 5309
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54164⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events020.exact5292RawTerms
def rightRaw : List Term := Proof.Events017.exact4569RawTerms
def group : MergeGroup := .operator 5292 4569
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5292) (leftOrdinal := 3)
    (rightResult := 4569) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54164⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54164⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5309

namespace LeftMerge5310
def owner : Owner := ⟨.program ⟨257⟩, ⟨67480⟩⟩
def mergeEvent : Nat := 5310
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51184⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events020.exact5292RawTerms
def rightRaw : List Term := Proof.Events017.exact4569RawTerms
def group : MergeGroup := .operator 5292 4569
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5292) (leftOrdinal := 4)
    (rightResult := 4569) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51184⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51184⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5310

namespace LeftMerge5311
def owner : Owner := ⟨.program ⟨257⟩, ⟨67480⟩⟩
def mergeEvent : Nat := 5311
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32120⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events020.exact5292RawTerms
def rightRaw : List Term := Proof.Events017.exact4569RawTerms
def group : MergeGroup := .operator 5292 4569
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5292) (leftOrdinal := 6)
    (rightResult := 4569) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32120⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32120⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5311

namespace LeftMerge5312
def owner : Owner := ⟨.program ⟨257⟩, ⟨67480⟩⟩
def mergeEvent : Nat := 5312
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22100⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events020.exact5292RawTerms
def rightRaw : List Term := Proof.Events017.exact4569RawTerms
def group : MergeGroup := .operator 5292 4569
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5292) (leftOrdinal := 10)
    (rightResult := 4569) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22100⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22100⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5312

namespace LeftMerge5313
def owner : Owner := ⟨.program ⟨257⟩, ⟨67480⟩⟩
def mergeEvent : Nat := 5313
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18880⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events020.exact5292RawTerms
def rightRaw : List Term := Proof.Events017.exact4569RawTerms
def group : MergeGroup := .operator 5292 4569
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5292) (leftOrdinal := 14)
    (rightResult := 4569) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18880⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18880⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5313

namespace LeftMerge5314
def owner : Owner := ⟨.program ⟨257⟩, ⟨67480⟩⟩
def mergeEvent : Nat := 5314
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16046⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events020.exact5292RawTerms
def rightRaw : List Term := Proof.Events017.exact4569RawTerms
def group : MergeGroup := .operator 5292 4569
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5292) (leftOrdinal := 17)
    (rightResult := 4569) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16046⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6753⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16046⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5314

namespace LeftMerge5819
def owner : Owner := ⟨.program ⟨257⟩, ⟨67383⟩⟩
def mergeEvent : Nat := 5819
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67382⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events022.exact5815RawTerms
def rightRaw : List Term := Proof.Events000.exact36RawTerms
def group : MergeGroup := .operator 5815 36
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5815) (leftOrdinal := 0)
    (rightResult := 36) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67382⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67382⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5819

namespace LeftMerge5827
def owner : Owner := ⟨.program ⟨257⟩, ⟨48308⟩⟩
def mergeEvent : Nat := 5827
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48307⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events022.exact5823RawTerms
def rightRaw : List Term := Proof.Events002.exact543RawTerms
def group : MergeGroup := .operator 5823 543
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5823) (leftOrdinal := 0)
    (rightResult := 543) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48307⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48307⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5827

namespace LeftMerge5835
def owner : Owner := ⟨.program ⟨257⟩, ⟨45628⟩⟩
def mergeEvent : Nat := 5835
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45627⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events022.exact5831RawTerms
def rightRaw : List Term := Proof.Events002.exact553RawTerms
def group : MergeGroup := .operator 5831 553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5831) (leftOrdinal := 0)
    (rightResult := 553) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45627⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45627⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5835

namespace LeftMerge5843
def owner : Owner := ⟨.program ⟨257⟩, ⟨42951⟩⟩
def mergeEvent : Nat := 5843
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events022.exact5839RawTerms
def rightRaw : List Term := Proof.Events002.exact563RawTerms
def group : MergeGroup := .operator 5839 563
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5839) (leftOrdinal := 0)
    (rightResult := 563) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42950⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5843

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
