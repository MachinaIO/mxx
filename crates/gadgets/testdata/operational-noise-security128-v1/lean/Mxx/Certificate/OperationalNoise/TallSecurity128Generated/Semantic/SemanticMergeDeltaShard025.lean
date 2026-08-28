import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge8207
def owner : Owner := ⟨.program ⟨257⟩, ⟨16095⟩⟩
def mergeEvent : Nat := 8207
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events032.exact8203RawTerms
def rightRaw : List Term := Proof.Events002.exact713RawTerms
def group : MergeGroup := .operator 8203 713
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8203) (leftOrdinal := 0)
    (rightResult := 713) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16094⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8207

namespace LeftMerge8288
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def mergeEvent : Nat := 8288
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67538⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events032.exact8284RawTerms
def rightRaw : List Term := Proof.Events029.exact7561RawTerms
def group : MergeGroup := .operator 8284 7561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8284) (leftOrdinal := 5)
    (rightResult := 7561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67538⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67538⟩⟩], []⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge8288

namespace LeftMerge8289
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def mergeEvent : Nat := 8289
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48411⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events032.exact8284RawTerms
def rightRaw : List Term := Proof.Events029.exact7561RawTerms
def group : MergeGroup := .operator 8284 7561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8284) (leftOrdinal := 7)
    (rightResult := 7561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48411⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48411⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8289

namespace LeftMerge8290
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def mergeEvent : Nat := 8290
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events032.exact8284RawTerms
def rightRaw : List Term := Proof.Events029.exact7561RawTerms
def group : MergeGroup := .operator 8284 7561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8284) (leftOrdinal := 8)
    (rightResult := 7561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8290

namespace LeftMerge8291
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def mergeEvent : Nat := 8291
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43054⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events032.exact8284RawTerms
def rightRaw : List Term := Proof.Events029.exact7561RawTerms
def group : MergeGroup := .operator 8284 7561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8284) (leftOrdinal := 9)
    (rightResult := 7561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43054⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43054⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8291

namespace LeftMerge8292
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def mergeEvent : Nat := 8292
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40374⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events032.exact8284RawTerms
def rightRaw : List Term := Proof.Events029.exact7561RawTerms
def group : MergeGroup := .operator 8284 7561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8284) (leftOrdinal := 11)
    (rightResult := 7561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40374⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40374⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8292

namespace LeftMerge8293
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def mergeEvent : Nat := 8293
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events032.exact8284RawTerms
def rightRaw : List Term := Proof.Events029.exact7561RawTerms
def group : MergeGroup := .operator 8284 7561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8284) (leftOrdinal := 12)
    (rightResult := 7561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8293

namespace LeftMerge8294
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def mergeEvent : Nat := 8294
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events032.exact8284RawTerms
def rightRaw : List Term := Proof.Events029.exact7561RawTerms
def group : MergeGroup := .operator 8284 7561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8284) (leftOrdinal := 13)
    (rightResult := 7561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8294

namespace LeftMerge8295
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def mergeEvent : Nat := 8295
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events032.exact8284RawTerms
def rightRaw : List Term := Proof.Events029.exact7561RawTerms
def group : MergeGroup := .operator 8284 7561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8284) (leftOrdinal := 15)
    (rightResult := 7561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8295

namespace LeftMerge8296
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def mergeEvent : Nat := 8296
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events032.exact8284RawTerms
def rightRaw : List Term := Proof.Events029.exact7561RawTerms
def group : MergeGroup := .operator 8284 7561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8284) (leftOrdinal := 16)
    (rightResult := 7561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8296

namespace LeftMerge8297
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def mergeEvent : Nat := 8297
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events032.exact8284RawTerms
def rightRaw : List Term := Proof.Events029.exact7561RawTerms
def group : MergeGroup := .operator 8284 7561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8284) (leftOrdinal := 18)
    (rightResult := 7561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8297

namespace LeftMerge8298
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def mergeEvent : Nat := 8298
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events032.exact8284RawTerms
def rightRaw : List Term := Proof.Events029.exact7561RawTerms
def group : MergeGroup := .operator 8284 7561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8284) (leftOrdinal := 0)
    (rightResult := 7561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8298

namespace LeftMerge8299
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def mergeEvent : Nat := 8299
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events032.exact8284RawTerms
def rightRaw : List Term := Proof.Events029.exact7561RawTerms
def group : MergeGroup := .operator 8284 7561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8284) (leftOrdinal := 1)
    (rightResult := 7561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8299

namespace LeftMerge8300
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def mergeEvent : Nat := 8300
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events032.exact8284RawTerms
def rightRaw : List Term := Proof.Events029.exact7561RawTerms
def group : MergeGroup := .operator 8284 7561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8284) (leftOrdinal := 2)
    (rightResult := 7561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8300

namespace LeftMerge8301
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def mergeEvent : Nat := 8301
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events032.exact8284RawTerms
def rightRaw : List Term := Proof.Events029.exact7561RawTerms
def group : MergeGroup := .operator 8284 7561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8284) (leftOrdinal := 3)
    (rightResult := 7561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8301

namespace LeftMerge8302
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def mergeEvent : Nat := 8302
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events032.exact8284RawTerms
def rightRaw : List Term := Proof.Events029.exact7561RawTerms
def group : MergeGroup := .operator 8284 7561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8284) (leftOrdinal := 4)
    (rightResult := 7561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8302

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
