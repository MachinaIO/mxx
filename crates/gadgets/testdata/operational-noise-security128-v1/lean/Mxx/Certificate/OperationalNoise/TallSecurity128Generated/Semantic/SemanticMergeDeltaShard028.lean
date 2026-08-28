import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge9045
def owner : Owner := ⟨.program ⟨257⟩, ⟨67518⟩⟩
def mergeEvent : Nat := 9045
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66798⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events035.exact9032RawTerms
def rightRaw : List Term := Proof.Events032.exact8309RawTerms
def group : MergeGroup := .operator 9032 8309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9032) (leftOrdinal := 18)
    (rightResult := 8309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66798⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66798⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9045

namespace LeftMerge9046
def owner : Owner := ⟨.program ⟨257⟩, ⟨67518⟩⟩
def mergeEvent : Nat := 9046
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨63142⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events035.exact9032RawTerms
def rightRaw : List Term := Proof.Events032.exact8309RawTerms
def group : MergeGroup := .operator 9032 8309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9032) (leftOrdinal := 0)
    (rightResult := 8309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63142⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨63142⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9046

namespace LeftMerge9047
def owner : Owner := ⟨.program ⟨257⟩, ⟨67518⟩⟩
def mergeEvent : Nat := 9047
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨60162⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events035.exact9032RawTerms
def rightRaw : List Term := Proof.Events032.exact8309RawTerms
def group : MergeGroup := .operator 9032 8309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9032) (leftOrdinal := 1)
    (rightResult := 8309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60162⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨60162⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9047

namespace LeftMerge9048
def owner : Owner := ⟨.program ⟨257⟩, ⟨67518⟩⟩
def mergeEvent : Nat := 9048
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events035.exact9032RawTerms
def rightRaw : List Term := Proof.Events032.exact8309RawTerms
def group : MergeGroup := .operator 9032 8309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9032) (leftOrdinal := 2)
    (rightResult := 8309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9048

namespace LeftMerge9049
def owner : Owner := ⟨.program ⟨257⟩, ⟨67518⟩⟩
def mergeEvent : Nat := 9049
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨54202⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events035.exact9032RawTerms
def rightRaw : List Term := Proof.Events032.exact8309RawTerms
def group : MergeGroup := .operator 9032 8309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9032) (leftOrdinal := 3)
    (rightResult := 8309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54202⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨54202⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9049

namespace LeftMerge9050
def owner : Owner := ⟨.program ⟨257⟩, ⟨67518⟩⟩
def mergeEvent : Nat := 9050
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨51222⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events035.exact9032RawTerms
def rightRaw : List Term := Proof.Events032.exact8309RawTerms
def group : MergeGroup := .operator 9032 8309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9032) (leftOrdinal := 4)
    (rightResult := 8309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51222⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨51222⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9050

namespace LeftMerge9051
def owner : Owner := ⟨.program ⟨257⟩, ⟨67518⟩⟩
def mergeEvent : Nat := 9051
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨32158⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events035.exact9032RawTerms
def rightRaw : List Term := Proof.Events032.exact8309RawTerms
def group : MergeGroup := .operator 9032 8309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9032) (leftOrdinal := 6)
    (rightResult := 8309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32158⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨32158⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9051

namespace LeftMerge9052
def owner : Owner := ⟨.program ⟨257⟩, ⟨67518⟩⟩
def mergeEvent : Nat := 9052
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events035.exact9032RawTerms
def rightRaw : List Term := Proof.Events032.exact8309RawTerms
def group : MergeGroup := .operator 9032 8309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9032) (leftOrdinal := 10)
    (rightResult := 8309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9052

namespace LeftMerge9053
def owner : Owner := ⟨.program ⟨257⟩, ⟨67518⟩⟩
def mergeEvent : Nat := 9053
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events035.exact9032RawTerms
def rightRaw : List Term := Proof.Events032.exact8309RawTerms
def group : MergeGroup := .operator 9032 8309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9032) (leftOrdinal := 14)
    (rightResult := 8309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9053

namespace LeftMerge9054
def owner : Owner := ⟨.program ⟨257⟩, ⟨67518⟩⟩
def mergeEvent : Nat := 9054
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events035.exact9032RawTerms
def rightRaw : List Term := Proof.Events032.exact8309RawTerms
def group : MergeGroup := .operator 9032 8309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9032) (leftOrdinal := 17)
    (rightResult := 8309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9054

namespace LeftMerge9559
def owner : Owner := ⟨.program ⟨257⟩, ⟨67495⟩⟩
def mergeEvent : Nat := 9559
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67494⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events037.exact9555RawTerms
def rightRaw : List Term := Proof.Events000.exact36RawTerms
def group : MergeGroup := .operator 9555 36
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9555) (leftOrdinal := 0)
    (rightResult := 36) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67494⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67494⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9559

namespace LeftMerge9567
def owner : Owner := ⟨.program ⟨257⟩, ⟨48386⟩⟩
def mergeEvent : Nat := 9567
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48385⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events037.exact9563RawTerms
def rightRaw : List Term := Proof.Events002.exact543RawTerms
def group : MergeGroup := .operator 9563 543
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9563) (leftOrdinal := 0)
    (rightResult := 543) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48385⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48385⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9567

namespace LeftMerge9575
def owner : Owner := ⟨.program ⟨257⟩, ⟨45706⟩⟩
def mergeEvent : Nat := 9575
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45705⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events037.exact9571RawTerms
def rightRaw : List Term := Proof.Events002.exact553RawTerms
def group : MergeGroup := .operator 9571 553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9571) (leftOrdinal := 0)
    (rightResult := 553) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45705⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45705⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9575

namespace LeftMerge9583
def owner : Owner := ⟨.program ⟨257⟩, ⟨43029⟩⟩
def mergeEvent : Nat := 9583
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43028⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events037.exact9579RawTerms
def rightRaw : List Term := Proof.Events002.exact563RawTerms
def group : MergeGroup := .operator 9579 563
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9579) (leftOrdinal := 0)
    (rightResult := 563) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨43028⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43028⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9583

namespace LeftMerge9591
def owner : Owner := ⟨.program ⟨257⟩, ⟨40349⟩⟩
def mergeEvent : Nat := 9591
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40348⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events037.exact9587RawTerms
def rightRaw : List Term := Proof.Events002.exact573RawTerms
def group : MergeGroup := .operator 9587 573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9587) (leftOrdinal := 0)
    (rightResult := 573) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40348⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40348⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9591

namespace LeftMerge9599
def owner : Owner := ⟨.program ⟨257⟩, ⟨37666⟩⟩
def mergeEvent : Nat := 9599
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events037.exact9595RawTerms
def rightRaw : List Term := Proof.Events002.exact583RawTerms
def group : MergeGroup := .operator 9595 583
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9595) (leftOrdinal := 0)
    (rightResult := 583) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37665⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9599

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
