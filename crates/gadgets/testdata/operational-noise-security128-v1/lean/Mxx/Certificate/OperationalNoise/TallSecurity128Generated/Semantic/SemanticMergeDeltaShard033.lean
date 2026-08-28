import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge10545
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def mergeEvent : Nat := 10545
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨54145⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events041.exact10528RawTerms
def rightRaw : List Term := Proof.Events038.exact9805RawTerms
def group : MergeGroup := .operator 10528 9805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10528) (leftOrdinal := 3)
    (rightResult := 9805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54145⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨54145⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10545

namespace LeftMerge10546
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def mergeEvent : Nat := 10546
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨51165⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events041.exact10528RawTerms
def rightRaw : List Term := Proof.Events038.exact9805RawTerms
def group : MergeGroup := .operator 10528 9805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10528) (leftOrdinal := 4)
    (rightResult := 9805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51165⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨51165⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10546

namespace LeftMerge10547
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def mergeEvent : Nat := 10547
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32101⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events041.exact10528RawTerms
def rightRaw : List Term := Proof.Events038.exact9805RawTerms
def group : MergeGroup := .operator 10528 9805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10528) (leftOrdinal := 6)
    (rightResult := 9805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32101⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32101⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10547

namespace LeftMerge10548
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def mergeEvent : Nat := 10548
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22081⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events041.exact10528RawTerms
def rightRaw : List Term := Proof.Events038.exact9805RawTerms
def group : MergeGroup := .operator 10528 9805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10528) (leftOrdinal := 10)
    (rightResult := 9805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22081⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22081⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10548

namespace LeftMerge10549
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def mergeEvent : Nat := 10549
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18861⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events041.exact10528RawTerms
def rightRaw : List Term := Proof.Events038.exact9805RawTerms
def group : MergeGroup := .operator 10528 9805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10528) (leftOrdinal := 14)
    (rightResult := 9805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18861⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18861⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10549

namespace LeftMerge10550
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def mergeEvent : Nat := 10550
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16030⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events041.exact10528RawTerms
def rightRaw : List Term := Proof.Events038.exact9805RawTerms
def group : MergeGroup := .operator 10528 9805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10528) (leftOrdinal := 17)
    (rightResult := 9805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16030⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16030⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10550

namespace LeftMerge11055
def owner : Owner := ⟨.program ⟨257⟩, ⟨67438⟩⟩
def mergeEvent : Nat := 11055
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67437⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events043.exact11051RawTerms
def rightRaw : List Term := Proof.Events000.exact36RawTerms
def group : MergeGroup := .operator 11051 36
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11051) (leftOrdinal := 0)
    (rightResult := 36) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67437⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67437⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11055

namespace LeftMerge11063
def owner : Owner := ⟨.program ⟨257⟩, ⟨48347⟩⟩
def mergeEvent : Nat := 11063
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48346⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events043.exact11059RawTerms
def rightRaw : List Term := Proof.Events002.exact543RawTerms
def group : MergeGroup := .operator 11059 543
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11059) (leftOrdinal := 0)
    (rightResult := 543) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48346⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48346⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11063

namespace LeftMerge11071
def owner : Owner := ⟨.program ⟨257⟩, ⟨45667⟩⟩
def mergeEvent : Nat := 11071
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45666⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events043.exact11067RawTerms
def rightRaw : List Term := Proof.Events002.exact553RawTerms
def group : MergeGroup := .operator 11067 553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11067) (leftOrdinal := 0)
    (rightResult := 553) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45666⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45666⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11071

namespace LeftMerge11079
def owner : Owner := ⟨.program ⟨257⟩, ⟨42990⟩⟩
def mergeEvent : Nat := 11079
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42989⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events043.exact11075RawTerms
def rightRaw : List Term := Proof.Events002.exact563RawTerms
def group : MergeGroup := .operator 11075 563
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11075) (leftOrdinal := 0)
    (rightResult := 563) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42989⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42989⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11079

namespace LeftMerge11087
def owner : Owner := ⟨.program ⟨257⟩, ⟨40310⟩⟩
def mergeEvent : Nat := 11087
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40309⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events043.exact11083RawTerms
def rightRaw : List Term := Proof.Events002.exact573RawTerms
def group : MergeGroup := .operator 11083 573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11083) (leftOrdinal := 0)
    (rightResult := 573) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40309⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40309⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11087

namespace LeftMerge11095
def owner : Owner := ⟨.program ⟨257⟩, ⟨37627⟩⟩
def mergeEvent : Nat := 11095
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37626⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events043.exact11091RawTerms
def rightRaw : List Term := Proof.Events002.exact583RawTerms
def group : MergeGroup := .operator 11091 583
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11091) (leftOrdinal := 0)
    (rightResult := 583) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37626⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37626⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11095

namespace LeftMerge11103
def owner : Owner := ⟨.program ⟨257⟩, ⟨34947⟩⟩
def mergeEvent : Nat := 11103
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34946⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events043.exact11099RawTerms
def rightRaw : List Term := Proof.Events002.exact593RawTerms
def group : MergeGroup := .operator 11099 593
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11099) (leftOrdinal := 0)
    (rightResult := 593) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34946⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34946⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11103

namespace LeftMerge11111
def owner : Owner := ⟨.program ⟨257⟩, ⟨29290⟩⟩
def mergeEvent : Nat := 11111
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29289⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events043.exact11107RawTerms
def rightRaw : List Term := Proof.Events002.exact603RawTerms
def group : MergeGroup := .operator 11107 603
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11107) (leftOrdinal := 0)
    (rightResult := 603) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29289⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29289⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11111

namespace LeftMerge11119
def owner : Owner := ⟨.program ⟨257⟩, ⟨26610⟩⟩
def mergeEvent : Nat := 11119
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26609⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events043.exact11115RawTerms
def rightRaw : List Term := Proof.Events002.exact613RawTerms
def group : MergeGroup := .operator 11115 613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11115) (leftOrdinal := 0)
    (rightResult := 613) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26609⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26609⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11119

namespace LeftMerge11127
def owner : Owner := ⟨.program ⟨257⟩, ⟨66519⟩⟩
def mergeEvent : Nat := 11127
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66518⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events043.exact11123RawTerms
def rightRaw : List Term := Proof.Events002.exact623RawTerms
def group : MergeGroup := .operator 11123 623
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11123) (leftOrdinal := 0)
    (rightResult := 623) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66518⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66518⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge11127

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
