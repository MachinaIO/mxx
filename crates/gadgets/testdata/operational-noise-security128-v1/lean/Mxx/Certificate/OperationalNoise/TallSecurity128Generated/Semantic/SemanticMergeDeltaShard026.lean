import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge8303
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def mergeEvent : Nat := 8303
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events032.exact8284RawTerms
def rightRaw : List Term := Proof.Events029.exact7561RawTerms
def group : MergeGroup := .operator 8284 7561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8284) (leftOrdinal := 6)
    (rightResult := 7561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8303

namespace LeftMerge8304
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def mergeEvent : Nat := 8304
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events032.exact8284RawTerms
def rightRaw : List Term := Proof.Events029.exact7561RawTerms
def group : MergeGroup := .operator 8284 7561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8284) (leftOrdinal := 10)
    (rightResult := 7561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8304

namespace LeftMerge8305
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def mergeEvent : Nat := 8305
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events032.exact8284RawTerms
def rightRaw : List Term := Proof.Events029.exact7561RawTerms
def group : MergeGroup := .operator 8284 7561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8284) (leftOrdinal := 14)
    (rightResult := 7561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8305

namespace LeftMerge8306
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def mergeEvent : Nat := 8306
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events032.exact8284RawTerms
def rightRaw : List Term := Proof.Events029.exact7561RawTerms
def group : MergeGroup := .operator 8284 7561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8284) (leftOrdinal := 17)
    (rightResult := 7561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8306

namespace LeftMerge8811
def owner : Owner := ⟨.program ⟨257⟩, ⟨67515⟩⟩
def mergeEvent : Nat := 8811
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67514⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events034.exact8807RawTerms
def rightRaw : List Term := Proof.Events000.exact36RawTerms
def group : MergeGroup := .operator 8807 36
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8807) (leftOrdinal := 0)
    (rightResult := 36) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67514⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67514⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8811

namespace LeftMerge8819
def owner : Owner := ⟨.program ⟨257⟩, ⟨48399⟩⟩
def mergeEvent : Nat := 8819
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48398⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events034.exact8815RawTerms
def rightRaw : List Term := Proof.Events002.exact543RawTerms
def group : MergeGroup := .operator 8815 543
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8815) (leftOrdinal := 0)
    (rightResult := 543) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48398⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48398⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8819

namespace LeftMerge8827
def owner : Owner := ⟨.program ⟨257⟩, ⟨45719⟩⟩
def mergeEvent : Nat := 8827
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45718⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events034.exact8823RawTerms
def rightRaw : List Term := Proof.Events002.exact553RawTerms
def group : MergeGroup := .operator 8823 553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8823) (leftOrdinal := 0)
    (rightResult := 553) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45718⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45718⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8827

namespace LeftMerge8835
def owner : Owner := ⟨.program ⟨257⟩, ⟨43042⟩⟩
def mergeEvent : Nat := 8835
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43041⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events034.exact8831RawTerms
def rightRaw : List Term := Proof.Events002.exact563RawTerms
def group : MergeGroup := .operator 8831 563
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8831) (leftOrdinal := 0)
    (rightResult := 563) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨43041⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43041⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8835

namespace LeftMerge8843
def owner : Owner := ⟨.program ⟨257⟩, ⟨40362⟩⟩
def mergeEvent : Nat := 8843
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40361⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events034.exact8839RawTerms
def rightRaw : List Term := Proof.Events002.exact573RawTerms
def group : MergeGroup := .operator 8839 573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8839) (leftOrdinal := 0)
    (rightResult := 573) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40361⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40361⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8843

namespace LeftMerge8851
def owner : Owner := ⟨.program ⟨257⟩, ⟨37679⟩⟩
def mergeEvent : Nat := 8851
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37678⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events034.exact8847RawTerms
def rightRaw : List Term := Proof.Events002.exact583RawTerms
def group : MergeGroup := .operator 8847 583
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8847) (leftOrdinal := 0)
    (rightResult := 583) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37678⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37678⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8851

namespace LeftMerge8859
def owner : Owner := ⟨.program ⟨257⟩, ⟨34999⟩⟩
def mergeEvent : Nat := 8859
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34998⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events034.exact8855RawTerms
def rightRaw : List Term := Proof.Events002.exact593RawTerms
def group : MergeGroup := .operator 8855 593
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8855) (leftOrdinal := 0)
    (rightResult := 593) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34998⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34998⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8859

namespace LeftMerge8867
def owner : Owner := ⟨.program ⟨257⟩, ⟨29342⟩⟩
def mergeEvent : Nat := 8867
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29341⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events034.exact8863RawTerms
def rightRaw : List Term := Proof.Events002.exact603RawTerms
def group : MergeGroup := .operator 8863 603
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8863) (leftOrdinal := 0)
    (rightResult := 603) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29341⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29341⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8867

namespace LeftMerge8875
def owner : Owner := ⟨.program ⟨257⟩, ⟨26662⟩⟩
def mergeEvent : Nat := 8875
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26661⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events034.exact8871RawTerms
def rightRaw : List Term := Proof.Events002.exact613RawTerms
def group : MergeGroup := .operator 8871 613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8871) (leftOrdinal := 0)
    (rightResult := 613) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26661⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26661⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8875

namespace LeftMerge8883
def owner : Owner := ⟨.program ⟨257⟩, ⟨66799⟩⟩
def mergeEvent : Nat := 8883
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66798⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events034.exact8879RawTerms
def rightRaw : List Term := Proof.Events002.exact623RawTerms
def group : MergeGroup := .operator 8879 623
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8879) (leftOrdinal := 0)
    (rightResult := 623) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66798⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66798⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8883

namespace LeftMerge8891
def owner : Owner := ⟨.program ⟨257⟩, ⟨63143⟩⟩
def mergeEvent : Nat := 8891
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63142⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events034.exact8887RawTerms
def rightRaw : List Term := Proof.Events002.exact633RawTerms
def group : MergeGroup := .operator 8887 633
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8887) (leftOrdinal := 0)
    (rightResult := 633) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63142⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63142⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8891

namespace LeftMerge8899
def owner : Owner := ⟨.program ⟨257⟩, ⟨60163⟩⟩
def mergeEvent : Nat := 8899
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60162⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events034.exact8895RawTerms
def rightRaw : List Term := Proof.Events002.exact643RawTerms
def group : MergeGroup := .operator 8895 643
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8895) (leftOrdinal := 0)
    (rightResult := 643) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60162⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60162⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8899

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
