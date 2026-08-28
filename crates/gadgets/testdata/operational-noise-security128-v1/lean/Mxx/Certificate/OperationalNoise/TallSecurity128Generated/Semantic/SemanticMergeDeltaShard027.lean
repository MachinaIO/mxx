import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge8907
def owner : Owner := ⟨.program ⟨257⟩, ⟨57183⟩⟩
def mergeEvent : Nat := 8907
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events034.exact8903RawTerms
def rightRaw : List Term := Proof.Events002.exact653RawTerms
def group : MergeGroup := .operator 8903 653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8903) (leftOrdinal := 0)
    (rightResult := 653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57182⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8907

namespace LeftMerge8915
def owner : Owner := ⟨.program ⟨257⟩, ⟨54203⟩⟩
def mergeEvent : Nat := 8915
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54202⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events034.exact8911RawTerms
def rightRaw : List Term := Proof.Events002.exact663RawTerms
def group : MergeGroup := .operator 8911 663
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8911) (leftOrdinal := 0)
    (rightResult := 663) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54202⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54202⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8915

namespace LeftMerge8923
def owner : Owner := ⟨.program ⟨257⟩, ⟨51223⟩⟩
def mergeEvent : Nat := 8923
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51222⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events034.exact8919RawTerms
def rightRaw : List Term := Proof.Events002.exact673RawTerms
def group : MergeGroup := .operator 8919 673
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8919) (leftOrdinal := 0)
    (rightResult := 673) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51222⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51222⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8923

namespace LeftMerge8931
def owner : Owner := ⟨.program ⟨257⟩, ⟨32159⟩⟩
def mergeEvent : Nat := 8931
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32158⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events034.exact8927RawTerms
def rightRaw : List Term := Proof.Events002.exact683RawTerms
def group : MergeGroup := .operator 8927 683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8927) (leftOrdinal := 0)
    (rightResult := 683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32158⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32158⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8931

namespace LeftMerge8939
def owner : Owner := ⟨.program ⟨257⟩, ⟨22139⟩⟩
def mergeEvent : Nat := 8939
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events034.exact8935RawTerms
def rightRaw : List Term := Proof.Events002.exact693RawTerms
def group : MergeGroup := .operator 8935 693
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8935) (leftOrdinal := 0)
    (rightResult := 693) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22138⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8939

namespace LeftMerge8947
def owner : Owner := ⟨.program ⟨257⟩, ⟨18919⟩⟩
def mergeEvent : Nat := 8947
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events034.exact8943RawTerms
def rightRaw : List Term := Proof.Events002.exact703RawTerms
def group : MergeGroup := .operator 8943 703
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8943) (leftOrdinal := 0)
    (rightResult := 703) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18918⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8947

namespace LeftMerge8955
def owner : Owner := ⟨.program ⟨257⟩, ⟨16079⟩⟩
def mergeEvent : Nat := 8955
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events034.exact8951RawTerms
def rightRaw : List Term := Proof.Events002.exact713RawTerms
def group : MergeGroup := .operator 8951 713
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8951) (leftOrdinal := 0)
    (rightResult := 713) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16078⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8955

namespace LeftMerge9036
def owner : Owner := ⟨.program ⟨257⟩, ⟨67518⟩⟩
def mergeEvent : Nat := 9036
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨67514⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events035.exact9032RawTerms
def rightRaw : List Term := Proof.Events032.exact8309RawTerms
def group : MergeGroup := .operator 9032 8309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9032) (leftOrdinal := 5)
    (rightResult := 8309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67514⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨67514⟩⟩], []⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge9036

namespace LeftMerge9037
def owner : Owner := ⟨.program ⟨257⟩, ⟨67518⟩⟩
def mergeEvent : Nat := 9037
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨48398⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events035.exact9032RawTerms
def rightRaw : List Term := Proof.Events032.exact8309RawTerms
def group : MergeGroup := .operator 9032 8309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9032) (leftOrdinal := 7)
    (rightResult := 8309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48398⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨48398⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9037

namespace LeftMerge9038
def owner : Owner := ⟨.program ⟨257⟩, ⟨67518⟩⟩
def mergeEvent : Nat := 9038
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45718⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events035.exact9032RawTerms
def rightRaw : List Term := Proof.Events032.exact8309RawTerms
def group : MergeGroup := .operator 9032 8309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9032) (leftOrdinal := 8)
    (rightResult := 8309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45718⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45718⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9038

namespace LeftMerge9039
def owner : Owner := ⟨.program ⟨257⟩, ⟨67518⟩⟩
def mergeEvent : Nat := 9039
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43041⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events035.exact9032RawTerms
def rightRaw : List Term := Proof.Events032.exact8309RawTerms
def group : MergeGroup := .operator 9032 8309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9032) (leftOrdinal := 9)
    (rightResult := 8309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43041⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43041⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9039

namespace LeftMerge9040
def owner : Owner := ⟨.program ⟨257⟩, ⟨67518⟩⟩
def mergeEvent : Nat := 9040
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40361⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events035.exact9032RawTerms
def rightRaw : List Term := Proof.Events032.exact8309RawTerms
def group : MergeGroup := .operator 9032 8309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9032) (leftOrdinal := 11)
    (rightResult := 8309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40361⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40361⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9040

namespace LeftMerge9041
def owner : Owner := ⟨.program ⟨257⟩, ⟨67518⟩⟩
def mergeEvent : Nat := 9041
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37678⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events035.exact9032RawTerms
def rightRaw : List Term := Proof.Events032.exact8309RawTerms
def group : MergeGroup := .operator 9032 8309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9032) (leftOrdinal := 12)
    (rightResult := 8309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37678⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37678⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9041

namespace LeftMerge9042
def owner : Owner := ⟨.program ⟨257⟩, ⟨67518⟩⟩
def mergeEvent : Nat := 9042
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34998⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events035.exact9032RawTerms
def rightRaw : List Term := Proof.Events032.exact8309RawTerms
def group : MergeGroup := .operator 9032 8309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9032) (leftOrdinal := 13)
    (rightResult := 8309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34998⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34998⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9042

namespace LeftMerge9043
def owner : Owner := ⟨.program ⟨257⟩, ⟨67518⟩⟩
def mergeEvent : Nat := 9043
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29341⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events035.exact9032RawTerms
def rightRaw : List Term := Proof.Events032.exact8309RawTerms
def group : MergeGroup := .operator 9032 8309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9032) (leftOrdinal := 15)
    (rightResult := 8309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29341⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29341⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9043

namespace LeftMerge9044
def owner : Owner := ⟨.program ⟨257⟩, ⟨67518⟩⟩
def mergeEvent : Nat := 9044
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26661⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events035.exact9032RawTerms
def rightRaw : List Term := Proof.Events032.exact8309RawTerms
def group : MergeGroup := .operator 9032 8309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9032) (leftOrdinal := 16)
    (rightResult := 8309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26661⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6806⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26661⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9044

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
