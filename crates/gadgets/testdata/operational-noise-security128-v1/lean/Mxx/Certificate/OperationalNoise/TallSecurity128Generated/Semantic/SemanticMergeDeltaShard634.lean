import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge105093
def owner : Owner := ⟨.program ⟨257⟩, ⟨67481⟩⟩
def mergeEvent : Nat := 105093
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54164⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events410.exact105038RawTerms
def rightRaw : List Term := Proof.Events020.exact5292RawTerms
def group : MergeGroup := .operator 105038 5292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105038) (leftOrdinal := 0)
    (rightResult := 5292) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54164⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105093

namespace LeftMerge105094
def owner : Owner := ⟨.program ⟨257⟩, ⟨67481⟩⟩
def mergeEvent : Nat := 105094
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51184⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events410.exact105038RawTerms
def rightRaw : List Term := Proof.Events020.exact5292RawTerms
def group : MergeGroup := .operator 105038 5292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105038) (leftOrdinal := 0)
    (rightResult := 5292) (rightOrdinal := 4) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51184⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51184⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105094

namespace LeftMerge105095
def owner : Owner := ⟨.program ⟨257⟩, ⟨67481⟩⟩
def mergeEvent : Nat := 105095
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events410.exact105038RawTerms
def rightRaw : List Term := Proof.Events020.exact5292RawTerms
def group : MergeGroup := .operator 105038 5292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105038) (leftOrdinal := 0)
    (rightResult := 5292) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32120⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105095

namespace LeftMerge105096
def owner : Owner := ⟨.program ⟨257⟩, ⟨67481⟩⟩
def mergeEvent : Nat := 105096
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events410.exact105038RawTerms
def rightRaw : List Term := Proof.Events020.exact5292RawTerms
def group : MergeGroup := .operator 105038 5292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105038) (leftOrdinal := 0)
    (rightResult := 5292) (rightOrdinal := 10) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22100⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105096

namespace LeftMerge105097
def owner : Owner := ⟨.program ⟨257⟩, ⟨67481⟩⟩
def mergeEvent : Nat := 105097
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events410.exact105038RawTerms
def rightRaw : List Term := Proof.Events020.exact5292RawTerms
def group : MergeGroup := .operator 105038 5292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105038) (leftOrdinal := 0)
    (rightResult := 5292) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18880⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105097

namespace LeftMerge105098
def owner : Owner := ⟨.program ⟨257⟩, ⟨67481⟩⟩
def mergeEvent : Nat := 105098
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16046⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events410.exact105038RawTerms
def rightRaw : List Term := Proof.Events020.exact5292RawTerms
def group : MergeGroup := .operator 105038 5292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105038) (leftOrdinal := 0)
    (rightResult := 5292) (rightOrdinal := 17) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16046⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16046⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105098

namespace LeftMerge105099
def owner : Owner := ⟨.program ⟨257⟩, ⟨67481⟩⟩
def mergeEvent : Nat := 105099
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨67476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7247⟩⟩] } }
def leftRaw : List Term := Proof.Events410.exact105038RawTerms
def rightRaw : List Term := Proof.Events020.exact5292RawTerms
def group : MergeGroup := .operator 105038 5292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105038) (leftOrdinal := 1)
    (rightResult := 5292) (rightOrdinal := 5) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7247⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67476⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨67476⟩⟩], [⟨.program ⟨257⟩, ⟨7247⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105099

namespace LeftMerge105100
def owner : Owner := ⟨.program ⟨257⟩, ⟨67481⟩⟩
def mergeEvent : Nat := 105100
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨48372⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7247⟩⟩] } }
def leftRaw : List Term := Proof.Events410.exact105038RawTerms
def rightRaw : List Term := Proof.Events020.exact5292RawTerms
def group : MergeGroup := .operator 105038 5292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105038) (leftOrdinal := 1)
    (rightResult := 5292) (rightOrdinal := 7) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7247⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48372⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨48372⟩⟩], [⟨.program ⟨257⟩, ⟨7247⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105100

namespace LeftMerge105101
def owner : Owner := ⟨.program ⟨257⟩, ⟨67481⟩⟩
def mergeEvent : Nat := 105101
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨45692⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7247⟩⟩] } }
def leftRaw : List Term := Proof.Events410.exact105038RawTerms
def rightRaw : List Term := Proof.Events020.exact5292RawTerms
def group : MergeGroup := .operator 105038 5292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105038) (leftOrdinal := 1)
    (rightResult := 5292) (rightOrdinal := 8) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7247⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45692⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨45692⟩⟩], [⟨.program ⟨257⟩, ⟨7247⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105101

namespace LeftMerge105102
def owner : Owner := ⟨.program ⟨257⟩, ⟨67481⟩⟩
def mergeEvent : Nat := 105102
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨43015⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7247⟩⟩] } }
def leftRaw : List Term := Proof.Events410.exact105038RawTerms
def rightRaw : List Term := Proof.Events020.exact5292RawTerms
def group : MergeGroup := .operator 105038 5292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105038) (leftOrdinal := 1)
    (rightResult := 5292) (rightOrdinal := 9) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7247⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43015⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨43015⟩⟩], [⟨.program ⟨257⟩, ⟨7247⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105102

namespace LeftMerge105103
def owner : Owner := ⟨.program ⟨257⟩, ⟨67481⟩⟩
def mergeEvent : Nat := 105103
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40335⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7247⟩⟩] } }
def leftRaw : List Term := Proof.Events410.exact105038RawTerms
def rightRaw : List Term := Proof.Events020.exact5292RawTerms
def group : MergeGroup := .operator 105038 5292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105038) (leftOrdinal := 1)
    (rightResult := 5292) (rightOrdinal := 11) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7247⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40335⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40335⟩⟩], [⟨.program ⟨257⟩, ⟨7247⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105103

namespace LeftMerge105104
def owner : Owner := ⟨.program ⟨257⟩, ⟨67481⟩⟩
def mergeEvent : Nat := 105104
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37652⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7247⟩⟩] } }
def leftRaw : List Term := Proof.Events410.exact105038RawTerms
def rightRaw : List Term := Proof.Events020.exact5292RawTerms
def group : MergeGroup := .operator 105038 5292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105038) (leftOrdinal := 1)
    (rightResult := 5292) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7247⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37652⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37652⟩⟩], [⟨.program ⟨257⟩, ⟨7247⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105104

namespace LeftMerge105105
def owner : Owner := ⟨.program ⟨257⟩, ⟨67481⟩⟩
def mergeEvent : Nat := 105105
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨34972⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7247⟩⟩] } }
def leftRaw : List Term := Proof.Events410.exact105038RawTerms
def rightRaw : List Term := Proof.Events020.exact5292RawTerms
def group : MergeGroup := .operator 105038 5292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105038) (leftOrdinal := 1)
    (rightResult := 5292) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7247⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34972⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨34972⟩⟩], [⟨.program ⟨257⟩, ⟨7247⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105105

namespace LeftMerge105106
def owner : Owner := ⟨.program ⟨257⟩, ⟨67481⟩⟩
def mergeEvent : Nat := 105106
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29315⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7247⟩⟩] } }
def leftRaw : List Term := Proof.Events410.exact105038RawTerms
def rightRaw : List Term := Proof.Events020.exact5292RawTerms
def group : MergeGroup := .operator 105038 5292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105038) (leftOrdinal := 1)
    (rightResult := 5292) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7247⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29315⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29315⟩⟩], [⟨.program ⟨257⟩, ⟨7247⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105106

namespace LeftMerge105107
def owner : Owner := ⟨.program ⟨257⟩, ⟨67481⟩⟩
def mergeEvent : Nat := 105107
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26635⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7247⟩⟩] } }
def leftRaw : List Term := Proof.Events410.exact105038RawTerms
def rightRaw : List Term := Proof.Events020.exact5292RawTerms
def group : MergeGroup := .operator 105038 5292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105038) (leftOrdinal := 1)
    (rightResult := 5292) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7247⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26635⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26635⟩⟩], [⟨.program ⟨257⟩, ⟨7247⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105107

namespace LeftMerge105108
def owner : Owner := ⟨.program ⟨257⟩, ⟨67481⟩⟩
def mergeEvent : Nat := 105108
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨66658⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7247⟩⟩] } }
def leftRaw : List Term := Proof.Events410.exact105038RawTerms
def rightRaw : List Term := Proof.Events020.exact5292RawTerms
def group : MergeGroup := .operator 105038 5292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105038) (leftOrdinal := 1)
    (rightResult := 5292) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7247⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66658⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨66658⟩⟩], [⟨.program ⟨257⟩, ⟨7247⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105108

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
