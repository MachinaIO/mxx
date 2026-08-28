import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge6045
def owner : Owner := ⟨.program ⟨257⟩, ⟨67386⟩⟩
def mergeEvent : Nat := 6045
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48307⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events023.exact6040RawTerms
def rightRaw : List Term := Proof.Events020.exact5317RawTerms
def group : MergeGroup := .operator 6040 5317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6040) (leftOrdinal := 7)
    (rightResult := 5317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48307⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48307⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6045

namespace LeftMerge6046
def owner : Owner := ⟨.program ⟨257⟩, ⟨67386⟩⟩
def mergeEvent : Nat := 6046
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45627⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events023.exact6040RawTerms
def rightRaw : List Term := Proof.Events020.exact5317RawTerms
def group : MergeGroup := .operator 6040 5317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6040) (leftOrdinal := 8)
    (rightResult := 5317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45627⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45627⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6046

namespace LeftMerge6047
def owner : Owner := ⟨.program ⟨257⟩, ⟨67386⟩⟩
def mergeEvent : Nat := 6047
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events023.exact6040RawTerms
def rightRaw : List Term := Proof.Events020.exact5317RawTerms
def group : MergeGroup := .operator 6040 5317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6040) (leftOrdinal := 9)
    (rightResult := 5317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6047

namespace LeftMerge6048
def owner : Owner := ⟨.program ⟨257⟩, ⟨67386⟩⟩
def mergeEvent : Nat := 6048
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events023.exact6040RawTerms
def rightRaw : List Term := Proof.Events020.exact5317RawTerms
def group : MergeGroup := .operator 6040 5317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6040) (leftOrdinal := 11)
    (rightResult := 5317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6048

namespace LeftMerge6049
def owner : Owner := ⟨.program ⟨257⟩, ⟨67386⟩⟩
def mergeEvent : Nat := 6049
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37587⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events023.exact6040RawTerms
def rightRaw : List Term := Proof.Events020.exact5317RawTerms
def group : MergeGroup := .operator 6040 5317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6040) (leftOrdinal := 12)
    (rightResult := 5317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37587⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37587⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6049

namespace LeftMerge6050
def owner : Owner := ⟨.program ⟨257⟩, ⟨67386⟩⟩
def mergeEvent : Nat := 6050
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34907⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events023.exact6040RawTerms
def rightRaw : List Term := Proof.Events020.exact5317RawTerms
def group : MergeGroup := .operator 6040 5317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6040) (leftOrdinal := 13)
    (rightResult := 5317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34907⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34907⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6050

namespace LeftMerge6051
def owner : Owner := ⟨.program ⟨257⟩, ⟨67386⟩⟩
def mergeEvent : Nat := 6051
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events023.exact6040RawTerms
def rightRaw : List Term := Proof.Events020.exact5317RawTerms
def group : MergeGroup := .operator 6040 5317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6040) (leftOrdinal := 15)
    (rightResult := 5317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6051

namespace LeftMerge6052
def owner : Owner := ⟨.program ⟨257⟩, ⟨67386⟩⟩
def mergeEvent : Nat := 6052
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events023.exact6040RawTerms
def rightRaw : List Term := Proof.Events020.exact5317RawTerms
def group : MergeGroup := .operator 6040 5317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6040) (leftOrdinal := 16)
    (rightResult := 5317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6052

namespace LeftMerge6053
def owner : Owner := ⟨.program ⟨257⟩, ⟨67386⟩⟩
def mergeEvent : Nat := 6053
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events023.exact6040RawTerms
def rightRaw : List Term := Proof.Events020.exact5317RawTerms
def group : MergeGroup := .operator 6040 5317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6040) (leftOrdinal := 18)
    (rightResult := 5317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6053

namespace LeftMerge6054
def owner : Owner := ⟨.program ⟨257⟩, ⟨67386⟩⟩
def mergeEvent : Nat := 6054
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events023.exact6040RawTerms
def rightRaw : List Term := Proof.Events020.exact5317RawTerms
def group : MergeGroup := .operator 6040 5317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6040) (leftOrdinal := 0)
    (rightResult := 5317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6054

namespace LeftMerge6055
def owner : Owner := ⟨.program ⟨257⟩, ⟨67386⟩⟩
def mergeEvent : Nat := 6055
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events023.exact6040RawTerms
def rightRaw : List Term := Proof.Events020.exact5317RawTerms
def group : MergeGroup := .operator 6040 5317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6040) (leftOrdinal := 1)
    (rightResult := 5317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6055

namespace LeftMerge6056
def owner : Owner := ⟨.program ⟨257⟩, ⟨67386⟩⟩
def mergeEvent : Nat := 6056
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events023.exact6040RawTerms
def rightRaw : List Term := Proof.Events020.exact5317RawTerms
def group : MergeGroup := .operator 6040 5317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6040) (leftOrdinal := 2)
    (rightResult := 5317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6056

namespace LeftMerge6057
def owner : Owner := ⟨.program ⟨257⟩, ⟨67386⟩⟩
def mergeEvent : Nat := 6057
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events023.exact6040RawTerms
def rightRaw : List Term := Proof.Events020.exact5317RawTerms
def group : MergeGroup := .operator 6040 5317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6040) (leftOrdinal := 3)
    (rightResult := 5317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6057

namespace LeftMerge6058
def owner : Owner := ⟨.program ⟨257⟩, ⟨67386⟩⟩
def mergeEvent : Nat := 6058
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events023.exact6040RawTerms
def rightRaw : List Term := Proof.Events020.exact5317RawTerms
def group : MergeGroup := .operator 6040 5317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6040) (leftOrdinal := 4)
    (rightResult := 5317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6058

namespace LeftMerge6059
def owner : Owner := ⟨.program ⟨257⟩, ⟨67386⟩⟩
def mergeEvent : Nat := 6059
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events023.exact6040RawTerms
def rightRaw : List Term := Proof.Events020.exact5317RawTerms
def group : MergeGroup := .operator 6040 5317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6040) (leftOrdinal := 6)
    (rightResult := 5317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6059

namespace LeftMerge6060
def owner : Owner := ⟨.program ⟨257⟩, ⟨67386⟩⟩
def mergeEvent : Nat := 6060
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events023.exact6040RawTerms
def rightRaw : List Term := Proof.Events020.exact5317RawTerms
def group : MergeGroup := .operator 6040 5317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6040) (leftOrdinal := 10)
    (rightResult := 5317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6060

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
