import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge247050
def owner : Owner := ⟨.program ⟨257⟩, ⟨71175⟩⟩
def mergeEvent : Nat := 247050
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact247024RawTerms
def rightRaw : List Term := Proof.Events959.exact245608RawTerms
def group : MergeGroup := .operator 247024 245608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 247024) (leftOrdinal := 6)
    (rightResult := 245608) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge247050

namespace LeftMerge247051
def owner : Owner := ⟨.program ⟨257⟩, ⟨71175⟩⟩
def mergeEvent : Nat := 247051
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨57083⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact247024RawTerms
def rightRaw : List Term := Proof.Events959.exact245608RawTerms
def group : MergeGroup := .operator 247024 245608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 247024) (leftOrdinal := 33)
    (rightResult := 245608) (rightOrdinal := 32) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨57083⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨57083⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨57083⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge247051

namespace LeftMerge247052
def owner : Owner := ⟨.program ⟨257⟩, ⟨71175⟩⟩
def mergeEvent : Nat := 247052
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact247024RawTerms
def rightRaw : List Term := Proof.Events959.exact245608RawTerms
def group : MergeGroup := .operator 247024 245608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 247024) (leftOrdinal := 5)
    (rightResult := 245608) (rightOrdinal := 5) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge247052

namespace LeftMerge247053
def owner : Owner := ⟨.program ⟨257⟩, ⟨71175⟩⟩
def mergeEvent : Nat := 247053
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨54103⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact247024RawTerms
def rightRaw : List Term := Proof.Events959.exact245608RawTerms
def group : MergeGroup := .operator 247024 245608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 247024) (leftOrdinal := 32)
    (rightResult := 245608) (rightOrdinal := 31) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨54103⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨54103⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨54103⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge247053

namespace LeftMerge247054
def owner : Owner := ⟨.program ⟨257⟩, ⟨71175⟩⟩
def mergeEvent : Nat := 247054
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact247024RawTerms
def rightRaw : List Term := Proof.Events959.exact245608RawTerms
def group : MergeGroup := .operator 247024 245608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 247024) (leftOrdinal := 4)
    (rightResult := 245608) (rightOrdinal := 4) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge247054

namespace LeftMerge247055
def owner : Owner := ⟨.program ⟨257⟩, ⟨71175⟩⟩
def mergeEvent : Nat := 247055
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨51123⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact247024RawTerms
def rightRaw : List Term := Proof.Events959.exact245608RawTerms
def group : MergeGroup := .operator 247024 245608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 247024) (leftOrdinal := 31)
    (rightResult := 245608) (rightOrdinal := 30) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨51123⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨51123⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨51123⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge247055

namespace LeftMerge247056
def owner : Owner := ⟨.program ⟨257⟩, ⟨71175⟩⟩
def mergeEvent : Nat := 247056
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact247024RawTerms
def rightRaw : List Term := Proof.Events959.exact245608RawTerms
def group : MergeGroup := .operator 247024 245608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 247024) (leftOrdinal := 3)
    (rightResult := 245608) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge247056

namespace LeftMerge247057
def owner : Owner := ⟨.program ⟨257⟩, ⟨71175⟩⟩
def mergeEvent : Nat := 247057
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨32068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact247024RawTerms
def rightRaw : List Term := Proof.Events959.exact245608RawTerms
def group : MergeGroup := .operator 247024 245608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 247024) (leftOrdinal := 24)
    (rightResult := 245608) (rightOrdinal := 23) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨32068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨32068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨32068⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge247057

namespace LeftMerge247058
def owner : Owner := ⟨.program ⟨257⟩, ⟨71175⟩⟩
def mergeEvent : Nat := 247058
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact247024RawTerms
def rightRaw : List Term := Proof.Events959.exact245608RawTerms
def group : MergeGroup := .operator 247024 245608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 247024) (leftOrdinal := 2)
    (rightResult := 245608) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge247058

namespace LeftMerge247059
def owner : Owner := ⟨.program ⟨257⟩, ⟨71175⟩⟩
def mergeEvent : Nat := 247059
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨22048⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact247024RawTerms
def rightRaw : List Term := Proof.Events959.exact245608RawTerms
def group : MergeGroup := .operator 247024 245608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 247024) (leftOrdinal := 21)
    (rightResult := 245608) (rightOrdinal := 20) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨22048⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨22048⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨22048⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge247059

namespace LeftMerge247060
def owner : Owner := ⟨.program ⟨257⟩, ⟨71175⟩⟩
def mergeEvent : Nat := 247060
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact247024RawTerms
def rightRaw : List Term := Proof.Events959.exact245608RawTerms
def group : MergeGroup := .operator 247024 245608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 247024) (leftOrdinal := 1)
    (rightResult := 245608) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge247060

namespace LeftMerge247061
def owner : Owner := ⟨.program ⟨257⟩, ⟨71175⟩⟩
def mergeEvent : Nat := 247061
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨18828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact247024RawTerms
def rightRaw : List Term := Proof.Events959.exact245608RawTerms
def group : MergeGroup := .operator 247024 245608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 247024) (leftOrdinal := 20)
    (rightResult := 245608) (rightOrdinal := 19) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨18828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨18828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨18828⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge247061

namespace LeftMerge247062
def owner : Owner := ⟨.program ⟨257⟩, ⟨71175⟩⟩
def mergeEvent : Nat := 247062
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact247024RawTerms
def rightRaw : List Term := Proof.Events959.exact245608RawTerms
def group : MergeGroup := .operator 247024 245608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 247024) (leftOrdinal := 0)
    (rightResult := 245608) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge247062

namespace LeftMerge247063
def owner : Owner := ⟨.program ⟨257⟩, ⟨71175⟩⟩
def mergeEvent : Nat := 247063
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨16003⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact247024RawTerms
def rightRaw : List Term := Proof.Events959.exact245608RawTerms
def group : MergeGroup := .operator 247024 245608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 247024) (leftOrdinal := 19)
    (rightResult := 245608) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨16003⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨16003⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨16003⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge247063

namespace LeftMerge247071
def owner : Owner := ⟨.program ⟨257⟩, ⟨71176⟩⟩
def mergeEvent : Nat := 247071
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩] } }
def leftRaw : List Term := Proof.Events965.exact247065RawTerms
def rightRaw : List Term := Proof.Events060.exact15522RawTerms
def group : MergeGroup := .operator 247065 15522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 247065) (leftOrdinal := 0)
    (rightResult := 15522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7139⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge247071

namespace LeftMerge247072
def owner : Owner := ⟨.program ⟨257⟩, ⟨71176⟩⟩
def mergeEvent : Nat := 247072
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨67417⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩] } }
def leftRaw : List Term := Proof.Events965.exact247065RawTerms
def rightRaw : List Term := Proof.Events060.exact15522RawTerms
def group : MergeGroup := .operator 247065 15522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 247065) (leftOrdinal := 1)
    (rightResult := 15522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨67417⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7139⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨67417⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge247072

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
