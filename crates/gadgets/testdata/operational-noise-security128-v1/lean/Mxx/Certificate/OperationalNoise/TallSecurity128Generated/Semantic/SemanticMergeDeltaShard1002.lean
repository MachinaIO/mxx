import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge163605
def owner : Owner := ⟨.program ⟨257⟩, ⟨67548⟩⟩
def mergeEvent : Nat := 163605
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163538RawTerms
def rightRaw : List Term := Proof.Events032.exact8284RawTerms
def group : MergeGroup := .operator 163538 8284
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163538) (leftOrdinal := 1)
    (rightResult := 8284) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163605

namespace LeftMerge163606
def owner : Owner := ⟨.program ⟨257⟩, ⟨67548⟩⟩
def mergeEvent : Nat := 163606
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163538RawTerms
def rightRaw : List Term := Proof.Events032.exact8284RawTerms
def group : MergeGroup := .operator 163538 8284
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163538) (leftOrdinal := 1)
    (rightResult := 8284) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163606

namespace LeftMerge163607
def owner : Owner := ⟨.program ⟨257⟩, ⟨67548⟩⟩
def mergeEvent : Nat := 163607
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163538RawTerms
def rightRaw : List Term := Proof.Events032.exact8284RawTerms
def group : MergeGroup := .operator 163538 8284
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163538) (leftOrdinal := 1)
    (rightResult := 8284) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163607

namespace LeftMerge163608
def owner : Owner := ⟨.program ⟨257⟩, ⟨67548⟩⟩
def mergeEvent : Nat := 163608
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163538RawTerms
def rightRaw : List Term := Proof.Events032.exact8284RawTerms
def group : MergeGroup := .operator 163538 8284
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163538) (leftOrdinal := 1)
    (rightResult := 8284) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163608

namespace LeftMerge163609
def owner : Owner := ⟨.program ⟨257⟩, ⟨67548⟩⟩
def mergeEvent : Nat := 163609
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163538RawTerms
def rightRaw : List Term := Proof.Events032.exact8284RawTerms
def group : MergeGroup := .operator 163538 8284
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163538) (leftOrdinal := 1)
    (rightResult := 8284) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163609

namespace LeftMerge163610
def owner : Owner := ⟨.program ⟨257⟩, ⟨67548⟩⟩
def mergeEvent : Nat := 163610
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163538RawTerms
def rightRaw : List Term := Proof.Events032.exact8284RawTerms
def group : MergeGroup := .operator 163538 8284
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163538) (leftOrdinal := 1)
    (rightResult := 8284) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163610

namespace LeftMerge163611
def owner : Owner := ⟨.program ⟨257⟩, ⟨67548⟩⟩
def mergeEvent : Nat := 163611
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163538RawTerms
def rightRaw : List Term := Proof.Events032.exact8284RawTerms
def group : MergeGroup := .operator 163538 8284
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163538) (leftOrdinal := 1)
    (rightResult := 8284) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163611

namespace LeftMerge163612
def owner : Owner := ⟨.program ⟨257⟩, ⟨67548⟩⟩
def mergeEvent : Nat := 163612
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163538RawTerms
def rightRaw : List Term := Proof.Events032.exact8284RawTerms
def group : MergeGroup := .operator 163538 8284
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163538) (leftOrdinal := 1)
    (rightResult := 8284) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163612

namespace LeftMerge163613
def owner : Owner := ⟨.program ⟨257⟩, ⟨67548⟩⟩
def mergeEvent : Nat := 163613
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163538RawTerms
def rightRaw : List Term := Proof.Events032.exact8284RawTerms
def group : MergeGroup := .operator 163538 8284
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163538) (leftOrdinal := 1)
    (rightResult := 8284) (rightOrdinal := 4) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163613

namespace LeftMerge163614
def owner : Owner := ⟨.program ⟨257⟩, ⟨67548⟩⟩
def mergeEvent : Nat := 163614
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163538RawTerms
def rightRaw : List Term := Proof.Events032.exact8284RawTerms
def group : MergeGroup := .operator 163538 8284
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163538) (leftOrdinal := 1)
    (rightResult := 8284) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163614

namespace LeftMerge163615
def owner : Owner := ⟨.program ⟨257⟩, ⟨67548⟩⟩
def mergeEvent : Nat := 163615
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163538RawTerms
def rightRaw : List Term := Proof.Events032.exact8284RawTerms
def group : MergeGroup := .operator 163538 8284
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163538) (leftOrdinal := 1)
    (rightResult := 8284) (rightOrdinal := 10) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163615

namespace LeftMerge163616
def owner : Owner := ⟨.program ⟨257⟩, ⟨67548⟩⟩
def mergeEvent : Nat := 163616
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163538RawTerms
def rightRaw : List Term := Proof.Events032.exact8284RawTerms
def group : MergeGroup := .operator 163538 8284
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163538) (leftOrdinal := 1)
    (rightResult := 8284) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163616

namespace LeftMerge163617
def owner : Owner := ⟨.program ⟨257⟩, ⟨67548⟩⟩
def mergeEvent : Nat := 163617
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163538RawTerms
def rightRaw : List Term := Proof.Events032.exact8284RawTerms
def group : MergeGroup := .operator 163538 8284
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163538) (leftOrdinal := 1)
    (rightResult := 8284) (rightOrdinal := 17) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163617

namespace LeftMerge163652
def owner : Owner := ⟨.program ⟨257⟩, ⟨7010⟩⟩
def mergeEvent : Nat := 163652
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163523RawTerms
def rightRaw : List Term := Proof.Events000.exact2RawTerms
def group : MergeGroup := .operator 163523 2
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163523) (leftOrdinal := 0)
    (rightResult := 2) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163652

namespace LeftMerge163657
def owner : Owner := ⟨.program ⟨257⟩, ⟨47933⟩⟩
def mergeEvent : Nat := 163657
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨47930⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events029.exact7574RawTerms
def rightRaw : List Term := Proof.Events639.exact163653RawTerms
def group : MergeGroup := .operator 7574 163653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7574) (leftOrdinal := 0)
    (rightResult := 163653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨47930⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨47930⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163657

namespace LeftMerge163662
def owner : Owner := ⟨.program ⟨257⟩, ⟨9047⟩⟩
def mergeEvent : Nat := 163662
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163523RawTerms
def rightRaw : List Term := Proof.Events066.exact17065RawTerms
def group : MergeGroup := .operator 163523 17065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163523) (leftOrdinal := 0)
    (rightResult := 17065) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163662

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
