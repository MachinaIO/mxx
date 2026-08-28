import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge163405
def owner : Owner := ⟨.program ⟨257⟩, ⟨71151⟩⟩
def mergeEvent : Nat := 163405
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163396RawTerms
def rightRaw : List Term := Proof.Events582.exact148993RawTerms
def group : MergeGroup := .operator 163396 148993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163396) (leftOrdinal := 13)
    (rightResult := 148993) (rightOrdinal := 31) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], [⟨.program ⟨257⟩, ⟨7253⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163405

namespace LeftMerge163406
def owner : Owner := ⟨.program ⟨257⟩, ⟨71151⟩⟩
def mergeEvent : Nat := 163406
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163396RawTerms
def rightRaw : List Term := Proof.Events582.exact148993RawTerms
def group : MergeGroup := .operator 163396 148993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163396) (leftOrdinal := 14)
    (rightResult := 148993) (rightOrdinal := 32) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], [⟨.program ⟨257⟩, ⟨7253⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163406

namespace LeftMerge163407
def owner : Owner := ⟨.program ⟨257⟩, ⟨71151⟩⟩
def mergeEvent : Nat := 163407
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163396RawTerms
def rightRaw : List Term := Proof.Events582.exact148993RawTerms
def group : MergeGroup := .operator 163396 148993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163396) (leftOrdinal := 16)
    (rightResult := 148993) (rightOrdinal := 34) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], [⟨.program ⟨257⟩, ⟨7253⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163407

namespace LeftMerge163408
def owner : Owner := ⟨.program ⟨257⟩, ⟨71151⟩⟩
def mergeEvent : Nat := 163408
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163396RawTerms
def rightRaw : List Term := Proof.Events582.exact148993RawTerms
def group : MergeGroup := .operator 163396 148993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163396) (leftOrdinal := 17)
    (rightResult := 148993) (rightOrdinal := 35) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], [⟨.program ⟨257⟩, ⟨7253⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163408

namespace LeftMerge163409
def owner : Owner := ⟨.program ⟨257⟩, ⟨71151⟩⟩
def mergeEvent : Nat := 163409
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163396RawTerms
def rightRaw : List Term := Proof.Events582.exact148993RawTerms
def group : MergeGroup := .operator 163396 148993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163396) (leftOrdinal := 19)
    (rightResult := 148993) (rightOrdinal := 37) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], [⟨.program ⟨257⟩, ⟨7253⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163409

namespace LeftMerge163410
def owner : Owner := ⟨.program ⟨257⟩, ⟨71151⟩⟩
def mergeEvent : Nat := 163410
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163396RawTerms
def rightRaw : List Term := Proof.Events582.exact148993RawTerms
def group : MergeGroup := .operator 163396 148993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163396) (leftOrdinal := 1)
    (rightResult := 148993) (rightOrdinal := 19) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], [⟨.program ⟨257⟩, ⟨7253⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163410

namespace LeftMerge163411
def owner : Owner := ⟨.program ⟨257⟩, ⟨71151⟩⟩
def mergeEvent : Nat := 163411
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163396RawTerms
def rightRaw : List Term := Proof.Events582.exact148993RawTerms
def group : MergeGroup := .operator 163396 148993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163396) (leftOrdinal := 2)
    (rightResult := 148993) (rightOrdinal := 20) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], [⟨.program ⟨257⟩, ⟨7253⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163411

namespace LeftMerge163412
def owner : Owner := ⟨.program ⟨257⟩, ⟨71151⟩⟩
def mergeEvent : Nat := 163412
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163396RawTerms
def rightRaw : List Term := Proof.Events582.exact148993RawTerms
def group : MergeGroup := .operator 163396 148993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163396) (leftOrdinal := 3)
    (rightResult := 148993) (rightOrdinal := 21) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], [⟨.program ⟨257⟩, ⟨7253⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163412

namespace LeftMerge163413
def owner : Owner := ⟨.program ⟨257⟩, ⟨71151⟩⟩
def mergeEvent : Nat := 163413
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163396RawTerms
def rightRaw : List Term := Proof.Events582.exact148993RawTerms
def group : MergeGroup := .operator 163396 148993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163396) (leftOrdinal := 4)
    (rightResult := 148993) (rightOrdinal := 22) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], [⟨.program ⟨257⟩, ⟨7253⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163413

namespace LeftMerge163414
def owner : Owner := ⟨.program ⟨257⟩, ⟨71151⟩⟩
def mergeEvent : Nat := 163414
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163396RawTerms
def rightRaw : List Term := Proof.Events582.exact148993RawTerms
def group : MergeGroup := .operator 163396 148993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163396) (leftOrdinal := 5)
    (rightResult := 148993) (rightOrdinal := 23) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], [⟨.program ⟨257⟩, ⟨7253⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163414

namespace LeftMerge163415
def owner : Owner := ⟨.program ⟨257⟩, ⟨71151⟩⟩
def mergeEvent : Nat := 163415
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163396RawTerms
def rightRaw : List Term := Proof.Events582.exact148993RawTerms
def group : MergeGroup := .operator 163396 148993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163396) (leftOrdinal := 7)
    (rightResult := 148993) (rightOrdinal := 25) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], [⟨.program ⟨257⟩, ⟨7253⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163415

namespace LeftMerge163416
def owner : Owner := ⟨.program ⟨257⟩, ⟨71151⟩⟩
def mergeEvent : Nat := 163416
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163396RawTerms
def rightRaw : List Term := Proof.Events582.exact148993RawTerms
def group : MergeGroup := .operator 163396 148993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163396) (leftOrdinal := 11)
    (rightResult := 148993) (rightOrdinal := 29) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], [⟨.program ⟨257⟩, ⟨7253⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163416

namespace LeftMerge163417
def owner : Owner := ⟨.program ⟨257⟩, ⟨71151⟩⟩
def mergeEvent : Nat := 163417
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163396RawTerms
def rightRaw : List Term := Proof.Events582.exact148993RawTerms
def group : MergeGroup := .operator 163396 148993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163396) (leftOrdinal := 15)
    (rightResult := 148993) (rightOrdinal := 33) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨7253⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163417

namespace LeftMerge163418
def owner : Owner := ⟨.program ⟨257⟩, ⟨71151⟩⟩
def mergeEvent : Nat := 163418
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163396RawTerms
def rightRaw : List Term := Proof.Events582.exact148993RawTerms
def group : MergeGroup := .operator 163396 148993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163396) (leftOrdinal := 18)
    (rightResult := 148993) (rightOrdinal := 36) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨7253⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163418

namespace LeftMerge163426
def owner : Owner := ⟨.program ⟨257⟩, ⟨71152⟩⟩
def mergeEvent : Nat := 163426
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67399⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7135⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163420RawTerms
def rightRaw : List Term := Proof.Events063.exact16334RawTerms
def group : MergeGroup := .operator 163420 16334
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163420) (leftOrdinal := 6)
    (rightResult := 16334) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67399⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7135⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67399⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7135⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163426

namespace LeftMerge163428
def owner : Owner := ⟨.program ⟨257⟩, ⟨71152⟩⟩
def mergeEvent : Nat := 163428
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67399⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events063.exact16327RawTerms
def group : MergeGroup := .relation 163427
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 163427) (rhsResult := 16327)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67399⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7135⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7135⟩⟩) ⟨7033⟩ 16327) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67399⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163428

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
