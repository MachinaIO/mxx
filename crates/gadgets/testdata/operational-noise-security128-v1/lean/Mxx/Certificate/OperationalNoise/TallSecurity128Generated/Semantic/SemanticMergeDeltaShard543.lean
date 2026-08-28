import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge90481
def owner : Owner := ⟨.program ⟨257⟩, ⟨67572⟩⟩
def mergeEvent : Nat := 90481
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29367⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90413RawTerms
def rightRaw : List Term := Proof.Events017.exact4544RawTerms
def group : MergeGroup := .operator 90413 4544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90413) (leftOrdinal := 1)
    (rightResult := 4544) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29367⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29367⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90481

namespace LeftMerge90482
def owner : Owner := ⟨.program ⟨257⟩, ⟨67572⟩⟩
def mergeEvent : Nat := 90482
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨26687⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90413RawTerms
def rightRaw : List Term := Proof.Events017.exact4544RawTerms
def group : MergeGroup := .operator 90413 4544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90413) (leftOrdinal := 1)
    (rightResult := 4544) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26687⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨26687⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90482

namespace LeftMerge90483
def owner : Owner := ⟨.program ⟨257⟩, ⟨67572⟩⟩
def mergeEvent : Nat := 90483
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨66938⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90413RawTerms
def rightRaw : List Term := Proof.Events017.exact4544RawTerms
def group : MergeGroup := .operator 90413 4544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90413) (leftOrdinal := 1)
    (rightResult := 4544) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66938⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨66938⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90483

namespace LeftMerge90484
def owner : Owner := ⟨.program ⟨257⟩, ⟨67572⟩⟩
def mergeEvent : Nat := 90484
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨63180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90413RawTerms
def rightRaw : List Term := Proof.Events017.exact4544RawTerms
def group : MergeGroup := .operator 90413 4544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90413) (leftOrdinal := 1)
    (rightResult := 4544) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63180⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨63180⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90484

namespace LeftMerge90485
def owner : Owner := ⟨.program ⟨257⟩, ⟨67572⟩⟩
def mergeEvent : Nat := 90485
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨60200⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90413RawTerms
def rightRaw : List Term := Proof.Events017.exact4544RawTerms
def group : MergeGroup := .operator 90413 4544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90413) (leftOrdinal := 1)
    (rightResult := 4544) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60200⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨60200⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90485

namespace LeftMerge90486
def owner : Owner := ⟨.program ⟨257⟩, ⟨67572⟩⟩
def mergeEvent : Nat := 90486
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨57220⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90413RawTerms
def rightRaw : List Term := Proof.Events017.exact4544RawTerms
def group : MergeGroup := .operator 90413 4544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90413) (leftOrdinal := 1)
    (rightResult := 4544) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57220⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨57220⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90486

namespace LeftMerge90487
def owner : Owner := ⟨.program ⟨257⟩, ⟨67572⟩⟩
def mergeEvent : Nat := 90487
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨54240⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90413RawTerms
def rightRaw : List Term := Proof.Events017.exact4544RawTerms
def group : MergeGroup := .operator 90413 4544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90413) (leftOrdinal := 1)
    (rightResult := 4544) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54240⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨54240⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90487

namespace LeftMerge90488
def owner : Owner := ⟨.program ⟨257⟩, ⟨67572⟩⟩
def mergeEvent : Nat := 90488
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90413RawTerms
def rightRaw : List Term := Proof.Events017.exact4544RawTerms
def group : MergeGroup := .operator 90413 4544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90413) (leftOrdinal := 1)
    (rightResult := 4544) (rightOrdinal := 4) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51260⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51260⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90488

namespace LeftMerge90489
def owner : Owner := ⟨.program ⟨257⟩, ⟨67572⟩⟩
def mergeEvent : Nat := 90489
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨32196⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90413RawTerms
def rightRaw : List Term := Proof.Events017.exact4544RawTerms
def group : MergeGroup := .operator 90413 4544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90413) (leftOrdinal := 1)
    (rightResult := 4544) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32196⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨32196⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90489

namespace LeftMerge90490
def owner : Owner := ⟨.program ⟨257⟩, ⟨67572⟩⟩
def mergeEvent : Nat := 90490
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90413RawTerms
def rightRaw : List Term := Proof.Events017.exact4544RawTerms
def group : MergeGroup := .operator 90413 4544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90413) (leftOrdinal := 1)
    (rightResult := 4544) (rightOrdinal := 10) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90490

namespace LeftMerge90491
def owner : Owner := ⟨.program ⟨257⟩, ⟨67572⟩⟩
def mergeEvent : Nat := 90491
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90413RawTerms
def rightRaw : List Term := Proof.Events017.exact4544RawTerms
def group : MergeGroup := .operator 90413 4544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90413) (leftOrdinal := 1)
    (rightResult := 4544) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90491

namespace LeftMerge90492
def owner : Owner := ⟨.program ⟨257⟩, ⟨67572⟩⟩
def mergeEvent : Nat := 90492
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90413RawTerms
def rightRaw : List Term := Proof.Events017.exact4544RawTerms
def group : MergeGroup := .operator 90413 4544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90413) (leftOrdinal := 1)
    (rightResult := 4544) (rightOrdinal := 17) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90492

namespace LeftMerge90527
def owner : Owner := ⟨.program ⟨257⟩, ⟨9904⟩⟩
def mergeEvent : Nat := 90527
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90398RawTerms
def rightRaw : List Term := Proof.Events000.exact2RawTerms
def group : MergeGroup := .operator 90398 2
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90398) (leftOrdinal := 0)
    (rightResult := 2) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90527

namespace LeftMerge90532
def owner : Owner := ⟨.program ⟨257⟩, ⟨47957⟩⟩
def mergeEvent : Nat := 90532
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events014.exact3834RawTerms
def rightRaw : List Term := Proof.Events353.exact90528RawTerms
def group : MergeGroup := .operator 3834 90528
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3834) (leftOrdinal := 0)
    (rightResult := 90528) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨47954⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90532

namespace LeftMerge90537
def owner : Owner := ⟨.program ⟨257⟩, ⟨9919⟩⟩
def mergeEvent : Nat := 90537
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90398RawTerms
def rightRaw : List Term := Proof.Events066.exact17065RawTerms
def group : MergeGroup := .operator 90398 17065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90398) (leftOrdinal := 0)
    (rightResult := 17065) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90537

namespace LeftMerge90554
def owner : Owner := ⟨.program ⟨257⟩, ⟨47960⟩⟩
def mergeEvent : Nat := 90554
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90548RawTerms
def rightRaw : List Term := Proof.Events014.exact3837RawTerms
def group : MergeGroup := .operator 90548 3837
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90548) (leftOrdinal := 1)
    (rightResult := 3837) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15156⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90554

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
