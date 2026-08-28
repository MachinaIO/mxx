import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge280597
def owner : Owner := ⟨.program ⟨257⟩, ⟨67348⟩⟩
def mergeEvent : Nat := 280597
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280538RawTerms
def rightRaw : List Term := Proof.Events055.exact14262RawTerms
def group : MergeGroup := .operator 280538 14262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280538) (leftOrdinal := 0)
    (rightResult := 14262) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge280597

namespace LeftMerge280598
def owner : Owner := ⟨.program ⟨257⟩, ⟨67348⟩⟩
def mergeEvent : Nat := 280598
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280538RawTerms
def rightRaw : List Term := Proof.Events055.exact14262RawTerms
def group : MergeGroup := .operator 280538 14262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280538) (leftOrdinal := 0)
    (rightResult := 14262) (rightOrdinal := 17) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge280598

namespace LeftMerge280599
def owner : Owner := ⟨.program ⟨257⟩, ⟨67348⟩⟩
def mergeEvent : Nat := 280599
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280538RawTerms
def rightRaw : List Term := Proof.Events055.exact14262RawTerms
def group : MergeGroup := .operator 280538 14262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280538) (leftOrdinal := 1)
    (rightResult := 14262) (rightOrdinal := 5) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], [⟨.program ⟨257⟩, ⟨7271⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge280599

namespace LeftMerge280600
def owner : Owner := ⟨.program ⟨257⟩, ⟨67348⟩⟩
def mergeEvent : Nat := 280600
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48281⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280538RawTerms
def rightRaw : List Term := Proof.Events055.exact14262RawTerms
def group : MergeGroup := .operator 280538 14262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280538) (leftOrdinal := 1)
    (rightResult := 14262) (rightOrdinal := 7) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48281⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48281⟩⟩], [⟨.program ⟨257⟩, ⟨7271⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280600

namespace LeftMerge280601
def owner : Owner := ⟨.program ⟨257⟩, ⟨67348⟩⟩
def mergeEvent : Nat := 280601
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280538RawTerms
def rightRaw : List Term := Proof.Events055.exact14262RawTerms
def group : MergeGroup := .operator 280538 14262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280538) (leftOrdinal := 1)
    (rightResult := 14262) (rightOrdinal := 8) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45601⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45601⟩⟩], [⟨.program ⟨257⟩, ⟨7271⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280601

namespace LeftMerge280602
def owner : Owner := ⟨.program ⟨257⟩, ⟨67348⟩⟩
def mergeEvent : Nat := 280602
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42924⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280538RawTerms
def rightRaw : List Term := Proof.Events055.exact14262RawTerms
def group : MergeGroup := .operator 280538 14262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280538) (leftOrdinal := 1)
    (rightResult := 14262) (rightOrdinal := 9) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42924⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42924⟩⟩], [⟨.program ⟨257⟩, ⟨7271⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280602

namespace LeftMerge280603
def owner : Owner := ⟨.program ⟨257⟩, ⟨67348⟩⟩
def mergeEvent : Nat := 280603
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40244⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280538RawTerms
def rightRaw : List Term := Proof.Events055.exact14262RawTerms
def group : MergeGroup := .operator 280538 14262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280538) (leftOrdinal := 1)
    (rightResult := 14262) (rightOrdinal := 11) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40244⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40244⟩⟩], [⟨.program ⟨257⟩, ⟨7271⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280603

namespace LeftMerge280604
def owner : Owner := ⟨.program ⟨257⟩, ⟨67348⟩⟩
def mergeEvent : Nat := 280604
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37561⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280538RawTerms
def rightRaw : List Term := Proof.Events055.exact14262RawTerms
def group : MergeGroup := .operator 280538 14262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280538) (leftOrdinal := 1)
    (rightResult := 14262) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37561⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37561⟩⟩], [⟨.program ⟨257⟩, ⟨7271⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280604

namespace LeftMerge280605
def owner : Owner := ⟨.program ⟨257⟩, ⟨67348⟩⟩
def mergeEvent : Nat := 280605
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34881⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280538RawTerms
def rightRaw : List Term := Proof.Events055.exact14262RawTerms
def group : MergeGroup := .operator 280538 14262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280538) (leftOrdinal := 1)
    (rightResult := 14262) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34881⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34881⟩⟩], [⟨.program ⟨257⟩, ⟨7271⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280605

namespace LeftMerge280606
def owner : Owner := ⟨.program ⟨257⟩, ⟨67348⟩⟩
def mergeEvent : Nat := 280606
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29224⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280538RawTerms
def rightRaw : List Term := Proof.Events055.exact14262RawTerms
def group : MergeGroup := .operator 280538 14262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280538) (leftOrdinal := 1)
    (rightResult := 14262) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29224⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29224⟩⟩], [⟨.program ⟨257⟩, ⟨7271⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280606

namespace LeftMerge280607
def owner : Owner := ⟨.program ⟨257⟩, ⟨67348⟩⟩
def mergeEvent : Nat := 280607
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280538RawTerms
def rightRaw : List Term := Proof.Events055.exact14262RawTerms
def group : MergeGroup := .operator 280538 14262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280538) (leftOrdinal := 1)
    (rightResult := 14262) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26544⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26544⟩⟩], [⟨.program ⟨257⟩, ⟨7271⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280607

namespace LeftMerge280608
def owner : Owner := ⟨.program ⟨257⟩, ⟨67348⟩⟩
def mergeEvent : Nat := 280608
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66168⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280538RawTerms
def rightRaw : List Term := Proof.Events055.exact14262RawTerms
def group : MergeGroup := .operator 280538 14262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280538) (leftOrdinal := 1)
    (rightResult := 14262) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66168⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66168⟩⟩], [⟨.program ⟨257⟩, ⟨7271⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280608

namespace LeftMerge280609
def owner : Owner := ⟨.program ⟨257⟩, ⟨67348⟩⟩
def mergeEvent : Nat := 280609
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280538RawTerms
def rightRaw : List Term := Proof.Events055.exact14262RawTerms
def group : MergeGroup := .operator 280538 14262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280538) (leftOrdinal := 1)
    (rightResult := 14262) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], [⟨.program ⟨257⟩, ⟨7271⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280609

namespace LeftMerge280610
def owner : Owner := ⟨.program ⟨257⟩, ⟨67348⟩⟩
def mergeEvent : Nat := 280610
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280538RawTerms
def rightRaw : List Term := Proof.Events055.exact14262RawTerms
def group : MergeGroup := .operator 280538 14262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280538) (leftOrdinal := 1)
    (rightResult := 14262) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], [⟨.program ⟨257⟩, ⟨7271⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280610

namespace LeftMerge280611
def owner : Owner := ⟨.program ⟨257⟩, ⟨67348⟩⟩
def mergeEvent : Nat := 280611
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280538RawTerms
def rightRaw : List Term := Proof.Events055.exact14262RawTerms
def group : MergeGroup := .operator 280538 14262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280538) (leftOrdinal := 1)
    (rightResult := 14262) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], [⟨.program ⟨257⟩, ⟨7271⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280611

namespace LeftMerge280612
def owner : Owner := ⟨.program ⟨257⟩, ⟨67348⟩⟩
def mergeEvent : Nat := 280612
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280538RawTerms
def rightRaw : List Term := Proof.Events055.exact14262RawTerms
def group : MergeGroup := .operator 280538 14262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280538) (leftOrdinal := 1)
    (rightResult := 14262) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], [⟨.program ⟨257⟩, ⟨7271⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280612

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
