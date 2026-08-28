import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge46597
def owner : Owner := ⟨.program ⟨257⟩, ⟨67632⟩⟩
def mergeEvent : Nat := 46597
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46538RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 46538 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46538) (leftOrdinal := 0)
    (rightResult := 2300) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46597

namespace LeftMerge46598
def owner : Owner := ⟨.program ⟨257⟩, ⟨67632⟩⟩
def mergeEvent : Nat := 46598
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46538RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 46538 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46538) (leftOrdinal := 0)
    (rightResult := 2300) (rightOrdinal := 17) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46598

namespace LeftMerge46599
def owner : Owner := ⟨.program ⟨257⟩, ⟨67632⟩⟩
def mergeEvent : Nat := 46599
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46538RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 46538 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46538) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 5) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67626⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67626⟩⟩], [⟨.program ⟨257⟩, ⟨7239⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46599

namespace LeftMerge46600
def owner : Owner := ⟨.program ⟨257⟩, ⟨67632⟩⟩
def mergeEvent : Nat := 46600
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48463⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46538RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 46538 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46538) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 7) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48463⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48463⟩⟩], [⟨.program ⟨257⟩, ⟨7239⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46600

namespace LeftMerge46601
def owner : Owner := ⟨.program ⟨257⟩, ⟨67632⟩⟩
def mergeEvent : Nat := 46601
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨45783⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46538RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 46538 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46538) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 8) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45783⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨45783⟩⟩], [⟨.program ⟨257⟩, ⟨7239⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46601

namespace LeftMerge46602
def owner : Owner := ⟨.program ⟨257⟩, ⟨67632⟩⟩
def mergeEvent : Nat := 46602
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨43106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46538RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 46538 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46538) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 9) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43106⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨43106⟩⟩], [⟨.program ⟨257⟩, ⟨7239⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46602

namespace LeftMerge46603
def owner : Owner := ⟨.program ⟨257⟩, ⟨67632⟩⟩
def mergeEvent : Nat := 46603
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40426⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46538RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 46538 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46538) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 11) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40426⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40426⟩⟩], [⟨.program ⟨257⟩, ⟨7239⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46603

namespace LeftMerge46604
def owner : Owner := ⟨.program ⟨257⟩, ⟨67632⟩⟩
def mergeEvent : Nat := 46604
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46538RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 46538 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46538) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37743⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37743⟩⟩], [⟨.program ⟨257⟩, ⟨7239⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46604

namespace LeftMerge46605
def owner : Owner := ⟨.program ⟨257⟩, ⟨67632⟩⟩
def mergeEvent : Nat := 46605
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35063⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46538RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 46538 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46538) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35063⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35063⟩⟩], [⟨.program ⟨257⟩, ⟨7239⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46605

namespace LeftMerge46606
def owner : Owner := ⟨.program ⟨257⟩, ⟨67632⟩⟩
def mergeEvent : Nat := 46606
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29406⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46538RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 46538 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46538) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29406⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨7239⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46606

namespace LeftMerge46607
def owner : Owner := ⟨.program ⟨257⟩, ⟨67632⟩⟩
def mergeEvent : Nat := 46607
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26726⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46538RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 46538 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46538) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26726⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26726⟩⟩], [⟨.program ⟨257⟩, ⟨7239⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46607

namespace LeftMerge46608
def owner : Owner := ⟨.program ⟨257⟩, ⟨67632⟩⟩
def mergeEvent : Nat := 46608
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46538RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 46538 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46538) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67148⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67148⟩⟩], [⟨.program ⟨257⟩, ⟨7239⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46608

namespace LeftMerge46609
def owner : Owner := ⟨.program ⟨257⟩, ⟨67632⟩⟩
def mergeEvent : Nat := 46609
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63237⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46538RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 46538 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46538) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63237⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63237⟩⟩], [⟨.program ⟨257⟩, ⟨7239⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46609

namespace LeftMerge46610
def owner : Owner := ⟨.program ⟨257⟩, ⟨67632⟩⟩
def mergeEvent : Nat := 46610
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60257⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46538RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 46538 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46538) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60257⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60257⟩⟩], [⟨.program ⟨257⟩, ⟨7239⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46610

namespace LeftMerge46611
def owner : Owner := ⟨.program ⟨257⟩, ⟨67632⟩⟩
def mergeEvent : Nat := 46611
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57277⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46538RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 46538 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46538) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57277⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57277⟩⟩], [⟨.program ⟨257⟩, ⟨7239⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46611

namespace LeftMerge46612
def owner : Owner := ⟨.program ⟨257⟩, ⟨67632⟩⟩
def mergeEvent : Nat := 46612
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54297⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46538RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 46538 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46538) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7239⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54297⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54297⟩⟩], [⟨.program ⟨257⟩, ⟨7239⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46612

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
