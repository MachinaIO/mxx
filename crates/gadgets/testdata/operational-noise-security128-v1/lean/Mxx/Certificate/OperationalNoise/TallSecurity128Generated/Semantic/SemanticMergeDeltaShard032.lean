import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge10435
def owner : Owner := ⟨.program ⟨257⟩, ⟨22082⟩⟩
def mergeEvent : Nat := 10435
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22081⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events040.exact10431RawTerms
def rightRaw : List Term := Proof.Events002.exact693RawTerms
def group : MergeGroup := .operator 10431 693
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10431) (leftOrdinal := 0)
    (rightResult := 693) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22081⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22081⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10435

namespace LeftMerge10443
def owner : Owner := ⟨.program ⟨257⟩, ⟨18862⟩⟩
def mergeEvent : Nat := 10443
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18861⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events040.exact10439RawTerms
def rightRaw : List Term := Proof.Events002.exact703RawTerms
def group : MergeGroup := .operator 10439 703
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10439) (leftOrdinal := 0)
    (rightResult := 703) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18861⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18861⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10443

namespace LeftMerge10451
def owner : Owner := ⟨.program ⟨257⟩, ⟨16031⟩⟩
def mergeEvent : Nat := 10451
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16030⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events040.exact10447RawTerms
def rightRaw : List Term := Proof.Events002.exact713RawTerms
def group : MergeGroup := .operator 10447 713
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10447) (leftOrdinal := 0)
    (rightResult := 713) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16030⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16030⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10451

namespace LeftMerge10532
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def mergeEvent : Nat := 10532
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events041.exact10528RawTerms
def rightRaw : List Term := Proof.Events038.exact9805RawTerms
def group : MergeGroup := .operator 10528 9805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10528) (leftOrdinal := 5)
    (rightResult := 9805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], []⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge10532

namespace LeftMerge10533
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def mergeEvent : Nat := 10533
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48359⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events041.exact10528RawTerms
def rightRaw : List Term := Proof.Events038.exact9805RawTerms
def group : MergeGroup := .operator 10528 9805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10528) (leftOrdinal := 7)
    (rightResult := 9805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48359⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48359⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10533

namespace LeftMerge10534
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def mergeEvent : Nat := 10534
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45679⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events041.exact10528RawTerms
def rightRaw : List Term := Proof.Events038.exact9805RawTerms
def group : MergeGroup := .operator 10528 9805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10528) (leftOrdinal := 8)
    (rightResult := 9805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45679⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45679⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10534

namespace LeftMerge10535
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def mergeEvent : Nat := 10535
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43002⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events041.exact10528RawTerms
def rightRaw : List Term := Proof.Events038.exact9805RawTerms
def group : MergeGroup := .operator 10528 9805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10528) (leftOrdinal := 9)
    (rightResult := 9805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43002⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43002⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10535

namespace LeftMerge10536
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def mergeEvent : Nat := 10536
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40322⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events041.exact10528RawTerms
def rightRaw : List Term := Proof.Events038.exact9805RawTerms
def group : MergeGroup := .operator 10528 9805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10528) (leftOrdinal := 11)
    (rightResult := 9805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40322⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40322⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10536

namespace LeftMerge10537
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def mergeEvent : Nat := 10537
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37639⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events041.exact10528RawTerms
def rightRaw : List Term := Proof.Events038.exact9805RawTerms
def group : MergeGroup := .operator 10528 9805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10528) (leftOrdinal := 12)
    (rightResult := 9805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37639⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37639⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10537

namespace LeftMerge10538
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def mergeEvent : Nat := 10538
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34959⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events041.exact10528RawTerms
def rightRaw : List Term := Proof.Events038.exact9805RawTerms
def group : MergeGroup := .operator 10528 9805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10528) (leftOrdinal := 13)
    (rightResult := 9805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34959⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34959⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10538

namespace LeftMerge10539
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def mergeEvent : Nat := 10539
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29302⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events041.exact10528RawTerms
def rightRaw : List Term := Proof.Events038.exact9805RawTerms
def group : MergeGroup := .operator 10528 9805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10528) (leftOrdinal := 15)
    (rightResult := 9805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29302⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29302⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10539

namespace LeftMerge10540
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def mergeEvent : Nat := 10540
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26622⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events041.exact10528RawTerms
def rightRaw : List Term := Proof.Events038.exact9805RawTerms
def group : MergeGroup := .operator 10528 9805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10528) (leftOrdinal := 16)
    (rightResult := 9805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26622⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26622⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10540

namespace LeftMerge10541
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def mergeEvent : Nat := 10541
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66588⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events041.exact10528RawTerms
def rightRaw : List Term := Proof.Events038.exact9805RawTerms
def group : MergeGroup := .operator 10528 9805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10528) (leftOrdinal := 18)
    (rightResult := 9805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66588⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66588⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10541

namespace LeftMerge10542
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def mergeEvent : Nat := 10542
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨63085⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events041.exact10528RawTerms
def rightRaw : List Term := Proof.Events038.exact9805RawTerms
def group : MergeGroup := .operator 10528 9805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10528) (leftOrdinal := 0)
    (rightResult := 9805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63085⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨63085⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10542

namespace LeftMerge10543
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def mergeEvent : Nat := 10543
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events041.exact10528RawTerms
def rightRaw : List Term := Proof.Events038.exact9805RawTerms
def group : MergeGroup := .operator 10528 9805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10528) (leftOrdinal := 1)
    (rightResult := 9805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10543

namespace LeftMerge10544
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def mergeEvent : Nat := 10544
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨57125⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events041.exact10528RawTerms
def rightRaw : List Term := Proof.Events038.exact9805RawTerms
def group : MergeGroup := .operator 10528 9805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10528) (leftOrdinal := 2)
    (rightResult := 9805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57125⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨57125⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10544

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
