import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge690
def owner : Owner := ⟨.program ⟨214⟩, ⟨15537⟩⟩
def mergeEvent : Nat := 690
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events002.exact686RawTerms
def rightRaw : List Term := Proof.Events002.exact683RawTerms
def group : MergeGroup := .operator 686 683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 686) (leftOrdinal := 0)
    (rightResult := 683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge690

namespace LeftMerge700
def owner : Owner := ⟨.program ⟨214⟩, ⟨15229⟩⟩
def mergeEvent : Nat := 700
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events002.exact696RawTerms
def rightRaw : List Term := Proof.Events002.exact693RawTerms
def group : MergeGroup := .operator 696 693
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 696) (leftOrdinal := 0)
    (rightResult := 693) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15228⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge700

namespace LeftMerge710
def owner : Owner := ⟨.program ⟨214⟩, ⟨15068⟩⟩
def mergeEvent : Nat := 710
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events002.exact706RawTerms
def rightRaw : List Term := Proof.Events002.exact703RawTerms
def group : MergeGroup := .operator 706 703
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 706) (leftOrdinal := 0)
    (rightResult := 703) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge710

namespace LeftMerge720
def owner : Owner := ⟨.program ⟨214⟩, ⟨14907⟩⟩
def mergeEvent : Nat := 720
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events002.exact716RawTerms
def rightRaw : List Term := Proof.Events002.exact713RawTerms
def group : MergeGroup := .operator 716 713
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 716) (leftOrdinal := 0)
    (rightResult := 713) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge720

namespace LeftMerge727
def owner : Owner := ⟨.program ⟨214⟩, ⟨6379⟩⟩
def mergeEvent : Nat := 727
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6378⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events002.exact723RawTerms
def rightRaw : List Term := Proof.Events002.exact723RawTerms
def group : MergeGroup := .operator 723 723
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 723) (leftOrdinal := 0)
    (rightResult := 723) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6378⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6378⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6378⟩⟩], []⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge727

namespace LeftMerge808
def owner : Owner := ⟨.program ⟨214⟩, ⟨18904⟩⟩
def mergeEvent : Nat := 808
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events003.exact804RawTerms
def rightRaw : List Term := Proof.Events000.exact34RawTerms
def group : MergeGroup := .operator 804 34
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 804) (leftOrdinal := 5)
    (rightResult := 34) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], []⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge808

namespace LeftMerge809
def owner : Owner := ⟨.program ⟨214⟩, ⟨18904⟩⟩
def mergeEvent : Nat := 809
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18140⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events003.exact804RawTerms
def rightRaw : List Term := Proof.Events000.exact34RawTerms
def group : MergeGroup := .operator 804 34
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 804) (leftOrdinal := 7)
    (rightResult := 34) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18140⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18140⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge809

namespace LeftMerge810
def owner : Owner := ⟨.program ⟨214⟩, ⟨18904⟩⟩
def mergeEvent : Nat := 810
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events003.exact804RawTerms
def rightRaw : List Term := Proof.Events000.exact34RawTerms
def group : MergeGroup := .operator 804 34
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 804) (leftOrdinal := 8)
    (rightResult := 34) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge810

namespace LeftMerge811
def owner : Owner := ⟨.program ⟨214⟩, ⟨18904⟩⟩
def mergeEvent : Nat := 811
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17510⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events003.exact804RawTerms
def rightRaw : List Term := Proof.Events000.exact34RawTerms
def group : MergeGroup := .operator 804 34
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 804) (leftOrdinal := 9)
    (rightResult := 34) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17510⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17510⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge811

namespace LeftMerge812
def owner : Owner := ⟨.program ⟨214⟩, ⟨18904⟩⟩
def mergeEvent : Nat := 812
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events003.exact804RawTerms
def rightRaw : List Term := Proof.Events000.exact34RawTerms
def group : MergeGroup := .operator 804 34
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 804) (leftOrdinal := 11)
    (rightResult := 34) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge812

namespace LeftMerge813
def owner : Owner := ⟨.program ⟨214⟩, ⟨18904⟩⟩
def mergeEvent : Nat := 813
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events003.exact804RawTerms
def rightRaw : List Term := Proof.Events000.exact34RawTerms
def group : MergeGroup := .operator 804 34
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 804) (leftOrdinal := 12)
    (rightResult := 34) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge813

namespace LeftMerge814
def owner : Owner := ⟨.program ⟨214⟩, ⟨18904⟩⟩
def mergeEvent : Nat := 814
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events003.exact804RawTerms
def rightRaw : List Term := Proof.Events000.exact34RawTerms
def group : MergeGroup := .operator 804 34
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 804) (leftOrdinal := 13)
    (rightResult := 34) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge814

namespace LeftMerge815
def owner : Owner := ⟨.program ⟨214⟩, ⟨18904⟩⟩
def mergeEvent : Nat := 815
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events003.exact804RawTerms
def rightRaw : List Term := Proof.Events000.exact34RawTerms
def group : MergeGroup := .operator 804 34
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 804) (leftOrdinal := 15)
    (rightResult := 34) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge815

namespace LeftMerge816
def owner : Owner := ⟨.program ⟨214⟩, ⟨18904⟩⟩
def mergeEvent : Nat := 816
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events003.exact804RawTerms
def rightRaw : List Term := Proof.Events000.exact34RawTerms
def group : MergeGroup := .operator 804 34
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 804) (leftOrdinal := 16)
    (rightResult := 34) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge816

namespace LeftMerge817
def owner : Owner := ⟨.program ⟨214⟩, ⟨18904⟩⟩
def mergeEvent : Nat := 817
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events003.exact804RawTerms
def rightRaw : List Term := Proof.Events000.exact34RawTerms
def group : MergeGroup := .operator 804 34
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 804) (leftOrdinal := 18)
    (rightResult := 34) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge817

namespace LeftMerge818
def owner : Owner := ⟨.program ⟨214⟩, ⟨18904⟩⟩
def mergeEvent : Nat := 818
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events003.exact804RawTerms
def rightRaw : List Term := Proof.Events000.exact34RawTerms
def group : MergeGroup := .operator 804 34
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 804) (leftOrdinal := 0)
    (rightResult := 34) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge818

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
