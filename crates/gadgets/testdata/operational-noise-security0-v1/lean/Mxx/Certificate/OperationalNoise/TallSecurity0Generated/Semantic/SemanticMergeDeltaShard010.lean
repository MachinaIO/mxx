import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge3623
def owner : Owner := ⟨.program ⟨214⟩, ⟨17547⟩⟩
def mergeEvent : Nat := 3623
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3619RawTerms
def rightRaw : List Term := Proof.Events002.exact593RawTerms
def group : MergeGroup := .operator 3619 593
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3619) (leftOrdinal := 0)
    (rightResult := 593) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17546⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3623

namespace LeftMerge3631
def owner : Owner := ⟨.program ⟨214⟩, ⟨18819⟩⟩
def mergeEvent : Nat := 3631
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18818⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3627RawTerms
def rightRaw : List Term := Proof.Events002.exact603RawTerms
def group : MergeGroup := .operator 3627 603
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3627) (leftOrdinal := 0)
    (rightResult := 603) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18818⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18818⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3631

namespace LeftMerge3639
def owner : Owner := ⟨.program ⟨214⟩, ⟨17603⟩⟩
def mergeEvent : Nat := 3639
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17602⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3635RawTerms
def rightRaw : List Term := Proof.Events002.exact613RawTerms
def group : MergeGroup := .operator 3635 613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3635) (leftOrdinal := 0)
    (rightResult := 613) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17602⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17602⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3639

namespace LeftMerge3647
def owner : Owner := ⟨.program ⟨214⟩, ⟨17659⟩⟩
def mergeEvent : Nat := 3647
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3643RawTerms
def rightRaw : List Term := Proof.Events002.exact623RawTerms
def group : MergeGroup := .operator 3643 623
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3643) (leftOrdinal := 0)
    (rightResult := 623) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17658⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3647

namespace LeftMerge3655
def owner : Owner := ⟨.program ⟨214⟩, ⟨18029⟩⟩
def mergeEvent : Nat := 3655
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3651RawTerms
def rightRaw : List Term := Proof.Events002.exact633RawTerms
def group : MergeGroup := .operator 3651 633
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3651) (leftOrdinal := 0)
    (rightResult := 633) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18028⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3655

namespace LeftMerge3663
def owner : Owner := ⟨.program ⟨214⟩, ⟨17162⟩⟩
def mergeEvent : Nat := 3663
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3659RawTerms
def rightRaw : List Term := Proof.Events002.exact643RawTerms
def group : MergeGroup := .operator 3659 643
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3659) (leftOrdinal := 0)
    (rightResult := 643) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17161⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3663

namespace LeftMerge3671
def owner : Owner := ⟨.program ⟨214⟩, ⟨17218⟩⟩
def mergeEvent : Nat := 3671
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17217⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3667RawTerms
def rightRaw : List Term := Proof.Events002.exact653RawTerms
def group : MergeGroup := .operator 3667 653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3667) (leftOrdinal := 0)
    (rightResult := 653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17217⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17217⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3671

namespace LeftMerge3679
def owner : Owner := ⟨.program ⟨214⟩, ⟨17435⟩⟩
def mergeEvent : Nat := 3679
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17434⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3675RawTerms
def rightRaw : List Term := Proof.Events002.exact663RawTerms
def group : MergeGroup := .operator 3675 663
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3675) (leftOrdinal := 0)
    (rightResult := 663) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17434⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17434⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3679

namespace LeftMerge3687
def owner : Owner := ⟨.program ⟨214⟩, ⟨17807⟩⟩
def mergeEvent : Nat := 3687
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17806⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3683RawTerms
def rightRaw : List Term := Proof.Events002.exact673RawTerms
def group : MergeGroup := .operator 3683 673
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3683) (leftOrdinal := 0)
    (rightResult := 673) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17806⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17806⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3687

namespace LeftMerge3695
def owner : Owner := ⟨.program ⟨214⟩, ⟨15512⟩⟩
def mergeEvent : Nat := 3695
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3691RawTerms
def rightRaw : List Term := Proof.Events002.exact683RawTerms
def group : MergeGroup := .operator 3691 683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3691) (leftOrdinal := 0)
    (rightResult := 683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15511⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3695

namespace LeftMerge3703
def owner : Owner := ⟨.program ⟨214⟩, ⟨15204⟩⟩
def mergeEvent : Nat := 3703
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3699RawTerms
def rightRaw : List Term := Proof.Events002.exact693RawTerms
def group : MergeGroup := .operator 3699 693
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3699) (leftOrdinal := 0)
    (rightResult := 693) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15203⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3703

namespace LeftMerge3711
def owner : Owner := ⟨.program ⟨214⟩, ⟨15043⟩⟩
def mergeEvent : Nat := 3711
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3707RawTerms
def rightRaw : List Term := Proof.Events002.exact703RawTerms
def group : MergeGroup := .operator 3707 703
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3707) (leftOrdinal := 0)
    (rightResult := 703) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15042⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3711

namespace LeftMerge3719
def owner : Owner := ⟨.program ⟨214⟩, ⟨14882⟩⟩
def mergeEvent : Nat := 3719
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3715RawTerms
def rightRaw : List Term := Proof.Events002.exact713RawTerms
def group : MergeGroup := .operator 3715 713
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3715) (leftOrdinal := 0)
    (rightResult := 713) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14881⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3719

namespace LeftMerge3800
def owner : Owner := ⟨.program ⟨214⟩, ⟨18829⟩⟩
def mergeEvent : Nat := 3800
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 5)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6542⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], []⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge3800

namespace LeftMerge3801
def owner : Owner := ⟨.program ⟨214⟩, ⟨18829⟩⟩
def mergeEvent : Nat := 3801
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18120⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 7)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18120⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6542⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18120⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3801

namespace LeftMerge3802
def owner : Owner := ⟨.program ⟨214⟩, ⟨18829⟩⟩
def mergeEvent : Nat := 3802
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨16923⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 8)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16923⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6542⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨16923⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3802

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
