import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge3803
def owner : Owner := ⟨.program ⟨214⟩, ⟨18829⟩⟩
def mergeEvent : Nat := 3803
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 9)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6542⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3803

namespace LeftMerge3804
def owner : Owner := ⟨.program ⟨214⟩, ⟨18829⟩⟩
def mergeEvent : Nat := 3804
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 11)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6542⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3804

namespace LeftMerge3805
def owner : Owner := ⟨.program ⟨214⟩, ⟨18829⟩⟩
def mergeEvent : Nat := 3805
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17945⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 12)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17945⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6542⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17945⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3805

namespace LeftMerge3806
def owner : Owner := ⟨.program ⟨214⟩, ⟨18829⟩⟩
def mergeEvent : Nat := 3806
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 13)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6542⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3806

namespace LeftMerge3807
def owner : Owner := ⟨.program ⟨214⟩, ⟨18829⟩⟩
def mergeEvent : Nat := 3807
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18818⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 15)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18818⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6542⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18818⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3807

namespace LeftMerge3808
def owner : Owner := ⟨.program ⟨214⟩, ⟨18829⟩⟩
def mergeEvent : Nat := 3808
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17602⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 16)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17602⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6542⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17602⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3808

namespace LeftMerge3809
def owner : Owner := ⟨.program ⟨214⟩, ⟨18829⟩⟩
def mergeEvent : Nat := 3809
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 18)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6542⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3809

namespace LeftMerge3810
def owner : Owner := ⟨.program ⟨214⟩, ⟨18829⟩⟩
def mergeEvent : Nat := 3810
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 0)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6542⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3810

namespace LeftMerge3811
def owner : Owner := ⟨.program ⟨214⟩, ⟨18829⟩⟩
def mergeEvent : Nat := 3811
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 1)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6542⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3811

namespace LeftMerge3812
def owner : Owner := ⟨.program ⟨214⟩, ⟨18829⟩⟩
def mergeEvent : Nat := 3812
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17217⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 2)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17217⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6542⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17217⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3812

namespace LeftMerge3813
def owner : Owner := ⟨.program ⟨214⟩, ⟨18829⟩⟩
def mergeEvent : Nat := 3813
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17434⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 3)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17434⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6542⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17434⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3813

namespace LeftMerge3814
def owner : Owner := ⟨.program ⟨214⟩, ⟨18829⟩⟩
def mergeEvent : Nat := 3814
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17806⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 4)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17806⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6542⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17806⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3814

namespace LeftMerge3815
def owner : Owner := ⟨.program ⟨214⟩, ⟨18829⟩⟩
def mergeEvent : Nat := 3815
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 6)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6542⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3815

namespace LeftMerge3816
def owner : Owner := ⟨.program ⟨214⟩, ⟨18829⟩⟩
def mergeEvent : Nat := 3816
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 10)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6542⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3816

namespace LeftMerge3817
def owner : Owner := ⟨.program ⟨214⟩, ⟨18829⟩⟩
def mergeEvent : Nat := 3817
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 14)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6542⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3817

namespace LeftMerge3818
def owner : Owner := ⟨.program ⟨214⟩, ⟨18829⟩⟩
def mergeEvent : Nat := 3818
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 17)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6542⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3818

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
