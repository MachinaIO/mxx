import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge14281
def owner : Owner := ⟨.program ⟨257⟩, ⟨67345⟩⟩
def mergeEvent : Nat := 14281
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events052.exact13545RawTerms
def group : MergeGroup := .operator 14262 13545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 6)
    (rightResult := 13545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14281

namespace LeftMerge14282
def owner : Owner := ⟨.program ⟨257⟩, ⟨67345⟩⟩
def mergeEvent : Nat := 14282
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events052.exact13545RawTerms
def group : MergeGroup := .operator 14262 13545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 10)
    (rightResult := 13545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14282

namespace LeftMerge14283
def owner : Owner := ⟨.program ⟨257⟩, ⟨67345⟩⟩
def mergeEvent : Nat := 14283
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events052.exact13545RawTerms
def group : MergeGroup := .operator 14262 13545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 14)
    (rightResult := 13545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14283

namespace LeftMerge14284
def owner : Owner := ⟨.program ⟨257⟩, ⟨67345⟩⟩
def mergeEvent : Nat := 14284
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events052.exact13545RawTerms
def group : MergeGroup := .operator 14262 13545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 17)
    (rightResult := 13545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14284

namespace LeftMerge14779
def owner : Owner := ⟨.program ⟨257⟩, ⟨67272⟩⟩
def mergeEvent : Nat := 14779
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67271⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events057.exact14775RawTerms
def rightRaw : List Term := Proof.Events000.exact36RawTerms
def group : MergeGroup := .operator 14775 36
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14775) (leftOrdinal := 0)
    (rightResult := 36) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67271⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67271⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14779

namespace LeftMerge14787
def owner : Owner := ⟨.program ⟨257⟩, ⟨48230⟩⟩
def mergeEvent : Nat := 14787
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48229⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events057.exact14783RawTerms
def rightRaw : List Term := Proof.Events002.exact543RawTerms
def group : MergeGroup := .operator 14783 543
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14783) (leftOrdinal := 0)
    (rightResult := 543) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48229⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48229⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14787

namespace LeftMerge14795
def owner : Owner := ⟨.program ⟨257⟩, ⟨45550⟩⟩
def mergeEvent : Nat := 14795
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45549⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events057.exact14791RawTerms
def rightRaw : List Term := Proof.Events002.exact553RawTerms
def group : MergeGroup := .operator 14791 553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14791) (leftOrdinal := 0)
    (rightResult := 553) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45549⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45549⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14795

namespace LeftMerge14803
def owner : Owner := ⟨.program ⟨257⟩, ⟨42873⟩⟩
def mergeEvent : Nat := 14803
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42872⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events057.exact14799RawTerms
def rightRaw : List Term := Proof.Events002.exact563RawTerms
def group : MergeGroup := .operator 14799 563
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14799) (leftOrdinal := 0)
    (rightResult := 563) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42872⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42872⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14803

namespace LeftMerge14811
def owner : Owner := ⟨.program ⟨257⟩, ⟨40193⟩⟩
def mergeEvent : Nat := 14811
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40192⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events057.exact14807RawTerms
def rightRaw : List Term := Proof.Events002.exact573RawTerms
def group : MergeGroup := .operator 14807 573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14807) (leftOrdinal := 0)
    (rightResult := 573) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40192⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40192⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14811

namespace LeftMerge14819
def owner : Owner := ⟨.program ⟨257⟩, ⟨37510⟩⟩
def mergeEvent : Nat := 14819
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37509⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events057.exact14815RawTerms
def rightRaw : List Term := Proof.Events002.exact583RawTerms
def group : MergeGroup := .operator 14815 583
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14815) (leftOrdinal := 0)
    (rightResult := 583) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37509⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37509⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14819

namespace LeftMerge14827
def owner : Owner := ⟨.program ⟨257⟩, ⟨34830⟩⟩
def mergeEvent : Nat := 14827
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34829⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events057.exact14823RawTerms
def rightRaw : List Term := Proof.Events002.exact593RawTerms
def group : MergeGroup := .operator 14823 593
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14823) (leftOrdinal := 0)
    (rightResult := 593) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34829⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34829⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14827

namespace LeftMerge14835
def owner : Owner := ⟨.program ⟨257⟩, ⟨29173⟩⟩
def mergeEvent : Nat := 14835
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29172⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events057.exact14831RawTerms
def rightRaw : List Term := Proof.Events002.exact603RawTerms
def group : MergeGroup := .operator 14831 603
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14831) (leftOrdinal := 0)
    (rightResult := 603) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29172⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29172⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14835

namespace LeftMerge14843
def owner : Owner := ⟨.program ⟨257⟩, ⟨26493⟩⟩
def mergeEvent : Nat := 14843
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26492⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events057.exact14839RawTerms
def rightRaw : List Term := Proof.Events002.exact613RawTerms
def group : MergeGroup := .operator 14839 613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14839) (leftOrdinal := 0)
    (rightResult := 613) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26492⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26492⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14843

namespace LeftMerge14851
def owner : Owner := ⟨.program ⟨257⟩, ⟨65889⟩⟩
def mergeEvent : Nat := 14851
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65888⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events057.exact14847RawTerms
def rightRaw : List Term := Proof.Events002.exact623RawTerms
def group : MergeGroup := .operator 14847 623
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14847) (leftOrdinal := 0)
    (rightResult := 623) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65888⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65888⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14851

namespace LeftMerge14859
def owner : Owner := ⟨.program ⟨257⟩, ⟨62896⟩⟩
def mergeEvent : Nat := 14859
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62895⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events058.exact14855RawTerms
def rightRaw : List Term := Proof.Events002.exact633RawTerms
def group : MergeGroup := .operator 14855 633
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14855) (leftOrdinal := 0)
    (rightResult := 633) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62895⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62895⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14859

namespace LeftMerge14867
def owner : Owner := ⟨.program ⟨257⟩, ⟨59916⟩⟩
def mergeEvent : Nat := 14867
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59915⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events058.exact14863RawTerms
def rightRaw : List Term := Proof.Events002.exact643RawTerms
def group : MergeGroup := .operator 14863 643
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14863) (leftOrdinal := 0)
    (rightResult := 643) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59915⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59915⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14867

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
