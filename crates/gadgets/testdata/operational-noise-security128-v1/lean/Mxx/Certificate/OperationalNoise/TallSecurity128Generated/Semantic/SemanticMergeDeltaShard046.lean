import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge14875
def owner : Owner := ⟨.program ⟨257⟩, ⟨56936⟩⟩
def mergeEvent : Nat := 14875
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56935⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events058.exact14871RawTerms
def rightRaw : List Term := Proof.Events002.exact653RawTerms
def group : MergeGroup := .operator 14871 653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14871) (leftOrdinal := 0)
    (rightResult := 653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56935⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56935⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14875

namespace LeftMerge14883
def owner : Owner := ⟨.program ⟨257⟩, ⟨53956⟩⟩
def mergeEvent : Nat := 14883
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53955⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events058.exact14879RawTerms
def rightRaw : List Term := Proof.Events002.exact663RawTerms
def group : MergeGroup := .operator 14879 663
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14879) (leftOrdinal := 0)
    (rightResult := 663) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53955⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53955⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14883

namespace LeftMerge14891
def owner : Owner := ⟨.program ⟨257⟩, ⟨50976⟩⟩
def mergeEvent : Nat := 14891
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50975⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events058.exact14887RawTerms
def rightRaw : List Term := Proof.Events002.exact673RawTerms
def group : MergeGroup := .operator 14887 673
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14887) (leftOrdinal := 0)
    (rightResult := 673) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50975⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50975⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14891

namespace LeftMerge14899
def owner : Owner := ⟨.program ⟨257⟩, ⟨31912⟩⟩
def mergeEvent : Nat := 14899
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31911⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events058.exact14895RawTerms
def rightRaw : List Term := Proof.Events002.exact683RawTerms
def group : MergeGroup := .operator 14895 683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14895) (leftOrdinal := 0)
    (rightResult := 683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31911⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31911⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14899

namespace LeftMerge14907
def owner : Owner := ⟨.program ⟨257⟩, ⟨21892⟩⟩
def mergeEvent : Nat := 14907
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21891⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events058.exact14903RawTerms
def rightRaw : List Term := Proof.Events002.exact693RawTerms
def group : MergeGroup := .operator 14903 693
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14903) (leftOrdinal := 0)
    (rightResult := 693) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21891⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21891⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14907

namespace LeftMerge14915
def owner : Owner := ⟨.program ⟨257⟩, ⟨18672⟩⟩
def mergeEvent : Nat := 14915
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18671⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events058.exact14911RawTerms
def rightRaw : List Term := Proof.Events002.exact703RawTerms
def group : MergeGroup := .operator 14911 703
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14911) (leftOrdinal := 0)
    (rightResult := 703) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18671⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18671⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14915

namespace LeftMerge14923
def owner : Owner := ⟨.program ⟨257⟩, ⟨15871⟩⟩
def mergeEvent : Nat := 14923
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15870⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events058.exact14919RawTerms
def rightRaw : List Term := Proof.Events002.exact713RawTerms
def group : MergeGroup := .operator 14919 713
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14919) (leftOrdinal := 0)
    (rightResult := 713) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15870⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15870⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14923

namespace LeftMerge15004
def owner : Owner := ⟨.program ⟨257⟩, ⟨67275⟩⟩
def mergeEvent : Nat := 15004
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6742⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67271⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events058.exact15000RawTerms
def rightRaw : List Term := Proof.Events055.exact14287RawTerms
def group : MergeGroup := .operator 15000 14287
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15000) (leftOrdinal := 5)
    (rightResult := 14287) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67271⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6742⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6742⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67271⟩⟩], []⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge15004

namespace LeftMerge15005
def owner : Owner := ⟨.program ⟨257⟩, ⟨67275⟩⟩
def mergeEvent : Nat := 15005
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6742⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48229⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events058.exact15000RawTerms
def rightRaw : List Term := Proof.Events055.exact14287RawTerms
def group : MergeGroup := .operator 15000 14287
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15000) (leftOrdinal := 7)
    (rightResult := 14287) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48229⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6742⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6742⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48229⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge15005

namespace LeftMerge15006
def owner : Owner := ⟨.program ⟨257⟩, ⟨67275⟩⟩
def mergeEvent : Nat := 15006
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6742⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45549⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events058.exact15000RawTerms
def rightRaw : List Term := Proof.Events055.exact14287RawTerms
def group : MergeGroup := .operator 15000 14287
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15000) (leftOrdinal := 8)
    (rightResult := 14287) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45549⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6742⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6742⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45549⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge15006

namespace LeftMerge15007
def owner : Owner := ⟨.program ⟨257⟩, ⟨67275⟩⟩
def mergeEvent : Nat := 15007
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6742⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42872⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events058.exact15000RawTerms
def rightRaw : List Term := Proof.Events055.exact14287RawTerms
def group : MergeGroup := .operator 15000 14287
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15000) (leftOrdinal := 9)
    (rightResult := 14287) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42872⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6742⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6742⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42872⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge15007

namespace LeftMerge15008
def owner : Owner := ⟨.program ⟨257⟩, ⟨67275⟩⟩
def mergeEvent : Nat := 15008
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6742⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40192⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events058.exact15000RawTerms
def rightRaw : List Term := Proof.Events055.exact14287RawTerms
def group : MergeGroup := .operator 15000 14287
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15000) (leftOrdinal := 11)
    (rightResult := 14287) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40192⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6742⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6742⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40192⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge15008

namespace LeftMerge15009
def owner : Owner := ⟨.program ⟨257⟩, ⟨67275⟩⟩
def mergeEvent : Nat := 15009
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6742⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37509⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events058.exact15000RawTerms
def rightRaw : List Term := Proof.Events055.exact14287RawTerms
def group : MergeGroup := .operator 15000 14287
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15000) (leftOrdinal := 12)
    (rightResult := 14287) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37509⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6742⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6742⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37509⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge15009

namespace LeftMerge15010
def owner : Owner := ⟨.program ⟨257⟩, ⟨67275⟩⟩
def mergeEvent : Nat := 15010
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6742⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34829⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events058.exact15000RawTerms
def rightRaw : List Term := Proof.Events055.exact14287RawTerms
def group : MergeGroup := .operator 15000 14287
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15000) (leftOrdinal := 13)
    (rightResult := 14287) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34829⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6742⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6742⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34829⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge15010

namespace LeftMerge15011
def owner : Owner := ⟨.program ⟨257⟩, ⟨67275⟩⟩
def mergeEvent : Nat := 15011
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6742⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29172⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events058.exact15000RawTerms
def rightRaw : List Term := Proof.Events055.exact14287RawTerms
def group : MergeGroup := .operator 15000 14287
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15000) (leftOrdinal := 15)
    (rightResult := 14287) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29172⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6742⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6742⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29172⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge15011

namespace LeftMerge15012
def owner : Owner := ⟨.program ⟨257⟩, ⟨67275⟩⟩
def mergeEvent : Nat := 15012
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6742⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26492⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events058.exact15000RawTerms
def rightRaw : List Term := Proof.Events055.exact14287RawTerms
def group : MergeGroup := .operator 15000 14287
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15000) (leftOrdinal := 16)
    (rightResult := 14287) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26492⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6742⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6742⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26492⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge15012

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
