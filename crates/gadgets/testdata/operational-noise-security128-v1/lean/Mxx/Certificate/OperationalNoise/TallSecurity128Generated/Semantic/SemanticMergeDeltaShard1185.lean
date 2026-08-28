import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge192845
def owner : Owner := ⟨.program ⟨257⟩, ⟨67500⟩⟩
def mergeEvent : Nat := 192845
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192788RawTerms
def rightRaw : List Term := Proof.Events038.exact9780RawTerms
def group : MergeGroup := .operator 192788 9780
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192788) (leftOrdinal := 0)
    (rightResult := 9780) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge192845

namespace LeftMerge192846
def owner : Owner := ⟨.program ⟨257⟩, ⟨67500⟩⟩
def mergeEvent : Nat := 192846
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192788RawTerms
def rightRaw : List Term := Proof.Events038.exact9780RawTerms
def group : MergeGroup := .operator 192788 9780
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192788) (leftOrdinal := 0)
    (rightResult := 9780) (rightOrdinal := 10) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge192846

namespace LeftMerge192847
def owner : Owner := ⟨.program ⟨257⟩, ⟨67500⟩⟩
def mergeEvent : Nat := 192847
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192788RawTerms
def rightRaw : List Term := Proof.Events038.exact9780RawTerms
def group : MergeGroup := .operator 192788 9780
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192788) (leftOrdinal := 0)
    (rightResult := 9780) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge192847

namespace LeftMerge192848
def owner : Owner := ⟨.program ⟨257⟩, ⟨67500⟩⟩
def mergeEvent : Nat := 192848
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192788RawTerms
def rightRaw : List Term := Proof.Events038.exact9780RawTerms
def group : MergeGroup := .operator 192788 9780
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192788) (leftOrdinal := 0)
    (rightResult := 9780) (rightOrdinal := 17) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge192848

namespace LeftMerge192849
def owner : Owner := ⟨.program ⟨257⟩, ⟨67500⟩⟩
def mergeEvent : Nat := 192849
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67494⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192788RawTerms
def rightRaw : List Term := Proof.Events038.exact9780RawTerms
def group : MergeGroup := .operator 192788 9780
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192788) (leftOrdinal := 1)
    (rightResult := 9780) (rightOrdinal := 5) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67494⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67494⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge192849

namespace LeftMerge192850
def owner : Owner := ⟨.program ⟨257⟩, ⟨67500⟩⟩
def mergeEvent : Nat := 192850
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48385⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192788RawTerms
def rightRaw : List Term := Proof.Events038.exact9780RawTerms
def group : MergeGroup := .operator 192788 9780
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192788) (leftOrdinal := 1)
    (rightResult := 9780) (rightOrdinal := 7) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48385⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48385⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192850

namespace LeftMerge192851
def owner : Owner := ⟨.program ⟨257⟩, ⟨67500⟩⟩
def mergeEvent : Nat := 192851
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45705⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192788RawTerms
def rightRaw : List Term := Proof.Events038.exact9780RawTerms
def group : MergeGroup := .operator 192788 9780
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192788) (leftOrdinal := 1)
    (rightResult := 9780) (rightOrdinal := 8) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45705⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45705⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192851

namespace LeftMerge192852
def owner : Owner := ⟨.program ⟨257⟩, ⟨67500⟩⟩
def mergeEvent : Nat := 192852
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192788RawTerms
def rightRaw : List Term := Proof.Events038.exact9780RawTerms
def group : MergeGroup := .operator 192788 9780
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192788) (leftOrdinal := 1)
    (rightResult := 9780) (rightOrdinal := 9) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43028⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43028⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192852

namespace LeftMerge192853
def owner : Owner := ⟨.program ⟨257⟩, ⟨67500⟩⟩
def mergeEvent : Nat := 192853
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192788RawTerms
def rightRaw : List Term := Proof.Events038.exact9780RawTerms
def group : MergeGroup := .operator 192788 9780
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192788) (leftOrdinal := 1)
    (rightResult := 9780) (rightOrdinal := 11) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40348⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40348⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192853

namespace LeftMerge192854
def owner : Owner := ⟨.program ⟨257⟩, ⟨67500⟩⟩
def mergeEvent : Nat := 192854
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192788RawTerms
def rightRaw : List Term := Proof.Events038.exact9780RawTerms
def group : MergeGroup := .operator 192788 9780
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192788) (leftOrdinal := 1)
    (rightResult := 9780) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192854

namespace LeftMerge192855
def owner : Owner := ⟨.program ⟨257⟩, ⟨67500⟩⟩
def mergeEvent : Nat := 192855
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34985⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192788RawTerms
def rightRaw : List Term := Proof.Events038.exact9780RawTerms
def group : MergeGroup := .operator 192788 9780
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192788) (leftOrdinal := 1)
    (rightResult := 9780) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34985⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34985⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192855

namespace LeftMerge192856
def owner : Owner := ⟨.program ⟨257⟩, ⟨67500⟩⟩
def mergeEvent : Nat := 192856
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192788RawTerms
def rightRaw : List Term := Proof.Events038.exact9780RawTerms
def group : MergeGroup := .operator 192788 9780
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192788) (leftOrdinal := 1)
    (rightResult := 9780) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192856

namespace LeftMerge192857
def owner : Owner := ⟨.program ⟨257⟩, ⟨67500⟩⟩
def mergeEvent : Nat := 192857
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26648⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192788RawTerms
def rightRaw : List Term := Proof.Events038.exact9780RawTerms
def group : MergeGroup := .operator 192788 9780
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192788) (leftOrdinal := 1)
    (rightResult := 9780) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26648⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26648⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192857

namespace LeftMerge192858
def owner : Owner := ⟨.program ⟨257⟩, ⟨67500⟩⟩
def mergeEvent : Nat := 192858
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192788RawTerms
def rightRaw : List Term := Proof.Events038.exact9780RawTerms
def group : MergeGroup := .operator 192788 9780
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192788) (leftOrdinal := 1)
    (rightResult := 9780) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192858

namespace LeftMerge192859
def owner : Owner := ⟨.program ⟨257⟩, ⟨67500⟩⟩
def mergeEvent : Nat := 192859
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192788RawTerms
def rightRaw : List Term := Proof.Events038.exact9780RawTerms
def group : MergeGroup := .operator 192788 9780
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192788) (leftOrdinal := 1)
    (rightResult := 9780) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192859

namespace LeftMerge192860
def owner : Owner := ⟨.program ⟨257⟩, ⟨67500⟩⟩
def mergeEvent : Nat := 192860
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60143⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192788RawTerms
def rightRaw : List Term := Proof.Events038.exact9780RawTerms
def group : MergeGroup := .operator 192788 9780
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192788) (leftOrdinal := 1)
    (rightResult := 9780) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60143⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60143⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192860

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
