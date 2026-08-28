import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge9787
def owner : Owner := ⟨.program ⟨257⟩, ⟨67498⟩⟩
def mergeEvent : Nat := 9787
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨43028⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events038.exact9780RawTerms
def rightRaw : List Term := Proof.Events035.exact9057RawTerms
def group : MergeGroup := .operator 9780 9057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9780) (leftOrdinal := 9)
    (rightResult := 9057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43028⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6907⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨43028⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9787

namespace LeftMerge9788
def owner : Owner := ⟨.program ⟨257⟩, ⟨67498⟩⟩
def mergeEvent : Nat := 9788
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨40348⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events038.exact9780RawTerms
def rightRaw : List Term := Proof.Events035.exact9057RawTerms
def group : MergeGroup := .operator 9780 9057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9780) (leftOrdinal := 11)
    (rightResult := 9057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40348⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6907⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨40348⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9788

namespace LeftMerge9789
def owner : Owner := ⟨.program ⟨257⟩, ⟨67498⟩⟩
def mergeEvent : Nat := 9789
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events038.exact9780RawTerms
def rightRaw : List Term := Proof.Events035.exact9057RawTerms
def group : MergeGroup := .operator 9780 9057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9780) (leftOrdinal := 12)
    (rightResult := 9057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6907⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9789

namespace LeftMerge9790
def owner : Owner := ⟨.program ⟨257⟩, ⟨67498⟩⟩
def mergeEvent : Nat := 9790
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨34985⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events038.exact9780RawTerms
def rightRaw : List Term := Proof.Events035.exact9057RawTerms
def group : MergeGroup := .operator 9780 9057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9780) (leftOrdinal := 13)
    (rightResult := 9057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34985⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6907⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨34985⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9790

namespace LeftMerge9791
def owner : Owner := ⟨.program ⟨257⟩, ⟨67498⟩⟩
def mergeEvent : Nat := 9791
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events038.exact9780RawTerms
def rightRaw : List Term := Proof.Events035.exact9057RawTerms
def group : MergeGroup := .operator 9780 9057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9780) (leftOrdinal := 15)
    (rightResult := 9057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6907⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9791

namespace LeftMerge9792
def owner : Owner := ⟨.program ⟨257⟩, ⟨67498⟩⟩
def mergeEvent : Nat := 9792
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨26648⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events038.exact9780RawTerms
def rightRaw : List Term := Proof.Events035.exact9057RawTerms
def group : MergeGroup := .operator 9780 9057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9780) (leftOrdinal := 16)
    (rightResult := 9057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26648⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6907⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨26648⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9792

namespace LeftMerge9793
def owner : Owner := ⟨.program ⟨257⟩, ⟨67498⟩⟩
def mergeEvent : Nat := 9793
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events038.exact9780RawTerms
def rightRaw : List Term := Proof.Events035.exact9057RawTerms
def group : MergeGroup := .operator 9780 9057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9780) (leftOrdinal := 18)
    (rightResult := 9057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6907⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9793

namespace LeftMerge9794
def owner : Owner := ⟨.program ⟨257⟩, ⟨67498⟩⟩
def mergeEvent : Nat := 9794
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events038.exact9780RawTerms
def rightRaw : List Term := Proof.Events035.exact9057RawTerms
def group : MergeGroup := .operator 9780 9057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9780) (leftOrdinal := 0)
    (rightResult := 9057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6907⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9794

namespace LeftMerge9795
def owner : Owner := ⟨.program ⟨257⟩, ⟨67498⟩⟩
def mergeEvent : Nat := 9795
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨60143⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events038.exact9780RawTerms
def rightRaw : List Term := Proof.Events035.exact9057RawTerms
def group : MergeGroup := .operator 9780 9057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9780) (leftOrdinal := 1)
    (rightResult := 9057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60143⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6907⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨60143⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9795

namespace LeftMerge9796
def owner : Owner := ⟨.program ⟨257⟩, ⟨67498⟩⟩
def mergeEvent : Nat := 9796
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨57163⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events038.exact9780RawTerms
def rightRaw : List Term := Proof.Events035.exact9057RawTerms
def group : MergeGroup := .operator 9780 9057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9780) (leftOrdinal := 2)
    (rightResult := 9057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57163⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6907⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨57163⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9796

namespace LeftMerge9797
def owner : Owner := ⟨.program ⟨257⟩, ⟨67498⟩⟩
def mergeEvent : Nat := 9797
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨54183⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events038.exact9780RawTerms
def rightRaw : List Term := Proof.Events035.exact9057RawTerms
def group : MergeGroup := .operator 9780 9057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9780) (leftOrdinal := 3)
    (rightResult := 9057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54183⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6907⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨54183⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9797

namespace LeftMerge9798
def owner : Owner := ⟨.program ⟨257⟩, ⟨67498⟩⟩
def mergeEvent : Nat := 9798
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events038.exact9780RawTerms
def rightRaw : List Term := Proof.Events035.exact9057RawTerms
def group : MergeGroup := .operator 9780 9057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9780) (leftOrdinal := 4)
    (rightResult := 9057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6907⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9798

namespace LeftMerge9799
def owner : Owner := ⟨.program ⟨257⟩, ⟨67498⟩⟩
def mergeEvent : Nat := 9799
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events038.exact9780RawTerms
def rightRaw : List Term := Proof.Events035.exact9057RawTerms
def group : MergeGroup := .operator 9780 9057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9780) (leftOrdinal := 6)
    (rightResult := 9057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6907⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9799

namespace LeftMerge9800
def owner : Owner := ⟨.program ⟨257⟩, ⟨67498⟩⟩
def mergeEvent : Nat := 9800
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events038.exact9780RawTerms
def rightRaw : List Term := Proof.Events035.exact9057RawTerms
def group : MergeGroup := .operator 9780 9057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9780) (leftOrdinal := 10)
    (rightResult := 9057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6907⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9800

namespace LeftMerge9801
def owner : Owner := ⟨.program ⟨257⟩, ⟨67498⟩⟩
def mergeEvent : Nat := 9801
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events038.exact9780RawTerms
def rightRaw : List Term := Proof.Events035.exact9057RawTerms
def group : MergeGroup := .operator 9780 9057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9780) (leftOrdinal := 14)
    (rightResult := 9057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6907⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9801

namespace LeftMerge9802
def owner : Owner := ⟨.program ⟨257⟩, ⟨67498⟩⟩
def mergeEvent : Nat := 9802
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events038.exact9780RawTerms
def rightRaw : List Term := Proof.Events035.exact9057RawTerms
def group : MergeGroup := .operator 9780 9057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9780) (leftOrdinal := 17)
    (rightResult := 9057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6907⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨6907⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge9802

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
