import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge12029
def owner : Owner := ⟨.program ⟨257⟩, ⟨67421⟩⟩
def mergeEvent : Nat := 12029
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48333⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events046.exact12024RawTerms
def rightRaw : List Term := Proof.Events044.exact11301RawTerms
def group : MergeGroup := .operator 12024 11301
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12024) (leftOrdinal := 7)
    (rightResult := 11301) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48333⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48333⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12029

namespace LeftMerge12030
def owner : Owner := ⟨.program ⟨257⟩, ⟨67421⟩⟩
def mergeEvent : Nat := 12030
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45653⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events046.exact12024RawTerms
def rightRaw : List Term := Proof.Events044.exact11301RawTerms
def group : MergeGroup := .operator 12024 11301
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12024) (leftOrdinal := 8)
    (rightResult := 11301) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45653⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45653⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12030

namespace LeftMerge12031
def owner : Owner := ⟨.program ⟨257⟩, ⟨67421⟩⟩
def mergeEvent : Nat := 12031
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42976⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events046.exact12024RawTerms
def rightRaw : List Term := Proof.Events044.exact11301RawTerms
def group : MergeGroup := .operator 12024 11301
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12024) (leftOrdinal := 9)
    (rightResult := 11301) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42976⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42976⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12031

namespace LeftMerge12032
def owner : Owner := ⟨.program ⟨257⟩, ⟨67421⟩⟩
def mergeEvent : Nat := 12032
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40296⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events046.exact12024RawTerms
def rightRaw : List Term := Proof.Events044.exact11301RawTerms
def group : MergeGroup := .operator 12024 11301
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12024) (leftOrdinal := 11)
    (rightResult := 11301) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40296⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40296⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12032

namespace LeftMerge12033
def owner : Owner := ⟨.program ⟨257⟩, ⟨67421⟩⟩
def mergeEvent : Nat := 12033
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37613⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events046.exact12024RawTerms
def rightRaw : List Term := Proof.Events044.exact11301RawTerms
def group : MergeGroup := .operator 12024 11301
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12024) (leftOrdinal := 12)
    (rightResult := 11301) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37613⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37613⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12033

namespace LeftMerge12034
def owner : Owner := ⟨.program ⟨257⟩, ⟨67421⟩⟩
def mergeEvent : Nat := 12034
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34933⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events046.exact12024RawTerms
def rightRaw : List Term := Proof.Events044.exact11301RawTerms
def group : MergeGroup := .operator 12024 11301
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12024) (leftOrdinal := 13)
    (rightResult := 11301) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34933⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34933⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12034

namespace LeftMerge12035
def owner : Owner := ⟨.program ⟨257⟩, ⟨67421⟩⟩
def mergeEvent : Nat := 12035
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29276⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events046.exact12024RawTerms
def rightRaw : List Term := Proof.Events044.exact11301RawTerms
def group : MergeGroup := .operator 12024 11301
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12024) (leftOrdinal := 15)
    (rightResult := 11301) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29276⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29276⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12035

namespace LeftMerge12036
def owner : Owner := ⟨.program ⟨257⟩, ⟨67421⟩⟩
def mergeEvent : Nat := 12036
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26596⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events046.exact12024RawTerms
def rightRaw : List Term := Proof.Events044.exact11301RawTerms
def group : MergeGroup := .operator 12024 11301
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12024) (leftOrdinal := 16)
    (rightResult := 11301) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26596⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26596⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12036

namespace LeftMerge12037
def owner : Owner := ⟨.program ⟨257⟩, ⟨67421⟩⟩
def mergeEvent : Nat := 12037
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66448⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events046.exact12024RawTerms
def rightRaw : List Term := Proof.Events044.exact11301RawTerms
def group : MergeGroup := .operator 12024 11301
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12024) (leftOrdinal := 18)
    (rightResult := 11301) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66448⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66448⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12037

namespace LeftMerge12038
def owner : Owner := ⟨.program ⟨257⟩, ⟨67421⟩⟩
def mergeEvent : Nat := 12038
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨63047⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events046.exact12024RawTerms
def rightRaw : List Term := Proof.Events044.exact11301RawTerms
def group : MergeGroup := .operator 12024 11301
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12024) (leftOrdinal := 0)
    (rightResult := 11301) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63047⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨63047⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12038

namespace LeftMerge12039
def owner : Owner := ⟨.program ⟨257⟩, ⟨67421⟩⟩
def mergeEvent : Nat := 12039
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨60067⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events046.exact12024RawTerms
def rightRaw : List Term := Proof.Events044.exact11301RawTerms
def group : MergeGroup := .operator 12024 11301
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12024) (leftOrdinal := 1)
    (rightResult := 11301) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60067⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨60067⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12039

namespace LeftMerge12040
def owner : Owner := ⟨.program ⟨257⟩, ⟨67421⟩⟩
def mergeEvent : Nat := 12040
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨57087⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events046.exact12024RawTerms
def rightRaw : List Term := Proof.Events044.exact11301RawTerms
def group : MergeGroup := .operator 12024 11301
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12024) (leftOrdinal := 2)
    (rightResult := 11301) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57087⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨57087⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12040

namespace LeftMerge12041
def owner : Owner := ⟨.program ⟨257⟩, ⟨67421⟩⟩
def mergeEvent : Nat := 12041
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨54107⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events046.exact12024RawTerms
def rightRaw : List Term := Proof.Events044.exact11301RawTerms
def group : MergeGroup := .operator 12024 11301
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12024) (leftOrdinal := 3)
    (rightResult := 11301) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54107⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨54107⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12041

namespace LeftMerge12042
def owner : Owner := ⟨.program ⟨257⟩, ⟨67421⟩⟩
def mergeEvent : Nat := 12042
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨51127⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events046.exact12024RawTerms
def rightRaw : List Term := Proof.Events044.exact11301RawTerms
def group : MergeGroup := .operator 12024 11301
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12024) (leftOrdinal := 4)
    (rightResult := 11301) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51127⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨51127⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12042

namespace LeftMerge12043
def owner : Owner := ⟨.program ⟨257⟩, ⟨67421⟩⟩
def mergeEvent : Nat := 12043
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32063⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events046.exact12024RawTerms
def rightRaw : List Term := Proof.Events044.exact11301RawTerms
def group : MergeGroup := .operator 12024 11301
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12024) (leftOrdinal := 6)
    (rightResult := 11301) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32063⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32063⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12043

namespace LeftMerge12044
def owner : Owner := ⟨.program ⟨257⟩, ⟨67421⟩⟩
def mergeEvent : Nat := 12044
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22043⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events046.exact12024RawTerms
def rightRaw : List Term := Proof.Events044.exact11301RawTerms
def group : MergeGroup := .operator 12024 11301
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12024) (leftOrdinal := 10)
    (rightResult := 11301) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22043⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22043⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12044

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
